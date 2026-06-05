from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.triton.cross_entropy.backward import cross_entropy_backward, cross_entropy_large_backward


@triton.jit
def _cross_entropy_forward_kernel(
    logits_ptr,
    labels_ptr,
    losses_ptr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    label = tl.load(labels_ptr + row)
    valid = label != ignore_index

    row_max = tl.max(logits, axis=0)
    logsumexp = tl.log(tl.sum(tl.exp(logits - row_max), axis=0)) + row_max
    target = tl.load(logits_ptr + row * vocab_size + label, mask=valid, other=0.0).to(tl.float32)
    loss = logsumexp - target
    tl.store(losses_ptr + row, tl.where(valid, loss, 0.0))


class _TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        logits_2d = logits.contiguous().reshape(-1, vocab_size)
        labels_1d = labels.contiguous().reshape(-1)
        rows = logits_2d.shape[0]
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)

        if rows:
            block_size = triton.next_power_of_2(vocab_size)
            num_warps = 4 if block_size <= 2048 else 8
            _cross_entropy_forward_kernel[(rows,)](
                logits_2d,
                labels_1d,
                losses,
                vocab_size,
                ignore_index,
                block_size,
                num_warps=num_warps,
            )

        valid_count = labels_1d.ne(ignore_index).sum()
        loss = losses.sum() / valid_count.to(losses.dtype)
        ctx.save_for_backward(logits_2d, labels_1d, valid_count)
        ctx.ignore_index = ignore_index
        ctx.original_shape = original_shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        logits_2d, labels_1d, valid_count = ctx.saved_tensors
        dlogits = cross_entropy_backward(logits_2d, labels_1d, valid_count, grad_output, ctx.ignore_index)
        return dlogits.reshape(ctx.original_shape), None, None


@triton.jit
def _cross_entropy_large_max_kernel(
    logits_ptr,
    partial_max_ptr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    tl.store(partial_max_ptr + row * num_blocks + block, tl.max(logits, axis=0))


@triton.jit
def _cross_entropy_large_reduce_max_kernel(
    partial_max_ptr,
    row_max_ptr,
    num_blocks: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_blocks
    values = tl.load(partial_max_ptr + row * num_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    tl.store(row_max_ptr + row, tl.max(values, axis=0))


@triton.jit
def _cross_entropy_large_sum_kernel(
    logits_ptr,
    row_max_ptr,
    partial_sum_ptr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < vocab_size
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = tl.exp(logits - row_max)
    values = tl.where(mask, values, 0.0)
    tl.store(partial_sum_ptr + row * num_blocks + block, tl.sum(values, axis=0))


@triton.jit
def _cross_entropy_large_reduce_loss_kernel(
    logits_ptr,
    labels_ptr,
    row_max_ptr,
    partial_sum_ptr,
    row_sum_ptr,
    losses_ptr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    ignore_index: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_blocks
    partial = tl.load(partial_sum_ptr + row * num_blocks + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(partial, axis=0)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    label = tl.load(labels_ptr + row)
    valid = label != ignore_index
    target = tl.load(logits_ptr + row * vocab_size + label, mask=valid, other=0.0).to(tl.float32)
    loss = tl.log(row_sum) + row_max - target
    tl.store(row_sum_ptr + row, row_sum)
    tl.store(losses_ptr + row, tl.where(valid, loss, 0.0))


class _TritonCrossEntropyLarge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, ignore_index: int, block_size: int) -> Tensor:
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        logits_2d = logits.contiguous().reshape(-1, vocab_size)
        labels_1d = labels.contiguous().reshape(-1)
        rows = logits_2d.shape[0]
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        num_blocks = triton.cdiv(vocab_size, block_size)
        partial_max = torch.empty((rows, num_blocks), device=logits.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        row_max = torch.empty(rows, device=logits.device, dtype=torch.float32)
        row_sum = torch.empty(rows, device=logits.device, dtype=torch.float32)

        if rows:
            reduce_block = triton.next_power_of_2(num_blocks)
            _cross_entropy_large_max_kernel[(rows, num_blocks)](
                logits_2d,
                partial_max,
                vocab_size,
                num_blocks,
                block_size,
                num_warps=4,
            )
            _cross_entropy_large_reduce_max_kernel[(rows,)](
                partial_max,
                row_max,
                num_blocks,
                reduce_block,
                num_warps=1,
            )
            _cross_entropy_large_sum_kernel[(rows, num_blocks)](
                logits_2d,
                row_max,
                partial_sum,
                vocab_size,
                num_blocks,
                block_size,
                num_warps=4,
            )
            _cross_entropy_large_reduce_loss_kernel[(rows,)](
                logits_2d,
                labels_1d,
                row_max,
                partial_sum,
                row_sum,
                losses,
                vocab_size,
                num_blocks,
                ignore_index,
                reduce_block,
                num_warps=1,
            )

        valid_count = labels_1d.ne(ignore_index).sum()
        loss = losses.sum() / valid_count.to(losses.dtype)
        ctx.save_for_backward(logits_2d, labels_1d, row_max, row_sum, valid_count)
        ctx.ignore_index = ignore_index
        ctx.block_size = block_size
        ctx.original_shape = original_shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        logits_2d, labels_1d, row_max, row_sum, valid_count = ctx.saved_tensors
        dlogits = cross_entropy_large_backward(
            logits_2d,
            labels_1d,
            row_max,
            row_sum,
            valid_count,
            grad_output,
            ctx.ignore_index,
            ctx.block_size,
        )
        return dlogits.reshape(ctx.original_shape), None, None, None


def cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    if not logits.is_cuda:
        return ref_cross_entropy(logits, labels, ignore_index=ignore_index)
    if logits.shape[-1] > 8192:
        return _TritonCrossEntropyLarge.apply(logits, labels, ignore_index, 1024)
    return _TritonCrossEntropy.apply(logits, labels, ignore_index)
