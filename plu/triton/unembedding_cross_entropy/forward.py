from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _logits_tile(
    x_ptr,
    weight_ptr,
    row_offsets,
    vocab_offsets,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    k_offsets = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in tl.range(0, hidden_size, block_k):
        k = k_start + k_offsets
        x = tl.load(
            x_ptr + row_offsets[:, None] * hidden_size + k[None, :],
            mask=k[None, :] < hidden_size,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + vocab_offsets[None, :] * hidden_size + k[:, None],
            mask=(vocab_offsets[None, :] < vocab_size) & (k[:, None] < hidden_size),
            other=0.0,
        )
        acc += tl.dot(x, weight, input_precision="tf32")
    return acc


@triton.jit
def _uce_max_kernel(
    x_ptr,
    weight_ptr,
    labels_ptr,
    partial_max_ptr,
    partial_target_ptr,
    partial_sum_ptr,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    ignore_index: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    row_block = tl.program_id(0)
    vocab_block = tl.program_id(1)
    row_offsets = row_block * block_m + tl.arange(0, block_m)
    vocab_offsets = vocab_block * block_n + tl.arange(0, block_n)
    logits = _logits_tile(x_ptr, weight_ptr, row_offsets, vocab_offsets, hidden_size, vocab_size, block_m, block_n, block_k)
    row_mask = row_offsets < rows
    vocab_mask = vocab_offsets < vocab_size
    logits = tl.where(row_mask[:, None] & vocab_mask[None, :], logits, -float("inf"))
    labels = tl.load(labels_ptr + row_offsets, mask=row_mask, other=ignore_index)
    valid = labels != ignore_index
    tile_max = tl.max(logits, axis=1)
    tile_sum = tl.sum(tl.exp(logits - tile_max[:, None]), axis=1)
    target_logits = tl.max(tl.where(vocab_offsets[None, :] == labels[:, None], logits, -float("inf")), axis=1)
    tl.store(partial_max_ptr + row_offsets * num_vocab_blocks + vocab_block, tile_max, mask=row_mask)
    tl.store(partial_target_ptr + row_offsets * num_vocab_blocks + vocab_block, tl.where(valid, target_logits, -float("inf")), mask=row_mask)
    tl.store(partial_sum_ptr + row_offsets * num_vocab_blocks + vocab_block, tile_sum, mask=row_mask)


@triton.jit
def _uce_reduce_max_kernel(
    partial_max_ptr,
    row_max_ptr,
    rows: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_vocab_blocks
    max_values = tl.load(partial_max_ptr + row * num_vocab_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(max_values, axis=0)
    tl.store(row_max_ptr + row, row_max)


@triton.jit
def _uce_reduce_loss_kernel(
    partial_max_ptr,
    partial_target_ptr,
    partial_sum_ptr,
    labels_ptr,
    row_max_ptr,
    row_sum_ptr,
    losses_ptr,
    rows: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    ignore_index: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_vocab_blocks
    partial_max = tl.load(partial_max_ptr + row * num_vocab_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    target_values = tl.load(partial_target_ptr + row * num_vocab_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    target = tl.max(target_values, axis=0)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    sums = tl.load(partial_sum_ptr + row * num_vocab_blocks + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(sums * tl.exp(partial_max - row_max), axis=0)
    label = tl.load(labels_ptr + row, mask=row < rows, other=ignore_index)
    valid = label != ignore_index
    loss = tl.log(row_sum) + row_max - target
    tl.store(row_sum_ptr + row, row_sum)
    tl.store(losses_ptr + row, tl.where(valid, loss, 0.0))


@triton.jit
def _uce_dx_kernel(
    x_ptr,
    weight_ptr,
    labels_ptr,
    row_max_ptr,
    row_sum_ptr,
    valid_count_ptr,
    grad_output_ptr,
    dx_ptr,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    row_block = tl.program_id(0)
    hidden_block = tl.program_id(1)
    row_offsets = row_block * block_m + tl.arange(0, block_m)
    h_offsets = hidden_block * block_h + tl.arange(0, block_h)
    vocab_offsets_base = tl.arange(0, block_n)
    row_mask = row_offsets < rows
    acc = tl.zeros((block_m, block_h), dtype=tl.float32)
    for vocab_start in tl.range(0, vocab_size, block_n):
        vocab_offsets = vocab_start + vocab_offsets_base
        logits = _logits_tile(x_ptr, weight_ptr, row_offsets, vocab_offsets, hidden_size, vocab_size, block_m, block_n, block_k)
        row_max = tl.load(row_max_ptr + row_offsets, mask=row_mask, other=-float("inf")).to(tl.float32)
        row_sum = tl.load(row_sum_ptr + row_offsets, mask=row_mask, other=1.0).to(tl.float32)
        probs = tl.exp(logits - row_max[:, None]) / row_sum[:, None]
        probs = tl.where((vocab_offsets[None, :] < vocab_size) & row_mask[:, None], probs, 0.0)
        weight = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_size + h_offsets[None, :],
            mask=(vocab_offsets[:, None] < vocab_size) & (h_offsets[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(probs, weight, input_precision="tf32")
    labels = tl.load(labels_ptr + row_offsets, mask=row_mask, other=ignore_index)
    valid = labels != ignore_index
    target_weight = tl.load(
        weight_ptr + labels[:, None] * hidden_size + h_offsets[None, :],
        mask=valid[:, None] & (h_offsets[None, :] < hidden_size),
        other=0.0,
    ).to(tl.float32)
    valid_count = tl.load(valid_count_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    scale = grad_output / valid_count
    dx = (acc - target_weight) * scale
    dx = tl.where(valid[:, None], dx, 0.0)
    tl.store(
        dx_ptr + row_offsets[:, None] * hidden_size + h_offsets[None, :],
        dx,
        mask=row_mask[:, None] & (h_offsets[None, :] < hidden_size),
    )


@triton.jit
def _uce_dweight_kernel(
    x_ptr,
    weight_ptr,
    labels_ptr,
    row_max_ptr,
    row_sum_ptr,
    valid_count_ptr,
    grad_output_ptr,
    dweight_ptr,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    vocab_block = tl.program_id(0)
    hidden_block = tl.program_id(1)
    vocab_offsets = vocab_block * block_n + tl.arange(0, block_n)
    h_offsets = hidden_block * block_h + tl.arange(0, block_h)
    row_offsets_base = tl.arange(0, block_m)
    acc = tl.zeros((block_n, block_h), dtype=tl.float32)
    valid_count = tl.load(valid_count_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    scale = grad_output / valid_count
    for row_start in tl.range(0, rows, block_m):
        row_offsets = row_start + row_offsets_base
        row_mask = row_offsets < rows
        logits = _logits_tile(x_ptr, weight_ptr, row_offsets, vocab_offsets, hidden_size, vocab_size, block_m, block_n, block_k)
        row_max = tl.load(row_max_ptr + row_offsets, mask=row_mask, other=-float("inf")).to(tl.float32)
        row_sum = tl.load(row_sum_ptr + row_offsets, mask=row_mask, other=1.0).to(tl.float32)
        labels = tl.load(labels_ptr + row_offsets, mask=row_mask, other=ignore_index)
        valid = labels != ignore_index
        probs = tl.exp(logits - row_max[:, None]) / row_sum[:, None]
        target = vocab_offsets[None, :] == labels[:, None]
        dlogits = (probs - target.to(tl.float32)) * scale
        dlogits = tl.where(valid[:, None] & (vocab_offsets[None, :] < vocab_size), dlogits, 0.0)
        x = tl.load(
            x_ptr + row_offsets[:, None] * hidden_size + h_offsets[None, :],
            mask=row_mask[:, None] & (h_offsets[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(tl.trans(dlogits), x, input_precision="tf32")
    tl.store(
        dweight_ptr + vocab_offsets[:, None] * hidden_size + h_offsets[None, :],
        acc,
        mask=(vocab_offsets[:, None] < vocab_size) & (h_offsets[None, :] < hidden_size),
    )


class _UnembeddingCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: Tensor, weight: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        x_2d = hidden_states.contiguous().reshape(-1, hidden_size)
        labels_1d = labels.contiguous().reshape(-1)
        weight = weight.contiguous()
        rows = x_2d.shape[0]
        vocab_size = weight.shape[0]
        block_m = 16
        block_n = 128
        block_k = 64
        num_vocab_blocks = triton.cdiv(vocab_size, block_n)
        partial_max = torch.empty((rows, num_vocab_blocks), device=x_2d.device, dtype=torch.float32)
        partial_target = torch.empty_like(partial_max)
        partial_sum = torch.empty_like(partial_max)
        row_max = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
        row_sum = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
        losses = torch.empty(rows, device=x_2d.device, dtype=torch.float32)
        if rows:
            grid = (triton.cdiv(rows, block_m), num_vocab_blocks)
            _uce_max_kernel[grid](
                x_2d,
                weight,
                labels_1d,
                partial_max,
                partial_target,
                partial_sum,
                rows,
                hidden_size,
                vocab_size,
                num_vocab_blocks,
                ignore_index,
                block_m,
                block_n,
                block_k,
                num_warps=4,
            )
            reduce_block = triton.next_power_of_2(num_vocab_blocks)
            _uce_reduce_max_kernel[(rows,)](
                partial_max,
                row_max,
                rows,
                num_vocab_blocks,
                reduce_block,
                num_warps=8,
            )
            _uce_reduce_loss_kernel[(rows,)](
                partial_max,
                partial_target,
                partial_sum,
                labels_1d,
                row_max,
                row_sum,
                losses,
                rows,
                num_vocab_blocks,
                ignore_index,
                reduce_block,
                num_warps=8,
            )
        valid_count = labels_1d.ne(ignore_index).sum()
        loss = losses.sum() / valid_count.to(losses.dtype)
        ctx.save_for_backward(x_2d, weight, labels_1d, row_max, row_sum, valid_count)
        ctx.ignore_index = ignore_index
        ctx.original_shape = original_shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_2d, weight, labels_1d, row_max, row_sum, valid_count = ctx.saved_tensors
        rows, hidden_size = x_2d.shape
        vocab_size = weight.shape[0]
        dx = torch.empty_like(x_2d)
        dweight = torch.empty_like(weight)
        if rows:
            block_m = 16
            block_n = 128
            block_h = 64
            block_k = 64
            _uce_dx_kernel[(triton.cdiv(rows, block_m), triton.cdiv(hidden_size, block_h))](
                x_2d,
                weight,
                labels_1d,
                row_max,
                row_sum,
                valid_count,
                grad_output.contiguous(),
                dx,
                rows,
                hidden_size,
                vocab_size,
                ctx.ignore_index,
                block_m,
                block_n,
                block_h,
                block_k,
                num_warps=4,
            )
            _uce_dweight_kernel[(triton.cdiv(vocab_size, block_n), triton.cdiv(hidden_size, block_h))](
                x_2d,
                weight,
                labels_1d,
                row_max,
                row_sum,
                valid_count,
                grad_output.contiguous(),
                dweight,
                rows,
                hidden_size,
                vocab_size,
                ctx.ignore_index,
                block_m,
                block_n,
                block_h,
                block_k,
                num_warps=4,
            )
        return dx.reshape(ctx.original_shape), dweight, None, None


def unembedding_cross_entropy(hidden_states: Tensor, weight: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    if not hidden_states.is_cuda:
        raise RuntimeError("plu.triton.unembedding_cross_entropy requires CUDA tensors")
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(f"hidden_states prefix shape {hidden_states.shape[:-1]} must match labels shape {labels.shape}")
    if hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError(f"hidden size {hidden_states.shape[-1]} does not match weight input features {weight.shape[1]}")
    return _UnembeddingCrossEntropy.apply(hidden_states, weight, labels, ignore_index)
