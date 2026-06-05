from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _cross_entropy_backward_kernel(
    logits_ptr,
    labels_ptr,
    valid_count_ptr,
    grad_output_ptr,
    dlogits_ptr,
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
    exp_logits = tl.exp(logits - row_max)
    denom = tl.sum(exp_logits, axis=0)
    probs = exp_logits / denom
    target = offsets == label
    valid_count = tl.load(valid_count_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad = (probs - target.to(tl.float32)) * (grad_output / valid_count)
    grad = tl.where(valid, grad, 0.0)
    tl.store(dlogits_ptr + row * vocab_size + offsets, grad, mask=mask)


def cross_entropy_backward(
    logits_2d: Tensor,
    labels_1d: Tensor,
    valid_count: Tensor,
    grad_output: Tensor,
    ignore_index: int,
) -> Tensor:
    if logits_2d.ndim != 2:
        raise ValueError("logits_2d must be rank 2")
    if labels_1d.ndim != 1:
        raise ValueError("labels_1d must be rank 1")

    rows, vocab_size = logits_2d.shape
    dlogits = torch.empty_like(logits_2d)
    if rows == 0:
        return dlogits

    block_size = triton.next_power_of_2(vocab_size)
    num_warps = 4 if block_size <= 2048 else 8
    _cross_entropy_backward_kernel[(rows,)](
        logits_2d,
        labels_1d,
        valid_count,
        grad_output.contiguous(),
        dlogits,
        vocab_size,
        ignore_index,
        block_size,
        num_warps=num_warps,
    )
    return dlogits


@triton.jit
def _cross_entropy_large_backward_kernel(
    logits_ptr,
    labels_ptr,
    row_max_ptr,
    row_sum_ptr,
    valid_count_ptr,
    grad_output_ptr,
    dlogits_ptr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    label = tl.load(labels_ptr + row)
    valid = label != ignore_index
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + row).to(tl.float32)
    probs = tl.exp(logits - row_max) / row_sum
    target = offsets == label
    valid_count = tl.load(valid_count_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad = (probs - target.to(tl.float32)) * (grad_output / valid_count)
    grad = tl.where(valid, grad, 0.0)
    tl.store(dlogits_ptr + row * vocab_size + offsets, grad, mask=mask)


def cross_entropy_large_backward(
    logits_2d: Tensor,
    labels_1d: Tensor,
    row_max: Tensor,
    row_sum: Tensor,
    valid_count: Tensor,
    grad_output: Tensor,
    ignore_index: int,
    block_size: int,
) -> Tensor:
    rows, vocab_size = logits_2d.shape
    dlogits = torch.empty_like(logits_2d)
    if rows == 0:
        return dlogits
    _cross_entropy_large_backward_kernel[(rows, triton.cdiv(vocab_size, block_size))](
        logits_2d,
        labels_1d,
        row_max,
        row_sum,
        valid_count,
        grad_output.contiguous(),
        dlogits,
        vocab_size,
        ignore_index,
        block_size,
        num_warps=4,
    )
    return dlogits
