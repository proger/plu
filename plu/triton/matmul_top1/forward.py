from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
from plu.triton.matmul_top1.backward import matmul_top1_backward


@triton.jit
def _matmul_top1_stage1_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    num_out_blocks: tl.constexpr,
    has_bias: tl.constexpr,
    block_v: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    out_block = tl.program_id(1)
    vocab_offsets = out_block * block_v + tl.arange(0, block_v)
    k_offsets = tl.arange(0, block_k)
    vocab_mask = vocab_offsets < out_features
    k_mask = k_offsets < in_features

    x = tl.load(x_ptr + row * in_features + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    weight = tl.load(
        weight_ptr + vocab_offsets[:, None] * in_features + k_offsets[None, :],
        mask=vocab_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(weight * x[None, :], axis=1)
    if has_bias:
        bias = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(tl.float32)
        logits += bias
    logits = tl.where(vocab_mask, logits, -float("inf"))

    max_value = tl.max(logits, axis=0)
    tie_indices = tl.where(logits == max_value, vocab_offsets, out_features)
    max_index = tl.min(tie_indices, axis=0)
    partial_offset = row * num_out_blocks + out_block
    tl.store(partial_values_ptr + partial_offset, max_value)
    tl.store(partial_indices_ptr + partial_offset, max_index)


@triton.jit
def _matmul_top1_stage2_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    values_ptr,
    indices_ptr,
    num_out_blocks: tl.constexpr,
    block_blocks: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_blocks)
    mask = offsets < num_out_blocks
    values = tl.load(partial_values_ptr + row * num_out_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    indices = tl.load(partial_indices_ptr + row * num_out_blocks + offsets, mask=mask, other=2147483647)
    max_value = tl.max(values, axis=0)
    tie_indices = tl.where(values == max_value, indices, 2147483647)
    max_index = tl.min(tie_indices, axis=0)
    tl.store(values_ptr + row, max_value)
    tl.store(indices_ptr + row, max_index)


def _matmul_top1_forward(x_2d: Tensor, weight: Tensor, bias: Tensor | None) -> tuple[Tensor, Tensor]:
    rows, in_features = x_2d.shape
    out_features = weight.shape[0]
    values = torch.empty(rows, device=x_2d.device, dtype=x_2d.dtype)
    indices = torch.empty(rows, device=x_2d.device, dtype=torch.long)
    if rows == 0:
        return values, indices

    block_v = 128
    block_k = triton.next_power_of_2(in_features)
    num_out_blocks = triton.cdiv(out_features, block_v)
    partial_values = torch.empty((rows, num_out_blocks), device=x_2d.device, dtype=torch.float32)
    partial_indices = torch.empty((rows, num_out_blocks), device=x_2d.device, dtype=torch.int64)
    _matmul_top1_stage1_kernel[(rows, num_out_blocks)](
        x_2d,
        weight,
        bias if bias is not None else x_2d,
        partial_values,
        partial_indices,
        in_features,
        out_features,
        num_out_blocks,
        bias is not None,
        block_v,
        block_k,
        num_warps=4,
    )

    block_blocks = triton.next_power_of_2(num_out_blocks)
    _matmul_top1_stage2_kernel[(rows,)](
        partial_values,
        partial_indices,
        values,
        indices,
        num_out_blocks,
        block_blocks,
        num_warps=1,
    )
    return values, indices


class _TritonMatmulTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None):
        original_x_shape = x.shape
        in_features = original_x_shape[-1]
        x_2d = x.contiguous().reshape(-1, in_features)
        weight_contiguous = weight.contiguous()
        bias_contiguous = None if bias is None else bias.contiguous()
        values, indices = _matmul_top1_forward(x_2d, weight_contiguous, bias_contiguous)
        ctx.save_for_backward(x_2d, weight_contiguous, indices)
        ctx.has_bias = bias is not None
        ctx.original_x_shape = original_x_shape
        return values.reshape(original_x_shape[:-1]), indices.reshape(original_x_shape[:-1])

    @staticmethod
    def backward(ctx, grad_values: Tensor, grad_indices: Tensor | None):
        x_2d, weight, indices = ctx.saved_tensors
        dx, dweight, dbias = matmul_top1_backward(
            x_2d,
            weight,
            indices.reshape(-1),
            grad_values.contiguous().reshape(-1),
            ctx.has_bias,
        )
        return dx.reshape(ctx.original_x_shape), dweight, dbias


def matmul_top1(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if not x.is_cuda:
        return ref_matmul_top1(x, weight, bias)
    return _TritonMatmulTop1.apply(x, weight, bias)

