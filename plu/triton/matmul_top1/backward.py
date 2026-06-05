from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _matmul_top1_dx_kernel(
    grad_values_ptr,
    weight_ptr,
    indices_ptr,
    dx_ptr,
    in_features: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_k + tl.arange(0, block_k)
    mask = offsets < in_features
    index = tl.load(indices_ptr + row)
    grad = tl.load(grad_values_ptr + row).to(tl.float32)
    weight = tl.load(weight_ptr + index * in_features + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(dx_ptr + row * in_features + offsets, grad * weight, mask=mask)


@triton.jit
def _matmul_top1_dw_kernel(
    grad_values_ptr,
    x_ptr,
    indices_ptr,
    dweight_ptr,
    in_features: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_k + tl.arange(0, block_k)
    mask = offsets < in_features
    index = tl.load(indices_ptr + row)
    grad = tl.load(grad_values_ptr + row).to(tl.float32)
    x = tl.load(x_ptr + row * in_features + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(dweight_ptr + index * in_features + offsets, grad * x, sem="relaxed", mask=mask)


@triton.jit
def _matmul_top1_dbias_kernel(grad_values_ptr, indices_ptr, dbias_ptr):
    row = tl.program_id(0)
    index = tl.load(indices_ptr + row)
    grad = tl.load(grad_values_ptr + row).to(tl.float32)
    tl.atomic_add(dbias_ptr + index, grad, sem="relaxed")


def matmul_top1_backward(
    x_2d: Tensor,
    weight: Tensor,
    indices_1d: Tensor,
    grad_values: Tensor,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    rows, in_features = x_2d.shape
    dx = torch.empty_like(x_2d)
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype) if has_bias else None
    if rows == 0:
        return dx, dweight, dbias

    block_k = min(1024, triton.next_power_of_2(in_features))
    grid = (rows, triton.cdiv(in_features, block_k))
    grad_values = grad_values.contiguous()
    _matmul_top1_dx_kernel[grid](grad_values, weight, indices_1d, dx, in_features, block_k, num_warps=4)
    _matmul_top1_dw_kernel[grid](grad_values, x_2d, indices_1d, dweight, in_features, block_k, num_warps=4)
    if dbias is not None:
        _matmul_top1_dbias_kernel[(rows,)](grad_values, indices_1d, dbias, num_warps=1)
    return dx, dweight, dbias

