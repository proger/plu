from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.linear.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _attention_output_to_2d_kernel(
    x_ptr,
    x_2d_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    k = offsets % (num_heads * head_dim)
    row = offsets // (num_heads * head_dim)
    b = row // seq_len
    t = row - b * seq_len
    h = k // head_dim
    d = k - h * head_dim
    value = tl.load(x_ptr + ((b * num_heads + h) * seq_len + t) * head_dim + d, mask=mask, other=0.0)
    tl.store(x_2d_ptr + row * (num_heads * head_dim) + k, value, mask=mask)


@triton.jit
def _attention_output_from_2d_kernel(
    x_2d_ptr,
    x_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    k = offsets % (num_heads * head_dim)
    row = offsets // (num_heads * head_dim)
    b = row // seq_len
    t = row - b * seq_len
    h = k // head_dim
    d = k - h * head_dim
    value = tl.load(x_2d_ptr + row * (num_heads * head_dim) + k, mask=mask, other=0.0)
    tl.store(x_ptr + ((b * num_heads + h) * seq_len + t) * head_dim + d, value, mask=mask)


def c_proj_backward(grad_out: Tensor, x: Tensor, weight: Tensor, has_bias: bool) -> tuple[Tensor, Tensor, Tensor | None]:
    batch, num_heads, seq_len, head_dim = x.shape
    rows = batch * seq_len
    in_features = num_heads * head_dim
    x_2d = torch.empty((rows, in_features), device=x.device, dtype=x.dtype)
    total = x_2d.numel()
    if total:
        block_size = 256
        _attention_output_to_2d_kernel[(triton.cdiv(total, block_size),)](
            x,
            x_2d,
            rows,
            seq_len,
            num_heads,
            head_dim,
            total,
            block_size,
            num_warps=4,
        )
    grad_out_2d = grad_out.contiguous().reshape(rows, weight.shape[0])
    grad_x_2d = linear_input_grad(grad_out_2d, weight)
    grad_x = torch.empty_like(x)
    if total:
        block_size = 256
        _attention_output_from_2d_kernel[(triton.cdiv(total, block_size),)](
            grad_x_2d,
            grad_x,
            rows,
            seq_len,
            num_heads,
            head_dim,
            total,
            block_size,
            num_warps=4,
        )
    grad_weight, grad_bias = linear_weight_bias_grad(grad_out_2d, x_2d, has_bias, dtype=weight.dtype)
    return grad_x, grad_weight, grad_bias
