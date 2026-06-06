from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.linear.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _qkv_proj_grad_to_2d_kernel(
    grad_ptr,
    grad_2d_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    n = offsets % (num_heads * head_dim)
    row = offsets // (num_heads * head_dim)
    b = row // seq_len
    t = row - b * seq_len
    h = n // head_dim
    d = n - h * head_dim
    grad = tl.load(grad_ptr + ((b * num_heads + h) * seq_len + t) * head_dim + d, mask=mask, other=0.0)
    tl.store(grad_2d_ptr + row * (num_heads * head_dim) + n, grad, mask=mask)


def qkv_proj_backward(
    grad_out: Tensor,
    x_2d: Tensor,
    weight: Tensor,
    has_bias: bool,
    original_shape: torch.Size,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor | None, None]:
    rows = x_2d.shape[0]
    out_features = weight.shape[0]
    grad_2d = torch.empty((rows, out_features), device=grad_out.device, dtype=grad_out.dtype)
    total = grad_2d.numel()
    if total:
        block_size = 256
        _qkv_proj_grad_to_2d_kernel[(triton.cdiv(total, block_size),)](
            grad_out.contiguous(),
            grad_2d,
            rows,
            seq_len,
            num_heads,
            head_dim,
            total,
            block_size,
            num_warps=4,
        )
    grad_x = linear_input_grad(grad_2d, weight)
    grad_weight, grad_bias = linear_weight_bias_grad(grad_2d, x_2d, has_bias, dtype=weight.dtype)
    return grad_x.reshape(original_shape), grad_weight, grad_bias, None
