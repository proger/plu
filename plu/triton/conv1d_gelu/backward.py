from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import gelu_backward
from plu.triton.linear.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _conv_to_linear_layout_kernel(
    x_ptr,
    x_2d_ptr,
    batch: tl.constexpr,
    out_channels: tl.constexpr,
    out_len: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    c = offsets % out_channels
    row = offsets // out_channels
    t = row % out_len
    b = row // out_len
    value = tl.load(x_ptr + (b * out_channels + c) * out_len + t, mask=mask, other=0.0)
    tl.store(x_2d_ptr + offsets, value, mask=mask)


@triton.jit
def _col2im_kernel(
    cols_ptr,
    grad_x_ptr,
    batch: tl.constexpr,
    in_channels: tl.constexpr,
    input_len: tl.constexpr,
    out_len: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    k_col: tl.constexpr = in_channels * kernel_size
    col = offsets % k_col
    row = offsets // k_col
    kk = col % kernel_size
    c = col // kernel_size
    t = row % out_len
    b = row // out_len
    input_t = t * stride + kk - padding
    valid = mask & (input_t >= 0) & (input_t < input_len)
    value = tl.load(cols_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(grad_x_ptr + (b * in_channels + c) * input_len + input_t, value, sem="relaxed", mask=valid)


def _conv_to_linear_layout(x: Tensor) -> Tensor:
    batch, out_channels, out_len = x.shape
    out = torch.empty((batch * out_len, out_channels), device=x.device, dtype=x.dtype)
    total = out.numel()
    if total:
        block_size = 256
        _conv_to_linear_layout_kernel[(triton.cdiv(total, block_size),)](
            x,
            out,
            batch,
            out_channels,
            out_len,
            total,
            block_size,
            num_warps=4,
        )
    return out


def conv1d_gelu_backward(
    grad_out: Tensor,
    cols: Tensor,
    weight_2d: Tensor,
    preact: Tensor,
    input_shape: torch.Size,
    weight_shape: torch.Size,
    has_bias: bool,
    out_len: int,
    stride: int,
    padding: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    grad_out_2d = _conv_to_linear_layout(grad_out.contiguous())
    grad_preact = gelu_backward(preact, grad_out_2d)
    grad_cols = linear_input_grad(grad_preact, weight_2d)
    grad_weight_2d, grad_bias = linear_weight_bias_grad(grad_preact, cols, has_bias, dtype=weight_2d.dtype)
    grad_x = torch.zeros(input_shape, device=grad_out.device, dtype=grad_out.dtype)
    total = grad_cols.numel()
    if total:
        block_size = 256
        batch, in_channels, input_len = input_shape
        kernel_size = weight_shape[2]
        _col2im_kernel[(triton.cdiv(total, block_size),)](
            grad_cols,
            grad_x,
            batch,
            in_channels,
            input_len,
            out_len,
            kernel_size,
            stride,
            padding,
            total,
            block_size,
            num_warps=4,
        )
    return grad_x, grad_weight_2d.reshape(weight_shape), grad_bias
