from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import gelu_backward, linear_input_grad, linear_weight_bias_grad
from plu.triton.linear.forward import linear_forward_2d


@triton.jit
def _im2col_kernel(
    x_ptr,
    cols_ptr,
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
    value = tl.load(x_ptr + (b * in_channels + c) * input_len + input_t, mask=valid, other=0.0)
    tl.store(cols_ptr + offsets, value, mask=mask)


@triton.jit
def _gelu_kernel(
    preact_ptr,
    out_ptr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    x = tl.load(preact_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    out = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _linear_to_conv_layout_kernel(
    x_2d_ptr,
    out_ptr,
    batch: tl.constexpr,
    out_channels: tl.constexpr,
    out_len: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    t = offsets % out_len
    c = (offsets // out_len) % out_channels
    b = offsets // (out_channels * out_len)
    value = tl.load(x_2d_ptr + (b * out_len + t) * out_channels + c, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, value, mask=mask)


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


def _make_cols(x: Tensor, out_len: int, kernel_size: int, stride: int, padding: int) -> Tensor:
    batch, in_channels, input_len = x.shape
    cols = torch.empty((batch * out_len, in_channels * kernel_size), device=x.device, dtype=x.dtype)
    total = cols.numel()
    if total:
        block_size = 256
        _im2col_kernel[(triton.cdiv(total, block_size),)](
            x,
            cols,
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
    return cols


def _linear_to_conv_layout(x_2d: Tensor, batch: int, out_channels: int, out_len: int) -> Tensor:
    out = torch.empty((batch, out_channels, out_len), device=x_2d.device, dtype=x_2d.dtype)
    total = out.numel()
    if total:
        block_size = 256
        _linear_to_conv_layout_kernel[(triton.cdiv(total, block_size),)](
            x_2d,
            out,
            batch,
            out_channels,
            out_len,
            total,
            block_size,
            num_warps=4,
        )
    return out


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


class _TritonConv1dGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, stride: int, padding: int):
        x = x.contiguous()
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        batch, in_channels, input_len = x.shape
        out_channels, _, kernel_size = weight.shape
        out_len = (input_len + 2 * padding - kernel_size) // stride + 1
        cols = _make_cols(x, out_len, kernel_size, stride, padding)
        weight_2d = weight.reshape(out_channels, in_channels * kernel_size)
        preact = linear_forward_2d(cols, weight_2d, bias)
        hidden_2d = torch.empty_like(preact)
        total = preact.numel()
        if total:
            block_size = 256
            _gelu_kernel[(triton.cdiv(total, block_size),)](preact, hidden_2d, total, block_size, num_warps=4)
        out = _linear_to_conv_layout(hidden_2d, batch, out_channels, out_len)
        ctx.save_for_backward(cols, weight_2d, preact)
        ctx.input_shape = x.shape
        ctx.weight_shape = weight.shape
        ctx.has_bias = bias is not None
        ctx.out_len = out_len
        ctx.stride = int(stride)
        ctx.padding = int(padding)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cols, weight_2d, preact = ctx.saved_tensors
        grad_out_2d = _conv_to_linear_layout(grad_out.contiguous())
        grad_preact = gelu_backward(preact, grad_out_2d)
        grad_cols = linear_input_grad(grad_preact, weight_2d)
        grad_weight_2d, grad_bias = linear_weight_bias_grad(grad_preact, cols, ctx.has_bias, dtype=weight_2d.dtype)
        grad_x = torch.zeros(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        total = grad_cols.numel()
        if total:
            block_size = 256
            batch, in_channels, input_len = ctx.input_shape
            kernel_size = ctx.weight_shape[2]
            _col2im_kernel[(triton.cdiv(total, block_size),)](
                grad_cols,
                grad_x,
                batch,
                in_channels,
                input_len,
                ctx.out_len,
                kernel_size,
                ctx.stride,
                ctx.padding,
                total,
                block_size,
                num_warps=4,
            )
        return grad_x, grad_weight_2d.reshape(ctx.weight_shape), grad_bias, None, None


def conv1d_gelu(x: Tensor, weight: Tensor, bias: Tensor | None, stride: int, padding: int) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.conv1d_gelu requires CUDA tensors")
    return _TritonConv1dGelu.apply(x, weight, bias, stride, padding)
