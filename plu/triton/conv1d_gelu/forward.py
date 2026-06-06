from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch import Tensor

from plu.triton.conv1d_gelu.backward import conv1d_gelu_backward


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
def _linear_gelu_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    has_bias: tl.constexpr,
    use_tf32: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in tl.range(0, in_features, block_k):
        k = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_n[:, None] * in_features + k[None, :],
            mask=(offs_n[:, None] < out_features) & (k[None, :] < in_features),
            other=0.0,
        )
        if use_tf32:
            acc += tl.dot(x, tl.trans(weight), input_precision="tf32")
        else:
            acc += tl.dot(x, tl.trans(weight), input_precision="ieee")
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        acc += bias[None, :]
    inv_sqrt2 = 0.7071067690849304
    out = acc * 0.5 * (1.0 + libdevice.erf(acc * inv_sqrt2))
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


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


def _linear_gelu_forward_2d(x_2d: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    rows, in_features = x_2d.shape
    out_features = weight.shape[0]
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 512 and in_features >= 256 and out_features >= 512 and x_2d.dtype == torch.float32
    block_m = 64 if use_large_tiles else 16
    block_n = 128 if use_large_tiles else 32
    block_k = 32
    _linear_gelu_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
        x_2d,
        weight,
        bias if bias is not None else x_2d,
        out,
        rows,
        in_features,
        out_features,
        bias is not None,
        use_large_tiles,
        block_m,
        block_n,
        block_k,
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
        hidden_2d = _linear_gelu_forward_2d(cols, weight_2d, bias)
        out = _linear_to_conv_layout(hidden_2d, batch, out_channels, out_len)
        bias_saved = bias if bias is not None else x.new_empty(0)
        ctx.save_for_backward(cols, weight_2d, bias_saved)
        ctx.input_shape = x.shape
        ctx.weight_shape = weight.shape
        ctx.has_bias = bias is not None
        ctx.out_len = out_len
        ctx.stride = int(stride)
        ctx.padding = int(padding)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cols, weight_2d, bias_saved = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        grad_x, grad_weight, grad_bias = conv1d_gelu_backward(
            grad_out,
            cols,
            weight_2d,
            bias,
            ctx.input_shape,
            ctx.weight_shape,
            ctx.has_bias,
            ctx.out_len,
            ctx.stride,
            ctx.padding,
        )
        return grad_x, grad_weight, grad_bias, None, None


def conv1d_gelu(x: Tensor, weight: Tensor, bias: Tensor | None, stride: int, padding: int) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.conv1d_gelu requires CUDA tensors")
    return _TritonConv1dGelu.apply(x, weight, bias, stride, padding)
