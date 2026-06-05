from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _qkv_proj_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    in_features: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
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
    out_features: tl.constexpr = num_heads * head_dim
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

    b = offs_m // seq_len
    t = offs_m - b * seq_len
    h = offs_n // head_dim
    d = offs_n - h * head_dim
    out_offsets = ((b[:, None] * num_heads + h[None, :]) * seq_len + t[:, None]) * head_dim + d[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


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


class _TritonQkvProj(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, num_heads: int):
        batch, seq_len, in_features = x.shape
        out_features = weight.shape[0]
        head_dim = out_features // num_heads
        x_2d = x.contiguous().reshape(-1, in_features)
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        out = torch.empty((batch, num_heads, seq_len, head_dim), device=x.device, dtype=x.dtype)
        rows = x_2d.shape[0]
        if rows:
            use_large_tiles = rows >= 512 and in_features >= 512 and out_features >= 512 and x.dtype == torch.float32
            block_m = 64 if use_large_tiles else 16
            block_n = 128 if use_large_tiles else 32
            block_k = 32 if use_large_tiles else 32
            _qkv_proj_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
                x_2d,
                weight,
                bias if bias is not None else x_2d,
                out,
                rows,
                seq_len,
                in_features,
                num_heads,
                head_dim,
                bias is not None,
                use_large_tiles,
                block_m,
                block_n,
                block_k,
                num_warps=4,
            )
        ctx.save_for_backward(x_2d, weight)
        ctx.has_bias = bias is not None
        ctx.original_shape = x.shape
        ctx.seq_len = seq_len
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight = ctx.saved_tensors
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
                ctx.seq_len,
                ctx.num_heads,
                ctx.head_dim,
                total,
                block_size,
                num_warps=4,
            )
        grad_x = linear_input_grad(grad_2d, weight)
        grad_weight, grad_bias = linear_weight_bias_grad(grad_2d, x_2d, ctx.has_bias, dtype=weight.dtype)
        return grad_x.reshape(ctx.original_shape), grad_weight, grad_bias, None


def qkv_proj(x: Tensor, weight: Tensor, bias: Tensor | None, num_heads: int) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.qkv_proj requires CUDA tensors")
    return _TritonQkvProj.apply(x, weight, bias, num_heads)
