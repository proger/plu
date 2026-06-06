from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.c_proj.backward import c_proj_backward


@triton.jit
def _c_proj_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
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
    in_features: tl.constexpr = num_heads * head_dim
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    b = offs_m // seq_len
    t = offs_m - b * seq_len
    for k_start in tl.range(0, in_features, block_k):
        k = k_start + offs_k
        h = k // head_dim
        d = k - h * head_dim
        x = tl.load(
            x_ptr + ((b[:, None] * num_heads + h[None, :]) * seq_len + t[:, None]) * head_dim + d[None, :],
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
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


class _TritonCProj(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None):
        batch, num_heads, seq_len, head_dim = x.shape
        rows = batch * seq_len
        in_features = num_heads * head_dim
        out_features = weight.shape[0]
        x = x.contiguous()
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        out = torch.empty((batch, seq_len, out_features), device=x.device, dtype=x.dtype)
        if rows:
            use_large_tiles = rows >= 512 and in_features >= 512 and out_features >= 512 and x.dtype == torch.float32
            block_m = 64 if use_large_tiles else 16
            block_n = 128 if use_large_tiles else 32
            block_k = 32 if use_large_tiles else 32
            _c_proj_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
                x,
                weight,
                bias if bias is not None else x,
                out,
                rows,
                seq_len,
                num_heads,
                head_dim,
                out_features,
                bias is not None,
                use_large_tiles,
                block_m,
                block_n,
                block_k,
                num_warps=4,
            )
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.seq_len = seq_len
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight = ctx.saved_tensors
        return c_proj_backward(grad_out, x, weight, ctx.has_bias)


def c_proj(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.c_proj requires CUDA tensors")
    return _TritonCProj.apply(x, weight, bias)
