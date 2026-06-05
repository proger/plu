from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _linear_forward_kernel(
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
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


def linear_forward_2d(x_2d: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    rows, in_features = x_2d.shape
    out_features = weight.shape[0]
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 512 and in_features >= 256 and out_features >= 512 and x_2d.dtype == torch.float32
    block_m = 64 if use_large_tiles else 16
    block_n = 128 if use_large_tiles else 32
    block_k = 32 if use_large_tiles else 32
    _linear_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
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


class _TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None):
        original_shape = x.shape
        in_features = original_shape[-1]
        x_2d = x.contiguous().reshape(-1, in_features)
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        out = linear_forward_2d(x_2d, weight, bias)
        ctx.save_for_backward(x_2d, weight)
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        return out.reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().reshape(-1, weight.shape[0])
        grad_x = linear_input_grad(grad_out_2d, weight)
        grad_weight, grad_bias = linear_weight_bias_grad(grad_out_2d, x_2d, ctx.has_bias, dtype=weight.dtype)
        return grad_x.reshape(ctx.original_shape), grad_weight, grad_bias


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.linear requires CUDA tensors")
    return _TritonLinear.apply(x, weight, bias)
