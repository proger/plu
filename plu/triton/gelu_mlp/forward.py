from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import linear_input_gelu_grad, linear_input_grad, linear_weight_bias_grad


@triton.jit
def _linear_gelu_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    preact_ptr,
    hidden_ptr,
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
    inv_sqrt2 = 0.7071067811865476
    hidden = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < out_features)
    offsets = offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(preact_ptr + offsets, acc, mask=mask)
    tl.store(hidden_ptr + offsets, hidden, mask=mask)


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


def _gelu_mlp_forward(
    x_2d: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
    fc2_weight: Tensor,
    fc2_bias: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    rows, in_features = x_2d.shape
    hidden_features = fc1_weight.shape[0]
    out_features = fc2_weight.shape[0]
    preact = torch.empty((rows, hidden_features), device=x_2d.device, dtype=x_2d.dtype)
    hidden = torch.empty_like(preact)
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype)
    if rows == 0:
        return out, preact, hidden

    use_large_tiles = rows >= 512 and in_features >= 512 and hidden_features >= 2048
    block_m = 64 if use_large_tiles else 16
    block_n = 128 if use_large_tiles else 32
    block_k = 32 if use_large_tiles else 32
    num_warps = 4
    _linear_gelu_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(hidden_features, block_n))](
        x_2d,
        fc1_weight,
        fc1_bias if fc1_bias is not None else x_2d,
        preact,
        hidden,
        rows,
        in_features,
        hidden_features,
        fc1_bias is not None,
        use_large_tiles,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
    )
    _linear_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
        hidden,
        fc2_weight,
        fc2_bias if fc2_bias is not None else x_2d,
        out,
        rows,
        hidden_features,
        out_features,
        fc2_bias is not None,
        use_large_tiles,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
    )
    return out, preact, hidden


class _TritonGeluMlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, fc1_weight: Tensor, fc1_bias: Tensor | None, fc2_weight: Tensor, fc2_bias: Tensor | None):
        original_x_shape = x.shape
        in_features = original_x_shape[-1]
        x_2d = x.contiguous().reshape(-1, in_features)
        fc1_weight = fc1_weight.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc1_bias = None if fc1_bias is None else fc1_bias.contiguous()
        fc2_bias = None if fc2_bias is None else fc2_bias.contiguous()
        out, preact, hidden = _gelu_mlp_forward(x_2d, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        ctx.save_for_backward(x_2d, fc1_weight, fc2_weight, preact, hidden)
        ctx.has_fc1_bias = fc1_bias is not None
        ctx.has_fc2_bias = fc2_bias is not None
        ctx.original_x_shape = original_x_shape
        return out.reshape(*original_x_shape[:-1], fc2_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, fc1_weight, fc2_weight, preact, hidden = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().reshape(-1, fc2_weight.shape[0])
        grad_fc2_weight, grad_fc2_bias = linear_weight_bias_grad(grad_out_2d, hidden, ctx.has_fc2_bias, dtype=fc2_weight.dtype)
        grad_preact = linear_input_gelu_grad(grad_out_2d, fc2_weight, preact)
        grad_x = linear_input_grad(grad_preact, fc1_weight)
        grad_fc1_weight, grad_fc1_bias = linear_weight_bias_grad(grad_preact, x_2d, ctx.has_fc1_bias, dtype=fc1_weight.dtype)
        return grad_x.reshape(ctx.original_x_shape), grad_fc1_weight, grad_fc1_bias, grad_fc2_weight, grad_fc2_bias


def gelu_mlp(x: Tensor, fc1_weight: Tensor, fc1_bias: Tensor | None, fc2_weight: Tensor, fc2_bias: Tensor | None) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.gelu_mlp requires CUDA tensors")
    return _TritonGeluMlp.apply(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
