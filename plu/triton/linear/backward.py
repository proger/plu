from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _linear_input_grad_kernel(
    grad_out_ptr,
    weight_ptr,
    grad_input_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    use_tf32: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_n = tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_k), dtype=tl.float32)
    for n_start in tl.range(0, out_features, block_n):
        n = n_start + offs_n
        grad = tl.load(
            grad_out_ptr + offs_m[:, None] * out_features + n[None, :],
            mask=(offs_m[:, None] < rows) & (n[None, :] < out_features),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + n[:, None] * in_features + offs_k[None, :],
            mask=(n[:, None] < out_features) & (offs_k[None, :] < in_features),
            other=0.0,
        )
        if use_tf32:
            acc += tl.dot(grad, weight, input_precision="tf32")
        else:
            acc += tl.dot(grad, weight, input_precision="ieee")
    tl.store(
        grad_input_ptr + offs_m[:, None] * in_features + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_k[None, :] < in_features),
    )


@triton.jit
def _linear_weight_grad_kernel(
    grad_out_ptr,
    input_ptr,
    grad_weight_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    use_tf32: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    grad = tl.load(
        grad_out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
        other=0.0,
    )
    x = tl.load(
        input_ptr + offs_m[:, None] * in_features + offs_k[None, :],
        mask=(offs_m[:, None] < rows) & (offs_k[None, :] < in_features),
        other=0.0,
    )
    if use_tf32:
        acc = tl.dot(tl.trans(grad), x, input_precision="tf32")
    else:
        acc = tl.dot(tl.trans(grad), x, input_precision="ieee")
    tl.atomic_add(
        grad_weight_ptr + offs_n[:, None] * in_features + offs_k[None, :],
        acc,
        sem="relaxed",
        mask=(offs_n[:, None] < out_features) & (offs_k[None, :] < in_features),
    )


@triton.jit
def _linear_weight_grad_reduce_kernel(
    grad_out_ptr,
    input_ptr,
    grad_weight_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    use_tf32: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_m = tl.arange(0, block_m)
    acc = tl.zeros((block_n, block_k), dtype=tl.float32)
    for m_start in tl.range(0, rows, block_m):
        m = m_start + offs_m
        grad = tl.load(
            grad_out_ptr + m[:, None] * out_features + offs_n[None, :],
            mask=(m[:, None] < rows) & (offs_n[None, :] < out_features),
            other=0.0,
        )
        x = tl.load(
            input_ptr + m[:, None] * in_features + offs_k[None, :],
            mask=(m[:, None] < rows) & (offs_k[None, :] < in_features),
            other=0.0,
        )
        if use_tf32:
            acc += tl.dot(tl.trans(grad), x, input_precision="tf32")
        else:
            acc += tl.dot(tl.trans(grad), x, input_precision="ieee")
    tl.store(
        grad_weight_ptr + offs_n[:, None] * in_features + offs_k[None, :],
        acc,
        mask=(offs_n[:, None] < out_features) & (offs_k[None, :] < in_features),
    )


@triton.jit
def _linear_bias_grad_kernel(
    grad_out_ptr,
    grad_bias_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    grad = tl.load(
        grad_out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
        other=0.0,
    )
    acc = tl.sum(grad, axis=0)
    tl.atomic_add(grad_bias_ptr + offs_n, acc, sem="relaxed", mask=offs_n < out_features)


def linear_input_grad(grad_out: Tensor, weight: Tensor) -> Tensor:
    rows, out_features = grad_out.shape
    in_features = weight.shape[1]
    grad_input = torch.empty((rows, in_features), device=grad_out.device, dtype=grad_out.dtype)
    if rows == 0:
        return grad_input
    use_large_tiles = rows >= 512 and out_features >= 512 and in_features >= 256
    use_tf32 = use_large_tiles and grad_out.dtype == torch.float32 and weight.dtype == torch.float32
    block_m = 64 if use_large_tiles else 16
    block_k = 128 if use_large_tiles else 32
    block_n = 32
    _linear_input_grad_kernel[(triton.cdiv(rows, block_m), triton.cdiv(in_features, block_k))](
        grad_out,
        weight,
        grad_input,
        rows,
        out_features,
        in_features,
        use_tf32,
        block_m,
        block_k,
        block_n,
        num_warps=4,
    )
    return grad_input


def linear_weight_bias_grad(grad_out: Tensor, x: Tensor, has_bias: bool, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor | None]:
    rows, out_features = grad_out.shape
    in_features = x.shape[1]
    grad_dtype = grad_out.dtype if dtype is None else dtype
    use_large_tiles = rows >= 512 and out_features >= 512 and in_features >= 256
    use_tf32 = use_large_tiles and grad_out.dtype == torch.float32 and x.dtype == torch.float32
    if use_large_tiles:
        grad_weight = torch.empty((out_features, in_features), device=grad_out.device, dtype=grad_dtype)
    else:
        grad_weight = torch.zeros((out_features, in_features), device=grad_out.device, dtype=grad_dtype)
    grad_bias = torch.zeros(out_features, device=grad_out.device, dtype=grad_dtype) if has_bias else None
    if rows == 0:
        grad_weight.zero_()
        return grad_weight, grad_bias

    block_n = 64 if use_large_tiles else 32
    block_k = 128 if use_large_tiles else 32
    block_m = 16 if use_large_tiles else 32
    if use_large_tiles:
        _linear_weight_grad_reduce_kernel[(triton.cdiv(out_features, block_n), triton.cdiv(in_features, block_k))](
            grad_out,
            x,
            grad_weight,
            rows,
            out_features,
            in_features,
            use_tf32,
            block_n,
            block_k,
            block_m,
            num_warps=4,
        )
    else:
        _linear_weight_grad_kernel[
            (triton.cdiv(out_features, block_n), triton.cdiv(in_features, block_k), triton.cdiv(rows, block_m))
        ](
            grad_out,
            x,
            grad_weight,
            rows,
            out_features,
            in_features,
            use_tf32,
            block_n,
            block_k,
            block_m,
            num_warps=4,
        )
    if grad_bias is not None:
        _linear_bias_grad_kernel[(triton.cdiv(out_features, block_n), triton.cdiv(rows, block_m))](
            grad_out,
            grad_bias,
            rows,
            out_features,
            block_n,
            block_m,
            num_warps=4,
        )
    return grad_weight, grad_bias
