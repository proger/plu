from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch import Tensor


@triton.jit
def _gelu_forward_kernel(
    preact_ptr,
    hidden_ptr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total
    x = tl.load(preact_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067690849304
    hidden = x * 0.5 * (1.0 + libdevice.erf(x * inv_sqrt2))
    tl.store(hidden_ptr + offsets, hidden, mask=mask)


@triton.jit
def _gelu_backward_kernel(
    preact_ptr,
    grad_hidden_ptr,
    grad_preact_ptr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total
    x = tl.load(preact_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067690849304
    inv_sqrt2pi = 0.3989422917366028
    cdf = 0.5 * (1.0 + libdevice.erf(x * inv_sqrt2))
    pdf_term = libdevice.exp(-0.5 * x * x) * inv_sqrt2pi
    tl.store(grad_preact_ptr + offsets, grad * (cdf + x * pdf_term), mask=mask)


@triton.jit
def _linear_input_gelu_grad_kernel(
    grad_out_ptr,
    weight_ptr,
    preact_ptr,
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

    preact = tl.load(
        preact_ptr + offs_m[:, None] * in_features + offs_k[None, :],
        mask=(offs_m[:, None] < rows) & (offs_k[None, :] < in_features),
        other=0.0,
    ).to(tl.float32)
    inv_sqrt2 = 0.7071067690849304
    inv_sqrt2pi = 0.3989422917366028
    cdf = 0.5 * (1.0 + libdevice.erf(preact * inv_sqrt2))
    pdf_term = libdevice.exp(-0.5 * preact * preact) * inv_sqrt2pi
    tl.store(
        grad_input_ptr + offs_m[:, None] * in_features + offs_k[None, :],
        acc * (cdf + preact * pdf_term),
        mask=(offs_m[:, None] < rows) & (offs_k[None, :] < in_features),
    )


@triton.jit
def _linear_input_gelu_grad_recompute_kernel(
    grad_out_ptr,
    fc2_weight_ptr,
    x_ptr,
    fc1_weight_ptr,
    fc1_bias_ptr,
    grad_input_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    hidden_features: tl.constexpr,
    input_features: tl.constexpr,
    has_fc1_bias: tl.constexpr,
    use_tf32_grad: tl.constexpr,
    use_tf32_preact: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    block_i: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_h = pid_h * block_h + tl.arange(0, block_h)
    offs_n = tl.arange(0, block_n)
    offs_i = tl.arange(0, block_i)

    grad_hidden_acc = tl.zeros((block_m, block_h), dtype=tl.float32)
    for n_start in tl.range(0, out_features, block_n):
        n = n_start + offs_n
        grad = tl.load(
            grad_out_ptr + offs_m[:, None] * out_features + n[None, :],
            mask=(offs_m[:, None] < rows) & (n[None, :] < out_features),
            other=0.0,
        )
        fc2_weight = tl.load(
            fc2_weight_ptr + n[:, None] * hidden_features + offs_h[None, :],
            mask=(n[:, None] < out_features) & (offs_h[None, :] < hidden_features),
            other=0.0,
        )
        if use_tf32_grad:
            grad_hidden_acc += tl.dot(grad, fc2_weight, input_precision="tf32")
        else:
            grad_hidden_acc += tl.dot(grad, fc2_weight, input_precision="ieee")

    preact = tl.zeros((block_m, block_h), dtype=tl.float32)
    for i_start in tl.range(0, input_features, block_i):
        i = i_start + offs_i
        x = tl.load(
            x_ptr + offs_m[:, None] * input_features + i[None, :],
            mask=(offs_m[:, None] < rows) & (i[None, :] < input_features),
            other=0.0,
        )
        fc1_weight = tl.load(
            fc1_weight_ptr + offs_h[:, None] * input_features + i[None, :],
            mask=(offs_h[:, None] < hidden_features) & (i[None, :] < input_features),
            other=0.0,
        )
        if use_tf32_preact:
            preact += tl.dot(x, tl.trans(fc1_weight), input_precision="tf32")
        else:
            preact += tl.dot(x, tl.trans(fc1_weight), input_precision="ieee")

    if has_fc1_bias:
        bias = tl.load(fc1_bias_ptr + offs_h, mask=offs_h < hidden_features, other=0.0).to(tl.float32)
        preact += bias[None, :]

    inv_sqrt2 = 0.7071067690849304
    inv_sqrt2pi = 0.3989422917366028
    cdf = 0.5 * (1.0 + libdevice.erf(preact * inv_sqrt2))
    pdf_term = libdevice.exp(-0.5 * preact * preact) * inv_sqrt2pi
    tl.store(
        grad_input_ptr + offs_m[:, None] * hidden_features + offs_h[None, :],
        grad_hidden_acc * (cdf + preact * pdf_term),
        mask=(offs_m[:, None] < rows) & (offs_h[None, :] < hidden_features),
    )


def gelu_backward(preact: Tensor, grad_hidden: Tensor) -> Tensor:
    grad_preact = torch.empty_like(preact)
    total = preact.numel()
    if total:
        block_size = 256
        _gelu_backward_kernel[(triton.cdiv(total, block_size),)](
            preact,
            grad_hidden,
            grad_preact,
            total,
            block_size,
            num_warps=4,
        )
    return grad_preact


def gelu_forward(preact: Tensor) -> Tensor:
    hidden = torch.empty_like(preact)
    total = preact.numel()
    if total:
        block_size = 256
        _gelu_forward_kernel[(triton.cdiv(total, block_size),)](
            preact,
            hidden,
            total,
            block_size,
            num_warps=4,
        )
    return hidden


def linear_input_gelu_grad(grad_out: Tensor, weight: Tensor, preact: Tensor) -> Tensor:
    rows, out_features = grad_out.shape
    in_features = weight.shape[1]
    grad_input = torch.empty((rows, in_features), device=grad_out.device, dtype=preact.dtype)
    if rows == 0:
        return grad_input
    use_large_tiles = rows >= 512 and out_features >= 512 and in_features >= 256
    use_tf32 = use_large_tiles and grad_out.dtype == torch.float32 and weight.dtype == torch.float32
    block_m = 64 if use_large_tiles else 16
    block_k = 128 if use_large_tiles else 32
    block_n = 32
    _linear_input_gelu_grad_kernel[(triton.cdiv(rows, block_m), triton.cdiv(in_features, block_k))](
        grad_out,
        weight,
        preact,
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


def linear_input_gelu_grad_recompute(
    grad_out: Tensor,
    fc2_weight: Tensor,
    x: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor | None,
) -> Tensor:
    rows, out_features = grad_out.shape
    hidden_features = fc2_weight.shape[1]
    input_features = x.shape[1]
    grad_dtype = torch.float32 if x.dtype is torch.bfloat16 else x.dtype
    grad_input = torch.empty((rows, hidden_features), device=grad_out.device, dtype=grad_dtype)
    if rows == 0:
        return grad_input

    use_large_tiles = rows >= 128 and out_features >= 512 and hidden_features >= 512 and input_features >= 512
    use_tf32_grad = use_large_tiles and grad_out.dtype == torch.float32 and fc2_weight.dtype == torch.float32
    use_tf32_preact = use_large_tiles and x.dtype == torch.float32 and fc1_weight.dtype == torch.float32
    block_m = 64 if use_large_tiles else 16
    block_h = 128 if use_large_tiles else 32
    block_n = 64 if use_large_tiles else 32
    block_i = 64 if use_large_tiles else 32
    _linear_input_gelu_grad_recompute_kernel[(triton.cdiv(rows, block_m), triton.cdiv(hidden_features, block_h))](
        grad_out,
        fc2_weight,
        x,
        fc1_weight,
        fc1_bias if fc1_bias is not None else x,
        grad_input,
        rows,
        out_features,
        hidden_features,
        input_features,
        fc1_bias is not None,
        use_tf32_grad,
        use_tf32_preact,
        block_m,
        block_h,
        block_n,
        block_i,
        num_warps=4,
    )
    return grad_input
