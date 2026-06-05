from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _layer_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    rows: tl.constexpr,
    features: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_n)
    mask = offs < features
    x = tl.load(x_ptr + row * features + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / features
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / features
    rstd = tl.rsqrt(var + eps)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight
    if has_bias:
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y += bias
    tl.store(out_ptr + row * features + offs, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layer_norm_backward_input_kernel(
    grad_out_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    grad_x_ptr,
    rows: tl.constexpr,
    features: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_n)
    mask = offs < features
    grad = tl.load(grad_out_ptr + row * features + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * features + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    xhat = (x - mean) * rstd
    dyw = tl.where(mask, grad * weight, 0.0)
    c1 = tl.sum(dyw, axis=0) / features
    c2 = tl.sum(dyw * xhat, axis=0) / features
    grad_x = (dyw - c1 - xhat * c2) * rstd
    tl.store(grad_x_ptr + row * features + offs, grad_x, mask=mask)


@triton.jit
def _layer_norm_backward_weight_bias_kernel(
    grad_out_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    rows: tl.constexpr,
    features: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m = tl.arange(0, block_m)
    mask_n = offs_n < features
    acc_w = tl.zeros((block_n,), dtype=tl.float32)
    acc_b = tl.zeros((block_n,), dtype=tl.float32)
    for m_start in tl.range(0, rows, block_m):
        m = m_start + offs_m
        mask = (m[:, None] < rows) & mask_n[None, :]
        grad = tl.load(grad_out_ptr + m[:, None] * features + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + m[:, None] * features + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + m, mask=m < rows, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + m, mask=m < rows, other=0.0).to(tl.float32)
        xhat = (x - mean[:, None]) * rstd[:, None]
        acc_w += tl.sum(grad * xhat, axis=0)
        acc_b += tl.sum(grad, axis=0)
    tl.store(grad_weight_ptr + offs_n, acc_w, mask=mask_n)
    if has_bias:
        tl.store(grad_bias_ptr + offs_n, acc_b, mask=mask_n)


class _TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, eps: float):
        original_shape = x.shape
        features = original_shape[-1]
        x_2d = x.contiguous().reshape(-1, features)
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        rows = x_2d.shape[0]
        out = torch.empty_like(x_2d)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
        if rows:
            block_n = triton.next_power_of_2(features)
            _layer_norm_forward_kernel[(rows,)](
                x_2d,
                weight,
                bias if bias is not None else weight,
                out,
                mean,
                rstd,
                rows,
                features,
                float(eps),
                bias is not None,
                block_n,
                num_warps=8,
            )
        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight, mean, rstd = ctx.saved_tensors
        features = x_2d.shape[1]
        rows = x_2d.shape[0]
        grad_out_2d = grad_out.contiguous().reshape(-1, features)
        grad_x = torch.empty_like(x_2d)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(weight) if ctx.has_bias else None
        if rows:
            block_n = triton.next_power_of_2(features)
            _layer_norm_backward_input_kernel[(rows,)](
                grad_out_2d,
                x_2d,
                weight,
                mean,
                rstd,
                grad_x,
                rows,
                features,
                block_n,
                num_warps=8,
            )
            block_grad_n = min(128, triton.next_power_of_2(features))
            _layer_norm_backward_weight_bias_kernel[(triton.cdiv(features, block_grad_n),)](
                grad_out_2d,
                x_2d,
                mean,
                rstd,
                grad_weight,
                grad_bias if grad_bias is not None else grad_weight,
                rows,
                features,
                ctx.has_bias,
                32,
                block_grad_n,
                num_warps=4,
            )
        return grad_x.reshape(ctx.original_shape), grad_weight, grad_bias, None


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.layer_norm requires CUDA tensors")
    return _TritonLayerNorm.apply(x, weight, bias, eps)
