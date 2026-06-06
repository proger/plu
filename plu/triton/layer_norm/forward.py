from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.layer_norm.backward import layer_norm_backward


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
        return layer_norm_backward(grad_out, x_2d, weight, mean, rstd, ctx.has_bias, ctx.original_shape)


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("plu.triton.layer_norm requires CUDA tensors")
    return _TritonLayerNorm.apply(x, weight, bias, eps)
