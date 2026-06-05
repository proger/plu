from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.ref.lora import lora_linear as ref_lora_linear
from plu.triton.lora.backward import lora_linear_backward


@triton.jit
def _lora_a_forward_kernel(
    adapter_input_ptr,
    lora_a_weight_ptr,
    hidden_ptr,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    rank: tl.constexpr,
    block_m: tl.constexpr,
    block_r: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_r = tl.arange(0, block_r)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_r), dtype=tl.float32)
    for k_start in tl.range(0, in_features, block_k):
        k = k_start + offs_k
        x = tl.load(
            adapter_input_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )
        weight = tl.load(
            lora_a_weight_ptr + offs_r[:, None] * in_features + k[None, :],
            mask=(offs_r[:, None] < rank) & (k[None, :] < in_features),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight), input_precision="ieee")
    tl.store(
        hidden_ptr + offs_m[:, None] * rank + offs_r[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_r[None, :] < rank),
    )


@triton.jit
def _lora_out_forward_kernel(
    x_ptr,
    hidden_ptr,
    base_weight_ptr,
    base_bias_ptr,
    lora_b_weight_ptr,
    out_ptr,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    rank: tl.constexpr,
    has_bias: tl.constexpr,
    scaling: tl.constexpr,
    block_m: tl.constexpr,
    block_o: tl.constexpr,
    block_k: tl.constexpr,
    block_r: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_o = pid_o * block_o + tl.arange(0, block_o)
    offs_k = tl.arange(0, block_k)
    offs_r = tl.arange(0, block_r)

    acc = tl.zeros((block_m, block_o), dtype=tl.float32)
    for k_start in tl.range(0, in_features, block_k):
        k = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )
        weight = tl.load(
            base_weight_ptr + offs_o[:, None] * in_features + k[None, :],
            mask=(offs_o[:, None] < out_features) & (k[None, :] < in_features),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight), input_precision="ieee")

    hidden = tl.load(
        hidden_ptr + offs_m[:, None] * rank + offs_r[None, :],
        mask=(offs_m[:, None] < rows) & (offs_r[None, :] < rank),
        other=0.0,
    )
    lora_b = tl.load(
        lora_b_weight_ptr + offs_o[:, None] * rank + offs_r[None, :],
        mask=(offs_o[:, None] < out_features) & (offs_r[None, :] < rank),
        other=0.0,
    )
    acc += tl.dot(hidden, tl.trans(lora_b), input_precision="ieee") * scaling
    if has_bias:
        bias = tl.load(base_bias_ptr + offs_o, mask=offs_o < out_features, other=0.0).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_o[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_o[None, :] < out_features),
    )


def _lora_linear_forward(
    x_2d: Tensor,
    adapter_input_2d: Tensor,
    base_weight: Tensor,
    base_bias: Tensor | None,
    lora_a_weight: Tensor,
    lora_b_weight: Tensor,
    scaling: float,
) -> tuple[Tensor, Tensor]:
    rows, in_features = x_2d.shape
    out_features = base_weight.shape[0]
    rank = lora_a_weight.shape[0]
    hidden = torch.empty((rows, rank), device=x_2d.device, dtype=x_2d.dtype)
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype)
    if rows == 0:
        return out, hidden

    block_m = 16
    block_k = 32
    block_r = max(16, triton.next_power_of_2(rank))
    block_o = 32
    _lora_a_forward_kernel[(triton.cdiv(rows, block_m),)](
        adapter_input_2d,
        lora_a_weight,
        hidden,
        rows,
        in_features,
        rank,
        block_m,
        block_r,
        block_k,
        num_warps=4,
    )
    _lora_out_forward_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_o))](
        x_2d,
        hidden,
        base_weight,
        base_bias if base_bias is not None else x_2d,
        lora_b_weight,
        out,
        rows,
        in_features,
        out_features,
        rank,
        base_bias is not None,
        float(scaling),
        block_m,
        block_o,
        block_k,
        block_r,
        num_warps=4,
    )
    return out, hidden


class _TritonLoraLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        adapter_input: Tensor,
        base_weight: Tensor,
        base_bias: Tensor | None,
        lora_a_weight: Tensor,
        lora_b_weight: Tensor,
        scaling: float,
    ):
        original_x_shape = x.shape
        in_features = original_x_shape[-1]
        x_2d = x.contiguous().reshape(-1, in_features)
        adapter_input_2d = adapter_input.contiguous().reshape(-1, in_features)
        base_weight = base_weight.contiguous()
        base_bias = None if base_bias is None else base_bias.contiguous()
        lora_a_weight = lora_a_weight.contiguous()
        lora_b_weight = lora_b_weight.contiguous()
        out, hidden = _lora_linear_forward(x_2d, adapter_input_2d, base_weight, base_bias, lora_a_weight, lora_b_weight, scaling)
        ctx.save_for_backward(x_2d, adapter_input_2d, hidden, base_weight, lora_a_weight, lora_b_weight)
        ctx.has_bias = base_bias is not None
        ctx.scaling = float(scaling)
        ctx.original_x_shape = original_x_shape
        return out.reshape(*original_x_shape[:-1], base_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, adapter_input_2d, hidden, base_weight, lora_a_weight, lora_b_weight = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().reshape(-1, base_weight.shape[0])
        grads = lora_linear_backward(
            x_2d,
            adapter_input_2d,
            hidden,
            base_weight,
            lora_a_weight,
            lora_b_weight,
            grad_out_2d,
            ctx.scaling,
            ctx.has_bias,
        )
        grad_x, grad_adapter_input, grad_base_weight, grad_base_bias, grad_lora_a_weight, grad_lora_b_weight = grads
        return (
            grad_x.reshape(ctx.original_x_shape),
            grad_adapter_input.reshape(ctx.original_x_shape),
            grad_base_weight,
            grad_base_bias,
            grad_lora_a_weight,
            grad_lora_b_weight,
            None,
        )


def lora_linear(
    x: Tensor,
    adapter_input: Tensor,
    base_weight: Tensor,
    base_bias: Tensor | None,
    lora_a_weight: Tensor,
    lora_b_weight: Tensor,
    scaling: float,
) -> Tensor:
    if not x.is_cuda:
        return ref_lora_linear(x, adapter_input, base_weight, base_bias, lora_a_weight, lora_b_weight, scaling)
    return _TritonLoraLinear.apply(x, adapter_input, base_weight, base_bias, lora_a_weight, lora_b_weight, scaling)
