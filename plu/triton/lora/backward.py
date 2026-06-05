from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.gelu_mlp.backward import linear_input_grad, linear_weight_bias_grad


@triton.jit
def _scale_kernel(
    input_ptr,
    output_ptr,
    total: tl.constexpr,
    scale: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values * scale, mask=mask)


def scale_tensor(x: Tensor, scale: float) -> Tensor:
    out = torch.empty_like(x)
    total = x.numel()
    if total:
        block_size = 256
        _scale_kernel[(triton.cdiv(total, block_size),)](x, out, total, float(scale), block_size, num_warps=4)
    return out


def lora_linear_backward(
    x_2d: Tensor,
    adapter_input_2d: Tensor,
    hidden: Tensor,
    base_weight: Tensor,
    lora_a_weight: Tensor,
    lora_b_weight: Tensor,
    grad_out_2d: Tensor,
    scaling: float,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    grad_x = linear_input_grad(grad_out_2d, base_weight)
    grad_base_weight, grad_base_bias = linear_weight_bias_grad(grad_out_2d, x_2d, has_bias, dtype=base_weight.dtype)

    grad_hidden = scale_tensor(linear_input_grad(grad_out_2d, lora_b_weight), scaling)
    grad_lora_b_weight = scale_tensor(
        linear_weight_bias_grad(grad_out_2d, hidden, False, dtype=lora_b_weight.dtype)[0],
        scaling,
    )
    grad_adapter_input = linear_input_grad(grad_hidden, lora_a_weight)
    grad_lora_a_weight = linear_weight_bias_grad(grad_hidden, adapter_input_2d, False, dtype=lora_a_weight.dtype)[0]
    return grad_x, grad_adapter_input, grad_base_weight, grad_base_bias, grad_lora_a_weight, grad_lora_b_weight

