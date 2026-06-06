from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.residual_add.backward import residual_add_backward


@triton.jit
def _residual_add_kernel(
    residual_ptr,
    hidden_ptr,
    out_ptr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, residual + hidden, mask=mask)


class _TritonResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: Tensor, hidden_states: Tensor) -> Tensor:
        residual = residual.contiguous()
        hidden_states = hidden_states.contiguous()
        out = torch.empty_like(residual)
        total = residual.numel()
        if total:
            block_size = 256
            _residual_add_kernel[(triton.cdiv(total, block_size),)](
                residual,
                hidden_states,
                out,
                total,
                block_size,
                num_warps=4,
            )
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        return residual_add_backward(grad_out)


def residual_add(residual: Tensor, hidden_states: Tensor) -> Tensor:
    if not residual.is_cuda:
        raise RuntimeError("plu.triton.residual_add requires CUDA tensors")
    return _TritonResidualAdd.apply(residual, hidden_states)
