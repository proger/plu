from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _decoder_embedding_backward_kernel(
    ids_ptr,
    grad_out_ptr,
    grad_token_ptr,
    grad_position_ptr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    d = offsets % hidden
    t = (offsets // hidden) % seq_len
    b = offsets // (seq_len * hidden)
    token_id = tl.load(ids_ptr + b * seq_len + t, mask=mask, other=0)
    grad = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(grad_token_ptr + token_id * hidden + d, grad, sem="relaxed", mask=mask)
    tl.atomic_add(grad_position_ptr + t * hidden + d, grad, sem="relaxed", mask=mask)


@triton.jit
def _encoder_position_backward_kernel(
    grad_out_ptr,
    grad_position_ptr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    d = offsets % hidden_size
    t = (offsets // hidden_size) % seq_len
    grad = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(grad_position_ptr + t * hidden_size + d, grad, sem="relaxed", mask=mask)


def decoder_embedding_backward(
    grad_out: Tensor,
    input_ids: Tensor,
    token_shape: torch.Size,
    position_shape: torch.Size,
) -> tuple[Tensor, Tensor]:
    grad_out = grad_out.contiguous()
    _, seq_len, hidden = grad_out.shape
    grad_token = torch.zeros(token_shape, device=grad_out.device, dtype=grad_out.dtype)
    grad_position = torch.zeros(position_shape, device=grad_out.device, dtype=grad_out.dtype)
    total = grad_out.numel()
    if total:
        block_size = 256
        _decoder_embedding_backward_kernel[(triton.cdiv(total, block_size),)](
            input_ids,
            grad_out,
            grad_token,
            grad_position,
            seq_len,
            hidden,
            total,
            block_size,
            num_warps=4,
        )
    return grad_token, grad_position


def encoder_position_embedding_backward(
    grad_out: Tensor,
    position_shape: torch.Size,
) -> tuple[Tensor, Tensor]:
    grad_out = grad_out.contiguous()
    batch, seq_len, hidden_size = grad_out.shape
    if batch == 1 and position_shape[0] == seq_len:
        return grad_out, grad_out.reshape(seq_len, hidden_size)

    grad_position = torch.zeros(position_shape, device=grad_out.device, dtype=grad_out.dtype)
    total = grad_out.numel()
    if total:
        block_size = 256
        _encoder_position_backward_kernel[(triton.cdiv(total, block_size),)](
            grad_out,
            grad_position,
            seq_len,
            hidden_size,
            total,
            block_size,
            num_warps=4,
        )
    return grad_out, grad_position
