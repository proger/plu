from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _decoder_embedding_forward_kernel(
    ids_ptr,
    token_ptr,
    position_ptr,
    out_ptr,
    batch: tl.constexpr,
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
    token = tl.load(token_ptr + token_id * hidden + d, mask=mask, other=0.0)
    position = tl.load(position_ptr + t * hidden + d, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, token + position, mask=mask)


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
def _encoder_position_forward_kernel(
    hidden_ptr,
    position_ptr,
    out_ptr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    d = offsets % hidden_size
    t = (offsets // hidden_size) % seq_len
    value = tl.load(hidden_ptr + offsets, mask=mask, other=0.0)
    position = tl.load(position_ptr + t * hidden_size + d, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, value + position, mask=mask)


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


class _TritonDecoderEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids: Tensor, token_weight: Tensor, position_weight: Tensor, dtype: torch.dtype):
        input_ids = input_ids.contiguous()
        token_weight = token_weight.contiguous()
        position_weight = position_weight.contiguous()
        batch, seq_len = input_ids.shape
        hidden = token_weight.shape[1]
        out = torch.empty((batch, seq_len, hidden), device=input_ids.device, dtype=dtype)
        total = out.numel()
        if total:
            block_size = 256
            _decoder_embedding_forward_kernel[(triton.cdiv(total, block_size),)](
                input_ids,
                token_weight,
                position_weight,
                out,
                batch,
                seq_len,
                hidden,
                total,
                block_size,
                num_warps=4,
            )
        ctx.save_for_backward(input_ids)
        ctx.token_shape = token_weight.shape
        ctx.position_shape = position_weight.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (input_ids,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        batch, seq_len, hidden = grad_out.shape
        grad_token = torch.zeros(ctx.token_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_position = torch.zeros(ctx.position_shape, device=grad_out.device, dtype=grad_out.dtype)
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
        return None, grad_token, grad_position, None


class _TritonEncoderPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: Tensor, position_weight: Tensor):
        hidden_states = hidden_states.contiguous()
        position_weight = position_weight.contiguous()
        out = torch.empty_like(hidden_states)
        batch, seq_len, hidden_size = hidden_states.shape
        total = out.numel()
        if total:
            block_size = 256
            _encoder_position_forward_kernel[(triton.cdiv(total, block_size),)](
                hidden_states,
                position_weight,
                out,
                batch,
                seq_len,
                hidden_size,
                total,
                block_size,
                num_warps=4,
            )
        ctx.position_shape = position_weight.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        grad_out = grad_out.contiguous()
        batch, seq_len, hidden_size = grad_out.shape
        if batch == 1 and ctx.position_shape[0] == seq_len:
            return grad_out, grad_out.reshape(seq_len, hidden_size)

        grad_position = torch.zeros(ctx.position_shape, device=grad_out.device, dtype=grad_out.dtype)
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


def decoder_embedding(input_ids: Tensor, token_weight: Tensor, position_weight: Tensor, dtype: torch.dtype) -> Tensor:
    if not input_ids.is_cuda:
        raise RuntimeError("plu.triton.embedding.decoder_embedding requires CUDA tensors")
    return _TritonDecoderEmbedding.apply(input_ids, token_weight, position_weight, dtype)


def encoder_position_embedding(hidden_states: Tensor, position_weight: Tensor) -> Tensor:
    if not hidden_states.is_cuda:
        raise RuntimeError("plu.triton.embedding.encoder_position_embedding requires CUDA tensors")
    return _TritonEncoderPositionEmbedding.apply(hidden_states, position_weight)
