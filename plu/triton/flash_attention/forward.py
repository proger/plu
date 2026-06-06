from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.flash_attention.backward import flash_attention_backward


@triton.jit
def _flash_attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    out_ptr,
    lse_ptr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    q_base = pid_bh * query_len * head_dim
    k_base = pid_bh * key_len * head_dim

    query = tl.load(
        query_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
        mask=(offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    acc = tl.zeros((block_m, block_d), dtype=tl.float32)
    row_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((block_m,), dtype=tl.float32)

    for n_start in tl.range(0, key_len, block_n):
        n = n_start + offs_n
        key = tl.load(
            key_ptr + k_base + n[:, None] * head_dim + offs_d[None, :],
            mask=(n[:, None] < key_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * scale
        valid = (offs_m[:, None] < query_len) & (n[None, :] < key_len)
        scores = tl.where(valid, scores, -float("inf"))
        if causal:
            scores = tl.where(offs_m[:, None] >= n[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_row_max)
        probs = tl.exp(scores - new_row_max[:, None])
        probs = tl.where(valid, probs, 0.0)
        if causal:
            probs = tl.where(offs_m[:, None] >= n[None, :], probs, 0.0)

        value = tl.load(
            value_ptr + k_base + n[:, None] * head_dim + offs_d[None, :],
            mask=(n[:, None] < key_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs, value.to(tl.float32), input_precision="ieee")
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_row_max

    out = acc / row_sum[:, None]
    lse = row_max + tl.log(row_sum)
    out_offsets = q_base + offs_m[:, None] * head_dim + offs_d[None, :]
    mask = (offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim)
    tl.store(out_ptr + out_offsets, out, mask=mask)
    tl.store(lse_ptr + pid_bh * query_len + offs_m, lse, mask=offs_m < query_len)


def _flash_attention_forward(query: Tensor, key: Tensor, value: Tensor, causal: bool) -> tuple[Tensor, Tensor]:
    batch, heads, query_len, head_dim = query.shape
    key_len = key.shape[2]
    batch_heads = batch * heads
    query_2d = query.reshape(batch_heads, query_len, head_dim)
    key_2d = key.reshape(batch_heads, key_len, head_dim)
    value_2d = value.reshape(batch_heads, key_len, head_dim)
    out = torch.empty_like(query_2d)
    lse = torch.empty((batch_heads, query_len), device=query.device, dtype=torch.float32)
    if batch_heads == 0 or query_len == 0:
        return out.reshape_as(query), lse

    block_m = 32
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    _flash_attention_forward_kernel[(batch_heads, triton.cdiv(query_len, block_m))](
        query_2d,
        key_2d,
        value_2d,
        out,
        lse,
        query_len,
        key_len,
        head_dim,
        head_dim**-0.5,
        causal,
        block_m,
        block_n,
        block_d,
        num_warps=4,
    )
    return out.reshape_as(query), lse


class _TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, causal: bool):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        out, lse = _flash_attention_forward(query, key, value, causal)
        ctx.save_for_backward(query, key, value, out, lse)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        query, key, value, out, lse = ctx.saved_tensors
        grad_query, grad_key, grad_value = flash_attention_backward(query, key, value, out, lse, grad_out, ctx.causal)
        return grad_query, grad_key, grad_value, None


def flash_attention(query: Tensor, key: Tensor, value: Tensor, causal_mask: Tensor | None = None) -> Tensor:
    if not query.is_cuda:
        raise RuntimeError("plu.triton.flash_attention requires CUDA tensors")
    if query.shape[-1] > 128:
        raise RuntimeError("plu.triton.flash_attention supports head_dim <= 128")
    causal = causal_mask is not None
    return _TritonFlashAttention.apply(query, key, value, causal)
