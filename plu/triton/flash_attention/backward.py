from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _flash_attention_delta_kernel(
    out_ptr,
    grad_out_ptr,
    delta_ptr,
    query_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, block_d)
    mask = (offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim)
    base = pid_bh * query_len * head_dim
    out = tl.load(out_ptr + base + offs_m[:, None] * head_dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + base + offs_m[:, None] * head_dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(out * grad, axis=1)
    tl.store(delta_ptr + pid_bh * query_len + offs_m, delta, mask=offs_m < query_len)


@triton.jit
def _flash_attention_backward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    grad_out_ptr,
    lse_ptr,
    delta_ptr,
    grad_query_ptr,
    grad_key_ptr,
    grad_value_ptr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_d: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    use_bf16_dot: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // heads
    pid_h = pid_bh - pid_b * heads
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q_base = pid_b * q_stride_b + pid_h * q_stride_h
    k_base = pid_b * k_stride_b + pid_h * k_stride_h
    v_base = pid_b * v_stride_b + pid_h * v_stride_h
    q_out_base = pid_bh * query_len * head_dim
    k_out_base = pid_bh * key_len * head_dim
    q = tl.load(
        query_ptr + q_base + offs_m[:, None] * q_stride_m + offs_d[None, :] * q_stride_d,
        mask=(offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    k = tl.load(
        key_ptr + k_base + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_d,
        mask=(offs_n[:, None] < key_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    v = tl.load(
        value_ptr + v_base + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d,
        mask=(offs_n[:, None] < key_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    grad_out = tl.load(
        grad_out_ptr + q_out_base + offs_m[:, None] * head_dim + offs_d[None, :],
        mask=(offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
    key_mask = offs_n[None, :] < key_len
    query_mask = offs_m[:, None] < query_len
    scores = tl.where(query_mask & key_mask, scores, -float("inf"))
    if causal:
        scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -float("inf"))

    lse = tl.load(lse_ptr + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=-float("inf")).to(tl.float32)
    probs = tl.exp(scores - lse[:, None])
    probs = tl.where(query_mask & key_mask, probs, 0.0)
    if causal:
        probs = tl.where(offs_m[:, None] >= offs_n[None, :], probs, 0.0)

    if use_bf16_dot:
        grad_value = tl.dot(tl.trans(probs.to(tl.bfloat16)), grad_out, input_precision="ieee")
    else:
        grad_value = tl.dot(tl.trans(probs), grad_out.to(tl.float32), input_precision="ieee")
    grad_probs = tl.dot(grad_out, tl.trans(v), input_precision="ieee")
    delta = tl.load(delta_ptr + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0).to(tl.float32)
    grad_scores = probs * (grad_probs - delta[:, None])
    grad_scores = tl.where(query_mask & key_mask, grad_scores, 0.0)
    if causal:
        grad_scores = tl.where(offs_m[:, None] >= offs_n[None, :], grad_scores, 0.0)

    if use_bf16_dot:
        grad_scores_dot = grad_scores.to(tl.bfloat16)
        grad_query = tl.dot(grad_scores_dot, k, input_precision="ieee") * scale
        grad_key = tl.dot(tl.trans(grad_scores_dot), q, input_precision="ieee") * scale
    else:
        grad_query = tl.dot(grad_scores, k.to(tl.float32), input_precision="ieee") * scale
        grad_key = tl.dot(tl.trans(grad_scores), q.to(tl.float32), input_precision="ieee") * scale

    q_offsets = q_out_base + offs_m[:, None] * head_dim + offs_d[None, :]
    k_offsets = k_out_base + offs_n[:, None] * head_dim + offs_d[None, :]
    q_store_mask = (offs_m[:, None] < query_len) & (offs_d[None, :] < head_dim)
    k_store_mask = (offs_n[:, None] < key_len) & (offs_d[None, :] < head_dim)
    tl.atomic_add(grad_query_ptr + q_offsets, grad_query, sem="relaxed", mask=q_store_mask)
    tl.atomic_add(grad_key_ptr + k_offsets, grad_key, sem="relaxed", mask=k_store_mask)
    tl.atomic_add(grad_value_ptr + k_offsets, grad_value, sem="relaxed", mask=k_store_mask)


def flash_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, heads, query_len, head_dim = query.shape
    key_len = key.shape[2]
    batch_heads = batch * heads
    out_2d = out.reshape(batch_heads, query_len, head_dim)
    grad_out_2d = grad_out.contiguous().reshape(batch_heads, query_len, head_dim)

    grad_query = torch.zeros((batch_heads, query_len, head_dim), device=query.device, dtype=query.dtype)
    grad_key = torch.zeros((batch_heads, key_len, head_dim), device=query.device, dtype=key.dtype)
    grad_value = torch.zeros((batch_heads, key_len, head_dim), device=value.device, dtype=value.dtype)
    delta = torch.empty((batch_heads, query_len), device=query.device, dtype=torch.float32)
    if batch_heads == 0 or query_len == 0:
        return grad_query.reshape_as(query), grad_key.reshape_as(key), grad_value.reshape_as(value)

    block_m = 128
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    _flash_attention_delta_kernel[(batch_heads, triton.cdiv(query_len, block_m))](
        out_2d,
        grad_out_2d,
        delta,
        query_len,
        head_dim,
        block_m,
        block_d,
        num_warps=4,
    )
    _flash_attention_backward_kernel[
        (batch_heads, triton.cdiv(query_len, block_m), triton.cdiv(key_len, block_n))
    ](
        query,
        key,
        value,
        grad_out_2d,
        lse,
        delta,
        grad_query,
        grad_key,
        grad_value,
        heads,
        query_len,
        key_len,
        head_dim,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        head_dim**-0.5,
        causal,
        query.dtype is torch.bfloat16,
        block_m,
        block_n,
        block_d,
        num_warps=4,
    )
    return grad_query.reshape_as(query), grad_key.reshape_as(key), grad_value.reshape_as(value)
