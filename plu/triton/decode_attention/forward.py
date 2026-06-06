from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _self_kv_cache_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    position_ptr,
    out_ptr,
    max_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offs_d = tl.arange(0, block_d)
    pos = tl.load(position_ptr + batch)
    query_base = (batch * tl.num_programs(1) + head) * head_dim
    cache_base = (batch * tl.num_programs(1) + head) * max_len * head_dim
    dim_mask = offs_d < head_dim

    query = tl.load(query_ptr + query_base + offs_d, mask=dim_mask, other=0.0).to(tl.float32)
    key = tl.load(key_ptr + query_base + offs_d, mask=dim_mask, other=0.0)
    value = tl.load(value_ptr + query_base + offs_d, mask=dim_mask, other=0.0)
    tl.store(key_cache_ptr + cache_base + pos * head_dim + offs_d, key, mask=dim_mask)
    tl.store(value_cache_ptr + cache_base + pos * head_dim + offs_d, value, mask=dim_mask)

    acc = tl.zeros((block_d,), dtype=tl.float32)
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    row_sum = tl.full((), 0.0, dtype=tl.float32)
    offs_n = tl.arange(0, block_n)

    for start in tl.range(0, max_len, block_n):
        n = start + offs_n
        valid_n = n <= pos
        cached_key = tl.load(
            key_cache_ptr + cache_base + n[:, None] * head_dim + offs_d[None, :],
            mask=valid_n[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(cached_key * query[None, :], axis=1) * scale
        scores = tl.where(valid_n, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_row_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_row_max)
        probs = tl.exp(scores - new_row_max)
        probs = tl.where(valid_n, probs, 0.0)

        cached_value = tl.load(
            value_cache_ptr + cache_base + n[:, None] * head_dim + offs_d[None, :],
            mask=valid_n[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * cached_value, axis=0)
        row_sum = row_sum * alpha + tl.sum(probs, axis=0)
        row_max = new_row_max

    out = acc / row_sum
    tl.store(out_ptr + query_base + offs_d, out, mask=dim_mask)


@triton.jit
def _cross_kv_cache_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    out_ptr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offs_d = tl.arange(0, block_d)
    query_base = (batch * tl.num_programs(1) + head) * head_dim
    cache_base = (batch * tl.num_programs(1) + head) * key_len * head_dim
    dim_mask = offs_d < head_dim

    query = tl.load(query_ptr + query_base + offs_d, mask=dim_mask, other=0.0).to(tl.float32)
    acc = tl.zeros((block_d,), dtype=tl.float32)
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    row_sum = tl.full((), 0.0, dtype=tl.float32)
    offs_n = tl.arange(0, block_n)

    for start in tl.range(0, key_len, block_n):
        n = start + offs_n
        valid_n = n < key_len
        cached_key = tl.load(
            key_cache_ptr + cache_base + n[:, None] * head_dim + offs_d[None, :],
            mask=valid_n[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(cached_key * query[None, :], axis=1) * scale
        scores = tl.where(valid_n, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_row_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_row_max)
        probs = tl.exp(scores - new_row_max)
        probs = tl.where(valid_n, probs, 0.0)

        cached_value = tl.load(
            value_cache_ptr + cache_base + n[:, None] * head_dim + offs_d[None, :],
            mask=valid_n[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * cached_value, axis=0)
        row_sum = row_sum * alpha + tl.sum(probs, axis=0)
        row_max = new_row_max

    out = acc / row_sum
    tl.store(out_ptr + query_base + offs_d, out, mask=dim_mask)


def _validate_query(query: Tensor) -> tuple[int, int, int, int]:
    if not query.is_cuda:
        raise RuntimeError("decode attention requires CUDA tensors")
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError(f"query must have shape [batch, heads, 1, head_dim], got {tuple(query.shape)}")
    batch, heads, _, head_dim = query.shape
    if head_dim > 128:
        raise ValueError("decode attention supports head_dim <= 128")
    return batch, heads, head_dim, max(16, triton.next_power_of_2(head_dim))


def self_kv_cache_attention(query: Tensor, key: Tensor, value: Tensor, key_cache: Tensor, value_cache: Tensor, position: Tensor) -> Tensor:
    batch, heads, head_dim, block_d = _validate_query(query)
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must have matching shapes")
    if key_cache.shape != value_cache.shape or key_cache.shape[:2] != (batch, heads) or key_cache.shape[3] != head_dim:
        raise ValueError("key/value caches must have shape [batch, heads, max_len, head_dim]")
    if position.numel() != batch or not position.is_cuda:
        raise ValueError("position must have one CUDA element per batch row")

    out = torch.empty_like(query)
    block_n = 64
    _self_kv_cache_attention_kernel[(batch, heads)](
        query,
        key,
        value,
        key_cache,
        value_cache,
        position,
        out,
        key_cache.shape[2],
        head_dim,
        head_dim**-0.5,
        block_n,
        block_d,
        num_warps=4,
    )
    return out


def cross_kv_cache_attention(query: Tensor, key_cache: Tensor, value_cache: Tensor) -> Tensor:
    batch, heads, head_dim, block_d = _validate_query(query)
    if key_cache.shape != value_cache.shape or key_cache.shape[:2] != (batch, heads) or key_cache.shape[3] != head_dim:
        raise ValueError("key/value caches must have shape [batch, heads, key_len, head_dim]")

    out = torch.empty_like(query)
    block_n = 64
    _cross_kv_cache_attention_kernel[(batch, heads)](
        query,
        key_cache,
        value_cache,
        out,
        key_cache.shape[2],
        head_dim,
        head_dim**-0.5,
        block_n,
        block_d,
        num_warps=4,
    )
    return out
