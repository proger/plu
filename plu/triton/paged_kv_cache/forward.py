from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def make_static_page_table(batch: int, max_pages: int, *, device: torch.device | str | None = None) -> Tensor:
    return torch.arange(batch * max_pages, device=device, dtype=torch.long).view(max_pages, batch).t().contiguous()


@triton.jit
def _paged_self_kv_cache_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    key_pages_ptr,
    value_pages_ptr,
    page_table_ptr,
    position_ptr,
    out_ptr,
    max_pages: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    heads = tl.num_programs(1)
    offs_d = tl.arange(0, block_d)
    dim_mask = offs_d < head_dim
    pos = tl.load(position_ptr + batch)
    page_index = pos // page_size
    page_offset = pos - page_index * page_size
    page_id = tl.load(page_table_ptr + batch * max_pages + page_index)

    query_base = (batch * heads + head) * head_dim
    write_base = ((page_id * heads + head) * page_size + page_offset) * head_dim
    query = tl.load(query_ptr + query_base + offs_d, mask=dim_mask, other=0.0).to(tl.float32)
    key = tl.load(key_ptr + query_base + offs_d, mask=dim_mask, other=0.0)
    value = tl.load(value_ptr + query_base + offs_d, mask=dim_mask, other=0.0)
    tl.store(key_pages_ptr + write_base + offs_d, key, mask=dim_mask)
    tl.store(value_pages_ptr + write_base + offs_d, value, mask=dim_mask)

    acc = tl.zeros((block_d,), dtype=tl.float32)
    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    row_sum = tl.full((), 0.0, dtype=tl.float32)
    offs_n = tl.arange(0, block_n)
    max_len: tl.constexpr = max_pages * page_size

    for start in tl.range(0, max_len, block_n):
        n = start + offs_n
        valid_n = n <= pos
        read_page_index = n // page_size
        read_page_offset = n - read_page_index * page_size
        read_page_id = tl.load(page_table_ptr + batch * max_pages + read_page_index, mask=valid_n, other=0)
        read_base = ((read_page_id[:, None] * heads + head) * page_size + read_page_offset[:, None]) * head_dim
        cached_key = tl.load(
            key_pages_ptr + read_base + offs_d[None, :],
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
            value_pages_ptr + read_base + offs_d[None, :],
            mask=valid_n[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * cached_value, axis=0)
        row_sum = row_sum * alpha + tl.sum(probs, axis=0)
        row_max = new_row_max

    out = acc / row_sum
    tl.store(out_ptr + query_base + offs_d, out, mask=dim_mask)


@triton.jit
def _resample_page_table_kernel(
    page_table_ptr,
    ancestors_ptr,
    next_page_ids_ptr,
    out_ptr,
    max_pages: tl.constexpr,
    next_page_index: tl.constexpr,
    has_next_page: tl.constexpr,
    block_m: tl.constexpr,
):
    batch = tl.program_id(0)
    offs = tl.arange(0, block_m)
    mask = offs < max_pages
    ancestor = tl.load(ancestors_ptr + batch)
    values = tl.load(page_table_ptr + ancestor * max_pages + offs, mask=mask, other=0)
    if has_next_page:
        next_page_id = tl.load(next_page_ids_ptr + batch)
        values = tl.where(offs == next_page_index, next_page_id, values)
    tl.store(out_ptr + batch * max_pages + offs, values, mask=mask)


@triton.jit
def _resample_state_kernel(
    state_ptr,
    ancestors_ptr,
    out_ptr,
    cols: tl.constexpr,
    block_c: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * block_c + tl.arange(0, block_c)
    mask = offs < cols
    ancestor = tl.load(ancestors_ptr + batch)
    values = tl.load(state_ptr + ancestor * cols + offs, mask=mask, other=0.0)
    tl.store(out_ptr + batch * cols + offs, values, mask=mask)


def _validate_query(query: Tensor) -> tuple[int, int, int, int]:
    if not query.is_cuda:
        raise RuntimeError("paged KV cache attention requires CUDA tensors")
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError(f"query must have shape [batch, heads, 1, head_dim], got {tuple(query.shape)}")
    batch, heads, _, head_dim = query.shape
    if head_dim > 128:
        raise ValueError("paged KV cache attention supports head_dim <= 128")
    return batch, heads, head_dim, max(16, triton.next_power_of_2(head_dim))


def paged_self_kv_cache_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_pages: Tensor,
    value_pages: Tensor,
    page_table: Tensor,
    position: Tensor,
) -> Tensor:
    batch, heads, head_dim, block_d = _validate_query(query)
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must have matching shapes")
    if key_pages.ndim != 4 or key_pages.shape != value_pages.shape:
        raise ValueError("key_pages and value_pages must have matching shape [pages, heads, page_size, head_dim]")
    if key_pages.shape[1] != heads or key_pages.shape[3] != head_dim:
        raise ValueError("page cache heads/head_dim must match query")
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError(f"page_table must have shape [batch, max_pages], got {tuple(page_table.shape)}")
    if position.ndim != 1 or position.numel() != batch or not position.is_cuda:
        raise ValueError("position must have one CUDA element per batch row")
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("key_pages", key_pages),
        ("value_pages", value_pages),
        ("page_table", page_table),
    ):
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    out = torch.empty_like(query)
    block_n = 64
    _paged_self_kv_cache_attention_kernel[(batch, heads)](
        query,
        key,
        value,
        key_pages,
        value_pages,
        page_table,
        position,
        out,
        page_table.shape[1],
        key_pages.shape[2],
        head_dim,
        head_dim**-0.5,
        block_n,
        block_d,
        num_warps=4,
    )
    return out


def resample_page_table(page_table: Tensor, ancestors: Tensor, next_page_ids: Tensor | None = None, next_page_index: int | None = None) -> Tensor:
    if not page_table.is_cuda or not ancestors.is_cuda:
        raise RuntimeError("page_table and ancestors must be CUDA tensors")
    if page_table.ndim != 2 or ancestors.ndim != 1 or ancestors.numel() != page_table.shape[0]:
        raise ValueError("page_table must be [batch, max_pages] and ancestors must be [batch]")
    if not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous")
    if next_page_ids is None:
        if next_page_index is not None:
            raise ValueError("next_page_index requires next_page_ids")
        next_page_ids = ancestors
        has_next_page = False
        next_page_index = 0
    else:
        if next_page_index is None:
            raise ValueError("next_page_ids requires next_page_index")
        if not next_page_ids.is_cuda or next_page_ids.ndim != 1 or next_page_ids.numel() != page_table.shape[0]:
            raise ValueError("next_page_ids must be a CUDA tensor with shape [batch]")
        has_next_page = True
    output = torch.empty_like(page_table)
    block_m = triton.next_power_of_2(page_table.shape[1])
    _resample_page_table_kernel[(page_table.shape[0],)](
        page_table,
        ancestors,
        next_page_ids,
        output,
        page_table.shape[1],
        next_page_index,
        has_next_page,
        block_m,
        num_warps=1,
    )
    return output


def resample_state(state: Tensor, ancestors: Tensor) -> Tensor:
    if not state.is_cuda or not ancestors.is_cuda:
        raise RuntimeError("state and ancestors must be CUDA tensors")
    if state.shape[0] != ancestors.numel() or ancestors.ndim != 1:
        raise ValueError("state batch dimension must match ancestors")
    if not state.is_contiguous():
        raise ValueError("state must be contiguous")
    output = torch.empty_like(state)
    cols = state.numel() // state.shape[0]
    block_c = 256
    grid = (state.shape[0], triton.cdiv(cols, block_c))
    _resample_state_kernel[grid](state, ancestors, output, cols, block_c, num_warps=4)
    return output
