from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def make_static_page_table(batch: int, max_pages: int, *, device: torch.device | str | None = None) -> Tensor:
    """Map logical (batch row, page index) to a unique physical page id."""
    return torch.arange(batch * max_pages, device=device, dtype=torch.long).view(max_pages, batch).t().contiguous()


def resample_state(state: Tensor, ancestors: Tensor) -> Tensor:
    """Gather per-particle metadata by ancestor row."""
    if ancestors.ndim != 1:
        raise ValueError(f"ancestors must be a 1D tensor, got {tuple(ancestors.shape)}")
    return state.index_select(0, ancestors.to(device=state.device, dtype=torch.long))


def resample_page_table(page_table: Tensor, ancestors: Tensor, next_page_ids: Tensor | None = None, next_page_index: int | None = None) -> Tensor:
    """Gather page-table rows and optionally assign fresh writable pages.

    Completed pages are immutable and may be shared by duplicate ancestors. When
    `next_page_ids` is provided, each output particle gets a unique fresh page at
    `next_page_index` for subsequent appends.
    """
    output = resample_state(page_table, ancestors).clone()
    if next_page_ids is None:
        if next_page_index is not None:
            raise ValueError("next_page_index requires next_page_ids")
        return output
    if next_page_index is None:
        raise ValueError("next_page_ids requires next_page_index")
    if next_page_ids.ndim != 1 or next_page_ids.numel() != output.shape[0]:
        raise ValueError(f"next_page_ids must have shape [{output.shape[0]}], got {tuple(next_page_ids.shape)}")
    output[:, next_page_index] = next_page_ids.to(device=output.device, dtype=output.dtype)
    return output


def _validate_decode_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_pages: Tensor,
    value_pages: Tensor,
    page_table: Tensor,
    position: Tensor,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError(f"query must have shape [batch, heads, 1, head_dim], got {tuple(query.shape)}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must have matching shapes")
    if key_pages.ndim != 4:
        raise ValueError(f"key_pages must have shape [pages, heads, page_size, head_dim], got {tuple(key_pages.shape)}")
    if key_pages.shape != value_pages.shape:
        raise ValueError("key_pages and value_pages must have matching shapes")
    batch, heads, _, head_dim = query.shape
    pages, page_heads, page_size, page_head_dim = key_pages.shape
    if page_heads != heads or page_head_dim != head_dim:
        raise ValueError("page cache heads/head_dim must match query")
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError(f"page_table must have shape [batch, max_pages], got {tuple(page_table.shape)}")
    if position.ndim != 1 or position.numel() != batch:
        raise ValueError(f"position must have shape [{batch}], got {tuple(position.shape)}")
    if int(position.max().item()) // page_size >= page_table.shape[1]:
        raise ValueError("position exceeds page_table capacity")
    if int(page_table.max().item()) >= pages or int(page_table.min().item()) < 0:
        raise ValueError("page_table contains an out-of-range page id")
    return batch, heads, head_dim, page_size, int(position.max().item()) + 1


def _write_current_kv(key: Tensor, value: Tensor, key_pages: Tensor, value_pages: Tensor, page_table: Tensor, position: Tensor) -> None:
    page_size = key_pages.shape[2]
    for batch_index, pos in enumerate(position.tolist()):
        page_index = pos // page_size
        page_offset = pos % page_size
        page_id = int(page_table[batch_index, page_index].item())
        key_pages[page_id, :, page_offset, :].copy_(key[batch_index, :, 0, :])
        value_pages[page_id, :, page_offset, :].copy_(value[batch_index, :, 0, :])


def _materialize_prefix(key_pages: Tensor, value_pages: Tensor, page_table: Tensor, position: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
    batch = page_table.shape[0]
    _, heads, _, head_dim = key_pages.shape
    page_size = key_pages.shape[2]
    key = key_pages.new_zeros((batch, heads, max_len, head_dim))
    value = value_pages.new_zeros((batch, heads, max_len, head_dim))
    for batch_index, pos in enumerate(position.tolist()):
        for token_index in range(pos + 1):
            page_index = token_index // page_size
            page_offset = token_index % page_size
            page_id = int(page_table[batch_index, page_index].item())
            key[batch_index, :, token_index, :].copy_(key_pages[page_id, :, page_offset, :])
            value[batch_index, :, token_index, :].copy_(value_pages[page_id, :, page_offset, :])
    return key, value


def paged_self_kv_cache_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_pages: Tensor,
    value_pages: Tensor,
    page_table: Tensor,
    position: Tensor,
) -> Tensor:
    """Reference paged self-attention decode step.

    This mirrors `self_kv_cache_attention`: it writes the current key/value at
    `position`, then attends each row over tokens `[0, position[row]]` through
    the page table.
    """
    _, _, _, _, max_len = _validate_decode_inputs(query, key, value, key_pages, value_pages, page_table, position)
    position = position.to(device=page_table.device, dtype=torch.long)
    _write_current_kv(key, value, key_pages, value_pages, page_table, position)
    prefix_key, prefix_value = _materialize_prefix(key_pages, value_pages, page_table, position, max_len)
    scores = torch.matmul(query.float(), prefix_key.float().transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    valid = torch.arange(max_len, device=query.device)[None, :] <= position.to(query.device)[:, None]
    valid = valid[:, None, None, :]
    scores = scores.masked_fill(~valid, -float("inf"))
    probs = F.softmax(scores, dim=-1).to(dtype=query.dtype)
    return torch.matmul(probs, prefix_value)
