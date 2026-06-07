from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.paged_kv_cache import make_static_page_table, paged_self_kv_cache_attention, resample_page_table, resample_state


def dense_decode_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    for batch_index, pos in enumerate(position.tolist()):
        key_cache[batch_index, :, pos, :].copy_(key[batch_index, :, 0, :])
        value_cache[batch_index, :, pos, :].copy_(value[batch_index, :, 0, :])
    max_len = int(position.max().item()) + 1
    prefix_key = key_cache[:, :, :max_len, :]
    prefix_value = value_cache[:, :, :max_len, :]
    scores = torch.matmul(query.float(), prefix_key.float().transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    valid = torch.arange(max_len)[None, :] <= position[:, None]
    scores = scores.masked_fill(~valid[:, None, None, :], -float("inf"))
    probs = F.softmax(scores, dim=-1).to(query.dtype)
    return torch.matmul(probs, prefix_value)


def test_paged_self_kv_cache_attention_matches_dense_cache():
    torch.manual_seed(0)
    batch, heads, max_len, head_dim, page_size = 3, 2, 7, 5, 3
    max_pages = (max_len + page_size - 1) // page_size
    page_table = make_static_page_table(batch, max_pages)
    key_pages = torch.empty(batch * max_pages, heads, page_size, head_dim)
    value_pages = torch.empty_like(key_pages)
    key_cache = torch.empty(batch, heads, max_len, head_dim)
    value_cache = torch.empty_like(key_cache)

    for pos in range(max_len):
        query = torch.randn(batch, heads, 1, head_dim)
        key = torch.randn(batch, heads, 1, head_dim)
        value = torch.randn(batch, heads, 1, head_dim)
        position = torch.full((batch,), pos, dtype=torch.long)

        actual = paged_self_kv_cache_attention(query, key, value, key_pages, value_pages, page_table, position)
        expected = dense_decode_step(query, key, value, key_cache, value_cache, position)

        torch.testing.assert_close(actual, expected)


def test_resampled_page_table_shares_prefix_pages_and_uses_fresh_writable_pages():
    torch.manual_seed(1)
    batch, heads, max_len, head_dim, page_size = 4, 2, 5, 4, 2
    max_pages = (max_len + page_size - 1) // page_size
    page_table = make_static_page_table(batch, max_pages)
    key_pages = torch.empty(batch * max_pages, heads, page_size, head_dim)
    value_pages = torch.empty_like(key_pages)
    key_cache = torch.empty(batch, heads, max_len, head_dim)
    value_cache = torch.empty_like(key_cache)

    for pos in range(page_size):
        query = torch.randn(batch, heads, 1, head_dim)
        key = torch.randn(batch, heads, 1, head_dim)
        value = torch.randn(batch, heads, 1, head_dim)
        position = torch.full((batch,), pos, dtype=torch.long)
        paged_self_kv_cache_attention(query, key, value, key_pages, value_pages, page_table, position)
        dense_decode_step(query, key, value, key_cache, value_cache, position)

    ancestors = torch.tensor([2, 2, 0, 1])
    next_page_ids = torch.arange(batch, dtype=torch.long) + batch
    page_table = resample_page_table(page_table, ancestors, next_page_ids=next_page_ids, next_page_index=1)
    key_cache = resample_state(key_cache, ancestors).clone()
    value_cache = resample_state(value_cache, ancestors).clone()

    query = torch.randn(batch, heads, 1, head_dim)
    key = torch.randn(batch, heads, 1, head_dim)
    value = torch.randn(batch, heads, 1, head_dim)
    position = torch.full((batch,), page_size, dtype=torch.long)

    actual = paged_self_kv_cache_attention(query, key, value, key_pages, value_pages, page_table, position)
    expected = dense_decode_step(query, key, value, key_cache, value_cache, position)

    torch.testing.assert_close(actual, expected)
    assert page_table[0, 0] == page_table[1, 0]
    assert page_table[0, 1] != page_table[1, 1]


def test_resample_state_gathers_batch_rows():
    state = torch.tensor([[0, 1], [2, 3], [4, 5]])
    ancestors = torch.tensor([2, 0, 2, 1])

    actual = resample_state(state, ancestors)

    torch.testing.assert_close(actual, torch.tensor([[4, 5], [0, 1], [4, 5], [2, 3]]))
