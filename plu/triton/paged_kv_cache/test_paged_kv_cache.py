from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.paged_kv_cache import paged_self_kv_cache_attention as ref_paged_self_kv_cache_attention
from plu.ref.paged_kv_cache import resample_page_table as ref_resample_page_table
from plu.ref.paged_kv_cache import resample_state as ref_resample_state
from plu.triton.paged_kv_cache import make_static_page_table, paged_self_kv_cache_attention, resample_page_table, resample_state


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.bfloat16, 6e-2, 6e-2)])
def test_paged_self_kv_cache_attention_matches_ref(dtype, atol, rtol):
    torch.manual_seed(0)
    batch, heads, max_len, head_dim, page_size = 3, 2, 9, 16, 4
    max_pages = (max_len + page_size - 1) // page_size
    page_table = make_static_page_table(batch, max_pages, device="cuda")
    ref_key_pages = torch.zeros(batch * max_pages, heads, page_size, head_dim, device="cuda", dtype=dtype)
    ref_value_pages = torch.zeros_like(ref_key_pages)
    triton_key_pages = torch.zeros_like(ref_key_pages)
    triton_value_pages = torch.zeros_like(ref_key_pages)

    for pos in range(max_len):
        query = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
        key = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
        value = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=dtype)
        position = torch.full((batch,), pos, device="cuda", dtype=torch.long)

        expected = ref_paged_self_kv_cache_attention(query, key, value, ref_key_pages, ref_value_pages, page_table, position)
        actual = paged_self_kv_cache_attention(query, key, value, triton_key_pages, triton_value_pages, page_table, position)

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(triton_key_pages, ref_key_pages, atol=0, rtol=0)
        torch.testing.assert_close(triton_value_pages, ref_value_pages, atol=0, rtol=0)


def test_resample_page_table_matches_ref_with_duplicate_ancestors():
    page_table = make_static_page_table(4, 5, device="cuda")
    ancestors = torch.tensor([2, 2, 0, 1], device="cuda")
    next_page_ids = torch.tensor([20, 21, 22, 23], device="cuda")

    expected = ref_resample_page_table(page_table.cpu(), ancestors.cpu(), next_page_ids=next_page_ids.cpu(), next_page_index=3).cuda()
    actual = resample_page_table(page_table, ancestors, next_page_ids=next_page_ids, next_page_index=3)

    torch.testing.assert_close(actual, expected)
    assert actual[0, 0] == actual[1, 0]
    assert actual[0, 3] != actual[1, 3]


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bfloat16])
def test_resample_state_matches_ref(dtype):
    if dtype.is_floating_point:
        state = torch.randn(4, 3, 2, device="cuda", dtype=dtype)
    else:
        state = torch.arange(24, device="cuda", dtype=dtype).view(4, 3, 2)
    ancestors = torch.tensor([3, 0, 3, 1], device="cuda")

    expected = ref_resample_state(state.cpu(), ancestors.cpu()).cuda()
    actual = resample_state(state, ancestors)

    torch.testing.assert_close(actual, expected)


def test_resampled_attention_matches_ref_after_page_boundary_resample():
    torch.manual_seed(1)
    batch, heads, max_len, head_dim, page_size = 4, 2, 5, 16, 2
    max_pages = (max_len + page_size - 1) // page_size
    page_table = make_static_page_table(batch, max_pages, device="cuda")
    ref_key_pages = torch.zeros(batch * max_pages, heads, page_size, head_dim, device="cuda")
    ref_value_pages = torch.zeros_like(ref_key_pages)
    triton_key_pages = torch.zeros_like(ref_key_pages)
    triton_value_pages = torch.zeros_like(ref_key_pages)

    for pos in range(page_size):
        query = torch.randn(batch, heads, 1, head_dim, device="cuda")
        key = torch.randn(batch, heads, 1, head_dim, device="cuda")
        value = torch.randn(batch, heads, 1, head_dim, device="cuda")
        position = torch.full((batch,), pos, device="cuda", dtype=torch.long)
        ref_paged_self_kv_cache_attention(query, key, value, ref_key_pages, ref_value_pages, page_table, position)
        paged_self_kv_cache_attention(query, key, value, triton_key_pages, triton_value_pages, page_table, position)

    ancestors = torch.tensor([2, 2, 0, 1], device="cuda")
    next_page_ids = torch.arange(batch, device="cuda", dtype=torch.long) + batch
    ref_page_table = ref_resample_page_table(page_table.cpu(), ancestors.cpu(), next_page_ids=next_page_ids.cpu(), next_page_index=1).cuda()
    triton_page_table = resample_page_table(page_table, ancestors, next_page_ids=next_page_ids, next_page_index=1)

    query = torch.randn(batch, heads, 1, head_dim, device="cuda")
    key = torch.randn(batch, heads, 1, head_dim, device="cuda")
    value = torch.randn(batch, heads, 1, head_dim, device="cuda")
    position = torch.full((batch,), page_size, device="cuda", dtype=torch.long)
    expected = ref_paged_self_kv_cache_attention(query, key, value, ref_key_pages, ref_value_pages, ref_page_table, position)
    actual = paged_self_kv_cache_attention(query, key, value, triton_key_pages, triton_value_pages, triton_page_table, position)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
