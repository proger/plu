from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.triton.decode_attention import cross_kv_cache_attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _ref_cross_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cache_index: torch.Tensor | None) -> torch.Tensor:
    selected_key = key if cache_index is None else key[cache_index]
    selected_value = value if cache_index is None else value[cache_index]
    scores = torch.einsum("bhqd,bhkd->bhqk", query.float(), selected_key.float()) * (query.shape[-1] ** -0.5)
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, selected_value.float())
    return out.to(dtype=query.dtype)


def test_cross_kv_cache_attention_indexed_cache_matches_reference():
    torch.manual_seed(0)
    query = torch.randn(6, 2, 1, 32, device="cuda", dtype=torch.float32)
    key = torch.randn(3, 2, 17, 32, device="cuda", dtype=torch.float32)
    value = torch.randn(3, 2, 17, 32, device="cuda", dtype=torch.float32)
    cache_index = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda", dtype=torch.long)

    expected = _ref_cross_attention(query, key, value, cache_index)
    actual = cross_kv_cache_attention(query, key, value, cache_index)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_cross_kv_cache_attention_identity_matches_reference():
    torch.manual_seed(1)
    query = torch.randn(4, 3, 1, 16, device="cuda", dtype=torch.float32)
    key = torch.randn(4, 3, 11, 16, device="cuda", dtype=torch.float32)
    value = torch.randn(4, 3, 11, 16, device="cuda", dtype=torch.float32)

    expected = _ref_cross_attention(query, key, value, None)
    actual = cross_kv_cache_attention(query, key, value)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
