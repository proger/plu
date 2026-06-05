from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.flash_attention import flash_attention


def test_flash_attention_matches_lifted_whisper_expression():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 5)
    mask = torch.empty(4, 4).fill_(-float("inf")).triu_(1)

    weights = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    weights = weights + mask[: weights.shape[-2], : weights.shape[-1]]
    expected = torch.matmul(F.softmax(weights.float(), dim=-1).to(query.dtype), value)

    torch.testing.assert_close(flash_attention(query, key, value, mask), expected)
