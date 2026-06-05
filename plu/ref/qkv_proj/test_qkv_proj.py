from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.qkv_proj import qkv_proj


def test_qkv_proj_matches_linear_with_internal_transpose():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    projected = F.linear(x, weight, bias)
    expected = projected.view(2, 4, 2, 4).transpose(1, 2)
    torch.testing.assert_close(qkv_proj(x, weight, bias, 2), expected)
