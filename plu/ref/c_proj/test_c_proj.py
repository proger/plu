from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.c_proj import c_proj


def test_c_proj_matches_internal_merge_then_linear():
    torch.manual_seed(0)
    attended = torch.randn(2, 2, 4, 4)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    merged = attended.transpose(1, 2).contiguous().view(2, 4, 8)
    torch.testing.assert_close(c_proj(attended, weight, bias), F.linear(merged, weight, bias))
