from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.matmul_top1 import matmul_top1


def test_matmul_top1_matches_linear_max():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)
    expected = F.linear(x, weight, bias).max(dim=-1)
    actual_values, actual_indices = matmul_top1(x, weight, bias)
    torch.testing.assert_close(actual_values, expected.values)
    torch.testing.assert_close(actual_indices, expected.indices)
