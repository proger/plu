from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.linear import linear


def test_linear_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8, 8)
    bias = torch.randn(8)
    torch.testing.assert_close(linear(x, weight, bias), F.linear(x, weight, bias))
