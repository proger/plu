from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.gelu_mlp import gelu_mlp


def test_gelu_mlp_matches_lifted_expression():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5)
    w1 = torch.randn(7, 5)
    b1 = torch.randn(7)
    w2 = torch.randn(4, 7)
    b2 = torch.randn(4)
    expected = F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
    torch.testing.assert_close(gelu_mlp(x, w1, b1, w2, b2), expected)
