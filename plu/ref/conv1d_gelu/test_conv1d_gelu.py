from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.conv1d_gelu import conv1d_gelu


def test_conv1d_gelu_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9)
    weight = torch.randn(5, 3, 3)
    bias = torch.randn(5)
    expected = F.gelu(F.conv1d(x, weight, bias, stride=2, padding=1))
    torch.testing.assert_close(conv1d_gelu(x, weight, bias, stride=2, padding=1), expected)
