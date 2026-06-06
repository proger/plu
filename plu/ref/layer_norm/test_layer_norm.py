from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.layer_norm import layer_norm


def test_layer_norm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    bias = torch.randn(8)
    expected = F.layer_norm(x.float(), (8,), weight, bias, 1e-5).to(x.dtype)
    torch.testing.assert_close(layer_norm(x, weight, bias, 1e-5), expected)


def test_layer_norm_bf16_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    expected = F.layer_norm(x.float(), (8,), weight.float(), bias.float(), 1e-5).to(x.dtype)
    torch.testing.assert_close(layer_norm(x, weight, bias, 1e-5), expected)
