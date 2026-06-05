from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.lora import lora_linear


def test_lora_matches_lifted_expression():
    torch.manual_seed(0)
    x = torch.randn(2, 5)
    adapter_input = torch.randn(2, 5)
    base_weight = torch.randn(7, 5)
    base_bias = torch.randn(7)
    lora_a = torch.randn(3, 5)
    lora_b = torch.randn(7, 3)
    scaling = 2.0

    expected = F.linear(x, base_weight, base_bias)
    expected = expected + (F.linear(F.linear(adapter_input, lora_a), lora_b) * scaling).to(expected.dtype)
    torch.testing.assert_close(lora_linear(x, adapter_input, base_weight, base_bias, lora_a, lora_b, scaling), expected)
