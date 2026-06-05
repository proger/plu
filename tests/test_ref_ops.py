from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.cross_entropy import cross_entropy
from plu.ref.flash_attention import flash_attention
from plu.ref.gelu_mlp import gelu_mlp
from plu.ref.lora import lora_linear
from plu.ref.matmul_top1 import matmul_top1


def test_ref_flash_attention_matches_lifted_whisper_expression():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 5)
    mask = torch.empty(4, 4).fill_(-float("inf")).triu_(1)

    weights = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    weights = weights + mask[: weights.shape[-2], : weights.shape[-1]]
    expected = torch.matmul(F.softmax(weights.float(), dim=-1).to(query.dtype), value)

    torch.testing.assert_close(flash_attention(query, key, value, mask), expected)


def test_ref_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 9)
    labels = torch.tensor([[1, 2, -100, 4], [3, 5, 6, -100]])
    expected = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(cross_entropy(logits, labels), expected)


def test_ref_matmul_top1_matches_linear_max():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)
    expected = F.linear(x, weight, bias).max(dim=-1)
    actual_values, actual_indices = matmul_top1(x, weight, bias)
    torch.testing.assert_close(actual_values, expected.values)
    torch.testing.assert_close(actual_indices, expected.indices)


def test_ref_gelu_mlp_matches_lifted_expression():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5)
    w1 = torch.randn(7, 5)
    b1 = torch.randn(7)
    w2 = torch.randn(4, 7)
    b2 = torch.randn(4)
    expected = F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
    torch.testing.assert_close(gelu_mlp(x, w1, b1, w2, b2), expected)


def test_ref_lora_matches_lifted_expression():
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
