from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.cross_entropy import cross_entropy


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 9)
    labels = torch.tensor([[1, 2, -100, 4], [3, 5, 6, -100]])
    expected = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(cross_entropy(logits, labels), expected)
