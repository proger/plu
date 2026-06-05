from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=ignore_index)

