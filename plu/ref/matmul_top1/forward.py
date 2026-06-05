from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def matmul_top1(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    logits = F.linear(x, weight, bias)
    return logits.max(dim=-1)
