from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return F.linear(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))
