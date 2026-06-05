from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def conv1d_gelu(x: Tensor, weight: Tensor, bias: Tensor | None, stride: int, padding: int) -> Tensor:
    conv = F.conv1d(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype), stride=stride, padding=padding)
    return F.gelu(conv)
