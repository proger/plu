from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def gelu_mlp(x: Tensor, fc1_weight: Tensor, fc1_bias: Tensor | None, fc2_weight: Tensor, fc2_bias: Tensor | None) -> Tensor:
    fc1_bias = None if fc1_bias is None else fc1_bias.to(x.dtype)
    fc2_bias = None if fc2_bias is None else fc2_bias.to(x.dtype)
    hidden = F.gelu(F.linear(x, fc1_weight.to(x.dtype), fc1_bias))
    return F.linear(hidden, fc2_weight.to(hidden.dtype), fc2_bias)
