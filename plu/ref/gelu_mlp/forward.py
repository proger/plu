from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


_LOW_PRECISION_FLOAT_DTYPES = (torch.bfloat16,)


def gelu_mlp(x: Tensor, fc1_weight: Tensor, fc1_bias: Tensor | None, fc2_weight: Tensor, fc2_bias: Tensor | None) -> Tensor:
    if x.dtype in _LOW_PRECISION_FLOAT_DTYPES and x.dtype == fc1_weight.dtype and x.dtype == fc2_weight.dtype:
        fc1_bias_float = None if fc1_bias is None else fc1_bias.float()
        hidden = F.gelu(F.linear(x.float(), fc1_weight.float(), fc1_bias_float)).to(x.dtype)
        fc2_bias = None if fc2_bias is None else fc2_bias.to(x.dtype)
        return F.linear(hidden, fc2_weight, fc2_bias)

    fc1_bias = None if fc1_bias is None else fc1_bias.to(x.dtype)
    fc2_bias = None if fc2_bias is None else fc2_bias.to(x.dtype)
    hidden = F.gelu(F.linear(x, fc1_weight.to(x.dtype), fc1_bias))
    return F.linear(hidden, fc2_weight.to(hidden.dtype), fc2_bias)
