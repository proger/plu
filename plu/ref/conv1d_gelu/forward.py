from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


_LOW_PRECISION_FLOAT_DTYPES = (torch.bfloat16,)


def conv1d_gelu(x: Tensor, weight: Tensor, bias: Tensor | None, stride: int, padding: int) -> Tensor:
    if x.dtype in _LOW_PRECISION_FLOAT_DTYPES and x.dtype == weight.dtype:
        bias_float = None if bias is None else bias.float()
        conv = F.conv1d(x.float(), weight.float(), bias_float, stride=stride, padding=padding)
        return F.gelu(conv).to(x.dtype)

    conv = F.conv1d(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype), stride=stride, padding=padding)
    return F.gelu(conv)
