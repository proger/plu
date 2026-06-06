from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float) -> Tensor:
    bias_float = None if bias is None else bias.float()
    return F.layer_norm(x.float(), (weight.shape[0],), weight.float(), bias_float, eps).to(x.dtype)
