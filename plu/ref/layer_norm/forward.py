from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float) -> Tensor:
    return F.layer_norm(x.float(), (weight.shape[0],), weight, bias, eps).to(x.dtype)
