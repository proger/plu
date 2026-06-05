from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def flash_attention(query: Tensor, key: Tensor, value: Tensor, causal_mask: Tensor | None = None) -> Tensor:
    scale = query.shape[-1] ** -0.5
    weights = torch.matmul(query, key.transpose(-1, -2)) * scale
    if causal_mask is not None:
        weights = weights + causal_mask[: weights.shape[-2], : weights.shape[-1]].to(weights.device)
    weights = F.softmax(weights.float(), dim=-1).to(query.dtype)
    return torch.matmul(weights, value)
