from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import Tensor


def flash_attention(query: Tensor, key: Tensor, value: Tensor, causal_mask: Tensor | None = None) -> Tensor:
    attn_mask = None
    if causal_mask is not None:
        attn_mask = causal_mask[: query.shape[-2], : key.shape[-2]].to(device=query.device, dtype=query.dtype)
    if query.is_cuda and query.dtype == torch.float32 and key.dtype == torch.float32 and value.dtype == torch.float32:
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
