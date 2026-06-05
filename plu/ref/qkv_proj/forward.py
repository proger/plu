from __future__ import annotations

from torch import Tensor

from plu.ref.linear import linear


def qkv_proj(x: Tensor, weight: Tensor, bias: Tensor | None, num_heads: int) -> Tensor:
    batch, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    projected = linear(x, weight, bias)
    return projected.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
