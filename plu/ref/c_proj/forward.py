from __future__ import annotations

from torch import Tensor

from plu.ref.linear import linear


def c_proj(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    batch, heads, seq_len, head_dim = x.shape
    merged = x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)
    return linear(merged, weight, bias)
