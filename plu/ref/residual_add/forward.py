from __future__ import annotations

from torch import Tensor


def residual_add(residual: Tensor, hidden_states: Tensor) -> Tensor:
    return residual + hidden_states
