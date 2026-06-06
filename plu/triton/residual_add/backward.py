from __future__ import annotations

from torch import Tensor


def residual_add_backward(grad_out: Tensor) -> tuple[Tensor, Tensor]:
    return grad_out, grad_out
