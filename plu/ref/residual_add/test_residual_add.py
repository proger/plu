from __future__ import annotations

import torch

from plu.ref.residual_add import residual_add


def test_residual_add_matches_torch_add():
    torch.manual_seed(0)
    residual = torch.randn(2, 4, 8)
    hidden_states = torch.randn(2, 4, 8)
    torch.testing.assert_close(residual_add(residual, hidden_states), residual + hidden_states)
