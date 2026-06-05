from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.residual_add import residual_add as ref_residual_add
from plu.triton.residual_add import residual_add as triton_residual_add


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_residual_add_forward_backward():
    torch.manual_seed(0)
    residual = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    hidden_states = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    triton_args = _clone_args(residual, hidden_states)

    ref_out = ref_residual_add(residual, hidden_states)
    triton_out = triton_residual_add(*triton_args)
    torch.testing.assert_close(triton_out, ref_out, atol=0, rtol=0)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((residual, hidden_states), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=0, rtol=0)
