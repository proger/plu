from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.flash_attention import flash_attention as ref_flash_attention
from plu.triton.flash_attention import flash_attention as triton_flash_attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_forward_backward(causal):
    torch.manual_seed(0)
    query = torch.randn(2, 3, 5, 16, device="cuda", requires_grad=True)
    key = torch.randn(2, 3, 5, 16, device="cuda", requires_grad=True)
    value = torch.randn(2, 3, 5, 16, device="cuda", requires_grad=True)
    triton_args = _clone_args(query, key, value)
    mask = torch.empty(5, 5, device="cuda").fill_(-float("inf")).triu_(1) if causal else None

    ref_out = ref_flash_attention(query, key, value, mask)
    triton_out = triton_flash_attention(*triton_args, mask)
    torch.testing.assert_close(triton_out, ref_out, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((query, key, value), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=5e-5, rtol=5e-5)
