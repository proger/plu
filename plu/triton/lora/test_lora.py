from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.lora import lora_linear as ref_lora_linear
from plu.triton.lora import lora_linear as triton_lora_linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_lora_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(6, 9, device="cuda", requires_grad=True)
    adapter_input = torch.randn(6, 9, device="cuda", requires_grad=True)
    base_weight = torch.randn(7, 9, device="cuda", requires_grad=True)
    base_bias = torch.randn(7, device="cuda", requires_grad=True)
    lora_a = torch.randn(3, 9, device="cuda", requires_grad=True)
    lora_b = torch.randn(7, 3, device="cuda", requires_grad=True)
    scaling = 2.5
    ref_args = (x, adapter_input, base_weight, base_bias, lora_a, lora_b)
    triton_args = _clone_args(*ref_args)

    ref_out = ref_lora_linear(*ref_args, scaling)
    triton_out = triton_lora_linear(*triton_args, scaling)
    torch.testing.assert_close(triton_out, ref_out, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip(ref_args, triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=5e-5, rtol=5e-5)
