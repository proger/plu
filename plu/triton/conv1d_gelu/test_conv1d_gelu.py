from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.conv1d_gelu import conv1d_gelu as ref_conv1d_gelu
from plu.triton.conv1d_gelu import conv1d_gelu as triton_conv1d_gelu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_conv1d_gelu_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, device="cuda", requires_grad=True)
    weight = torch.randn(5, 3, 3, device="cuda", requires_grad=True)
    bias = torch.randn(5, device="cuda", requires_grad=True)
    triton_args = _clone_args(x, weight, bias)

    ref_out = ref_conv1d_gelu(x, weight, bias, stride=2, padding=1)
    triton_out = triton_conv1d_gelu(*triton_args, stride=2, padding=1)
    torch.testing.assert_close(triton_out, ref_out, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((x, weight, bias), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=1e-5, rtol=1e-5)


def test_conv1d_gelu_bf16_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(5, 3, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    triton_args = _clone_args(x, weight, bias)

    ref_out = ref_conv1d_gelu(x, weight, bias, stride=2, padding=1)
    triton_out = triton_conv1d_gelu(*triton_args, stride=2, padding=1)
    torch.testing.assert_close(triton_out, ref_out, atol=4e-2, rtol=2e-2)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((x, weight, bias), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=4e-2, rtol=2e-2)
