from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.qkv_proj import qkv_proj as ref_qkv_proj
from plu.triton.qkv_proj import qkv_proj as triton_qkv_proj


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_qkv_proj_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    weight = torch.randn(8, 8, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)
    triton_args = _clone_args(x, weight, bias)

    ref_out = ref_qkv_proj(x, weight, bias, 2)
    triton_out = triton_qkv_proj(*triton_args, 2)
    torch.testing.assert_close(triton_out, ref_out, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((x, weight, bias), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=1e-5, rtol=1e-5)
