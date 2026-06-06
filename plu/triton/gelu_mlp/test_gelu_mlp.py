from __future__ import annotations

from contextlib import contextmanager
import math

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.gelu_mlp import gelu_mlp as ref_gelu_mlp
from plu.triton.gelu_mlp import gelu_mlp as triton_gelu_mlp


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


@contextmanager
def _tf32(enabled: bool):
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        old_global = getattr(torch.backends, "fp32_precision", None)
        old_matmul = torch.backends.cuda.matmul.fp32_precision
        if old_global is not None:
            torch.backends.fp32_precision = "tf32" if enabled else "ieee"
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
        try:
            yield
        finally:
            torch.backends.cuda.matmul.fp32_precision = old_matmul
            if old_global is not None:
                torch.backends.fp32_precision = old_global
    else:
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = enabled
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def _assert_grads_close(ref_args, triton_args, names, atol=5e-5, rtol=5e-5):
    for name, ref_arg, triton_arg in zip(names, ref_args, triton_args):
        if torch.is_tensor(ref_arg) and ref_arg.requires_grad:
            torch.testing.assert_close(ref_arg.grad, triton_arg.grad, atol=atol, rtol=rtol, msg=name)


def test_gelu_mlp_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, device="cuda", requires_grad=True)
    w1 = torch.randn(11, 7, device="cuda", requires_grad=True)
    b1 = torch.randn(11, device="cuda", requires_grad=True)
    w2 = torch.randn(5, 11, device="cuda", requires_grad=True)
    b2 = torch.randn(5, device="cuda", requires_grad=True)
    triton_args = _clone_args(x, w1, b1, w2, b2)

    ref_out = ref_gelu_mlp(x, w1, b1, w2, b2)
    triton_out = triton_gelu_mlp(*triton_args)
    torch.testing.assert_close(triton_out, ref_out, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    _assert_grads_close((x, w1, b1, w2, b2), triton_args, ("x", "w1", "b1", "w2", "b2"))


def test_gelu_mlp_large_tf32_forward_backward():
    torch.manual_seed(0)
    with _tf32(True):
        x = torch.randn(512, 512, device="cuda", requires_grad=True)
        # Keep the turbo-like dimensions without inflating activations into the thousands.
        w1 = (torch.randn(2048, 512, device="cuda") / math.sqrt(512)).requires_grad_()
        b1 = torch.zeros(2048, device="cuda", requires_grad=True)
        w2 = (torch.randn(512, 2048, device="cuda") / math.sqrt(2048)).requires_grad_()
        b2 = torch.zeros(512, device="cuda", requires_grad=True)
        triton_args = _clone_args(x, w1, b1, w2, b2)

        ref_out = ref_gelu_mlp(x, w1, b1, w2, b2)
        triton_out = triton_gelu_mlp(*triton_args)
        torch.testing.assert_close(triton_out, ref_out, atol=1e-2, rtol=7e-3)

        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        triton_out.backward(grad)
        _assert_grads_close((x, w1, b1, w2, b2), triton_args, ("x", "w1", "b1", "w2", "b2"), atol=2e-1, rtol=7e-3)
