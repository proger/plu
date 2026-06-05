from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
from plu.triton.matmul_top1 import matmul_top1 as triton_matmul_top1


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


def test_matmul_top1_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 13, device="cuda", requires_grad=True)
    weight = torch.randn(17, 13, device="cuda", requires_grad=True)
    bias = torch.randn(17, device="cuda", requires_grad=True)
    triton_args = _clone_args(x, weight, bias)

    ref_values, ref_indices = ref_matmul_top1(x, weight, bias)
    triton_values, triton_indices = triton_matmul_top1(*triton_args)
    torch.testing.assert_close(triton_values, ref_values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_indices, ref_indices)

    grad = torch.randn_like(ref_values)
    ref_values.backward(grad)
    triton_values.backward(grad)
    _assert_grads_close((x, weight, bias), triton_args, ("x", "weight", "bias"))


def test_matmul_top1_large_tf32_forward_backward():
    torch.manual_seed(0)
    with _tf32(True):
        x = torch.randn(16, 512, device="cuda", requires_grad=True)
        weight = torch.randn(512, 512, device="cuda", requires_grad=True)
        bias = torch.randn(512, device="cuda", requires_grad=True)
        triton_args = _clone_args(x, weight, bias)

        ref_values, ref_indices = ref_matmul_top1(x, weight, bias)
        triton_values, triton_indices = triton_matmul_top1(*triton_args)
        torch.testing.assert_close(triton_values, ref_values, atol=2e-1, rtol=5e-3)
        torch.testing.assert_close(triton_indices, ref_indices)

        grad = torch.randn_like(ref_values)
        ref_values.backward(grad)
        triton_values.backward(grad)
        _assert_grads_close((x, weight, bias), triton_args, ("x", "weight", "bias"), atol=5e-3, rtol=1e-3)
