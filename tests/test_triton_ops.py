from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.ref.flash_attention import flash_attention as ref_flash_attention
from plu.ref.gelu_mlp import gelu_mlp as ref_gelu_mlp
from plu.ref.lora import lora_linear as ref_lora_linear
from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
from plu.triton.cross_entropy import cross_entropy as triton_cross_entropy
from plu.triton.flash_attention import flash_attention as triton_flash_attention
from plu.triton.gelu_mlp import gelu_mlp as triton_gelu_mlp
from plu.triton.lora import lora_linear as triton_lora_linear
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


@pytest.mark.parametrize("causal", [False, True])
def test_triton_flash_attention_forward_backward(causal):
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
    _assert_grads_close((query, key, value), triton_args, ("query", "key", "value"))


def test_triton_cross_entropy_forward_backward():
    torch.manual_seed(0)
    logits = torch.randn(6, 19, device="cuda", requires_grad=True)
    labels = torch.tensor([1, 3, -100, 9, 18, 2], device="cuda")
    triton_logits = logits.detach().clone().requires_grad_()

    ref_loss = ref_cross_entropy(logits, labels)
    triton_loss = triton_cross_entropy(triton_logits, labels)
    torch.testing.assert_close(triton_loss, ref_loss, atol=1e-5, rtol=1e-5)

    ref_loss.backward()
    triton_loss.backward()
    torch.testing.assert_close(triton_logits.grad, logits.grad, atol=1e-5, rtol=1e-5)


def test_triton_matmul_top1_forward_backward():
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


def test_triton_matmul_top1_large_tf32_forward_backward():
    torch.manual_seed(0)
    with _tf32(True):
        x = torch.randn(16, 512, device="cuda", requires_grad=True)
        weight = torch.randn(512, 512, device="cuda", requires_grad=True)
        bias = torch.randn(512, device="cuda", requires_grad=True)
        triton_args = _clone_args(x, weight, bias)

        ref_values, ref_indices = ref_matmul_top1(x, weight, bias)
        triton_values, triton_indices = triton_matmul_top1(*triton_args)
        torch.testing.assert_close(triton_values, ref_values, atol=0, rtol=0)
        torch.testing.assert_close(triton_indices, ref_indices)

        grad = torch.randn_like(ref_values)
        ref_values.backward(grad)
        triton_values.backward(grad)
        _assert_grads_close((x, weight, bias), triton_args, ("x", "weight", "bias"), atol=5e-3, rtol=1e-3)


def test_triton_gelu_mlp_forward_backward():
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


def test_triton_gelu_mlp_large_tf32_forward_backward():
    torch.manual_seed(0)
    with _tf32(True):
        x = torch.randn(512, 512, device="cuda", requires_grad=True)
        w1 = torch.randn(2048, 512, device="cuda", requires_grad=True)
        b1 = torch.randn(2048, device="cuda", requires_grad=True)
        w2 = torch.randn(512, 2048, device="cuda", requires_grad=True)
        b2 = torch.randn(512, device="cuda", requires_grad=True)
        triton_args = _clone_args(x, w1, b1, w2, b2)

        ref_out = ref_gelu_mlp(x, w1, b1, w2, b2)
        triton_out = triton_gelu_mlp(*triton_args)
        torch.testing.assert_close(triton_out, ref_out, atol=0, rtol=0)

        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        triton_out.backward(grad)
        _assert_grads_close((x, w1, b1, w2, b2), triton_args, ("x", "w1", "b1", "w2", "b2"), atol=8e-2, rtol=1e-5)


def test_triton_lora_forward_backward():
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
    _assert_grads_close(ref_args, triton_args, ("x", "adapter_input", "base_weight", "base_bias", "lora_a", "lora_b"))
