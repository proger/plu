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


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_bf16_forward_backward(causal):
    torch.manual_seed(0)
    query = torch.randn(2, 3, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 3, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(2, 3, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    triton_args = _clone_args(query, key, value)
    mask = torch.empty(5, 5, device="cuda").fill_(-float("inf")).triu_(1) if causal else None

    ref_out = ref_flash_attention(query, key, value, mask)
    triton_out = triton_flash_attention(*triton_args, mask)
    torch.testing.assert_close(triton_out, ref_out, atol=4e-2, rtol=4e-2)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((query, key, value), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=6e-2, rtol=6e-2)


def test_flash_attention_bf16_non_contiguous_qkv_matches_contiguous():
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 1, 4, 7, 16
    query_base = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_base = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_base = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    query = query_base.permute(0, 2, 1, 3).requires_grad_()
    key = key_base.permute(0, 2, 1, 3).requires_grad_()
    value = value_base.permute(0, 2, 1, 3).requires_grad_()
    query_contiguous, key_contiguous, value_contiguous = _clone_args(query.contiguous(), key.contiguous(), value.contiguous())

    out = triton_flash_attention(query, key, value)
    expected = triton_flash_attention(query_contiguous, key_contiguous, value_contiguous)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(query.grad, query_contiguous.grad, atol=0, rtol=0)
    torch.testing.assert_close(key.grad, key_contiguous.grad, atol=0, rtol=0)
    torch.testing.assert_close(value.grad, value_contiguous.grad, atol=0, rtol=0)
