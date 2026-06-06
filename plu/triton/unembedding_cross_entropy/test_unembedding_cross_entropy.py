from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from plu.triton.unembedding_cross_entropy import unembedding_cross_entropy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def test_unembedding_cross_entropy_forward_backward():
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(97, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 97, (2, 3), device="cuda")
    labels[0, 2] = -100
    triton_hidden = hidden.detach().clone().requires_grad_()
    triton_weight = weight.detach().clone().requires_grad_()

    logits = F.linear(hidden, weight).float()
    ref_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    actual_loss = unembedding_cross_entropy(triton_hidden, triton_weight, labels)
    torch.testing.assert_close(actual_loss, ref_loss, atol=2e-3, rtol=2e-3)

    ref_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(triton_hidden.grad, hidden.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(triton_weight.grad, weight.grad, atol=5e-3, rtol=5e-3)
