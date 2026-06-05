from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.triton.cross_entropy import cross_entropy as triton_cross_entropy


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def test_cross_entropy_forward_backward():
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
