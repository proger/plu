from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.linear import linear as ref_linear
from plu.triton.linear import linear as triton_linear
from plu.triton.linear.backward import collect_linear_grad_norms, summarize_linear_grad_norms


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_linear_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    weight = torch.randn(8, 8, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)
    triton_args = _clone_args(x, weight, bias)

    ref_out = ref_linear(x, weight, bias)
    triton_out = triton_linear(*triton_args)
    torch.testing.assert_close(triton_out, ref_out, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((x, weight, bias), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=1e-5, rtol=1e-5)


def test_linear_grad_norms_are_collected_from_backward_kernel():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    weight = torch.randn(8, 8, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)

    with collect_linear_grad_norms() as records:
        out = triton_linear(x, weight, bias)
        out.backward(torch.randn_like(out))

    metrics = summarize_linear_grad_norms(records)
    expected_weight = float(weight.grad.float().norm().detach().cpu())
    expected_bias = float(bias.grad.float().norm().detach().cpu())
    expected_total = (expected_weight**2 + expected_bias**2) ** 0.5

    assert metrics["train/grad_norm/linear_calls"] == 1.0
    assert metrics["train/grad_norm/linear_weight"] == pytest.approx(expected_weight, rel=1e-5, abs=1e-5)
    assert metrics["train/grad_norm/linear_bias"] == pytest.approx(expected_bias, rel=1e-5, abs=1e-5)
    assert metrics["train/grad_norm/linear_total"] == pytest.approx(expected_total, rel=1e-5, abs=1e-5)


def test_linear_grad_norm_collection_is_deterministic_for_tiled_reductions():
    torch.manual_seed(2)
    rows, in_features, out_features = 512, 512, 512
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(rows, out_features, device="cuda", dtype=torch.bfloat16)

    totals = []
    weight_totals = []
    bias_totals = []
    for _ in range(3):
        current_x, current_weight, current_bias = _clone_args(
            x.requires_grad_(),
            weight.requires_grad_(),
            bias.requires_grad_(),
        )
        with collect_linear_grad_norms() as records:
            out = triton_linear(current_x, current_weight, current_bias)
            out.backward(grad)
        torch.cuda.synchronize()
        metrics = summarize_linear_grad_norms(records)
        totals.append(metrics["train/grad_norm/linear_total"])
        weight_totals.append(metrics["train/grad_norm/linear_weight"])
        bias_totals.append(metrics["train/grad_norm/linear_bias"])

    assert totals[1:] == totals[:1] * (len(totals) - 1)
    assert weight_totals[1:] == weight_totals[:1] * (len(weight_totals) - 1)
    assert bias_totals[1:] == bias_totals[:1] * (len(bias_totals) - 1)
