from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from plu.triton.mx_linear import (
    mxfp4_linear,
    mxfp4_linear_with_master_weight,
    mxfp8_linear,
    mxfp8_linear_from_heads_with_master_weight,
    mxfp8_linear_heads_with_master_weight,
    mxfp8_linear_with_master_weight,
    nvfp4_linear,
    nvfp4_linear_with_master_weight,
    pack_mxfp4_weight,
    pack_mxfp8_weight,
    pack_nvfp4_weight,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def test_mxfp8_pack_layout():
    torch.manual_seed(0)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp8_weight(weight)

    assert packed.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.uint8
    assert packed.shape == (37, 96)
    assert scales.shape == (1, 1, 2, 2, 256)
    assert in_features == 65


def test_mxfp8_e5m2_pack_and_linear_matches_bf16_reference():
    torch.manual_seed(0)
    x = torch.randn(5, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp8_weight(weight, element_format="e5m2")

    assert packed.dtype == torch.float8_e5m2
    assert _relative_l2(mxfp8_linear(x, packed, scales, in_features), F.linear(x, weight)) < 8.0e-2


def test_mxfp8_linear_from_heads_with_master_weight_matches_merged_input():
    torch.manual_seed(7)
    batch = 2
    heads = 4
    seq_len = 5
    head_dim = 16
    out_features = 37
    x_heads = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_merged = x_heads.detach().permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, heads * head_dim).requires_grad_()
    master_weight = (torch.randn(out_features, heads * head_dim, device="cuda", dtype=torch.bfloat16) / math.sqrt(heads * head_dim)).requires_grad_()
    merged_weight = master_weight.detach().clone().requires_grad_()
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    merged_bias = bias.detach().clone().requires_grad_()
    packed, scales, in_features = pack_mxfp8_weight(master_weight.detach())
    input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp8_weight(master_weight.detach().T.contiguous())

    out = mxfp8_linear_from_heads_with_master_weight(
        x_heads,
        packed,
        scales,
        in_features,
        master_weight,
        bias,
        input_grad_packed,
        input_grad_scales,
        input_grad_in_features,
    )
    ref = mxfp8_linear_with_master_weight(
        x_merged,
        packed,
        scales,
        in_features,
        merged_weight,
        merged_bias,
        input_grad_packed,
        input_grad_scales,
        input_grad_in_features,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    expected_x_grad = x_merged.grad.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(x_heads.grad, expected_x_grad, atol=0, rtol=0)
    torch.testing.assert_close(master_weight.grad, merged_weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(bias.grad, merged_bias.grad, atol=0, rtol=0)


def test_mxfp8_linear_heads_with_master_weight_matches_merged_output():
    torch.manual_seed(8)
    batch = 2
    seq_len = 5
    in_features = 31
    heads = 4
    head_dim = 16
    out_features = heads * head_dim
    x = torch.randn(batch, seq_len, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    merged_x = x.detach().clone().requires_grad_()
    master_weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) / math.sqrt(in_features)).requires_grad_()
    merged_weight = master_weight.detach().clone().requires_grad_()
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    merged_bias = bias.detach().clone().requires_grad_()
    packed, scales, packed_in_features = pack_mxfp8_weight(master_weight.detach())
    input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp8_weight(master_weight.detach().T.contiguous())

    out = mxfp8_linear_heads_with_master_weight(
        x,
        packed,
        scales,
        packed_in_features,
        heads,
        master_weight,
        bias,
        input_grad_packed,
        input_grad_scales,
        input_grad_in_features,
    )
    ref = mxfp8_linear_with_master_weight(
        merged_x,
        packed,
        scales,
        packed_in_features,
        merged_weight,
        merged_bias,
        input_grad_packed,
        input_grad_scales,
        input_grad_in_features,
    ).reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(x.grad, merged_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(master_weight.grad, merged_weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(bias.grad, merged_bias.grad, atol=4e-2, rtol=3e-1)


def test_mxfp4_pack_layout():
    torch.manual_seed(0)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp4_weight(weight)

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert packed.shape == (37, 48)
    assert scales.shape == (1, 1, 2, 2, 256)
    assert in_features == 65


def test_nvfp4_pack_layout():
    torch.manual_seed(0)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_nvfp4_weight(weight)

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float8_e4m3fn
    assert packed.shape == (37, 40)
    assert scales.shape == (1, 1, 2, 2, 256)
    assert in_features == 65


@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp8_linear_matches_bf16_reference(with_bias: bool):
    torch.manual_seed(1)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16) if with_bias else None
    packed, scales, in_features = pack_mxfp8_weight(weight)

    ref = F.linear(x, weight, bias)
    out = mxfp8_linear(x, packed, scales, in_features, bias)
    assert _relative_l2(out, ref) < 5.0e-2


@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp4_linear_matches_bf16_reference(with_bias: bool):
    torch.manual_seed(2)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16) if with_bias else None
    packed, scales, in_features = pack_mxfp4_weight(weight)

    ref = F.linear(x, weight, bias)
    out = mxfp4_linear(x, packed, scales, in_features, bias)
    assert _relative_l2(out, ref) < 2.5e-1


@pytest.mark.parametrize("with_bias", [False, True])
def test_nvfp4_linear_matches_bf16_reference(with_bias: bool):
    torch.manual_seed(2)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16) if with_bias else None
    packed, scales, in_features = pack_nvfp4_weight(weight)

    ref = F.linear(x, weight, bias)
    out = nvfp4_linear(x, packed, scales, in_features, bias)
    assert _relative_l2(out, ref) < 2.5e-1


@pytest.mark.parametrize("format_name", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp_linear_backward_matches_bf16_reference(format_name: str, with_bias: bool):
    torch.manual_seed(4)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16, requires_grad=True) if with_bias else None

    if format_name == "mxfp8":
        packed, scales, in_features = pack_mxfp8_weight(weight)
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp8_weight(weight.T.contiguous())
        out = mxfp8_linear(x, packed, scales, in_features, bias, input_grad_packed, input_grad_scales, input_grad_in_features)
        grad_threshold = 5.0e-2
    elif format_name == "mxfp4":
        packed, scales, in_features = pack_mxfp4_weight(weight)
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp4_weight(weight.T.contiguous())
        out = mxfp4_linear(x, packed, scales, in_features, bias, input_grad_packed, input_grad_scales, input_grad_in_features)
        grad_threshold = 2.5e-1
    else:
        packed, scales, in_features = pack_nvfp4_weight(weight)
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_nvfp4_weight(weight.T.contiguous())
        out = nvfp4_linear(x, packed, scales, in_features, bias, input_grad_packed, input_grad_scales, input_grad_in_features)
        grad_threshold = 2.5e-1

    grad = torch.randn_like(out)
    out.backward(grad)
    expected_x_grad = F.linear(grad.reshape(-1, 37), weight.T.to(grad.dtype)).reshape_as(x)
    assert _relative_l2(x.grad, expected_x_grad) < grad_threshold
    if with_bias:
        expected_bias_grad = grad.reshape(-1, 37).float().sum(dim=0).to(torch.bfloat16)
        torch.testing.assert_close(bias.grad, expected_bias_grad, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("format_name", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp_linear_with_master_weight_backward_matches_linear_gradients(format_name: str, with_bias: bool):
    torch.manual_seed(5)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    master_weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous().requires_grad_()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16, requires_grad=True) if with_bias else None

    if format_name == "mxfp8":
        packed, scales, in_features = pack_mxfp8_weight(master_weight.detach())
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp8_weight(master_weight.detach().T.contiguous())
        out = mxfp8_linear_with_master_weight(
            x,
            packed,
            scales,
            in_features,
            master_weight,
            bias,
            input_grad_packed,
            input_grad_scales,
            input_grad_in_features,
        )
        grad_threshold = 5.0e-2
    elif format_name == "mxfp4":
        packed, scales, in_features = pack_mxfp4_weight(master_weight.detach())
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_mxfp4_weight(master_weight.detach().T.contiguous())
        out = mxfp4_linear_with_master_weight(
            x,
            packed,
            scales,
            in_features,
            master_weight,
            bias,
            input_grad_packed,
            input_grad_scales,
            input_grad_in_features,
        )
        grad_threshold = 2.5e-1
    else:
        packed, scales, in_features = pack_nvfp4_weight(master_weight.detach())
        input_grad_packed, input_grad_scales, input_grad_in_features = pack_nvfp4_weight(master_weight.detach().T.contiguous())
        out = nvfp4_linear_with_master_weight(
            x,
            packed,
            scales,
            in_features,
            master_weight,
            bias,
            input_grad_packed,
            input_grad_scales,
            input_grad_in_features,
        )
        grad_threshold = 2.5e-1

    grad = torch.randn_like(out)
    out.backward(grad)
    expected_x_grad = F.linear(grad.reshape(-1, 37), master_weight.detach().T.to(grad.dtype)).reshape_as(x)
    expected_w_grad = (grad.reshape(-1, 37).float().T @ x.detach().reshape(-1, 65).float()).to(torch.bfloat16)
    assert _relative_l2(x.grad, expected_x_grad) < grad_threshold
    torch.testing.assert_close(master_weight.grad, expected_w_grad, atol=2e-3, rtol=2e-3)
    if with_bias:
        expected_bias_grad = grad.reshape(-1, 37).float().sum(dim=0).to(torch.bfloat16)
        torch.testing.assert_close(bias.grad, expected_bias_grad, atol=2e-3, rtol=2e-3)


def test_mxfp_linear_reports_quantization_error_against_original_weight():
    torch.manual_seed(3)
    x = torch.randn(2, 11, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.zeros(37, device="cuda", dtype=torch.bfloat16)
    fp8_weight, fp8_scales, fp8_in = pack_mxfp8_weight(weight)
    fp4_weight, fp4_scales, fp4_in = pack_mxfp4_weight(weight)
    nvfp4_weight, nvfp4_scales, nvfp4_in = pack_nvfp4_weight(weight)

    ref = F.linear(x, weight, bias)
    fp8_out = mxfp8_linear(x, fp8_weight, fp8_scales, fp8_in, bias)
    fp4_out = mxfp4_linear(x, fp4_weight, fp4_scales, fp4_in, bias)
    nvfp4_out = nvfp4_linear(x, nvfp4_weight, nvfp4_scales, nvfp4_in, bias)

    assert _relative_l2(fp8_out, ref) < 5.0e-2
    assert _relative_l2(fp4_out, ref) < 2.5e-1
    assert _relative_l2(nvfp4_out, ref) < 2.5e-1
