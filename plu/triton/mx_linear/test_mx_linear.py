from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from plu.triton.mx_linear import (
    dequantize_mxfp4_weight,
    dequantize_mxfp8_weight,
    mxfp4_linear,
    mxfp8_linear,
    pack_mxfp4_weight,
    pack_mxfp8_weight,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def test_mxfp8_pack_dequant_quality():
    torch.manual_seed(0)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp8_weight(weight)
    dequant = dequantize_mxfp8_weight(packed, scales, in_features)

    assert packed.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.uint8
    assert packed.shape == (37, 96)
    assert scales.shape == (37, 3)
    assert in_features == 65
    assert _relative_l2(dequant, weight) < 3.5e-2


def test_mxfp8_e5m2_pack_and_linear_match_dequantized_reference():
    torch.manual_seed(0)
    x = torch.randn(5, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp8_weight(weight, element_format="e5m2")
    dequant = dequantize_mxfp8_weight(packed, scales, in_features).to(x.dtype)

    assert packed.dtype == torch.float8_e5m2
    assert _relative_l2(dequant, weight) < 7.0e-2
    torch.testing.assert_close(mxfp8_linear(x, packed, scales, in_features), F.linear(x, dequant), atol=2e-3, rtol=2e-3)


def test_mxfp4_pack_dequant_quality():
    torch.manual_seed(0)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    packed, scales, in_features = pack_mxfp4_weight(weight)
    dequant = dequantize_mxfp4_weight(packed, scales, in_features)

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert packed.shape == (37, 48)
    assert scales.shape == (37, 3)
    assert in_features == 65
    assert _relative_l2(dequant, weight) < 1.8e-1


@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp8_linear_matches_dequantized_reference(with_bias: bool):
    torch.manual_seed(1)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16) if with_bias else None
    packed, scales, in_features = pack_mxfp8_weight(weight)
    dequant = dequantize_mxfp8_weight(packed, scales, in_features).to(x.dtype)

    ref = F.linear(x, dequant, bias)
    out = mxfp8_linear(x, packed, scales, in_features, bias)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp4_linear_matches_dequantized_reference(with_bias: bool):
    torch.manual_seed(2)
    x = torch.randn(3, 7, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.bfloat16) if with_bias else None
    packed, scales, in_features = pack_mxfp4_weight(weight)
    dequant = dequantize_mxfp4_weight(packed, scales, in_features).to(x.dtype)

    ref = F.linear(x, dequant, bias)
    out = mxfp4_linear(x, packed, scales, in_features, bias)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


def test_mxfp_linear_reports_quantization_error_against_original_weight():
    torch.manual_seed(3)
    x = torch.randn(2, 11, 65, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(37, 65, device="cuda", dtype=torch.bfloat16) / math.sqrt(65)).contiguous()
    bias = torch.zeros(37, device="cuda", dtype=torch.bfloat16)
    fp8_weight, fp8_scales, fp8_in = pack_mxfp8_weight(weight)
    fp4_weight, fp4_scales, fp4_in = pack_mxfp4_weight(weight)

    ref = F.linear(x, weight, bias)
    fp8_out = mxfp8_linear(x, fp8_weight, fp8_scales, fp8_in, bias)
    fp4_out = mxfp4_linear(x, fp4_weight, fp4_scales, fp4_in, bias)

    assert _relative_l2(fp8_out, ref) < 5.0e-2
    assert _relative_l2(fp4_out, ref) < 2.5e-1
