from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


MXFP_BLOCK_SIZE = 32
_E8M0_BIAS = 127
_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_E2M1_MAX = 6.0


def _padded_in_features(in_features: int, block_size: int = MXFP_BLOCK_SIZE) -> int:
    return triton.cdiv(in_features, block_size) * block_size


def _e8m0_scale_codes(blocks: Tensor, max_value: float) -> tuple[Tensor, Tensor]:
    amax = blocks.abs().amax(dim=-1)
    safe_ratio = torch.clamp(amax.float() / max_value, min=2.0**-126)
    exponent = torch.ceil(torch.log2(safe_ratio))
    code = torch.clamp(exponent + _E8M0_BIAS, 0, 255)
    code = torch.where(amax > 0, code, torch.full_like(code, _E8M0_BIAS))
    code_u8 = code.to(torch.uint8)
    scale = torch.pow(2.0, code_u8.float() - _E8M0_BIAS).to(blocks.dtype)
    return code_u8.contiguous(), scale


def _e8m0_decode(scale_codes: Tensor) -> Tensor:
    return torch.pow(2.0, scale_codes.float() - _E8M0_BIAS)


def _fp8_dtype_and_max(element_format: str) -> tuple[torch.dtype, float]:
    if element_format == "e4m3":
        return torch.float8_e4m3fn, _E4M3_MAX
    if element_format == "e5m2":
        return torch.float8_e5m2, _E5M2_MAX
    raise ValueError(f"unsupported MXFP8 element format: {element_format}")


def pack_mxfp8_weight(weight: Tensor, element_format: str = "e4m3") -> tuple[Tensor, Tensor, int]:
    """Pack a weight matrix into MXFP8 FP8 values plus E8M0 block scales.

    Returns (values, scale_codes, original_in_features).  Values are padded along
    the input-feature dimension to a multiple of 32.  `element_format` may be
    "e4m3" or "e5m2"; E4M3 is the default because it is normally better for
    forward weight storage.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    fp8_dtype, fp8_max = _fp8_dtype_and_max(element_format)
    out_features, in_features = weight.shape
    padded_in = _padded_in_features(in_features)
    padded = torch.zeros((out_features, padded_in), device=weight.device, dtype=weight.dtype)
    padded[:, :in_features] = weight
    blocks = padded.reshape(out_features, padded_in // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE)
    scale_codes, scales = _e8m0_scale_codes(blocks, fp8_max)
    values = torch.clamp(blocks / scales[..., None], min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return values.reshape(out_features, padded_in).contiguous(), scale_codes, in_features


def dequantize_mxfp8_weight(values: Tensor, scale_codes: Tensor, in_features: int | None = None) -> Tensor:
    if values.ndim != 2 or scale_codes.ndim != 2:
        raise ValueError("values and scale_codes must be 2D")
    out_features, padded_in = values.shape
    blocks = values.reshape(out_features, padded_in // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE).float()
    scales = _e8m0_decode(scale_codes).to(blocks.device)
    weight = (blocks * scales[..., None]).reshape(out_features, padded_in)
    return weight if in_features is None else weight[:, :in_features].contiguous()


def pack_mxfp4_weight(weight: Tensor) -> tuple[Tensor, Tensor, int]:
    """Pack a weight matrix into MXFP4 E2M1 nibbles plus E8M0 block scales."""
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    out_features, in_features = weight.shape
    padded_in = _padded_in_features(in_features)
    padded = torch.zeros((out_features, padded_in), device=weight.device, dtype=weight.dtype)
    padded[:, :in_features] = weight
    blocks = padded.reshape(out_features, padded_in // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE)
    scale_codes, scales = _e8m0_scale_codes(blocks, _E2M1_MAX)
    normalized = blocks / scales[..., None]

    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device, dtype=torch.float32)
    nearest = (normalized.abs().float()[..., None] - levels).abs().argmin(dim=-1).to(torch.uint8)
    sign = torch.where(normalized < 0, torch.full_like(nearest, 8), torch.zeros_like(nearest))
    codes = (nearest | sign).reshape(out_features, padded_in)
    low = codes[:, 0::2]
    high = codes[:, 1::2] * 16
    return (low | high).contiguous(), scale_codes, in_features


def dequantize_mxfp4_weight(packed_values: Tensor, scale_codes: Tensor, in_features: int | None = None) -> Tensor:
    if packed_values.ndim != 2 or scale_codes.ndim != 2:
        raise ValueError("packed_values and scale_codes must be 2D")
    out_features, packed_in = packed_values.shape
    low = packed_values & 0x0F
    high = torch.bitwise_right_shift(packed_values, 4)
    codes = torch.stack((low, high), dim=-1).reshape(out_features, packed_in * 2)
    magnitude_code = codes & 0x07
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed_values.device, dtype=torch.float32)
    values = levels[magnitude_code.long()]
    values = torch.where((codes & 0x08) != 0, -values, values)
    scales = _e8m0_decode(scale_codes).to(values.device).repeat_interleave(MXFP_BLOCK_SIZE, dim=1)
    weight = values * scales[:, : values.shape[1]]
    return weight if in_features is None else weight[:, :in_features].contiguous()


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    padded_in_features: tl.constexpr,
    out_features: tl.constexpr,
    scale_blocks: tl.constexpr,
    has_bias: tl.constexpr,
    use_tf32: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, 32)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for block_idx in tl.range(0, scale_blocks):
        k = block_idx * 32 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + offs_n[:, None] * padded_in_features + k[None, :],
            mask=(offs_n[:, None] < out_features) & (k[None, :] < padded_in_features),
            other=0.0,
        ).to(tl.float32)
        scale_code = tl.load(
            scale_ptr + offs_n * scale_blocks + block_idx,
            mask=offs_n < out_features,
            other=127,
        ).to(tl.float32)
        weight *= tl.exp2(scale_code[:, None] - 127.0)
        if use_tf32:
            acc += tl.dot(x, tl.trans(weight), input_precision="tf32")
        else:
            acc += tl.dot(x, tl.trans(weight), input_precision="ieee")

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


@triton.jit
def _mxfp4_linear_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    padded_in_features: tl.constexpr,
    out_features: tl.constexpr,
    scale_blocks: tl.constexpr,
    has_bias: tl.constexpr,
    use_tf32: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, 32)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    packed_in_features: tl.constexpr = padded_in_features // 2

    for block_idx in tl.range(0, scale_blocks):
        k = block_idx * 32 + offs_k
        packed = tl.load(
            weight_ptr + offs_n[:, None] * packed_in_features + (k[None, :] // 2),
            mask=(offs_n[:, None] < out_features) & (k[None, :] < padded_in_features),
            other=0,
        )
        low_code = packed & 0x0F
        high_code = (packed >> 4) & 0x0F
        code = tl.where((k[None, :] % 2) == 0, low_code, high_code).to(tl.int32)
        magnitude = code & 0x07
        value = tl.full((block_n, 32), 0.0, dtype=tl.float32)
        value = tl.where(magnitude == 1, 0.5, value)
        value = tl.where(magnitude == 2, 1.0, value)
        value = tl.where(magnitude == 3, 1.5, value)
        value = tl.where(magnitude == 4, 2.0, value)
        value = tl.where(magnitude == 5, 3.0, value)
        value = tl.where(magnitude == 6, 4.0, value)
        value = tl.where(magnitude == 7, 6.0, value)
        value = tl.where((code & 0x08) != 0, -value, value)
        scale_code = tl.load(
            scale_ptr + offs_n * scale_blocks + block_idx,
            mask=offs_n < out_features,
            other=127,
        ).to(tl.float32)
        value *= tl.exp2(scale_code[:, None] - 127.0)

        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        ).to(tl.float32)
        if use_tf32:
            acc += tl.dot(x, tl.trans(value), input_precision="tf32")
        else:
            acc += tl.dot(x, tl.trans(value), input_precision="ieee")

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


def _check_packed_shapes(x_2d: Tensor, packed_weight: Tensor, scale_codes: Tensor, in_features: int) -> tuple[int, int, int, int]:
    if x_2d.ndim != 2:
        raise ValueError("x_2d must be 2D")
    if x_2d.shape[1] != in_features:
        raise ValueError(f"x_2d has {x_2d.shape[1]} input features, expected {in_features}")
    out_features = scale_codes.shape[0]
    scale_blocks = scale_codes.shape[1]
    padded_in = scale_blocks * MXFP_BLOCK_SIZE
    return x_2d.shape[0], in_features, padded_in, out_features


def mxfp8_linear_2d(
    x_2d: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    bias: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    if not x_2d.is_cuda:
        raise RuntimeError("mxfp8_linear requires CUDA tensors")
    rows, in_features, padded_in, out_features = _check_packed_shapes(x_2d, packed_weight, scale_codes, in_features)
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype if out_dtype is None else out_dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 128 and in_features >= 512 and out_features >= 512
    block_m = 64 if use_large_tiles else 16
    block_n = 64 if use_large_tiles else 32
    _mxfp8_linear_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
        x_2d,
        packed_weight,
        scale_codes,
        bias if bias is not None else x_2d,
        out,
        rows,
        in_features,
        padded_in,
        out_features,
        scale_codes.shape[1],
        bias is not None,
        use_large_tiles,
        block_m,
        block_n,
        num_warps=4,
    )
    return out


def mxfp4_linear_2d(
    x_2d: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    bias: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    if not x_2d.is_cuda:
        raise RuntimeError("mxfp4_linear requires CUDA tensors")
    rows, in_features, padded_in, out_features = _check_packed_shapes(x_2d, packed_weight, scale_codes, in_features)
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype if out_dtype is None else out_dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 128 and in_features >= 512 and out_features >= 512
    block_m = 64 if use_large_tiles else 16
    block_n = 64 if use_large_tiles else 32
    _mxfp4_linear_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
        x_2d,
        packed_weight,
        scale_codes,
        bias if bias is not None else x_2d,
        out,
        rows,
        in_features,
        padded_in,
        out_features,
        scale_codes.shape[1],
        bias is not None,
        use_large_tiles,
        block_m,
        block_n,
        num_warps=4,
    )
    return out


def mxfp8_linear(x: Tensor, packed_weight: Tensor, scale_codes: Tensor, in_features: int, bias: Tensor | None = None) -> Tensor:
    original_shape = x.shape
    x_2d = x.contiguous().reshape(-1, original_shape[-1])
    out = mxfp8_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
    return out.reshape(*original_shape[:-1], scale_codes.shape[0])


def mxfp4_linear(x: Tensor, packed_weight: Tensor, scale_codes: Tensor, in_features: int, bias: Tensor | None = None) -> Tensor:
    original_shape = x.shape
    x_2d = x.contiguous().reshape(-1, original_shape[-1])
    out = mxfp4_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
    return out.reshape(*original_shape[:-1], scale_codes.shape[0])
