from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.triton.linear.backward import linear_weight_bias_grad


MXFP_BLOCK_SIZE = 32
_E8M0_BIAS = 127
_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_E2M1_MAX = 6.0
_BLACKWELL_SCALE_ALIGN_K = 8
_BLACKWELL_SCALE_ALIGN_N = 128
_BLACKWELL_SCALE_SWIZZLE_K = 4


@triton.jit
def _grouped_pids(pid, grid_m: tl.constexpr, grid_n: tl.constexpr, group_m: tl.constexpr):
    group_width: tl.constexpr = group_m * grid_n
    group_id = pid // group_width
    first_m = group_id * group_m
    group_size_m = tl.minimum(grid_m - first_m, group_m)
    pid_in_group = pid - group_id * group_width
    pid_m = first_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _load_unswizzled_mx_scale_bw(scale_ptr, base):
    offs_n = tl.arange(0, 128)
    offs_k = tl.arange(0, 4)
    row_base = ((offs_n % 32) * 16 + (offs_n // 32) * 4)[:, None]
    return tl.load(scale_ptr + base + row_base + offs_k)


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


def _blackwell_swizzle_scale_codes(scale_codes: Tensor) -> Tensor:
    if scale_codes.ndim != 2:
        raise ValueError("scale_codes must be 2D")
    out_features, scale_blocks = scale_codes.shape
    data = scale_codes.T.contiguous()
    k_pad = triton.cdiv(scale_blocks, _BLACKWELL_SCALE_ALIGN_K) * _BLACKWELL_SCALE_ALIGN_K
    n_pad = triton.cdiv(out_features, _BLACKWELL_SCALE_ALIGN_N) * _BLACKWELL_SCALE_ALIGN_N
    data = torch.nn.functional.pad(data, (0, n_pad - out_features, 0, k_pad - scale_blocks), value=127)
    data = data.transpose(-1, -2).contiguous()
    data = data.reshape(
        n_pad // _BLACKWELL_SCALE_ALIGN_N,
        _BLACKWELL_SCALE_ALIGN_N // 32,
        32,
        k_pad // _BLACKWELL_SCALE_SWIZZLE_K,
        _BLACKWELL_SCALE_SWIZZLE_K,
    )
    data = data.transpose(1, 3).contiguous()
    return data.view(
        1,
        n_pad // _BLACKWELL_SCALE_ALIGN_N,
        k_pad // _BLACKWELL_SCALE_SWIZZLE_K,
        2,
        256,
    )


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
    return values.reshape(out_features, padded_in).contiguous(), _blackwell_swizzle_scale_codes(scale_codes), in_features


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
    return (low | high).contiguous(), _blackwell_swizzle_scale_codes(scale_codes), in_features


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
    scale_k_groups: tl.constexpr,
    has_bias: tl.constexpr,
    input_format: tl.constexpr,
    weight_format: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    group_m: tl.constexpr,
):
    pid_m, pid_n = _grouped_pids(tl.program_id(0), grid_m, grid_n, group_m)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for scale_group in tl.range(0, tl.cdiv(scale_blocks, 4)):
        k = scale_group * 128 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )
        x_scaled = x.to(tl.bfloat16) if input_format == "bf16" else x.to(tl.float16)
        weight_t = tl.load(
            weight_ptr + offs_n[None, :] * padded_in_features + k[:, None],
            mask=(offs_n[None, :] < out_features) & (k[:, None] < padded_in_features),
            other=0.0,
        )
        scale_base = pid_n * scale_k_groups * 512 + scale_group * 512
        scale_code = _load_unswizzled_mx_scale_bw(scale_ptr, scale_base)
        acc = tl.dot_scaled(
            x_scaled,
            None,
            input_format,
            weight_t,
            scale_code,
            weight_format,
            acc=acc,
            fast_math=True,
        )

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
    scale_k_groups: tl.constexpr,
    has_bias: tl.constexpr,
    input_format: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    group_m: tl.constexpr,
):
    pid_m, pid_n = _grouped_pids(tl.program_id(0), grid_m, grid_n, group_m)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, 128)
    offs_packed_k = tl.arange(0, 64)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    packed_in_features: tl.constexpr = padded_in_features // 2

    for scale_group in tl.range(0, tl.cdiv(scale_blocks, 4)):
        k = scale_group * 128 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )
        x_scaled = x.to(tl.bfloat16) if input_format == "bf16" else x.to(tl.float16)
        packed_k = scale_group * 64 + offs_packed_k
        packed_t = tl.load(
            weight_ptr + offs_n[None, :] * packed_in_features + packed_k[:, None],
            mask=(offs_n[None, :] < out_features) & (packed_k[:, None] < packed_in_features),
            other=0,
        )
        scale_base = pid_n * scale_k_groups * 512 + scale_group * 512
        scale_code = _load_unswizzled_mx_scale_bw(scale_ptr, scale_base)
        acc = tl.dot_scaled(
            x_scaled,
            None,
            input_format,
            packed_t,
            scale_code,
            "e2m1",
            acc=acc,
            fast_math=True,
        )

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


@triton.jit
def _mx_linear_bias_grad_kernel(
    grad_out_ptr,
    grad_bias_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    grad = tl.load(
        grad_out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(grad, axis=0)
    tl.atomic_add(grad_bias_ptr + offs_n, acc, sem="relaxed", mask=offs_n < out_features)


def _check_blackwell_scale_shape(scale_codes: Tensor, out_features: int, scale_blocks: int) -> int:
    if scale_codes.ndim != 5:
        raise ValueError("scale_codes must use the Blackwell-swizzled 5D layout")
    expected_n_tiles = triton.cdiv(out_features, _BLACKWELL_SCALE_ALIGN_N)
    expected_k_groups = triton.cdiv(
        triton.cdiv(scale_blocks, _BLACKWELL_SCALE_ALIGN_K) * _BLACKWELL_SCALE_ALIGN_K,
        _BLACKWELL_SCALE_SWIZZLE_K,
    )
    expected_shape = (1, expected_n_tiles, expected_k_groups, 2, 256)
    if tuple(scale_codes.shape) != expected_shape:
        raise ValueError(f"scale_codes has shape {tuple(scale_codes.shape)}, expected {expected_shape}")
    if not scale_codes.is_contiguous():
        raise ValueError("scale_codes must be contiguous")
    return expected_k_groups


def _check_packed_shapes(
    x_2d: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    packed_values_per_byte: int,
) -> tuple[int, int, int, int, int, int]:
    if x_2d.ndim != 2:
        raise ValueError("x_2d must be 2D")
    if x_2d.shape[1] != in_features:
        raise ValueError(f"x_2d has {x_2d.shape[1]} input features, expected {in_features}")
    if packed_weight.ndim != 2:
        raise ValueError("packed_weight must be 2D")
    out_features = packed_weight.shape[0]
    padded_in = packed_weight.shape[1] * packed_values_per_byte
    scale_blocks = triton.cdiv(padded_in, MXFP_BLOCK_SIZE)
    scale_k_groups = _check_blackwell_scale_shape(scale_codes, out_features, scale_blocks)
    return x_2d.shape[0], in_features, padded_in, out_features, scale_blocks, scale_k_groups


def _dot_scaled_input_format(x: Tensor, op_name: str) -> str:
    if x.dtype == torch.bfloat16:
        return "bf16"
    if x.dtype == torch.float16:
        return "fp16"
    raise RuntimeError(f"{op_name} requires bf16 or fp16 inputs")


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
    rows, in_features, padded_in, out_features, scale_blocks, scale_k_groups = _check_packed_shapes(
        x_2d,
        packed_weight,
        scale_codes,
        in_features,
        packed_values_per_byte=1,
    )
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype if out_dtype is None else out_dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 128 and in_features >= 512 and out_features >= 512
    input_format = _dot_scaled_input_format(x_2d, "mxfp8_linear")
    weight_format = "e5m2" if packed_weight.dtype == torch.float8_e5m2 else "e4m3"
    block_m = 64 if use_large_tiles else 16
    block_n = 128
    grid_m = triton.cdiv(rows, block_m)
    grid_n = triton.cdiv(out_features, block_n)
    group_m = 8 if use_large_tiles else 4
    _mxfp8_linear_kernel[(grid_m * grid_n,)](
        x_2d,
        packed_weight,
        scale_codes,
        bias if bias is not None else x_2d,
        out,
        rows,
        in_features,
        padded_in,
        out_features,
        scale_blocks,
        scale_k_groups,
        bias is not None,
        input_format,
        weight_format,
        block_m,
        block_n,
        grid_m,
        grid_n,
        group_m,
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
    rows, in_features, padded_in, out_features, scale_blocks, scale_k_groups = _check_packed_shapes(
        x_2d,
        packed_weight,
        scale_codes,
        in_features,
        packed_values_per_byte=2,
    )
    out = torch.empty((rows, out_features), device=x_2d.device, dtype=x_2d.dtype if out_dtype is None else out_dtype)
    if rows == 0:
        return out
    use_large_tiles = rows >= 128 and in_features >= 512 and out_features >= 512
    input_format = _dot_scaled_input_format(x_2d, "mxfp4_linear")
    block_m = 64 if use_large_tiles else 16
    block_n = 128
    grid_m = triton.cdiv(rows, block_m)
    grid_n = triton.cdiv(out_features, block_n)
    group_m = 8 if use_large_tiles else 4
    _mxfp4_linear_kernel[(grid_m * grid_n,)](
        x_2d,
        packed_weight,
        scale_codes,
        bias if bias is not None else x_2d,
        out,
        rows,
        in_features,
        padded_in,
        out_features,
        scale_blocks,
        scale_k_groups,
        bias is not None,
        input_format,
        block_m,
        block_n,
        grid_m,
        grid_n,
        group_m,
        num_warps=4,
    )
    return out


def mxfp8_linear_input_grad_2d(
    grad_out_2d: Tensor,
    packed_transposed_weight: Tensor,
    transposed_scale_codes: Tensor,
    transposed_in_features: int,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    return mxfp8_linear_2d(
        grad_out_2d,
        packed_transposed_weight,
        transposed_scale_codes,
        transposed_in_features,
        None,
        out_dtype,
    )


def mxfp4_linear_input_grad_2d(
    grad_out_2d: Tensor,
    packed_transposed_weight: Tensor,
    transposed_scale_codes: Tensor,
    transposed_in_features: int,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    return mxfp4_linear_2d(
        grad_out_2d,
        packed_transposed_weight,
        transposed_scale_codes,
        transposed_in_features,
        None,
        out_dtype,
    )


def mx_linear_bias_grad_2d(grad_out_2d: Tensor, out_features: int, dtype: torch.dtype | None = None) -> Tensor:
    rows = grad_out_2d.shape[0]
    grad_bias = torch.zeros(out_features, device=grad_out_2d.device, dtype=grad_out_2d.dtype if dtype is None else dtype)
    if rows == 0:
        return grad_bias
    use_large_tiles = rows >= 128 and out_features >= 512
    block_n = 64 if use_large_tiles else 32
    block_m = 64 if use_large_tiles else 32
    _mx_linear_bias_grad_kernel[(triton.cdiv(out_features, block_n), triton.cdiv(rows, block_m))](
        grad_out_2d,
        grad_bias,
        rows,
        out_features,
        block_n,
        block_m,
        num_warps=4,
    )
    return grad_bias


class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        packed_weight: Tensor,
        scale_codes: Tensor,
        in_features: int,
        bias: Tensor | None,
        input_grad_packed_weight: Tensor | None,
        input_grad_scale_codes: Tensor | None,
        input_grad_in_features: int | None,
    ):
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        packed_weight = packed_weight.contiguous()
        scale_codes = scale_codes.contiguous()
        bias = None if bias is None else bias.contiguous()
        out_features = packed_weight.shape[0]
        has_input_grad_pack = input_grad_packed_weight is not None
        if x.requires_grad and (input_grad_packed_weight is None or input_grad_scale_codes is None or input_grad_in_features is None):
            raise RuntimeError("mxfp8_linear backward requires packed transposed weight; pass input_grad_packed_weight/input_grad_scale_codes")
        if input_grad_packed_weight is not None:
            input_grad_packed_weight = input_grad_packed_weight.contiguous()
            input_grad_scale_codes = input_grad_scale_codes.contiguous()
        out = mxfp8_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
        if has_input_grad_pack:
            ctx.save_for_backward(packed_weight, scale_codes, input_grad_packed_weight, input_grad_scale_codes)
        else:
            ctx.save_for_backward(packed_weight, scale_codes)
        ctx.in_features = in_features
        ctx.input_grad_in_features = input_grad_in_features
        ctx.has_input_grad_pack = has_input_grad_pack
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        ctx.input_dtype = x.dtype
        ctx.out_features = out_features
        return out.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().reshape(-1, ctx.out_features)
        grad_x = None
        if ctx.needs_input_grad[0]:
            if not ctx.has_input_grad_pack:
                raise RuntimeError("mxfp8_linear backward requires packed transposed weight")
            input_grad_packed_weight, input_grad_scale_codes = saved[2], saved[3]
            grad_x = mxfp8_linear_input_grad_2d(
                grad_out_2d,
                input_grad_packed_weight,
                input_grad_scale_codes,
                ctx.input_grad_in_features,
                ctx.input_dtype,
            ).reshape(ctx.original_shape)
        grad_bias = mx_linear_bias_grad_2d(grad_out_2d, ctx.out_features, grad_out.dtype) if ctx.has_bias else None
        return grad_x, None, None, None, grad_bias, None, None, None


class _MXFP8LinearWithMasterWeight(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        packed_weight: Tensor,
        scale_codes: Tensor,
        in_features: int,
        master_weight: Tensor,
        bias: Tensor | None,
        input_grad_packed_weight: Tensor | None,
        input_grad_scale_codes: Tensor | None,
        input_grad_in_features: int | None,
    ):
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        packed_weight = packed_weight.contiguous()
        scale_codes = scale_codes.contiguous()
        bias = None if bias is None else bias.contiguous()
        out_features = packed_weight.shape[0]
        has_input_grad_pack = input_grad_packed_weight is not None
        if x.requires_grad and (input_grad_packed_weight is None or input_grad_scale_codes is None or input_grad_in_features is None):
            raise RuntimeError("mxfp8_linear_with_master_weight backward requires packed transposed weight")
        if input_grad_packed_weight is not None:
            input_grad_packed_weight = input_grad_packed_weight.contiguous()
            input_grad_scale_codes = input_grad_scale_codes.contiguous()
        out = mxfp8_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
        if has_input_grad_pack:
            ctx.save_for_backward(x_2d, packed_weight, scale_codes, input_grad_packed_weight, input_grad_scale_codes)
        else:
            ctx.save_for_backward(x_2d, packed_weight, scale_codes)
        ctx.in_features = in_features
        ctx.input_grad_in_features = input_grad_in_features
        ctx.has_input_grad_pack = has_input_grad_pack
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        ctx.input_dtype = x.dtype
        ctx.master_weight_dtype = master_weight.dtype
        ctx.out_features = out_features
        return out.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        x_2d = saved[0]
        grad_out_2d = grad_out.contiguous().reshape(-1, ctx.out_features)
        grad_x = None
        if ctx.needs_input_grad[0]:
            if not ctx.has_input_grad_pack:
                raise RuntimeError("mxfp8_linear_with_master_weight backward requires packed transposed weight")
            input_grad_packed_weight, input_grad_scale_codes = saved[3], saved[4]
            grad_x = mxfp8_linear_input_grad_2d(
                grad_out_2d,
                input_grad_packed_weight,
                input_grad_scale_codes,
                ctx.input_grad_in_features,
                ctx.input_dtype,
            ).reshape(ctx.original_shape)
        grad_weight, _ = linear_weight_bias_grad(grad_out_2d, x_2d, False, dtype=ctx.master_weight_dtype)
        grad_bias = mx_linear_bias_grad_2d(grad_out_2d, ctx.out_features, grad_out.dtype) if ctx.has_bias else None
        return grad_x, None, None, None, grad_weight, grad_bias, None, None, None


class _MXFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        packed_weight: Tensor,
        scale_codes: Tensor,
        in_features: int,
        bias: Tensor | None,
        input_grad_packed_weight: Tensor | None,
        input_grad_scale_codes: Tensor | None,
        input_grad_in_features: int | None,
    ):
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        packed_weight = packed_weight.contiguous()
        scale_codes = scale_codes.contiguous()
        bias = None if bias is None else bias.contiguous()
        out_features = packed_weight.shape[0]
        has_input_grad_pack = input_grad_packed_weight is not None
        if x.requires_grad and (input_grad_packed_weight is None or input_grad_scale_codes is None or input_grad_in_features is None):
            raise RuntimeError("mxfp4_linear backward requires packed transposed weight; pass input_grad_packed_weight/input_grad_scale_codes")
        if input_grad_packed_weight is not None:
            input_grad_packed_weight = input_grad_packed_weight.contiguous()
            input_grad_scale_codes = input_grad_scale_codes.contiguous()
        out = mxfp4_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
        if has_input_grad_pack:
            ctx.save_for_backward(packed_weight, scale_codes, input_grad_packed_weight, input_grad_scale_codes)
        else:
            ctx.save_for_backward(packed_weight, scale_codes)
        ctx.in_features = in_features
        ctx.input_grad_in_features = input_grad_in_features
        ctx.has_input_grad_pack = has_input_grad_pack
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        ctx.input_dtype = x.dtype
        ctx.out_features = out_features
        return out.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        grad_out_2d = grad_out.contiguous().reshape(-1, ctx.out_features)
        grad_x = None
        if ctx.needs_input_grad[0]:
            if not ctx.has_input_grad_pack:
                raise RuntimeError("mxfp4_linear backward requires packed transposed weight")
            input_grad_packed_weight, input_grad_scale_codes = saved[2], saved[3]
            grad_x = mxfp4_linear_input_grad_2d(
                grad_out_2d,
                input_grad_packed_weight,
                input_grad_scale_codes,
                ctx.input_grad_in_features,
                ctx.input_dtype,
            ).reshape(ctx.original_shape)
        grad_bias = mx_linear_bias_grad_2d(grad_out_2d, ctx.out_features, grad_out.dtype) if ctx.has_bias else None
        return grad_x, None, None, None, grad_bias, None, None, None


class _MXFP4LinearWithMasterWeight(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        packed_weight: Tensor,
        scale_codes: Tensor,
        in_features: int,
        master_weight: Tensor,
        bias: Tensor | None,
        input_grad_packed_weight: Tensor | None,
        input_grad_scale_codes: Tensor | None,
        input_grad_in_features: int | None,
    ):
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        packed_weight = packed_weight.contiguous()
        scale_codes = scale_codes.contiguous()
        bias = None if bias is None else bias.contiguous()
        out_features = packed_weight.shape[0]
        has_input_grad_pack = input_grad_packed_weight is not None
        if x.requires_grad and (input_grad_packed_weight is None or input_grad_scale_codes is None or input_grad_in_features is None):
            raise RuntimeError("mxfp4_linear_with_master_weight backward requires packed transposed weight")
        if input_grad_packed_weight is not None:
            input_grad_packed_weight = input_grad_packed_weight.contiguous()
            input_grad_scale_codes = input_grad_scale_codes.contiguous()
        out = mxfp4_linear_2d(x_2d, packed_weight, scale_codes, in_features, bias)
        if has_input_grad_pack:
            ctx.save_for_backward(x_2d, packed_weight, scale_codes, input_grad_packed_weight, input_grad_scale_codes)
        else:
            ctx.save_for_backward(x_2d, packed_weight, scale_codes)
        ctx.in_features = in_features
        ctx.input_grad_in_features = input_grad_in_features
        ctx.has_input_grad_pack = has_input_grad_pack
        ctx.has_bias = bias is not None
        ctx.original_shape = original_shape
        ctx.input_dtype = x.dtype
        ctx.master_weight_dtype = master_weight.dtype
        ctx.out_features = out_features
        return out.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        x_2d = saved[0]
        grad_out_2d = grad_out.contiguous().reshape(-1, ctx.out_features)
        grad_x = None
        if ctx.needs_input_grad[0]:
            if not ctx.has_input_grad_pack:
                raise RuntimeError("mxfp4_linear_with_master_weight backward requires packed transposed weight")
            input_grad_packed_weight, input_grad_scale_codes = saved[3], saved[4]
            grad_x = mxfp4_linear_input_grad_2d(
                grad_out_2d,
                input_grad_packed_weight,
                input_grad_scale_codes,
                ctx.input_grad_in_features,
                ctx.input_dtype,
            ).reshape(ctx.original_shape)
        grad_weight, _ = linear_weight_bias_grad(grad_out_2d, x_2d, False, dtype=ctx.master_weight_dtype)
        grad_bias = mx_linear_bias_grad_2d(grad_out_2d, ctx.out_features, grad_out.dtype) if ctx.has_bias else None
        return grad_x, None, None, None, grad_weight, grad_bias, None, None, None


def mxfp8_linear(
    x: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    bias: Tensor | None = None,
    input_grad_packed_weight: Tensor | None = None,
    input_grad_scale_codes: Tensor | None = None,
    input_grad_in_features: int | None = None,
) -> Tensor:
    return _MXFP8Linear.apply(
        x,
        packed_weight,
        scale_codes,
        in_features,
        bias,
        input_grad_packed_weight,
        input_grad_scale_codes,
        input_grad_in_features,
    )


def mxfp4_linear(
    x: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    bias: Tensor | None = None,
    input_grad_packed_weight: Tensor | None = None,
    input_grad_scale_codes: Tensor | None = None,
    input_grad_in_features: int | None = None,
) -> Tensor:
    return _MXFP4Linear.apply(
        x,
        packed_weight,
        scale_codes,
        in_features,
        bias,
        input_grad_packed_weight,
        input_grad_scale_codes,
        input_grad_in_features,
    )


def mxfp8_linear_with_master_weight(
    x: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    master_weight: Tensor,
    bias: Tensor | None = None,
    input_grad_packed_weight: Tensor | None = None,
    input_grad_scale_codes: Tensor | None = None,
    input_grad_in_features: int | None = None,
) -> Tensor:
    return _MXFP8LinearWithMasterWeight.apply(
        x,
        packed_weight,
        scale_codes,
        in_features,
        master_weight,
        bias,
        input_grad_packed_weight,
        input_grad_scale_codes,
        input_grad_in_features,
    )


def mxfp4_linear_with_master_weight(
    x: Tensor,
    packed_weight: Tensor,
    scale_codes: Tensor,
    in_features: int,
    master_weight: Tensor,
    bias: Tensor | None = None,
    input_grad_packed_weight: Tensor | None = None,
    input_grad_scale_codes: Tensor | None = None,
    input_grad_in_features: int | None = None,
) -> Tensor:
    return _MXFP4LinearWithMasterWeight.apply(
        x,
        packed_weight,
        scale_codes,
        in_features,
        master_weight,
        bias,
        input_grad_packed_weight,
        input_grad_scale_codes,
        input_grad_in_features,
    )
