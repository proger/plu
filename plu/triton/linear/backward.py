from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor


@dataclass
class LinearGradNormRecord:
    weight_norm_sq: Tensor
    bias_norm_sq: Tensor | None
    weight_elements: int
    bias_elements: int


_LINEAR_GRAD_NORM_RECORDS: list[LinearGradNormRecord] | None = None


@contextmanager
def collect_linear_grad_norms() -> Iterator[list[LinearGradNormRecord]]:
    global _LINEAR_GRAD_NORM_RECORDS
    previous = _LINEAR_GRAD_NORM_RECORDS
    records: list[LinearGradNormRecord] = []
    _LINEAR_GRAD_NORM_RECORDS = records
    try:
        yield records
    finally:
        _LINEAR_GRAD_NORM_RECORDS = previous


def _record_linear_grad_norms(
    weight_norm_sq: Tensor,
    bias_norm_sq: Tensor | None,
    weight_elements: int,
    bias_elements: int,
) -> None:
    if _LINEAR_GRAD_NORM_RECORDS is not None:
        _LINEAR_GRAD_NORM_RECORDS.append(
            LinearGradNormRecord(
                weight_norm_sq=weight_norm_sq,
                bias_norm_sq=bias_norm_sq,
                weight_elements=weight_elements,
                bias_elements=bias_elements,
            )
        )


@torch.no_grad()
def summarize_linear_grad_norms(records: list[LinearGradNormRecord]) -> dict[str, float]:
    if not records:
        return {
            "train/grad_norm/linear_calls": 0.0,
            "train/grad_norm/linear_total": 0.0,
            "train/grad_norm/linear_weight": 0.0,
            "train/grad_norm/linear_bias": 0.0,
        }

    weight_norm_sq = torch.stack([record.weight_norm_sq.reshape(-1).sum() for record in records]).sum()
    bias_parts = [record.bias_norm_sq.reshape(-1).sum() for record in records if record.bias_norm_sq is not None]
    bias_norm_sq = torch.stack(bias_parts).sum() if bias_parts else weight_norm_sq.new_zeros(())
    weight_value = float(weight_norm_sq.detach().cpu())
    bias_value = float(bias_norm_sq.detach().cpu())
    total_value = weight_value + bias_value
    return {
        "train/grad_norm/linear_calls": float(len(records)),
        "train/grad_norm/linear_weight_elements": float(sum(record.weight_elements for record in records)),
        "train/grad_norm/linear_bias_elements": float(sum(record.bias_elements for record in records)),
        "train/grad_norm/linear_weight": weight_value**0.5,
        "train/grad_norm/linear_bias": bias_value**0.5,
        "train/grad_norm/linear_total": total_value**0.5,
    }


@triton.jit
def _linear_input_grad_kernel(
    grad_out_ptr,
    weight_ptr,
    grad_input_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    use_tf32: tl.constexpr,
    cast_grad_to_float: tl.constexpr,
    cast_weight_to_float: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_n = tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_k), dtype=tl.float32)
    for n_start in tl.range(0, out_features, block_n):
        n = n_start + offs_n
        grad = tl.load(
            grad_out_ptr + offs_m[:, None] * out_features + n[None, :],
            mask=(offs_m[:, None] < rows) & (n[None, :] < out_features),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + n[:, None] * in_features + offs_k[None, :],
            mask=(n[:, None] < out_features) & (offs_k[None, :] < in_features),
            other=0.0,
        )
        if cast_grad_to_float:
            grad = grad.to(tl.float32)
        if cast_weight_to_float:
            weight = weight.to(tl.float32)
        if use_tf32:
            acc += tl.dot(grad, weight, input_precision="tf32")
        else:
            acc += tl.dot(grad, weight, input_precision="ieee")
    tl.store(
        grad_input_ptr + offs_m[:, None] * in_features + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_k[None, :] < in_features),
    )


@triton.jit
def _linear_weight_grad_reduce_kernel(
    grad_out_ptr,
    input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    grad_weight_norm_sq_ptr,
    grad_bias_norm_sq_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    has_bias: tl.constexpr,
    collect_norms: tl.constexpr,
    use_tf32: tl.constexpr,
    cast_grad_to_float: tl.constexpr,
    cast_input_to_float: tl.constexpr,
    grad_out_is_heads: tl.constexpr,
    grad_seq_len: tl.constexpr,
    grad_num_heads: tl.constexpr,
    grad_head_dim: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    block_m: tl.constexpr,
    grid_k: tl.constexpr,
    input_is_heads: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_m = tl.arange(0, block_m)
    acc = tl.zeros((block_n, block_k), dtype=tl.float32)
    bias_acc = tl.zeros((block_n,), dtype=tl.float32)
    for m_start in tl.range(0, rows, block_m):
        m = m_start + offs_m
        if grad_out_is_heads:
            grad_batch_offsets = m // grad_seq_len
            grad_time_offsets = m - grad_batch_offsets * grad_seq_len
            grad_head_offsets = offs_n // grad_head_dim
            grad_dim_offsets = offs_n - grad_head_offsets * grad_head_dim
            grad_offsets = (
                (grad_batch_offsets[:, None] * grad_num_heads + grad_head_offsets[None, :]) * grad_seq_len
                + grad_time_offsets[:, None]
            ) * grad_head_dim + grad_dim_offsets[None, :]
            grad = tl.load(
                grad_out_ptr + grad_offsets,
                mask=(m[:, None] < rows) & (offs_n[None, :] < out_features),
                other=0.0,
            )
        else:
            grad = tl.load(
                grad_out_ptr + m[:, None] * out_features + offs_n[None, :],
                mask=(m[:, None] < rows) & (offs_n[None, :] < out_features),
                other=0.0,
            )
        if has_bias:
            bias_acc += tl.sum(grad, axis=0)
        if input_is_heads:
            batch_offsets = m // seq_len
            time_offsets = m - batch_offsets * seq_len
            head_offsets = offs_k // head_dim
            dim_offsets = offs_k - head_offsets * head_dim
            input_offsets = (
                (batch_offsets[:, None] * num_heads + head_offsets[None, :]) * seq_len
                + time_offsets[:, None]
            ) * head_dim + dim_offsets[None, :]
            x = tl.load(
                input_ptr + input_offsets,
                mask=(m[:, None] < rows) & (offs_k[None, :] < in_features),
                other=0.0,
            )
        else:
            x = tl.load(
                input_ptr + m[:, None] * in_features + offs_k[None, :],
                mask=(m[:, None] < rows) & (offs_k[None, :] < in_features),
                other=0.0,
            )
        if cast_grad_to_float:
            grad = grad.to(tl.float32)
        if cast_input_to_float:
            x = x.to(tl.float32)
        if use_tf32:
            acc += tl.dot(tl.trans(grad), x, input_precision="tf32")
        else:
            acc += tl.dot(tl.trans(grad), x, input_precision="ieee")
    tl.store(
        grad_weight_ptr + offs_n[:, None] * in_features + offs_k[None, :],
        acc,
        mask=(offs_n[:, None] < out_features) & (offs_k[None, :] < in_features),
    )
    if has_bias:
        tl.store(grad_bias_ptr + offs_n, bias_acc, mask=(pid_k == 0) & (offs_n < out_features))
    if collect_norms:
        weight_mask = (offs_n[:, None] < out_features) & (offs_k[None, :] < in_features)
        weight_norm_sq = tl.sum(tl.where(weight_mask, acc * acc, 0.0))
        tl.store(grad_weight_norm_sq_ptr + pid_n * grid_k + pid_k, weight_norm_sq)
        if has_bias and pid_k == 0:
            bias_mask = offs_n < out_features
            bias_norm_sq = tl.sum(tl.where(bias_mask, bias_acc * bias_acc, 0.0))
            tl.store(grad_bias_norm_sq_ptr + pid_n, bias_norm_sq)


def linear_input_grad(grad_out: Tensor, weight: Tensor, out_dtype: torch.dtype | None = None) -> Tensor:
    rows, out_features = grad_out.shape
    in_features = weight.shape[1]
    grad_input = torch.empty((rows, in_features), device=grad_out.device, dtype=grad_out.dtype if out_dtype is None else out_dtype)
    if rows == 0:
        return grad_input
    use_large_tiles = rows >= 128 and out_features >= 512 and in_features >= 512
    cast_grad_to_float = grad_out.dtype != weight.dtype and grad_out.dtype != torch.float32
    cast_weight_to_float = grad_out.dtype != weight.dtype and weight.dtype != torch.float32
    inputs_are_fp32 = (grad_out.dtype == torch.float32 or cast_grad_to_float) and (
        weight.dtype == torch.float32 or cast_weight_to_float
    )
    use_tf32 = inputs_are_fp32 and (use_large_tiles or cast_grad_to_float or cast_weight_to_float)
    block_m = 64 if use_large_tiles else 16
    block_k = 128 if use_large_tiles else 32
    block_n = 32 if not use_large_tiles or out_features >= 16384 else 64
    _linear_input_grad_kernel[(triton.cdiv(rows, block_m), triton.cdiv(in_features, block_k))](
        grad_out,
        weight,
        grad_input,
        rows,
        out_features,
        in_features,
        use_tf32,
        cast_grad_to_float,
        cast_weight_to_float,
        block_m,
        block_k,
        block_n,
        num_warps=4,
    )
    return grad_input


def _linear_weight_bias_grad(
    grad_out: Tensor,
    x: Tensor,
    has_bias: bool,
    dtype: torch.dtype | None,
    *,
    input_is_heads: bool,
    rows: int,
    in_features: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    grad_out_is_heads: bool = False,
    grad_seq_len: int = 0,
    grad_num_heads: int = 0,
    grad_head_dim: int = 0,
) -> tuple[Tensor, Tensor | None]:
    out_features = grad_num_heads * grad_head_dim if grad_out_is_heads else grad_out.shape[1]
    grad_dtype = grad_out.dtype if dtype is None else dtype
    use_large_tiles = rows >= 512 and out_features >= 512 and in_features >= 256
    cast_grad_to_float = grad_out.dtype != x.dtype and grad_out.dtype != torch.float32
    cast_input_to_float = grad_out.dtype != x.dtype and x.dtype != torch.float32
    inputs_are_fp32 = (grad_out.dtype == torch.float32 or cast_grad_to_float) and (x.dtype == torch.float32 or cast_input_to_float)
    use_wide_reduce = rows >= 128 and out_features >= 512 and in_features >= 512
    use_tf32 = inputs_are_fp32 and (use_large_tiles or use_wide_reduce or cast_grad_to_float or cast_input_to_float)
    grad_weight = torch.empty((out_features, in_features), device=grad_out.device, dtype=grad_dtype)
    grad_bias = torch.empty(out_features, device=grad_out.device, dtype=grad_dtype) if has_bias else None
    if rows == 0:
        grad_weight.zero_()
        if grad_bias is not None:
            grad_bias.zero_()
        if _LINEAR_GRAD_NORM_RECORDS is not None:
            zero = torch.zeros((), device=grad_out.device, dtype=torch.float32)
            _record_linear_grad_norms(zero, zero if has_bias else None, grad_weight.numel(), grad_bias.numel() if grad_bias is not None else 0)
        return grad_weight, grad_bias

    if rows < 512 and out_features >= 16384:
        block_n = 64
        block_k = 128
        block_m = 16
    elif rows < 512:
        block_n = 128
        block_k = 64
        block_m = 32
    else:
        block_n = 64
        block_k = 128
        block_m = 64 if out_features >= 4096 and in_features <= 2048 else 32

    grid_n = triton.cdiv(out_features, block_n)
    grid_k = triton.cdiv(in_features, block_k)
    collect_norms = _LINEAR_GRAD_NORM_RECORDS is not None
    grad_weight_norm_sq = torch.empty((grid_n, grid_k), device=grad_out.device, dtype=torch.float32) if collect_norms else grad_weight
    grad_bias_norm_sq = torch.empty((grid_n,), device=grad_out.device, dtype=torch.float32) if collect_norms and has_bias else grad_weight
    inputs_are_bf16 = grad_out.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    num_stages = 1 if inputs_are_bf16 and has_bias else 3
    _linear_weight_grad_reduce_kernel[(grid_n, grid_k)](
        grad_out,
        x,
        grad_weight,
        grad_bias if grad_bias is not None else grad_weight,
        grad_weight_norm_sq,
        grad_bias_norm_sq,
        rows,
        out_features,
        in_features,
        has_bias,
        collect_norms,
        use_tf32,
        cast_grad_to_float,
        cast_input_to_float,
        grad_out_is_heads,
        grad_seq_len,
        grad_num_heads,
        grad_head_dim,
        block_n,
        block_k,
        block_m,
        grid_k,
        input_is_heads,
        seq_len,
        num_heads,
        head_dim,
        num_warps=4,
        num_stages=num_stages,
    )
    if collect_norms:
        _record_linear_grad_norms(
            grad_weight_norm_sq,
            grad_bias_norm_sq if has_bias else None,
            grad_weight.numel(),
            grad_bias.numel() if grad_bias is not None else 0,
        )
    return grad_weight, grad_bias


def linear_weight_bias_grad(grad_out: Tensor, x: Tensor, has_bias: bool, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor | None]:
    rows, _ = grad_out.shape
    return _linear_weight_bias_grad(
        grad_out,
        x,
        has_bias,
        dtype,
        input_is_heads=False,
        rows=rows,
        in_features=x.shape[1],
        seq_len=0,
        num_heads=0,
        head_dim=0,
    )


def linear_weight_bias_grad_heads_grad(
    grad_out_heads: Tensor,
    x: Tensor,
    has_bias: bool,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor | None]:
    if grad_out_heads.ndim != 4:
        raise ValueError("grad_out_heads must have shape [batch, heads, seq_len, head_dim]")
    batch, num_heads, seq_len, head_dim = grad_out_heads.shape
    rows = batch * seq_len
    if x.shape[0] != rows:
        raise ValueError(f"x rows {x.shape[0]} must equal batch * seq_len {rows}")
    return _linear_weight_bias_grad(
        grad_out_heads,
        x,
        has_bias,
        dtype,
        input_is_heads=False,
        rows=rows,
        in_features=x.shape[1],
        seq_len=0,
        num_heads=0,
        head_dim=0,
        grad_out_is_heads=True,
        grad_seq_len=seq_len,
        grad_num_heads=num_heads,
        grad_head_dim=head_dim,
    )


def linear_weight_bias_grad_heads_input(
    grad_out: Tensor,
    x_heads: Tensor,
    has_bias: bool,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor | None]:
    if x_heads.ndim != 4:
        raise ValueError("x_heads must have shape [batch, heads, seq_len, head_dim]")
    batch, num_heads, seq_len, head_dim = x_heads.shape
    rows = batch * seq_len
    if grad_out.shape[0] != rows:
        raise ValueError(f"grad_out rows {grad_out.shape[0]} must equal batch * seq_len {rows}")
    return _linear_weight_bias_grad(
        grad_out,
        x_heads,
        has_bias,
        dtype,
        input_is_heads=True,
        rows=rows,
        in_features=num_heads * head_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
