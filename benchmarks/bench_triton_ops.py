from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.ref.flash_attention import flash_attention as ref_flash_attention
from plu.ref.gelu_mlp import gelu_mlp as ref_gelu_mlp
from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
from plu.ref.paged_kv_cache import make_static_page_table as ref_make_static_page_table
from plu.ref.paged_kv_cache import paged_self_kv_cache_attention as ref_paged_self_kv_cache_attention
from plu.ref.paged_kv_cache import resample_page_table as ref_resample_page_table
from plu.ref.paged_kv_cache import resample_state as ref_resample_state
from plu.ref.c_proj import c_proj as ref_c_proj
from plu.ref.conv1d_gelu import conv1d_gelu as ref_conv1d_gelu
from plu.ref.embedding import decoder_embedding as ref_decoder_embedding
from plu.ref.embedding import encoder_position_embedding as ref_encoder_position_embedding
from plu.ref.layer_norm import layer_norm as ref_layer_norm
from plu.ref.linear import linear as ref_linear
from plu.ref.qkv_proj import qkv_proj as ref_qkv_proj
from plu.ref.residual_add import residual_add as ref_residual_add
from plu.triton.cross_entropy import cross_entropy as triton_cross_entropy
from plu.triton.flash_attention import flash_attention as triton_flash_attention
from plu.triton.gelu_mlp import gelu_mlp as triton_gelu_mlp
from plu.triton.matmul_top1 import matmul_top1 as triton_matmul_top1
from plu.triton.c_proj import c_proj as triton_c_proj
from plu.triton.conv1d_gelu import conv1d_gelu as triton_conv1d_gelu
from plu.triton.embedding import decoder_embedding as triton_decoder_embedding
from plu.triton.embedding import encoder_position_embedding as triton_encoder_position_embedding
from plu.triton.layer_norm import layer_norm as triton_layer_norm
from plu.triton.linear import linear as triton_linear
from plu.triton.linear.forward import linear_forward_2d as triton_linear_forward_2d
from plu.triton.mx_linear import (
    mxfp4_linear,
    mxfp8_linear,
    nvfp4_linear,
    pack_mxfp4_weight,
    pack_mxfp8_weight,
    pack_nvfp4_weight,
)
from plu.triton.paged_kv_cache import make_static_page_table as triton_make_static_page_table
from plu.triton.paged_kv_cache import paged_self_kv_cache_attention as triton_paged_self_kv_cache_attention
from plu.triton.paged_kv_cache import resample_page_table as triton_resample_page_table
from plu.triton.paged_kv_cache import resample_state as triton_resample_state
from plu.triton.qkv_proj import qkv_proj as triton_qkv_proj
from plu.triton.residual_add import residual_add as triton_residual_add
from plu.triton.unembedding_cross_entropy import unembedding_cross_entropy as triton_unembedding_cross_entropy


def set_tf32(enabled: bool) -> None:
    precision = "tf32" if enabled else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = enabled
        try:
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        except AttributeError:
            pass


def cuda_time_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def fresh_like(tensor: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(requires_grad)


def bench_flash_attention(args: argparse.Namespace) -> list[dict]:
    query = torch.randn(args.batch, args.heads, args.query_len, args.head_dim, device="cuda", requires_grad=True)
    key = torch.randn(args.batch, args.heads, args.key_len, args.head_dim, device="cuda", requires_grad=True)
    value = torch.randn(args.batch, args.heads, args.key_len, args.head_dim, device="cuda", requires_grad=True)
    mask = None
    if args.causal:
        if args.query_len != args.key_len:
            raise ValueError("causal flash_attention benchmark requires query_len == key_len")
        mask = torch.empty(args.query_len, args.key_len, device="cuda").fill_(-float("inf")).triu_(1)

    def ref_step():
        q, k, v = fresh_like(query), fresh_like(key), fresh_like(value)
        out = ref_flash_attention(q, k, v, mask)
        out.backward(torch.randn_like(out))

    def triton_step():
        q, k, v = fresh_like(query), fresh_like(key), fresh_like(value)
        out = triton_flash_attention(q, k, v, mask)
        out.backward(torch.randn_like(out))

    return [
        {"op": "flash_attention", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "flash_attention", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_paged_kv_cache(args: argparse.Namespace) -> list[dict]:
    dtype = torch.bfloat16
    page_size = args.page_size
    max_pages = (args.target_len + page_size - 1) // page_size
    pages = args.batch * max_pages
    query = torch.randn(args.batch, args.heads, 1, args.head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    position = torch.full((args.batch,), args.target_len - 1, device="cuda", dtype=torch.long)
    ref_page_table = ref_make_static_page_table(args.batch, max_pages, device="cuda")
    triton_page_table = triton_make_static_page_table(args.batch, max_pages, device="cuda")
    ref_key_pages = torch.randn(pages, args.heads, page_size, args.head_dim, device="cuda", dtype=dtype)
    ref_value_pages = torch.randn_like(ref_key_pages)
    triton_key_pages = ref_key_pages.clone()
    triton_value_pages = ref_value_pages.clone()
    ancestors = torch.arange(args.batch, device="cuda", dtype=torch.long).flip(0)
    next_page_ids = torch.arange(args.batch, device="cuda", dtype=torch.long) + pages
    page_table_arena = torch.cat(
        [triton_page_table, torch.empty(args.batch, 1, device="cuda", dtype=triton_page_table.dtype)],
        dim=1,
    )
    ref_page_table_arena = page_table_arena.clone()
    state = torch.randn(args.batch, max_pages, args.head_dim, device="cuda", dtype=dtype)

    def ref_attention():
        ref_paged_self_kv_cache_attention(query, key, value, ref_key_pages, ref_value_pages, ref_page_table, position)

    def triton_attention():
        triton_paged_self_kv_cache_attention(query, key, value, triton_key_pages, triton_value_pages, triton_page_table, position)

    def ref_page_table_resample():
        ref_resample_page_table(ref_page_table_arena, ancestors, next_page_ids=next_page_ids, next_page_index=max_pages)

    def triton_page_table_resample():
        triton_resample_page_table(page_table_arena, ancestors, next_page_ids=next_page_ids, next_page_index=max_pages)

    def ref_state_resample():
        ref_resample_state(state, ancestors)

    def triton_state_resample():
        triton_resample_state(state, ancestors)

    return [
        {"op": "paged_kv_cache_attention", "target": "ref", "mode": "forward", "ms": cuda_time_ms(ref_attention, args.warmup, args.iters)},
        {"op": "paged_kv_cache_attention", "target": "triton", "mode": "forward", "ms": cuda_time_ms(triton_attention, args.warmup, args.iters)},
        {"op": "paged_kv_cache_page_table_resample", "target": "ref", "mode": "forward", "ms": cuda_time_ms(ref_page_table_resample, args.warmup, args.iters)},
        {"op": "paged_kv_cache_page_table_resample", "target": "triton", "mode": "forward", "ms": cuda_time_ms(triton_page_table_resample, args.warmup, args.iters)},
        {"op": "paged_kv_cache_state_resample", "target": "ref", "mode": "forward", "ms": cuda_time_ms(ref_state_resample, args.warmup, args.iters)},
        {"op": "paged_kv_cache_state_resample", "target": "triton", "mode": "forward", "ms": cuda_time_ms(triton_state_resample, args.warmup, args.iters)},
    ]


def bench_cross_entropy(args: argparse.Namespace) -> list[dict]:
    logits = torch.randn(args.rows, args.vocab, device="cuda", requires_grad=True)
    labels = torch.randint(0, args.vocab, (args.rows,), device="cuda")
    labels[::17] = -100

    def ref_step():
        x = fresh_like(logits)
        ref_cross_entropy(x, labels).backward()

    def triton_step():
        x = fresh_like(logits)
        triton_cross_entropy(x, labels).backward()

    return [
        {"op": "cross_entropy", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "cross_entropy", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_unembedding_cross_entropy(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(args.vocab, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, args.vocab, (args.rows,), device="cuda")
    labels[::17] = -100

    def materialized_forward():
        logits = triton_linear_forward_2d(x, weight, None, out_dtype=torch.bfloat16).float()
        triton_cross_entropy(logits, labels)

    def fused_forward():
        triton_unembedding_cross_entropy(x, weight, labels)

    def materialized_step():
        x_, weight_ = fresh_like(x), fresh_like(weight)
        logits = triton_linear(x_, weight_, None).float()
        triton_cross_entropy(logits, labels).backward()

    def fused_step():
        x_, weight_ = fresh_like(x), fresh_like(weight)
        triton_unembedding_cross_entropy(x_, weight_, labels).backward()

    mat_forward_ms = cuda_time_ms(materialized_forward, args.warmup, args.iters)
    fused_forward_ms = cuda_time_ms(fused_forward, args.warmup, args.iters)
    mat_ms = cuda_time_ms(materialized_step, args.warmup, args.iters)
    fused_ms = cuda_time_ms(fused_step, args.warmup, args.iters)
    logits_bytes = args.rows * args.vocab * torch.tensor([], dtype=torch.bfloat16).element_size()
    return [
        {
            "op": "unembedding_cross_entropy",
            "target": "materialized_triton",
            "mode": "forward",
            "ms": mat_forward_ms,
            "rows": args.rows,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "materialized_logits_bytes": logits_bytes,
        },
        {
            "op": "unembedding_cross_entropy",
            "target": "fused_triton",
            "mode": "forward",
            "ms": fused_forward_ms,
            "speedup_vs_materialized": mat_forward_ms / fused_forward_ms,
            "rows": args.rows,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "avoided_logits_bytes": logits_bytes,
        },
        {
            "op": "unembedding_cross_entropy",
            "target": "materialized_triton",
            "mode": "forward_backward",
            "ms": mat_ms,
            "rows": args.rows,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "materialized_logits_bytes": logits_bytes,
        },
        {
            "op": "unembedding_cross_entropy",
            "target": "fused_triton",
            "mode": "forward_backward",
            "ms": fused_ms,
            "speedup_vs_materialized": mat_ms / fused_ms,
            "rows": args.rows,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "avoided_logits_bytes": logits_bytes,
        },
    ]


def bench_matmul_top1(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    weight = torch.randn(args.vocab, args.hidden, device="cuda", requires_grad=True)
    bias = torch.randn(args.vocab, device="cuda", requires_grad=True)

    def ref_forward():
        ref_matmul_top1(x, weight, bias)

    def triton_forward():
        triton_matmul_top1(x, weight, bias)

    def ref_backward():
        x_, w_, b_ = fresh_like(x), fresh_like(weight), fresh_like(bias)
        values, _ = ref_matmul_top1(x_, w_, b_)
        values.sum().backward()

    def triton_backward():
        x_, w_, b_ = fresh_like(x), fresh_like(weight), fresh_like(bias)
        values, _ = triton_matmul_top1(x_, w_, b_)
        values.sum().backward()

    return [
        {"op": "matmul_top1", "target": "ref", "mode": "forward", "ms": cuda_time_ms(ref_forward, args.warmup, args.iters)},
        {"op": "matmul_top1", "target": "triton", "mode": "forward", "ms": cuda_time_ms(triton_forward, args.warmup, args.iters)},
        {"op": "matmul_top1", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_backward, args.warmup, args.iters)},
        {"op": "matmul_top1", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_backward, args.warmup, args.iters)},
    ]


def bench_gelu_mlp(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    w1 = torch.randn(args.ffn, args.hidden, device="cuda", requires_grad=True)
    b1 = torch.randn(args.ffn, device="cuda", requires_grad=True)
    w2 = torch.randn(args.hidden, args.ffn, device="cuda", requires_grad=True)
    b2 = torch.randn(args.hidden, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, w1, b1, w2, b2)]
        ref_gelu_mlp(*xs).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, w1, b1, w2, b2)]
        triton_gelu_mlp(*xs).sum().backward()

    return [
        {"op": "gelu_mlp", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "gelu_mlp", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_linear(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    weight = torch.randn(args.linear_out, args.hidden, device="cuda", requires_grad=True)
    bias = torch.randn(args.linear_out, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        ref_linear(*xs).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        triton_linear(*xs).sum().backward()

    return [
        {"op": "linear", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "linear", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.detach().float() - expected.detach().float()).norm() / expected.detach().float().norm().clamp_min(1e-12))


def bench_mx_linear(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(args.linear_out, args.hidden, device="cuda", dtype=torch.bfloat16) / (args.hidden**0.5)).contiguous()
    bias = torch.zeros(args.linear_out, device="cuda", dtype=torch.bfloat16)
    fp8_weight, fp8_scales, fp8_in_features = pack_mxfp8_weight(weight)
    fp4_weight, fp4_scales, fp4_in_features = pack_mxfp4_weight(weight)
    nvfp4_weight, nvfp4_scales, nvfp4_in_features = pack_nvfp4_weight(weight)

    bf16_out = triton_linear_forward_2d(x, weight, bias)
    fp8_out = mxfp8_linear(x, fp8_weight, fp8_scales, fp8_in_features, bias)
    fp4_out = mxfp4_linear(x, fp4_weight, fp4_scales, fp4_in_features, bias)
    nvfp4_out = nvfp4_linear(x, nvfp4_weight, nvfp4_scales, nvfp4_in_features, bias)

    bf16_ms = cuda_time_ms(lambda: triton_linear_forward_2d(x, weight, bias), args.warmup, args.iters)
    fp8_ms = cuda_time_ms(lambda: mxfp8_linear(x, fp8_weight, fp8_scales, fp8_in_features, bias), args.warmup, args.iters)
    fp4_ms = cuda_time_ms(lambda: mxfp4_linear(x, fp4_weight, fp4_scales, fp4_in_features, bias), args.warmup, args.iters)
    nvfp4_ms = cuda_time_ms(lambda: nvfp4_linear(x, nvfp4_weight, nvfp4_scales, nvfp4_in_features, bias), args.warmup, args.iters)
    bf16_bytes = weight.numel() * weight.element_size()
    fp8_bytes = fp8_weight.numel() * fp8_weight.element_size() + fp8_scales.numel() * fp8_scales.element_size()
    fp4_bytes = fp4_weight.numel() * fp4_weight.element_size() + fp4_scales.numel() * fp4_scales.element_size()
    nvfp4_bytes = nvfp4_weight.numel() * nvfp4_weight.element_size() + nvfp4_scales.numel() * nvfp4_scales.element_size()

    return [
        {
            "op": "mx_linear",
            "target": "bf16",
            "format": "bf16",
            "mode": "forward",
            "ms": bf16_ms,
            "rows": args.rows,
            "in_features": args.hidden,
            "out_features": args.linear_out,
            "weight_storage_bytes": bf16_bytes,
        },
        {
            "op": "mx_linear",
            "target": "mxfp8",
            "format": "mxfp8_e4m3_e8m0_block32",
            "mode": "forward",
            "ms": fp8_ms,
            "speedup_vs_bf16": bf16_ms / fp8_ms,
            "rel_l2_vs_bf16_output": _rel_l2(fp8_out, bf16_out),
            "weight_storage_bytes": fp8_bytes,
            "compression_vs_bf16": bf16_bytes / fp8_bytes,
        },
        {
            "op": "mx_linear",
            "target": "mxfp4",
            "format": "mxfp4_e2m1_e8m0_block32",
            "mode": "forward",
            "ms": fp4_ms,
            "speedup_vs_bf16": bf16_ms / fp4_ms,
            "rel_l2_vs_bf16_output": _rel_l2(fp4_out, bf16_out),
            "weight_storage_bytes": fp4_bytes,
            "compression_vs_bf16": bf16_bytes / fp4_bytes,
        },
        {
            "op": "mx_linear",
            "target": "nvfp4",
            "format": "nvfp4_e2m1_e4m3_block16",
            "mode": "forward",
            "ms": nvfp4_ms,
            "speedup_vs_bf16": bf16_ms / nvfp4_ms,
            "rel_l2_vs_bf16_output": _rel_l2(nvfp4_out, bf16_out),
            "weight_storage_bytes": nvfp4_bytes,
            "compression_vs_bf16": bf16_bytes / nvfp4_bytes,
        },
    ]


def bench_conv1d_gelu(args: argparse.Namespace) -> list[dict]:
    input_frames = args.input_frames or args.query_len * 2
    conv1_x = torch.randn(args.batch, args.mel_bins, input_frames, device="cuda", requires_grad=True)
    conv1_weight = torch.randn(args.hidden, args.mel_bins, 3, device="cuda", requires_grad=True)
    conv1_bias = torch.randn(args.hidden, device="cuda", requires_grad=True)
    conv2_x = torch.randn(args.batch, args.hidden, input_frames, device="cuda", requires_grad=True)
    conv2_weight = torch.randn(args.hidden, args.hidden, 3, device="cuda", requires_grad=True)
    conv2_bias = torch.randn(args.hidden, device="cuda", requires_grad=True)

    def run(target: Callable, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], stride: int, padding: int):
        xs = [fresh_like(t) for t in tensors]
        target(*xs, stride, padding).sum().backward()

    rows = []
    for name, tensors, stride, padding in (
        ("conv1d_gelu.conv1", (conv1_x, conv1_weight, conv1_bias), 1, 1),
        ("conv1d_gelu.conv2", (conv2_x, conv2_weight, conv2_bias), 2, 1),
    ):
        rows.extend(
            [
                {"op": name, "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(lambda: run(ref_conv1d_gelu, tensors, stride, padding), args.warmup, args.iters)},
                {"op": name, "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(lambda: run(triton_conv1d_gelu, tensors, stride, padding), args.warmup, args.iters)},
            ]
        )
    return rows


def bench_layer_norm(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    weight = torch.randn(args.hidden, device="cuda", requires_grad=True)
    bias = torch.randn(args.hidden, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        ref_layer_norm(*xs, args.layer_norm_eps).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        triton_layer_norm(*xs, args.layer_norm_eps).sum().backward()

    return [
        {"op": "layer_norm", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "layer_norm", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_qkv_proj(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.batch, args.query_len, args.hidden, device="cuda", requires_grad=True)
    weight = torch.randn(args.hidden, args.hidden, device="cuda", requires_grad=True)
    bias = torch.randn(args.hidden, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        ref_qkv_proj(*xs, args.heads).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        triton_qkv_proj(*xs, args.heads).sum().backward()

    return [
        {"op": "qkv_proj", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "qkv_proj", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_c_proj(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.batch, args.heads, args.query_len, args.head_dim, device="cuda", requires_grad=True)
    weight = torch.randn(args.hidden, args.hidden, device="cuda", requires_grad=True)
    bias = torch.randn(args.hidden, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        ref_c_proj(*xs).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, weight, bias)]
        triton_c_proj(*xs).sum().backward()

    return [
        {"op": "c_proj", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "c_proj", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def bench_embedding(args: argparse.Namespace) -> list[dict]:
    input_ids = torch.randint(0, args.vocab, (args.batch, args.target_len), device="cuda")
    token_weight = torch.randn(args.vocab, args.hidden, device="cuda", requires_grad=True)
    position_weight = torch.randn(args.max_target_positions, args.hidden, device="cuda", requires_grad=True)
    hidden = torch.randn(args.batch, args.query_len, args.hidden, device="cuda", requires_grad=True)
    encoder_position_weight = torch.randn(args.query_len, args.hidden, device="cuda", requires_grad=True)

    def ref_decoder_step():
        token, position = fresh_like(token_weight), fresh_like(position_weight)
        ref_decoder_embedding(input_ids, token, position, torch.float32).sum().backward()

    def triton_decoder_step():
        token, position = fresh_like(token_weight), fresh_like(position_weight)
        triton_decoder_embedding(input_ids, token, position, torch.float32).sum().backward()

    def ref_encoder_step():
        x, position = fresh_like(hidden), fresh_like(encoder_position_weight)
        ref_encoder_position_embedding(x, position).sum().backward()

    def triton_encoder_step():
        x, position = fresh_like(hidden), fresh_like(encoder_position_weight)
        triton_encoder_position_embedding(x, position).sum().backward()

    return [
        {"op": "decoder_embedding", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_decoder_step, args.warmup, args.iters)},
        {"op": "decoder_embedding", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_decoder_step, args.warmup, args.iters)},
        {"op": "encoder_position_embedding", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_encoder_step, args.warmup, args.iters)},
        {"op": "encoder_position_embedding", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_encoder_step, args.warmup, args.iters)},
    ]


def bench_residual_add(args: argparse.Namespace) -> list[dict]:
    residual = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    hidden_states = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (residual, hidden_states)]
        ref_residual_add(*xs).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (residual, hidden_states)]
        triton_residual_add(*xs).sum().backward()

    return [
        {"op": "residual_add", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "residual_add", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PLU PyTorch reference ops against Triton targets.")
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "flash_attention",
            "paged_kv_cache",
            "cross_entropy",
            "unembedding_cross_entropy",
            "matmul_top1",
            "gelu_mlp",
            "linear",
            "mx_linear",
            "conv1d_gelu",
            "layer_norm",
            "qkv_proj",
            "c_proj",
            "embedding",
            "residual_add",
        ],
        default="all",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--query-len", type=int, default=1500)
    parser.add_argument("--key-len", type=int, default=1500)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--rows", type=int, default=None, help="Flattened token rows for linear ops. Defaults to batch * query_len.")
    parser.add_argument("--hidden", type=int, default=1280)
    parser.add_argument("--ffn", type=int, default=5120)
    parser.add_argument("--vocab", type=int, default=51865)
    parser.add_argument("--mel-bins", type=int, default=128)
    parser.add_argument("--input-frames", type=int, default=None, help="Input frame count for convolution benchmarks. Defaults to query_len * 2.")
    parser.add_argument("--target-len", type=int, default=448)
    parser.add_argument("--max-target-positions", type=int, default=448)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--linear-out", type=int, default=1280)
    parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 for PyTorch reference matmuls to match large Triton tensor-core paths.")
    args = parser.parse_args()
    if args.rows is None:
        args.rows = args.batch * args.query_len
    return args


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton benchmarks")
    args = parse_args()
    set_tf32(args.tf32)
    torch.manual_seed(0)
    benches = {
        "flash_attention": bench_flash_attention,
        "paged_kv_cache": bench_paged_kv_cache,
        "cross_entropy": bench_cross_entropy,
        "unembedding_cross_entropy": bench_unembedding_cross_entropy,
        "matmul_top1": bench_matmul_top1,
        "gelu_mlp": bench_gelu_mlp,
        "linear": bench_linear,
        "mx_linear": bench_mx_linear,
        "conv1d_gelu": bench_conv1d_gelu,
        "layer_norm": bench_layer_norm,
        "qkv_proj": bench_qkv_proj,
        "c_proj": bench_c_proj,
        "embedding": bench_embedding,
        "residual_add": bench_residual_add,
    }
    selected = benches if args.op == "all" else {args.op: benches[args.op]}
    for bench in selected.values():
        for row in bench(args):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
