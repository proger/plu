from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.ref.flash_attention import flash_attention as ref_flash_attention
from plu.ref.gelu_mlp import gelu_mlp as ref_gelu_mlp
from plu.ref.lora import lora_linear as ref_lora_linear
from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
from plu.triton.cross_entropy import cross_entropy as triton_cross_entropy
from plu.triton.flash_attention import flash_attention as triton_flash_attention
from plu.triton.gelu_mlp import gelu_mlp as triton_gelu_mlp
from plu.triton.lora import lora_linear as triton_lora_linear
from plu.triton.matmul_top1 import matmul_top1 as triton_matmul_top1


def set_tf32(enabled: bool) -> None:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = "tf32" if enabled else "ieee"
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
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


def bench_lora(args: argparse.Namespace) -> list[dict]:
    x = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    adapter_input = torch.randn(args.rows, args.hidden, device="cuda", requires_grad=True)
    base_weight = torch.randn(args.hidden, args.hidden, device="cuda", requires_grad=True)
    base_bias = torch.randn(args.hidden, device="cuda", requires_grad=True)
    lora_a = torch.randn(args.rank, args.hidden, device="cuda", requires_grad=True)
    lora_b = torch.randn(args.hidden, args.rank, device="cuda", requires_grad=True)

    def ref_step():
        xs = [fresh_like(t) for t in (x, adapter_input, base_weight, base_bias, lora_a, lora_b)]
        ref_lora_linear(*xs, args.lora_scaling).sum().backward()

    def triton_step():
        xs = [fresh_like(t) for t in (x, adapter_input, base_weight, base_bias, lora_a, lora_b)]
        triton_lora_linear(*xs, args.lora_scaling).sum().backward()

    return [
        {"op": "lora", "target": "ref", "mode": "forward_backward", "ms": cuda_time_ms(ref_step, args.warmup, args.iters)},
        {"op": "lora", "target": "triton", "mode": "forward_backward", "ms": cuda_time_ms(triton_step, args.warmup, args.iters)},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PLU PyTorch reference ops against Triton targets.")
    parser.add_argument("--op", choices=["all", "flash_attention", "cross_entropy", "matmul_top1", "gelu_mlp", "lora"], default="all")
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
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-scaling", type=float, default=4.0)
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
        "cross_entropy": bench_cross_entropy,
        "matmul_top1": bench_matmul_top1,
        "gelu_mlp": bench_gelu_mlp,
        "lora": bench_lora,
    }
    selected = benches if args.op == "all" else {args.op: benches[args.op]}
    for bench in selected.values():
        for row in bench(args):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
