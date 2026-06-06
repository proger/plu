from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_gelu_mlp


def benchmark(args: Namespace):
    return bench_gelu_mlp(args)
