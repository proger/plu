from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_conv1d_gelu


def benchmark(args: Namespace):
    return bench_conv1d_gelu(args)
