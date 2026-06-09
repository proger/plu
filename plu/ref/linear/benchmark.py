from __future__ import annotations

from argparse import Namespace

from plu.benchmarks.bench_triton_ops import bench_linear


def benchmark(args: Namespace):
    return bench_linear(args)
