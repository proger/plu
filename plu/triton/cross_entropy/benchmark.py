from __future__ import annotations

from argparse import Namespace

from plu.benchmarks.bench_triton_ops import bench_cross_entropy


def benchmark(args: Namespace):
    return bench_cross_entropy(args)
