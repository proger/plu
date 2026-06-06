from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_matmul_top1


def benchmark(args: Namespace):
    return bench_matmul_top1(args)
