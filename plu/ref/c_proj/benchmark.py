from __future__ import annotations

from argparse import Namespace

from plu.benchmarks.bench_triton_ops import bench_c_proj


def benchmark(args: Namespace):
    return bench_c_proj(args)
