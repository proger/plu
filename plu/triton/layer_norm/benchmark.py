from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_layer_norm


def benchmark(args: Namespace):
    return bench_layer_norm(args)
