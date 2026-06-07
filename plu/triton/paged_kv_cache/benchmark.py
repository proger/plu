from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_paged_kv_cache


def benchmark(args: Namespace):
    return bench_paged_kv_cache(args)
