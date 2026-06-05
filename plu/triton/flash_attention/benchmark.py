from __future__ import annotations

from argparse import Namespace

from benchmarks.bench_triton_ops import bench_flash_attention


def benchmark(args: Namespace):
    return bench_flash_attention(args)
