from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from plu.train_data import HOP_LENGTH, SAMPLE_RATE


DEFAULT_FEATURE_FPS = SAMPLE_RATE / HOP_LENGTH


def set_tf32(enabled: bool) -> None:
    precision = "tf32" if enabled else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = enabled
        try:
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        except AttributeError:
            pass


def parse_dtype(name: str) -> torch.dtype:
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def encoder_tokens_after_conv(input_feature_frames: int) -> int:
    return (input_feature_frames + 1) // 2


def audio_seconds(input_feature_frames: int, feature_fps: float) -> float:
    return input_feature_frames / feature_fps


def extend_encoder_positions(model: torch.nn.Module, positions: int) -> bool:
    from plu.whisper import sinusoids

    encoder = model.model.encoder
    if encoder.config.max_source_positions >= positions and encoder.embed_positions.weight.shape[0] >= positions:
        return False

    encoder.config.max_source_positions = positions
    old_weight = encoder.embed_positions.weight
    embedding = torch.nn.Embedding(
        positions,
        model.config.d_model,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    embedding.weight.requires_grad = False
    with torch.no_grad():
        embedding.weight.copy_(sinusoids(positions, model.config.d_model).to(device=old_weight.device, dtype=old_weight.dtype))
    encoder.embed_positions = embedding
    return True


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PLU_OPS_BACKEND"] = args.child_backend
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for end-to-end benchmarks")
    if args.cuda_graph and not args.device.startswith("cuda"):
        raise RuntimeError("CUDA graph capture requires a CUDA device")

    set_tf32(args.tf32)
    torch.manual_seed(args.seed)

    from plu.whisper import WhisperForConditionalGeneration

    dtype = parse_dtype(args.dtype)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device=args.device, dtype=dtype)
    model.train()

    encoder_tokens = encoder_tokens_after_conv(args.input_features)
    extended_positions = False
    if encoder_tokens > model.config.max_source_positions:
        if not args.extend_encoder_positions:
            raise ValueError(
                f"input_features={args.input_features} becomes {encoder_tokens} encoder tokens, "
                f"but model supports only {model.config.max_source_positions}"
            )
        extended_positions = extend_encoder_positions(model, encoder_tokens)

    input_features = torch.randn(
        args.batch,
        model.config.num_mel_bins,
        args.input_features,
        device=args.device,
        dtype=dtype,
    )
    labels = torch.randint(model.config.vocab_size, (args.batch, args.decoder_len), device=args.device)

    def step() -> tuple[torch.Tensor, torch.Size]:
        model.zero_grad(set_to_none=True)
        output = model(input_features, labels=labels)
        if output.loss is None:
            raise RuntimeError("expected loss for end-to-end benchmark")
        output.loss.backward()
        return output.loss.detach(), output.logits.shape

    times_ms: list[float] = []
    loss_value = 0.0
    logits_shape: torch.Size | None = None
    cuda_graph_capture_ms: float | None = None

    for _ in range(args.warmup):
        loss, logits_shape = step()
        loss_value = float(loss.cpu())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if args.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        capture_start = time.perf_counter()
        with torch.cuda.graph(graph):
            static_loss, logits_shape = step()
        torch.cuda.synchronize()
        cuda_graph_capture_ms = (time.perf_counter() - capture_start) * 1000.0

        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        loss_value = float(static_loss.detach().cpu())
    else:
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss, logits_shape = step()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
            loss_value = float(loss.cpu())

    mean_ms = sum(times_ms) / len(times_ms)
    audio_secs = audio_seconds(args.input_features, args.feature_fps)
    mean_seconds = mean_ms / 1000.0
    return {
        "op": "end_to_end",
        "target": args.child_backend,
        "mode": "forward_backward_cuda_graph" if args.cuda_graph else "forward_backward",
        "cuda_graph": args.cuda_graph,
        "model": args.model,
        "dtype": str(dtype).replace("torch.", ""),
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "batch": args.batch,
        "input_features": args.input_features,
        "feature_fps": args.feature_fps,
        "audio_seconds": audio_secs,
        "encoder_tokens_after_conv": encoder_tokens,
        "decoder_tokens": args.decoder_len,
        "input_features_shape": list(input_features.shape),
        "labels_shape": list(labels.shape),
        "logits_shape": list(logits_shape) if logits_shape is not None else None,
        "extended_encoder_positions": extended_positions,
        "tf32": args.tf32,
        "warmup": args.warmup,
        "iters": args.iters,
        "times_ms": times_ms,
        "mean_ms": mean_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "cuda_graph_capture_ms": cuda_graph_capture_ms,
        "x_real_time": audio_secs / mean_seconds,
        "real_time_factor": mean_seconds / audio_secs,
        "loss": loss_value,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def child_argv(args: argparse.Namespace, backend: str) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-backend",
        backend,
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch",
        str(args.batch),
        "--input-features",
        str(args.input_features),
        "--feature-fps",
        str(args.feature_fps),
        "--decoder-len",
        str(args.decoder_len),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
    ]
    argv.append("--tf32" if args.tf32 else "--no-tf32")
    argv.append("--extend-encoder-positions" if args.extend_encoder_positions else "--no-extend-encoder-positions")
    argv.append("--cuda-graph" if args.cuda_graph else "--no-cuda-graph")
    return argv


def run_parent(args: argparse.Namespace) -> None:
    backends = ["ref", "triton"] if args.backend == "all" else [args.backend]
    rows: list[dict[str, Any]] = []
    for backend in backends:
        env = os.environ.copy()
        env["PLU_OPS_BACKEND"] = backend
        completed = subprocess.run(child_argv(args, backend), env=env, text=True, capture_output=True)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            raise SystemExit(completed.returncode)

        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            print(json.dumps(row), flush=True)

    by_target = {row["target"]: row for row in rows}
    if "ref" in by_target and "triton" in by_target:
        ref_ms = by_target["ref"]["mean_ms"]
        triton_ms = by_target["triton"]["mean_ms"]
        summary = {
            "op": "end_to_end",
            "target": "summary",
            "mode": "forward_backward_cuda_graph" if args.cuda_graph else "forward_backward",
            "cuda_graph": args.cuda_graph,
            "model": args.model,
            "dtype": by_target["triton"]["dtype"],
            "batch": args.batch,
            "input_features": args.input_features,
            "feature_fps": by_target["triton"]["feature_fps"],
            "audio_seconds": by_target["triton"]["audio_seconds"],
            "encoder_tokens_after_conv": by_target["triton"]["encoder_tokens_after_conv"],
            "decoder_tokens": args.decoder_len,
            "ref_mean_ms": ref_ms,
            "triton_mean_ms": triton_ms,
            "speedup": ref_ms / triton_ms,
            "wall_time_reduction_pct": (1.0 - triton_ms / ref_ms) * 100.0,
            "ref_x_real_time": by_target["ref"]["x_real_time"],
            "triton_x_real_time": by_target["triton"]["x_real_time"],
            "ref_real_time_factor": by_target["ref"]["real_time_factor"],
            "triton_real_time_factor": by_target["triton"]["real_time_factor"],
            "real_time_speedup": by_target["triton"]["x_real_time"] / by_target["ref"]["x_real_time"],
            "ref_peak_allocated_gib": by_target["ref"]["peak_allocated_gib"],
            "triton_peak_allocated_gib": by_target["triton"]["peak_allocated_gib"],
        }
        print(json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full PLU Whisper forward+backward passes.")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo", help="Local model path or Hugging Face repo id.")
    parser.add_argument("--backend", choices=["all", "ref", "triton"], default="all")
    parser.add_argument("--child-backend", choices=["ref", "triton"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "fp32", "bfloat16", "bf16"], default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-features", type=int, default=3000, help="Input feature frames before Whisper conv downsampling.")
    parser.add_argument("--feature-fps", type=float, default=DEFAULT_FEATURE_FPS, help="Input feature frames per second of audio.")
    parser.add_argument("--decoder-len", type=int, default=448)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extend-encoder-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.feature_fps <= 0:
        raise ValueError("--feature-fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.child_backend is not None:
        print(json.dumps(run_child(args)), flush=True)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
