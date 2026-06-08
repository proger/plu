#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential and batched UK1E2 decode speed benchmarks.")
    parser.add_argument("--wait-pid", type=int, default=None, help="Optional PID to wait for before starting GPU benchmarks.")
    parser.add_argument("--wait-poll-seconds", type=float, default=60.0)
    parser.add_argument("--wav-scp", type=Path, default=Path("egs/uk1e2/exp/bench_1pct/wav.parallel_1pct.scp"))
    parser.add_argument("--out-root", type=Path, default=Path("egs/uk1e2/exp/bench_1pct"))
    parser.add_argument(
        "--exp",
        type=Path,
        default=Path("egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8/train"),
    )
    parser.add_argument("--language", default="uk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--decode-batch-size", type=int, default=8, help="Alternatives per decoded window.")
    parser.add_argument("--window-batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--batched-audio-load-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--run-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--skip-cuda-precheck", action="store_true")
    return parser.parse_args()


def pid_running(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def wait_for_pid(pid: int, poll_seconds: float, log_path: Path) -> None:
    while pid_running(pid):
        with log_path.open("a", encoding="utf-8") as f:
            print(json.dumps({"event": "waiting", "pid": pid, "time": time.time()}), file=f, flush=True)
        time.sleep(poll_seconds)


def decode_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--wav-scp",
        args.wav_scp.as_posix(),
        "--exp",
        args.exp.as_posix(),
        "--language",
        args.language,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--window-seconds",
        "30",
        "--hop-seconds",
        "15",
        "--decode-batch-size",
        str(args.decode_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timestamps-after-sentence-end",
        "--sentence-hop",
        "--sentence-hop-mode",
        "last",
        "--condition-on-previous-utterance",
    ]


def run_decode(name: str, cmd: list[str], out_dir: Path, timeout_seconds: float) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    start = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_f:
        print("+ " + " ".join(cmd), file=log_f, flush=True)
        try:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, check=False, timeout=timeout_seconds)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -124
            print(json.dumps({"event": "timeout", "timeout_seconds": timeout_seconds}), file=log_f, flush=True)
    elapsed = time.perf_counter() - start
    summary_path = out_dir / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = validate_decode_jsonl(out_dir / "decode.jsonl")
    return {
        "name": name,
        "out_dir": out_dir.as_posix(),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds_driver": elapsed,
        "summary": summary,
        "validation": validation,
    }


def run_precheck(run_root: Path, timeout_seconds: float) -> dict[str, object]:
    log_path = run_root / "cuda_precheck.log"
    cmd = [sys.executable, "-m", "pytest", "-q", "plu/triton/decode_attention/test_decode_attention.py"]
    start = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_f:
        print("+ " + " ".join(cmd), file=log_f, flush=True)
        try:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, check=False, timeout=timeout_seconds)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -124
            print(json.dumps({"event": "timeout", "timeout_seconds": timeout_seconds}), file=log_f, flush=True)
    return {
        "name": "cuda_precheck",
        "cmd": cmd,
        "log": log_path.as_posix(),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds_driver": time.perf_counter() - start,
    }


def validate_decode_jsonl(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    rows = 0
    selected_rows = 0
    counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    sample_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            utt_id = str(row.get("utt_id"))
            counts[utt_id] = counts.get(utt_id, 0) + 1
            for key in row:
                if key.startswith("sample_"):
                    sample_keys.add(key)
            if row.get("sample_selected"):
                selected_rows += 1
                selected_counts[utt_id] = selected_counts.get(utt_id, 0) + 1
    per_window_counts = sorted(set(counts.values()))
    selected_per_window_counts = sorted(set(selected_counts.get(utt_id, 0) for utt_id in counts))
    return {
        "exists": True,
        "rows": rows,
        "windows": len(counts),
        "selected_rows": selected_rows,
        "per_window_counts": per_window_counts,
        "selected_per_window_counts": selected_per_window_counts,
        "sample_keys": sorted(sample_keys),
    }


def speedup(base: dict[str, object], candidate: dict[str, object], key: str) -> float | None:
    base_value = (base.get("summary") or {}).get(key)
    candidate_value = (candidate.get("summary") or {}).get(key)
    if not isinstance(base_value, int | float) or not isinstance(candidate_value, int | float) or candidate_value <= 0:
        return None
    return float(base_value) / float(candidate_value)


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.out_root / f"speed_bench_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    controller_log = run_root / "controller.log"

    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid, args.wait_poll_seconds, controller_log)

    cuda_precheck = None if args.skip_cuda_precheck else run_precheck(run_root, min(args.run_timeout_seconds, 600.0))
    common = decode_flags(args)
    baseline_out = run_root / "baseline_seq"
    baseline_cmd = [
        sys.executable,
        "egs/uk1e2/local/decode_windows.py",
        *common,
        "--out-dir",
        baseline_out.as_posix(),
        "--no-resume",
    ]
    results = [run_decode("baseline_seq", baseline_cmd, baseline_out, args.run_timeout_seconds)]

    for window_batch_size in args.window_batch_sizes:
        out_dir = run_root / f"batched_wb{window_batch_size}"
        cmd = [
            sys.executable,
            "egs/uk1e2/local/decode_windows_batched.py",
            *common,
            "--out-dir",
            out_dir.as_posix(),
            "--window-batch-size",
            str(window_batch_size),
            "--audio-load-workers",
            str(args.batched_audio_load_workers),
        ]
        results.append(run_decode(f"batched_wb{window_batch_size}", cmd, out_dir, args.run_timeout_seconds))

    baseline = results[0]
    comparisons = []
    for result in results[1:]:
        comparisons.append(
            {
                "name": result["name"],
                "returncode": result["returncode"],
                "elapsed_speedup": speedup(baseline, result, "elapsed_seconds"),
                "decode_seconds_speedup": speedup(baseline, result, "total_decode_seconds"),
                "model_seconds_speedup": speedup(baseline, result, "total_model_seconds"),
                "wall_rtf_speedup": speedup(baseline, result, "wall_decode_rtf"),
                "wall_model_rtf_speedup": speedup(baseline, result, "wall_model_rtf"),
                "elapsed_rtf_speedup": speedup(baseline, result, "elapsed_rtf"),
                "effective_rtf_speedup": speedup(baseline, result, "effective_decode_rtf"),
                "effective_model_rtf_speedup": speedup(baseline, result, "effective_model_rtf"),
                "tokens_per_second_ratio": speedup(result, baseline, "tokens_per_second"),
            }
        )
    successful = [comparison for comparison in comparisons if comparison["returncode"] == 0]
    best_by_elapsed = max(
        successful,
        key=lambda comparison: float(comparison["elapsed_speedup"] or 0.0),
        default=None,
    )

    report = {
        "run_root": run_root.as_posix(),
        "wav_scp": args.wav_scp.as_posix(),
        "decode_batch_size": args.decode_batch_size,
        "window_batch_sizes": args.window_batch_sizes,
        "batched_audio_load_workers": args.batched_audio_load_workers,
        "run_timeout_seconds": args.run_timeout_seconds,
        "cuda_precheck": cuda_precheck,
        "results": results,
        "comparisons": comparisons,
        "best_by_elapsed": best_by_elapsed,
        "target_speedup": 3.0,
        "target_met": bool(best_by_elapsed and float(best_by_elapsed.get("elapsed_speedup") or 0.0) >= 3.0),
    }
    (run_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
