#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    learning_rate: float
    beta1: float
    beta2: float
    clip_grad_norm: float | None = None
    frozen_encoder_layers: int | None = None


FAST_FAIL_CONFIGS = [
    Config(lr, beta1, beta2)
    for beta1, beta2 in ((0.0, 0.0), (0.0, 0.9999), (0.9, 0.999))
    for lr in (1.0e-5, 3.0e-6, 1.0e-6, 1.0e-7)
]
STABLE_CONFIGS = [
    Config(lr, beta1, beta2)
    for beta1, beta2 in ((0.0, 0.9999), (0.5, 0.999), (0.9, 0.999), (0.9, 0.9999))
    for lr in (1.0e-6, 3.0e-7, 1.0e-7)
]
FOCUSED_CONFIGS = [
    Config(lr, beta1, beta2)
    for beta1, beta2 in ((0.0, 0.9999), (0.5, 0.999), (0.9, 0.999), (0.9, 0.9999))
    for lr in (2.0e-6, 1.0e-6, 7.0e-7)
]
DEFAULT_CONFIGS = STABLE_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an ATC one-epoch LR/beta sweep.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-jsonl", type=Path, default=Path("egs/atc/data/train.jsonl"))
    parser.add_argument("--validation-jsonl", type=Path, default=Path("egs/atc/data/validation.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path("egs/atc/exp"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Read trained run directories from this sweep root. Intended for --eval-only cross-evaluation.",
    )
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--eval-name", default="validation", help="Name used for decode directories and non-default result artifacts.")
    parser.add_argument("--init", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--baseline-model", default="large-v3-turbo")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--grid", default="stable", choices=["stable", "focused", "fast-fail"], help="Hyperparameter grid to run.")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="Override --grid with these learning rates crossed with --beta1-values and --beta2-values.",
    )
    parser.add_argument("--beta1-values", type=float, nargs="+", default=None, help="Beta1 values for --learning-rates.")
    parser.add_argument("--beta2-values", type=float, nargs="+", default=None, help="Beta2 values for --learning-rates.")
    parser.add_argument("--clip-grad-norm-values", type=float, nargs="+", default=None, help="Clip grad norm values for --learning-rates.")
    parser.add_argument(
        "--frozen-encoder-layers-values",
        type=int,
        nargs="+",
        default=None,
        help="Frozen encoder prefix layer counts for --learning-rates.",
    )
    parser.add_argument("--lr-scheduler-type", default="constant", choices=["constant", "linear"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-bin", default=None)
    parser.add_argument("--test-bin", default=None)
    parser.add_argument("--test-retries", type=int, default=0, help="Retry failed decode subprocesses this many times.")
    parser.add_argument("--test-retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--expected-runs", type=int, default=12)
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path, root)


def command_path(name: str, override: str | None) -> str:
    if override:
        return override
    local = Path("/home/vol/.local/bin") / name
    return str(local) if local.exists() else name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    chars = [char if char.isalnum() or char == "'" else " " for char in text]
    return " ".join("".join(chars).split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            cost = previous[j - 1] if ref_word == hyp_word else previous[j - 1] + 1
            current.append(min(cost, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1]


def path_key(path: str, root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())


def score_rows(refs: list[dict[str, Any]], hyps: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row in hyps:
        by_path[path_key(str(row["path"]), root)].append((int(row.get("i", 0)), row))

    total_words = 0
    total_edits = 0
    missing = 0
    details = []
    for ref_row in refs:
        key = path_key(str(ref_row["path"]), root)
        choices = sorted(by_path.get(key, []), key=lambda item: item[0])
        if not choices:
            missing += 1
        hyp_text = " ".join(str(row.get("text", "")) for _, row in choices)
        ref = normalize(str(ref_row["text"]))
        hyp = normalize(hyp_text)
        ref_words = ref.split()
        hyp_words = hyp.split()
        edits = edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_edits += edits
        details.append(
            {
                "id": ref_row.get("id"),
                "path": ref_row["path"],
                "ref": ref,
                "hyp": hyp,
                "edits": edits,
                "words": len(ref_words),
            }
        )
    wer = total_edits / total_words if total_words else 0.0
    return {
        "wer": wer,
        "wer_percent": 100.0 * wer,
        "total_edits": total_edits,
        "total_words": total_words,
        "references": len(refs),
        "hypotheses": len(by_path),
        "missing_hypotheses": missing,
        "normalization": "NFKC + casefold + keep unicode alnum/apostrophe",
        "details": details,
    }


def select_i(rows: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("i", 0)) == index]


def select_best_avg_logprob(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = path_key(str(row["path"]), root)
        current = best.get(key)
        if current is None or float(row.get("avg_logprob", -1.0e9)) > float(current.get("avg_logprob", -1.0e9)):
            best[key] = row
    return [best[key] for key in sorted(best)]


def has_model_files(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any((path / filename).exists() for filename in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"))


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in run_dir.glob("step_*/training_state.pt"):
        try:
            step = int(path.parent.name.removeprefix("step_"))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates)[1]


def fmt_float(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.4g}"
    text = text.replace("e-0", "e-").replace("e+0", "e")
    return text.replace(".", "p")


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned or "eval"


def eval_artifact(out_root: Path, filename: str, eval_name: str) -> Path:
    if eval_name == "validation":
        return out_root / filename
    path = Path(filename)
    return out_root / f"{path.stem}_{safe_name(eval_name)}{path.suffix}"


def run_name(config: Config) -> str:
    parts = [f"lr{fmt_float(config.learning_rate)}", f"b1{fmt_float(config.beta1)}", f"b2{fmt_float(config.beta2)}"]
    if config.clip_grad_norm is not None:
        parts.append(f"clip{fmt_float(config.clip_grad_norm)}")
    if config.frozen_encoder_layers is not None:
        parts.append(f"frz{config.frozen_encoder_layers}")
    return "_".join(parts)


def build_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(root)
    return env


def run_subprocess(command: list[str], *, root: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=root, env=build_env(root), stdout=stdout, stderr=stderr, check=True)


def train_run(args: argparse.Namespace, root: Path, run_dir: Path, config: Config, train_bin: str) -> None:
    if has_model_files(run_dir):
        return
    init = latest_checkpoint(run_dir)
    init_arg = rel(init, root) if init is not None else args.init
    command = [
        *shlex.split(train_bin),
        "--train",
        rel(root / args.train_jsonl, root),
        "--init",
        init_arg,
        "--exp",
        rel(run_dir, root),
        "--learning_rate",
        str(config.learning_rate),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--weight_decay",
        str(args.weight_decay),
        "--beta1",
        str(config.beta1),
        "--beta2",
        str(config.beta2),
    ]
    if config.clip_grad_norm is not None:
        command.extend(["--clip_grad_norm", str(config.clip_grad_norm)])
    if config.frozen_encoder_layers is not None:
        command.extend(["--frozen_encoder_layers", str(config.frozen_encoder_layers)])
    (run_dir / "train.command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    run_subprocess(command, root=root, stdout_path=run_dir / "train.stdout.log", stderr_path=run_dir / "train.stderr.log")


def run_test(
    *,
    args: argparse.Namespace,
    root: Path,
    out_dir: Path,
    test_bin: str,
    model_arg: tuple[str, str],
    wav_paths: list[str],
) -> Path:
    out_jsonl = out_dir / "decode.jsonl"
    if out_jsonl.exists() and out_jsonl.stat().st_size > 0:
        rows = read_jsonl(out_jsonl)
        expected_rows = len(wav_paths) * args.decode_batch_size
        if len(rows) == expected_rows:
            return out_jsonl
        incomplete = out_dir / f"decode.incomplete.{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        out_jsonl.rename(incomplete)
    command = [
        *shlex.split(test_bin),
        model_arg[0],
        model_arg[1],
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--language",
        args.language,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--decode_batch_size",
        str(args.decode_batch_size),
        *wav_paths,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decode.command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    stderr_path = out_dir / "decode.stderr.log"
    for attempt in range(args.test_retries + 1):
        with out_jsonl.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, cwd=root, env=build_env(root), stdout=stdout, stderr=stderr, check=False)
        if proc.returncode == 0:
            break
        if attempt >= args.test_retries:
            raise subprocess.CalledProcessError(proc.returncode, command)
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(
                f"\nrun_lr_beta_sweep.py: decode failed with return code {proc.returncode}; "
                f"retrying attempt {attempt + 2}/{args.test_retries + 1} "
                f"after {args.test_retry_sleep_seconds:g}s\n"
            )
        time.sleep(args.test_retry_sleep_seconds)
    return out_jsonl


def score_decode(refs: list[dict[str, Any]], decode_jsonl: Path, out_dir: Path, root: Path, decode_batch_size: int) -> dict[str, float]:
    rows = read_jsonl(decode_jsonl)
    scores: dict[str, float] = {}
    for index in range(decode_batch_size):
        selected = select_i(rows, index)
        write_jsonl(out_dir / f"decode_i{index}.jsonl", selected)
        result = score_rows(refs, selected, root)
        (out_dir / f"wer_i{index}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        scores[f"i{index}"] = float(result["wer_percent"])
    best = select_best_avg_logprob(rows, root)
    write_jsonl(out_dir / "decode_best_avg_logprob.jsonl", best)
    best_result = score_rows(refs, best, root)
    (out_dir / "wer_best_avg_logprob.json").write_text(json.dumps(best_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scores["best_avg_logprob"] = float(best_result["wer_percent"])
    return scores


def read_losses(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        return [], []
    steps: list[int] = []
    losses: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "train/loss" not in row or row.get("train/update_skipped"):
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["train/loss"]))
    return steps, losses


def rolling_mean(values: list[float], width: int = 100) -> list[float]:
    if not values:
        return []
    result = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= width:
            running -= values[index - width]
        result.append(running / min(index + 1, width))
    return result


def write_summary(summary_path: Path, baseline: dict[str, float] | None, rows: list[dict[str, Any]]) -> None:
    keys = ["i0", "best_avg_logprob"]
    lines = ["run\tlearning_rate\tbeta1\tbeta2\tclip_grad_norm\tfrozen_encoder_layers\t" + "\t".join(keys)]
    if baseline is not None:
        lines.append("baseline\t\t\t\t\t\t" + "\t".join(f"{baseline.get(key, 0.0):.6f}" for key in keys))
    for row in rows:
        scores = row["scores"]
        lines.append(
            "\t".join(
                [
                    row["run"],
                    f"{row['learning_rate']:.10g}",
                    f"{row['beta1']:.10g}",
                    f"{row['beta2']:.10g}",
                    "" if row.get("clip_grad_norm") is None else f"{row['clip_grad_norm']:.10g}",
                    "" if row.get("frozen_encoder_layers") is None else str(row["frozen_encoder_layers"]),
                    *[f"{scores.get(key, 0.0):.6f}" for key in keys],
                ]
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(out_root: Path, rows: list[dict[str, Any]], baseline: dict[str, float] | None, eval_name: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (loss_ax, wer_ax) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3.0, 1.4]})
    for row in rows:
        steps, losses = read_losses(Path(row["path"]))
        if not losses:
            continue
        label = f"{row['run']} WER {row['scores']['i0']:.2f}%"
        loss_ax.plot(steps, rolling_mean(losses, 100), linewidth=1.2, label=label)
    loss_ax.set_title("ATC one-epoch LR/beta sweep")
    loss_ax.set_xlabel("training step")
    loss_ax.set_ylabel("train loss, rolling mean")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend(fontsize=7, ncols=2)

    sorted_rows = sorted(rows, key=lambda row: row["scores"]["i0"])
    labels = [row["run"].replace("_", "\n") for row in sorted_rows]
    values = [row["scores"]["i0"] for row in sorted_rows]
    xs = list(range(len(sorted_rows)))
    wer_ax.bar(xs, values, width=0.72)
    if baseline is not None:
        wer_ax.axhline(baseline["i0"], color="black", linestyle="--", linewidth=1.0, label=f"baseline {baseline['i0']:.2f}%")
        wer_ax.legend(fontsize=8)
    for x, value in zip(xs, values, strict=True):
        wer_ax.text(x, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=7, rotation=90)
    wer_ax.set_ylabel("validation WER %")
    wer_ax.set_xticks(xs, labels, fontsize=7)
    wer_ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(eval_artifact(out_root, "loss_and_wer.png", eval_name), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.learning_rates is None:
        if (
            args.beta1_values is not None
            or args.beta2_values is not None
            or args.clip_grad_norm_values is not None
            or args.frozen_encoder_layers_values is not None
        ):
            raise ValueError("--beta1-values/--beta2-values/--clip-grad-norm-values/--frozen-encoder-layers-values require --learning-rates")
        configs = {"stable": STABLE_CONFIGS, "focused": FOCUSED_CONFIGS, "fast-fail": FAST_FAIL_CONFIGS}[args.grid]
    else:
        beta1_values = args.beta1_values if args.beta1_values is not None else [0.0]
        beta2_values = args.beta2_values if args.beta2_values is not None else [0.9999]
        clip_grad_norm_values = args.clip_grad_norm_values if args.clip_grad_norm_values is not None else [None]
        frozen_encoder_layers_values = args.frozen_encoder_layers_values if args.frozen_encoder_layers_values is not None else [None]
        configs = [
            Config(lr, beta1, beta2, clip_grad_norm, frozen_encoder_layers)
            for beta1 in beta1_values
            for beta2 in beta2_values
            for lr in args.learning_rates
            for clip_grad_norm in clip_grad_norm_values
            for frozen_encoder_layers in frozen_encoder_layers_values
        ]
    if len(configs) < args.expected_runs:
        raise ValueError(f"{args.grid} config grid has {len(configs)} runs, expected at least {args.expected_runs}")

    root = args.root.resolve()
    train_jsonl = root / args.train_jsonl
    validation_jsonl = root / args.validation_jsonl
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)
    if not validation_jsonl.exists():
        raise FileNotFoundError(validation_jsonl)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_name = args.sweep_name or f"{timestamp}_atc_lr_beta_sweep_notimestamps"
    out_root = args.out_root if args.out_root.is_absolute() else root / args.out_root / sweep_name
    model_root = out_root if args.model_root is None else (args.model_root if args.model_root.is_absolute() else root / args.model_root)
    if not args.eval_only and model_root != out_root:
        raise ValueError("--model-root is only valid with --eval-only")
    out_root.mkdir(parents=True, exist_ok=True)
    train_bin = command_path("+train", args.train_bin)
    test_bin = command_path("+test", args.test_bin)
    refs = read_jsonl(validation_jsonl)
    wav_paths = [str(row["path"]) for row in refs]

    (out_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "train_jsonl": rel(train_jsonl, root),
                "validation_jsonl": rel(validation_jsonl, root),
                "eval_name": args.eval_name,
                "model_root": rel(model_root, root),
                "init": args.init,
                "baseline_model": args.baseline_model,
                "language": args.language,
                "max_new_tokens": args.max_new_tokens,
                "decode_batch_size": args.decode_batch_size,
                "grid": args.grid,
                "learning_rates": args.learning_rates,
                "beta1_values": args.beta1_values,
                "beta2_values": args.beta2_values,
                "clip_grad_norm_values": args.clip_grad_norm_values,
                "frozen_encoder_layers_values": args.frozen_encoder_layers_values,
                "expected_runs": args.expected_runs,
                "lr_scheduler_type": args.lr_scheduler_type,
                "weight_decay": args.weight_decay,
                "uses_timestamp_tokens": False,
                "configs": [config.__dict__ for config in configs],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    baseline_scores: dict[str, float] | None = None
    if not args.skip_baseline:
        baseline_dir = out_root / "baseline_large-v3-turbo"
        if args.eval_name != "validation":
            baseline_dir = out_root / f"baseline_large-v3-turbo.{safe_name(args.eval_name)}"
        baseline_jsonl = run_test(
            args=args,
            root=root,
            out_dir=baseline_dir,
            test_bin=test_bin,
            model_arg=("--model", args.baseline_model),
            wav_paths=wav_paths,
        )
        baseline_scores = score_decode(refs, baseline_jsonl, baseline_dir, root, args.decode_batch_size)

    result_rows: list[dict[str, Any]] = []
    results_path = eval_artifact(out_root, "results.jsonl", args.eval_name)
    summary_path = eval_artifact(out_root, "wer_summary.tsv", args.eval_name)
    best_path = eval_artifact(out_root, "best.json", args.eval_name)
    for config in configs:
        name = run_name(config)
        run_dir = model_root / name
        if not args.eval_only:
            run_dir = out_root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            train_run(args, root, run_dir, config, train_bin)
        if not has_model_files(run_dir):
            raise RuntimeError(f"training run did not produce model files: {run_dir}")
        decode_dir = out_root / f"{name}.{safe_name(args.eval_name)}"
        decode_jsonl = run_test(
            args=args,
            root=root,
            out_dir=decode_dir,
            test_bin=test_bin,
            model_arg=("--exp", rel(run_dir, root)),
            wav_paths=wav_paths,
        )
        scores = score_decode(refs, decode_jsonl, decode_dir, root, args.decode_batch_size)
        row = {
            "run": name,
            "path": str(run_dir),
            "learning_rate": config.learning_rate,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "clip_grad_norm": config.clip_grad_norm,
            "frozen_encoder_layers": config.frozen_encoder_layers,
            "scores": scores,
            "eval_name": args.eval_name,
            "eval_path": str(decode_dir),
        }
        result_rows.append(row)
        write_jsonl(results_path, result_rows)
        write_summary(summary_path, baseline_scores, result_rows)
        plot_results(out_root, result_rows, baseline_scores, args.eval_name)

    best = min(result_rows, key=lambda row: row["scores"]["i0"])
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": rel(out_root, root), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
