#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from run_lr_beta_sweep import command_path, read_jsonl, rel, run_test, score_decode


MODELS: list[tuple[str, str]] = [
    ("small", "openai/whisper-small"),
    ("medium", "openai/whisper-medium"),
    ("large-v1", "openai/whisper-large"),
    ("large-v2", "openai/whisper-large-v2"),
    ("large-v3", "openai/whisper-large-v3"),
    ("turbo", "openai/whisper-large-v3-turbo"),
]

EVALS: list[tuple[str, str, str, str]] = [
    ("atc_validation", "ATC validation", "egs/atc/data/validation.jsonl", "validation"),
    ("atc_test", "ATC test", "egs/atc/data/test.jsonl", "test"),
    ("atcosim_test", "ATCOSIM test", "egs/atc/data/atcosim/test.jsonl", "atcosim/test"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode baseline Whisper models on the ATC evaluation matrix.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=Path("egs/atc/exp"))
    parser.add_argument("--run-name", default="20260617_atc_baselines_all_models_notimestamps_max96")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--test-bin", default=None)
    parser.add_argument("--test-retries", type=int, default=0)
    parser.add_argument("--test-retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[label for label, _ in MODELS],
        help="Model labels to run. Defaults to all finetuned model-size variants.",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=[name for name, _, _, _ in EVALS],
        help="Eval names to run. Defaults to atc_validation atc_test atcosim_test.",
    )
    return parser.parse_args()


def mean_avg_logprob(decode_i0: Path) -> float | None:
    values = []
    for row in read_jsonl(decode_i0):
        value = row.get("avg_logprob")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_id",
        "eval_name",
        "eval_set",
        "manifest",
        "split",
        "wer_percent",
        "best_avg_logprob_wer_percent",
        "mean_avg_logprob",
        "references",
        "hypotheses",
        "missing_hypotheses",
        "total_edits",
        "total_words",
        "decode_rows",
        "eval_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_root = args.out_root if args.out_root.is_absolute() else root / args.out_root / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    models = dict(MODELS)
    evals = {name: (label, manifest, split) for name, label, manifest, split in EVALS}
    unknown_models = [label for label in args.models if label not in models]
    unknown_evals = [name for name in args.evals if name not in evals]
    if unknown_models:
        raise ValueError(f"unknown model labels: {', '.join(unknown_models)}")
    if unknown_evals:
        raise ValueError(f"unknown eval names: {', '.join(unknown_evals)}")

    test_bin = command_path("+test", args.test_bin)
    config = {
        "run_name": args.run_name,
        "models": [{"label": label, "model_id": models[label]} for label in args.models],
        "evals": [
            {"eval_name": name, "eval_set": evals[name][0], "manifest": evals[name][1], "split": evals[name][2]}
            for name in args.evals
        ],
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "decode_batch_size": args.decode_batch_size,
        "test_bin": test_bin,
        "uses_timestamp_tokens": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_root / "baseline_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    for model_label in args.models:
        model_id = models[model_label]
        for eval_name in args.evals:
            eval_set, manifest, split = evals[eval_name]
            manifest_path = root / manifest
            refs = read_jsonl(manifest_path)
            wav_paths = [str(row["path"]) for row in refs]
            decode_dir = out_root / f"baseline_{model_label}.{eval_name}"
            decode_jsonl = run_test(
                args=args,
                root=root,
                out_dir=decode_dir,
                test_bin=test_bin,
                model_arg=("--model", model_id),
                wav_paths=wav_paths,
            )
            scores = score_decode(refs, decode_jsonl, decode_dir, root, args.decode_batch_size)
            wer_path = decode_dir / "wer_i0.json"
            with wer_path.open("r", encoding="utf-8") as f:
                wer = json.load(f)
            best_wer_path = decode_dir / "wer_best_avg_logprob.json"
            with best_wer_path.open("r", encoding="utf-8") as f:
                best_wer = json.load(f)
            decode_rows = len(read_jsonl(decode_jsonl))
            row = {
                "model": model_label,
                "model_id": model_id,
                "eval_name": eval_name,
                "eval_set": eval_set,
                "manifest": manifest,
                "split": split,
                "wer_percent": scores["i0"],
                "best_avg_logprob_wer_percent": scores["best_avg_logprob"],
                "mean_avg_logprob": mean_avg_logprob(decode_dir / "decode_i0.jsonl"),
                "references": int(wer["references"]),
                "hypotheses": int(wer["hypotheses"]),
                "missing_hypotheses": int(wer["missing_hypotheses"]),
                "total_edits": int(wer["total_edits"]),
                "total_words": int(wer["total_words"]),
                "decode_rows": decode_rows,
                "eval_path": rel(decode_dir, root),
            }
            if row["best_avg_logprob_wer_percent"] != float(best_wer["wer_percent"]):
                raise RuntimeError(f"best avg-logprob score mismatch for {decode_dir}")
            result_rows.append(row)
            write_jsonl(out_root / "baseline_results.jsonl", result_rows)
            write_tsv(out_root / "baseline_summary.tsv", result_rows)
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
