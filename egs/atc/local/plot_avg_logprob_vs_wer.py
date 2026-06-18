#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from plot_lr_wer_and_lengths import (
    eval_dataset_label,
    excluded_plot_row_keys,
    filter_plot_series_rows,
    ordered_train_labels,
    parse_mapping,
    plot_style_by_train,
    read_series,
    standalone_series,
    train_series_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean decode avg_logprob vs WER by evaluation dataset.")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Use the standard ATC/ATCOSIM sweep/eval layout and discover result files.",
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("egs/atc/exp"))
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=RESULTS_JSONL[,RESULTS_JSONL...]",
        help="Result series to plot. Can be provided multiple times.",
    )
    parser.add_argument("--out", type=Path, default=Path("egs/atc/exp/avg_logprob_vs_wer_by_eval.png"))
    parser.add_argument("--summary-tsv", type=Path, default=Path("egs/atc/exp/avg_logprob_vs_wer_by_eval.tsv"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def mean_avg_logprob(eval_path: Path) -> tuple[float, int]:
    decode_path = eval_path / "decode_i0.jsonl"
    if not decode_path.exists():
        decode_path = eval_path / "decode.jsonl"
    if not decode_path.exists():
        raise FileNotFoundError(f"missing decode JSONL for {eval_path}")

    values = [
        float(row["avg_logprob"])
        for row in read_jsonl(decode_path)
        if int(row.get("i", 0)) == 0 and row.get("avg_logprob") is not None
    ]
    if not values:
        raise ValueError(f"no avg_logprob values in {decode_path}")
    return statistics.fmean(values), len(values)


def has_decode_jsonl(eval_path: Path) -> bool:
    return (eval_path / "decode_i0.jsonl").exists() or (eval_path / "decode.jsonl").exists()


def row_eval_path(row: dict[str, Any], root: Path, eval_name: str) -> Path | None:
    if row.get("eval_path"):
        return resolve_path(str(row["eval_path"]), root)
    if not row.get("path"):
        return None

    base_path = resolve_path(str(row["path"]), root)
    suffixes = [eval_name]
    if eval_name in {"validation", "atc_validation"}:
        suffixes = ["validation", "atc_validation"]
    for suffix in dict.fromkeys(suffixes):
        candidate = Path(f"{base_path}.{suffix}")
        if has_decode_jsonl(candidate):
            return candidate
    return None


def read_points(label: str, result_path: Path, root: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in read_jsonl(result_path):
        scores = row.get("scores", {})
        if "i0" not in scores:
            continue
        eval_name = str(row.get("eval_name") or "validation")
        eval_path = row_eval_path(row, root, eval_name)
        if eval_path is None:
            continue
        try:
            avg_logprob, decoded_rows = mean_avg_logprob(eval_path)
        except (FileNotFoundError, ValueError):
            continue
        points.append(
            {
                "series": label,
                "train_series": train_series_label(label),
                "eval_name": eval_name,
                "eval_set": eval_dataset_label(eval_name),
                "run": row["run"],
                "learning_rate": float(row["learning_rate"]),
                "wer_percent": float(scores["i0"]),
                "mean_avg_logprob": avg_logprob,
                "decoded_rows": decoded_rows,
                "eval_path": str(eval_path),
                "results_path": str(result_path),
            }
        )
    return points


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0.0 or y_den == 0.0:
        return None
    return numerator / (x_den * y_den)


def fmt_lr(value: float) -> str:
    return f"{value:.2g}".replace("e-0", "e-").replace("e+0", "e")


def make_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_order = ["ATC validation", "ATC test", "ATCOSIM test"]
    eval_sets = [name for name in eval_order if any(row["eval_set"] == name for row in rows)]
    eval_sets.extend(name for name in sorted({row["eval_set"] for row in rows}) if name not in eval_sets)
    if not eval_sets:
        raise SystemExit("no rows with avg_logprob were found")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(eval_sets), figsize=(7.0 * len(eval_sets), 5.4), squeeze=False, constrained_layout=True)
    train_labels = ordered_train_labels([str(row["train_series"]) for row in rows])
    cmap = plt.get_cmap("tab10")
    style = plot_style_by_train(train_labels, cmap)
    legend_handles = {}

    for ax, eval_set in zip(axes[0], eval_sets, strict=True):
        pane_rows = [row for row in rows if row["eval_set"] == eval_set]
        for train_label in train_labels:
            group = [row for row in pane_rows if row["train_series"] == train_label]
            if not group:
                continue
            color, marker = style[train_label]
            scatter = ax.scatter(
                [row["mean_avg_logprob"] for row in group],
                [row["wer_percent"] for row in group],
                s=34,
                alpha=0.78,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                label=train_label,
            )
            legend_handles.setdefault(train_label, scatter)
            best = min(group, key=lambda row: row["wer_percent"])
            ax.annotate(
                fmt_lr(float(best["learning_rate"])),
                (best["mean_avg_logprob"], best["wer_percent"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        xs = [row["mean_avg_logprob"] for row in pane_rows]
        ys = [row["wer_percent"] for row in pane_rows]
        corr = pearson(xs, ys)
        suffix = "" if corr is None else f"  r={corr:.2f}"
        ax.set_title(f"{eval_set}{suffix}")
        ax.set_xlabel("mean avg_logprob (higher is better)")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("WER (%)")

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles),
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            fontsize=8,
        )
    fig.suptitle("Mean Decode Avg Log Probability vs WER", fontsize=14)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    if args.standalone or not args.series:
        series_pairs = standalone_series(args.exp_dir)
    else:
        series_pairs = parse_mapping(args.series, "--series")

    rows: list[dict[str, Any]] = []
    filter_source_rows: list[dict[str, Any]] = []
    for label, paths in series_pairs:
        for result_path in paths:
            rows.extend(read_points(label, result_path, root))
            filter_source_rows.extend(read_series(label, result_path))

    rows = sorted(rows, key=lambda row: (row["eval_set"], row["train_series"], row["learning_rate"], row["run"]))
    write_tsv(args.summary_tsv, rows)
    make_plot(args.out, filter_plot_series_rows(rows, excluded_plot_row_keys(filter_source_rows)))
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_tsv}")
    print(f"points {len(rows)}")


if __name__ == "__main__":
    main()
