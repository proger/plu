#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]
PLOT_TRAIN_LABEL_ORDER = [
    "ATC 100% train",
    "ATC+ATCOSIM train",
    "ATCOSIM train",
    "ATC aug4 train",
    "ATC 10% train",
    "ATC 25% train",
    "ATC 50% train",
    "ATC 75% train",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LR/WER sweeps and training length histograms.")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Use the standard ATC/ATCOSIM sweep/eval layout and discover the newest cross-eval result files.",
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("egs/atc/exp"), help="Experiment directory for --standalone discovery.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=RESULTS_JSONL[,RESULTS_JSONL...]",
        help="Result series to plot. Can be provided multiple times; paths sharing a label are merged.",
    )
    parser.add_argument(
        "--length-set",
        action="append",
        default=[],
        metavar="LABEL=TRAIN_JSONL",
        help="Training JSONL whose duration/token lengths should be histogrammed. Can be provided multiple times.",
    )
    parser.add_argument("--out", type=Path, default=Path("egs/atc/exp/lr_wer_and_lengths.png"))
    parser.add_argument("--eval-pane-out", type=Path, default=None, help="Optional LR/WER plot with one pane per evaluation dataset.")
    parser.add_argument("--summary-tsv", type=Path, default=Path("egs/atc/exp/lr_wer_summary.tsv"))
    return parser.parse_args()


def result_row_count(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def newest_result(exp_dir: Path, pattern: str, filename: str, *, prefer_most_rows: bool = False) -> Path | None:
    candidates = [directory / filename for directory in exp_dir.glob(pattern) if (directory / filename).exists()]
    if not candidates:
        return None
    if prefer_most_rows:
        return max(candidates, key=lambda path: (result_row_count(path), path.parent.name))
    for path in sorted(candidates, key=lambda path: path.parent.name, reverse=True):
        return path
    return None


def standalone_series(exp_dir: Path) -> list[tuple[str, list[Path]]]:
    static: list[tuple[str, list[Path]]] = [
        (
            "ATC train -> ATC val",
            [
                exp_dir / "20260610_224128_atc_linear_lr_b10_b20p9999_notimestamps_max96" / "results.jsonl",
                exp_dir / "20260611_014153_atc_linear_lr_b10_b20p9999_ext_notimestamps_max96" / "results.jsonl",
            ],
        ),
        (
            "ATC+ATCOSIM train -> ATC val",
            [exp_dir / "20260611_230217_atc_mixed_atcosim_lr10_notimestamps_max96" / "results.jsonl"],
        ),
        (
            "ATC+ATCOSIM train -> ATCOSIM test",
            [exp_dir / "20260612_090127_atc_mixed_lr10_atcosim_test" / "results_atcosim_test.jsonl"],
        ),
        (
            "ATCOSIM train -> ATCOSIM test",
            [exp_dir / "20260612_090127_atcosim_only_lr10_atcosim_test" / "results_atcosim_test.jsonl"],
        ),
    ]
    discovered_specs = [
        ("ATCOSIM train -> ATC val", "*_atcosim_only_lr10_atc_validation", "results_atc_validation.jsonl"),
        ("ATC train -> ATC test", "*_atc_only_lr10_base_atc_test", "results_atc_test.jsonl"),
        ("ATC train -> ATC test", "*_atc_only_lr10_ext_atc_test", "results_atc_test.jsonl"),
        ("ATC+ATCOSIM train -> ATC test", "*_atc_mixed_lr10_atc_test", "results_atc_test.jsonl"),
        ("ATCOSIM train -> ATC test", "*_atcosim_only_lr10_atc_test", "results_atc_test.jsonl"),
    ]
    optional_specs = [
        ("ATC train -> ATCOSIM test", "*_atc_only_lr10_base_atcosim_test", "results_atcosim_test.jsonl", False),
        ("ATC train -> ATCOSIM test", "*_atc_only_lr10_ext_atcosim_test", "results_atcosim_test.jsonl", False),
        ("ATC aug4 train -> ATC val", "*_atc_aug4_lr*_validation", "results_atc_validation.jsonl", True),
        ("ATC aug4 train -> ATC test", "*_atc_aug4_lr*_atc_test", "results_atc_test.jsonl", True),
        ("ATC aug4 train -> ATCOSIM test", "*_atc_aug4_lr*_atcosim_test", "results_atcosim_test.jsonl", True),
        ("ATC 10% train -> ATC val", "*_atc_subsample_10pct_lr*_validation", "results_atc_validation.jsonl", False),
        ("ATC 10% train -> ATC test", "*_atc_subsample_10pct_lr*_atc_test", "results_atc_test.jsonl", False),
        ("ATC 10% train -> ATCOSIM test", "*_atc_subsample_10pct_lr*_atcosim_test", "results_atcosim_test.jsonl", False),
        ("ATC 25% train -> ATC val", "*_atc_subsample_25pct_lr*_validation", "results_atc_validation.jsonl", False),
        ("ATC 25% train -> ATC test", "*_atc_subsample_25pct_lr*_atc_test", "results_atc_test.jsonl", False),
        ("ATC 25% train -> ATCOSIM test", "*_atc_subsample_25pct_lr*_atcosim_test", "results_atcosim_test.jsonl", False),
        ("ATC 50% train -> ATC val", "*_atc_subsample_50pct_lr*_validation", "results_atc_validation.jsonl", False),
        ("ATC 50% train -> ATC test", "*_atc_subsample_50pct_lr*_atc_test", "results_atc_test.jsonl", False),
        ("ATC 50% train -> ATCOSIM test", "*_atc_subsample_50pct_lr*_atcosim_test", "results_atcosim_test.jsonl", False),
        ("ATC 75% train -> ATC val", "*_atc_subsample_75pct_lr*_validation", "results_atc_validation.jsonl", False),
        ("ATC 75% train -> ATC test", "*_atc_subsample_75pct_lr*_atc_test", "results_atc_test.jsonl", False),
        ("ATC 75% train -> ATCOSIM test", "*_atc_subsample_75pct_lr*_atcosim_test", "results_atcosim_test.jsonl", False),
    ]
    grouped: dict[str, list[Path]] = {label: list(paths) for label, paths in static}
    missing: list[str] = []
    for label, pattern, filename in discovered_specs:
        path = newest_result(exp_dir, pattern, filename)
        if path is None:
            missing.append(f"{label}: {pattern}/{filename}")
            continue
        grouped.setdefault(label, []).append(path)
    for label, pattern, filename, prefer_most_rows in optional_specs:
        path = newest_result(exp_dir, pattern, filename, prefer_most_rows=prefer_most_rows)
        if path is not None:
            grouped.setdefault(label, []).append(path)
    pairs = [(label, paths) for label, paths in grouped.items()]
    absent = [str(path) for _, paths in pairs for path in paths if not path.exists()]
    if missing or absent:
        details = "\n".join([*missing, *absent])
        raise FileNotFoundError(f"standalone plot is missing required result files:\n{details}")
    return pairs


def standalone_lengths() -> list[tuple[str, list[Path]]]:
    return [
        ("ATC train", [Path("egs/atc/data/train.jsonl")]),
        ("ATC 10% train", [Path("egs/atc/data/train_subsample_10pct.jsonl")]),
        ("ATC 25% train", [Path("egs/atc/data/train_subsample_25pct.jsonl")]),
        ("ATC 50% train", [Path("egs/atc/data/train_subsample_50pct.jsonl")]),
        ("ATC 75% train", [Path("egs/atc/data/train_subsample_75pct.jsonl")]),
        ("ATC aug4 train", [Path("egs/atc/data/train_aug4.jsonl")]),
        ("ATC+ATCOSIM train", [Path("egs/atc/data/train_mixed_atcosim.jsonl")]),
        ("ATCOSIM train", [Path("egs/atc/data/atcosim/train.jsonl")]),
    ]


def parse_mapping(values: list[str], option: str) -> list[tuple[str, list[Path]]]:
    pairs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects LABEL=PATH, got {value!r}")
        label, paths_text = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"{option} label cannot be empty")
        paths = [Path(path.strip()) for path in paths_text.split(",") if path.strip()]
        if not paths:
            raise ValueError(f"{option} requires at least one path")
        pairs.append((label, paths))
    return pairs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_series(label: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        scores = row.get("scores", {})
        rows.append(
            {
                "series": label,
                "run": row["run"],
                "learning_rate": float(row["learning_rate"]),
                "wer_percent": float(scores["i0"]),
                "best_avg_logprob_wer_percent": float(scores.get("best_avg_logprob", scores["i0"])),
                "eval_name": row.get("eval_name", "validation"),
                "path": row.get("path", ""),
                "eval_path": row.get("eval_path", ""),
            }
        )
    return rows


def read_lengths(label: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        duration = row.get("duration")
        if duration is None:
            continue
        input_ids = row.get("input_ids") or row.get("labels") or []
        rows.append(
            {
                "set": label,
                "duration": float(duration),
                "label_tokens": len(input_ids),
                "source_dataset": row.get("source_dataset", ""),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(out: Path, series_rows: list[dict[str, Any]], length_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), constrained_layout=True)

    ax = axes[0]
    series_labels = list(dict.fromkeys(row["series"] for row in series_rows))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]
    for index, label in enumerate(series_labels):
        rows = sorted((row for row in series_rows if row["series"] == label), key=lambda row: row["learning_rate"])
        xs = [row["learning_rate"] for row in rows]
        ys = [row["wer_percent"] for row in rows]
        ax.plot(xs, ys, marker=markers[index % len(markers)], linewidth=1.4, markersize=4.5, label=label, color=cmap(index % 10))
        for row in rows:
            ax.annotate(f"{row['wer_percent']:.2f}", (row["learning_rate"], row["wer_percent"]), xytext=(2, 3), textcoords="offset points", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("WER (%)")
    ax.set_title("Learning Rate vs WER")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    length_labels = list(dict.fromkeys(row["set"] for row in length_rows))
    bins_duration = [i * 1.0 for i in range(0, 31)]
    bins_tokens = [i * 4 for i in range(0, 31)]

    ax = axes[1]
    for index, label in enumerate(length_labels):
        values = [row["duration"] for row in length_rows if row["set"] == label]
        ax.hist(values, bins=bins_duration, histtype="step", linewidth=1.5, density=True, label=label, color=cmap(index % 10))
    ax.set_xlabel("audio duration (s)")
    ax.set_ylabel("density")
    ax.set_title("Training Audio Lengths")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    for index, label in enumerate(length_labels):
        values = [row["label_tokens"] for row in length_rows if row["set"] == label]
        ax.hist(values, bins=bins_tokens, histtype="step", linewidth=1.5, density=True, label=label, color=cmap(index % 10))
    ax.set_xlabel("label tokens")
    ax.set_ylabel("density")
    ax.set_title("Training Label Lengths")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.savefig(out, dpi=300)
    plt.close(fig)


def eval_dataset_label(eval_name: str) -> str:
    if eval_name in {"validation", "atc_validation"}:
        return "ATC validation"
    if eval_name == "atc_test":
        return "ATC test"
    if eval_name == "atcosim_test":
        return "ATCOSIM test"
    return eval_name.replace("_", " ")


def train_series_label(series: str) -> str:
    label = series.split("->", 1)[0].strip()
    if label == "ATC train":
        return "ATC 100% train"
    return label


def ordered_train_labels(labels: list[str]) -> list[str]:
    seen = set(labels)
    ordered = [label for label in PLOT_TRAIN_LABEL_ORDER if label in seen]
    ordered_set = set(ordered)
    ordered.extend(label for label in labels if label not in ordered_set)
    return list(dict.fromkeys(ordered))


def plot_style_by_train(labels: list[str], cmap: Any) -> dict[str, tuple[Any, str]]:
    return {
        label: (cmap(index % 10), PLOT_MARKERS[index % len(PLOT_MARKERS)])
        for index, label in enumerate(ordered_train_labels(labels))
    }


def row_train_label(row: dict[str, Any]) -> str:
    if row.get("train_series"):
        return str(row["train_series"])
    return train_series_label(str(row.get("series", "")))


def row_eval_label(row: dict[str, Any]) -> str:
    if row.get("eval_set"):
        return str(row["eval_set"])
    return eval_dataset_label(str(row.get("eval_name", "validation")))


def plot_row_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (row_train_label(row), row_eval_label(row), str(row["run"]), float(row["learning_rate"]))


def excluded_plot_row_keys(series_rows: list[dict[str, Any]]) -> set[tuple[str, str, str, float]]:
    """Select row keys hidden from plots without changing summaries."""
    rows_by_plot_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in series_rows:
        rows_by_plot_group.setdefault((row_train_label(row), row_eval_label(row)), []).append(row)

    excluded: set[tuple[str, str, str, float]] = set()
    for (train_label, _eval_label), rows in rows_by_plot_group.items():
        if train_label == "ATC aug4 train" and len(rows) > 2:
            excluded.update(
                plot_row_key(row)
                for row in sorted(rows, key=lambda row: float(row["wer_percent"]), reverse=True)[:2]
            )
    return excluded


def filter_plot_series_rows(
    series_rows: list[dict[str, Any]],
    excluded_keys: set[tuple[str, str, str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Hide filtered row keys while preserving input order."""
    if excluded_keys is None:
        excluded_keys = excluded_plot_row_keys(series_rows)
    return [row for row in series_rows if plot_row_key(row) not in excluded_keys]


def make_eval_pane_plot(out: Path, series_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_order = ["ATC validation", "ATC test", "ATCOSIM test"]
    eval_labels = [label for label in eval_order if any(eval_dataset_label(str(row["eval_name"])) == label for row in series_rows)]
    eval_labels.extend(
        label
        for label in sorted({eval_dataset_label(str(row["eval_name"])) for row in series_rows})
        if label not in eval_labels
    )
    if not eval_labels:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(eval_labels), figsize=(7.0 * len(eval_labels), 5.4), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    train_labels = ordered_train_labels([train_series_label(str(row["series"])) for row in series_rows])
    style = plot_style_by_train(train_labels, cmap)
    legend_handles = {}

    for ax, eval_label in zip(axes[0], eval_labels, strict=True):
        pane_rows = [row for row in series_rows if eval_dataset_label(str(row["eval_name"])) == eval_label]
        for train_label in train_labels:
            rows = sorted(
                (row for row in pane_rows if train_series_label(str(row["series"])) == train_label),
                key=lambda row: row["learning_rate"],
            )
            if not rows:
                continue
            color, marker = style[train_label]
            xs = [row["learning_rate"] for row in rows]
            ys = [row["wer_percent"] for row in rows]
            line = ax.plot(
                xs,
                ys,
                marker=marker,
                linewidth=1.5,
                markersize=4.8,
                label=train_label,
                color=color,
            )[0]
            legend_handles.setdefault(train_label, line)
            best = min(rows, key=lambda row: row["wer_percent"])
            ax.annotate(
                f"{best['wer_percent']:.2f}",
                (best["learning_rate"], best["wer_percent"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("peak learning rate")
        ax.set_title(eval_label)
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
    fig.suptitle("Learning Rate vs WER by Evaluation Dataset", fontsize=14)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.standalone or (not args.series and not args.length_set):
        series_pairs = standalone_series(args.exp_dir)
        length_pairs = standalone_lengths()
    else:
        series_pairs = parse_mapping(args.series, "--series")
        length_pairs = parse_mapping(args.length_set, "--length-set")
    series_rows = []
    for label, paths in series_pairs:
        for path in paths:
            series_rows.extend(read_series(label, path))
    length_rows = []
    for label, paths in length_pairs:
        if len(paths) != 1:
            raise ValueError("--length-set expects exactly one JSONL path per label")
        length_rows.extend(read_lengths(label, paths[0]))
    if not series_rows:
        raise SystemExit("provide at least one --series LABEL=RESULTS_JSONL")
    if not length_rows:
        raise SystemExit("provide at least one --length-set LABEL=TRAIN_JSONL")
    write_tsv(args.summary_tsv, sorted(series_rows, key=lambda row: (row["series"], row["learning_rate"])))
    plot_series_rows = filter_plot_series_rows(series_rows)
    make_plot(args.out, plot_series_rows, length_rows)
    if args.eval_pane_out is not None:
        make_eval_pane_plot(args.eval_pane_out, plot_series_rows)
    best = min(series_rows, key=lambda row: row["wer_percent"])
    print(f"wrote {args.out}")
    if args.eval_pane_out is not None:
        print(f"wrote {args.eval_pane_out}")
    print(f"wrote {args.summary_tsv}")
    print(f"best {best['series']} {best['run']} WER={best['wer_percent']:.6f}")


if __name__ == "__main__":
    main()
