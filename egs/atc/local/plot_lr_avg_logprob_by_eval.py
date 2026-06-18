#!/usr/bin/env python3

import argparse
from pathlib import Path

from plot_avg_logprob_vs_wer import fmt_lr, read_points, write_tsv
from plot_lr_wer_and_lengths import (
    excluded_plot_row_keys,
    filter_plot_series_rows,
    ordered_train_labels,
    parse_mapping,
    plot_style_by_train,
    read_series,
    standalone_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot peak learning rate vs mean decode avg_logprob by evaluation dataset.")
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
    parser.add_argument("--out", type=Path, default=Path("egs/atc/exp/peak_lr_vs_mean_avg_logprob_by_eval.png"))
    parser.add_argument("--summary-tsv", type=Path, default=Path("egs/atc/exp/peak_lr_vs_mean_avg_logprob_by_eval.tsv"))
    return parser.parse_args()


def make_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_order = ["ATC validation", "ATC test", "ATCOSIM test"]
    eval_sets = [name for name in eval_order if any(row["eval_set"] == name for row in rows)]
    eval_sets.extend(name for name in sorted({str(row["eval_set"]) for row in rows}) if name not in eval_sets)
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
            group = sorted(
                (row for row in pane_rows if row["train_series"] == train_label),
                key=lambda row: float(row["learning_rate"]),
            )
            if not group:
                continue
            color, marker = style[train_label]
            line = ax.plot(
                [float(row["learning_rate"]) for row in group],
                [float(row["mean_avg_logprob"]) for row in group],
                marker=marker,
                linewidth=1.3,
                markersize=4.8,
                alpha=0.82,
                color=color,
                label=train_label,
            )[0]
            legend_handles.setdefault(train_label, line)
            best = max(group, key=lambda row: float(row["mean_avg_logprob"]))
            ax.annotate(
                fmt_lr(float(best["learning_rate"])),
                (float(best["learning_rate"]), float(best["mean_avg_logprob"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_xlabel("peak learning rate")
        ax.set_title(eval_set)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("mean decode avg_logprob")

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles),
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            fontsize=8,
        )
    fig.suptitle("Peak Learning Rate vs Mean Decode Avg Log Probability", fontsize=14)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    if args.standalone or not args.series:
        series_pairs = standalone_series(args.exp_dir)
    else:
        series_pairs = parse_mapping(args.series, "--series")

    rows: list[dict[str, object]] = []
    filter_source_rows: list[dict[str, object]] = []
    for label, paths in series_pairs:
        for result_path in paths:
            rows.extend(read_points(label, result_path, root))
            filter_source_rows.extend(read_series(label, result_path))

    rows = sorted(rows, key=lambda row: (str(row["eval_set"]), str(row["train_series"]), float(row["learning_rate"]), str(row["run"])))
    write_tsv(args.summary_tsv, rows)
    make_plot(args.out, filter_plot_series_rows(rows, excluded_plot_row_keys(filter_source_rows)))
    print(f"wrote {args.out}")
    print(f"wrote {args.summary_tsv}")
    print(f"points {len(rows)}")


if __name__ == "__main__":
    main()
