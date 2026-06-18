#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Any

from plot_avg_logprob_vs_wer import fmt_lr, parse_mapping, pearson, read_points


MODEL_ORDER = ["small", "medium", "large-v1", "large", "large-v2", "large-v3", "turbo"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MODEL_METADATA: dict[str, dict[str, int]] = {
    "small": {"trainable_params": 241_734_912, "width": 768, "encoder_depth": 12, "decoder_depth": 12},
    "medium": {"trainable_params": 763_857_920, "width": 1024, "encoder_depth": 24, "decoder_depth": 24},
    "large-v1": {"trainable_params": 1_543_304_960, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large_v1": {"trainable_params": 1_543_304_960, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large": {"trainable_params": 1_543_490_560, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large-v2": {"trainable_params": 1_543_304_960, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large_v2": {"trainable_params": 1_543_304_960, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large-v3": {"trainable_params": 1_543_490_560, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "large_v3": {"trainable_params": 1_543_490_560, "width": 1280, "encoder_depth": 32, "decoder_depth": 32},
    "turbo": {"trainable_params": 808_878_080, "width": 1280, "encoder_depth": 32, "decoder_depth": 4},
}
BEST_WER_EVAL_SETS = ["ATC validation", "ATC test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ATC full-train LR basin comparisons by Whisper model size.")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="MODEL=RESULTS_JSONL[,RESULTS_JSONL...]",
        help="Model label and result JSONL path(s). Can be provided multiple times.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("egs/atc/exp"))
    parser.add_argument("--prefix", default="atc_model_lr_comparison")
    parser.add_argument("--summary-tsv", type=Path, default=None)
    parser.add_argument(
        "--no-invert-logprob-y",
        action="store_true",
        help="Do not invert the y axis on the peak-LR vs avg-logprob plot.",
    )
    return parser.parse_args()


def ordered_models(labels: list[str]) -> list[str]:
    seen = set(labels)
    ordered = [label for label in MODEL_ORDER if label in seen]
    ordered_set = set(ordered)
    ordered.extend(label for label in labels if label not in ordered_set)
    return list(dict.fromkeys(ordered))


def style_by_model(labels: list[str], cmap: Any) -> dict[str, tuple[Any, str]]:
    return {
        label: (cmap(index % 10), MARKERS[index % len(MARKERS)])
        for index, label in enumerate(ordered_models(labels))
    }


def eval_sets(rows: list[dict[str, Any]]) -> list[str]:
    eval_order = ["ATC validation", "ATC test", "ATCOSIM test"]
    labels = [name for name in eval_order if any(row["eval_set"] == name for row in rows)]
    labels.extend(name for name in sorted({row["eval_set"] for row in rows}) if name not in labels)
    return labels


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_model_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        label = str(row["train_series"])
        metadata = MODEL_METADATA.get(label)
        if metadata is None:
            metadata = {"trainable_params": "", "width": "", "encoder_depth": "", "decoder_depth": ""}
        annotated.append(
            {
                "series": row["series"],
                "train_series": row["train_series"],
                **metadata,
                **{key: value for key, value in row.items() if key not in {"series", "train_series"}},
            }
        )
    return annotated


def plot_peak_lr_vs_wer(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = eval_sets(rows)
    models = ordered_models([str(row["train_series"]) for row in rows])
    cmap = plt.get_cmap("tab10")
    style = style_by_model(models, cmap)
    fig, axes = plt.subplots(1, len(labels), figsize=(7.0 * len(labels), 5.4), squeeze=False, constrained_layout=True)
    legend_handles = {}
    for ax, eval_set in zip(axes[0], labels, strict=True):
        pane_rows = [row for row in rows if row["eval_set"] == eval_set]
        for model in models:
            group = sorted((row for row in pane_rows if row["train_series"] == model), key=lambda row: float(row["learning_rate"]))
            if not group:
                continue
            color, marker = style[model]
            line = ax.plot(
                [float(row["learning_rate"]) for row in group],
                [float(row["wer_percent"]) for row in group],
                marker=marker,
                linewidth=1.5,
                markersize=4.8,
                color=color,
                label=model,
            )[0]
            legend_handles.setdefault(model, line)
            best = min(group, key=lambda row: float(row["wer_percent"]))
            ax.annotate(
                f"{float(best['wer_percent']):.2f}",
                (float(best["learning_rate"]), float(best["wer_percent"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_xlabel("peak learning rate")
        ax.set_title(eval_set)
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("WER (%)")
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles), loc="lower center", ncol=min(4, len(legend_handles)), fontsize=8)
    fig.suptitle("Peak Learning Rate vs WER by Model", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_peak_lr_vs_logprob(path: Path, rows: list[dict[str, Any]], *, invert_y: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = eval_sets(rows)
    models = ordered_models([str(row["train_series"]) for row in rows])
    cmap = plt.get_cmap("tab10")
    style = style_by_model(models, cmap)
    fig, axes = plt.subplots(1, len(labels), figsize=(7.0 * len(labels), 5.4), squeeze=False, constrained_layout=True)
    legend_handles = {}
    for ax, eval_set in zip(axes[0], labels, strict=True):
        pane_rows = [row for row in rows if row["eval_set"] == eval_set]
        for model in models:
            group = sorted((row for row in pane_rows if row["train_series"] == model), key=lambda row: float(row["learning_rate"]))
            if not group:
                continue
            color, marker = style[model]
            line = ax.plot(
                [float(row["learning_rate"]) for row in group],
                [float(row["mean_avg_logprob"]) for row in group],
                marker=marker,
                linewidth=1.5,
                markersize=4.8,
                color=color,
                label=model,
            )[0]
            legend_handles.setdefault(model, line)
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
        if invert_y:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("mean decode avg_logprob")
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles), loc="lower center", ncol=min(4, len(legend_handles)), fontsize=8)
    fig.suptitle("Peak Learning Rate vs Mean Decode Avg Log Probability by Model", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_logprob_vs_wer(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = eval_sets(rows)
    models = ordered_models([str(row["train_series"]) for row in rows])
    cmap = plt.get_cmap("tab10")
    style = style_by_model(models, cmap)
    fig, axes = plt.subplots(1, len(labels), figsize=(7.0 * len(labels), 5.4), squeeze=False, constrained_layout=True)
    legend_handles = {}
    for ax, eval_set in zip(axes[0], labels, strict=True):
        pane_rows = [row for row in rows if row["eval_set"] == eval_set]
        for model in models:
            group = [row for row in pane_rows if row["train_series"] == model]
            if not group:
                continue
            color, marker = style[model]
            scatter = ax.scatter(
                [float(row["mean_avg_logprob"]) for row in group],
                [float(row["wer_percent"]) for row in group],
                s=34,
                alpha=0.8,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                label=model,
            )
            legend_handles.setdefault(model, scatter)
            best = min(group, key=lambda row: float(row["wer_percent"]))
            ax.annotate(
                fmt_lr(float(best["learning_rate"])),
                (float(best["mean_avg_logprob"]), float(best["wer_percent"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        xs = [float(row["mean_avg_logprob"]) for row in pane_rows]
        ys = [float(row["wer_percent"]) for row in pane_rows]
        corr = pearson(xs, ys)
        suffix = "" if corr is None else f"  r={corr:.2f}"
        ax.set_title(f"{eval_set}{suffix}")
        ax.set_xlabel("mean avg_logprob (higher is better)")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("WER (%)")
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles), loc="lower center", ncol=min(4, len(legend_handles)), fontsize=8)
    fig.suptitle("Mean Decode Avg Log Probability vs WER by Model", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def best_rows_by_model_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["train_series"]), str(row["eval_set"]))
        if key not in best or float(row["wer_percent"]) < float(best[key]["wer_percent"]):
            best[key] = row

    labels = ordered_models([model for model, _ in best])
    eval_labels = eval_sets(rows)
    result: list[dict[str, Any]] = []
    for model in labels:
        for eval_set in eval_labels:
            row = best.get((model, eval_set))
            if row is None:
                continue
            result.append(
                {
                    "model": model,
                    "trainable_params": row["trainable_params"],
                    "width": row["width"],
                    "encoder_depth": row["encoder_depth"],
                    "decoder_depth": row["decoder_depth"],
                    "eval_set": eval_set,
                    "best_learning_rate": row["learning_rate"],
                    "best_wer_percent": row["wer_percent"],
                    "mean_avg_logprob": row["mean_avg_logprob"],
                    "run": row["run"],
                    "results_path": row["results_path"],
                }
            )
    return result


def model_order_by_trainable_params(rows: list[dict[str, Any]]) -> list[str]:
    models = ordered_models([str(row["model"]) for row in rows])
    order = {model: index for index, model in enumerate(MODEL_ORDER)}

    def key(model: str) -> tuple[float, int, str]:
        params = next((row["trainable_params"] for row in rows if row["model"] == model), "")
        if params == "":
            return (float("inf"), order.get(model, len(MODEL_ORDER)), model)
        return (float(params), order.get(model, len(MODEL_ORDER)), model)

    return sorted(models, key=key)


def format_params(value: Any) -> str:
    if value == "":
        return ""
    params = float(value)
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    return f"{params / 1_000_000:.0f}M"


def model_axis_label(row: dict[str, Any]) -> str:
    if row["width"] == "" or row["encoder_depth"] == "" or row["decoder_depth"] == "":
        return str(row["model"])
    params = format_params(row["trainable_params"])
    params_label = "" if not params else f"\n{params} params"
    return f"{row['model']}\n{row['width']}w {row['encoder_depth']}/{row['decoder_depth']}{params_label}"


def plot_best_wer_by_model(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_rows = [
        row for row in best_rows_by_model_eval(rows)
        if row["eval_set"] in BEST_WER_EVAL_SETS
    ]
    if not best_rows:
        raise SystemExit("no best rows were found")

    models = model_order_by_trainable_params(best_rows)
    eval_labels = [label for label in BEST_WER_EVAL_SETS if any(row["eval_set"] == label for row in best_rows)]
    model_to_index = {model: index for index, model in enumerate(models)}
    label_by_model = {}
    for model in models:
        row = next(item for item in best_rows if item["model"] == model)
        label_by_model[model] = model_axis_label(row)

    fig, ax = plt.subplots(figsize=(2.6 * max(3, len(models)), 5.6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for index, eval_set in enumerate(eval_labels):
        group = [row for row in best_rows if row["eval_set"] == eval_set]
        if not group:
            continue
        group = sorted(group, key=lambda row: model_to_index[str(row["model"])])
        color = cmap(index % 10)
        ax.plot(
            [model_to_index[str(row["model"])] for row in group],
            [float(row["best_wer_percent"]) for row in group],
            marker=markers[index % len(markers)],
            linewidth=1.7,
            markersize=6.0,
            color=color,
            label=eval_set,
        )
        for row in group:
            ax.annotate(
                fmt_lr(float(row["best_learning_rate"])),
                (model_to_index[str(row["model"])], float(row["best_wer_percent"])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xticks(range(len(models)), [label_by_model[model] for model in models])
    ax.set_xlabel("Model size / variant, ordered by trainable parameters")
    ax.set_ylabel("Best WER (%)")
    ax.set_title("Best WER by Model Size / Variant")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Eval set", loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    eval_index = {eval_set: index for index, eval_set in enumerate(eval_labels)}
    return sorted(
        best_rows,
        key=lambda row: (model_to_index[str(row["model"])], eval_index[str(row["eval_set"])]),
    )


def main() -> None:
    args = parse_args()
    if not args.series:
        raise SystemExit("provide at least one --series MODEL=RESULTS_JSONL")

    root = Path(".").resolve()
    rows: list[dict[str, Any]] = []
    for label, paths in parse_mapping(args.series, "--series"):
        for result_path in paths:
            rows.extend(read_points(label, result_path, root))
    rows = add_model_metadata(rows)
    rows = sorted(rows, key=lambda row: (str(row["eval_set"]), str(row["train_series"]), float(row["learning_rate"]), str(row["run"])))
    if not rows:
        raise SystemExit("no rows with WER and avg_logprob were found")

    summary_tsv = args.summary_tsv or args.out_dir / f"{args.prefix}.tsv"
    peak_lr_wer = args.out_dir / f"{args.prefix}_peak_lr_vs_wer_by_eval.png"
    peak_lr_logprob = args.out_dir / f"{args.prefix}_peak_lr_vs_mean_avg_logprob_by_eval.png"
    logprob_wer = args.out_dir / f"{args.prefix}_avg_logprob_vs_wer_by_eval.png"
    best_wer_tsv = args.out_dir / f"{args.prefix}_best_wer_by_eval.tsv"
    best_wer_by_model = args.out_dir / f"{args.prefix}_best_wer_by_eval.png"

    write_tsv(summary_tsv, rows)
    plot_peak_lr_vs_wer(peak_lr_wer, rows)
    plot_peak_lr_vs_logprob(peak_lr_logprob, rows, invert_y=not args.no_invert_logprob_y)
    plot_logprob_vs_wer(logprob_wer, rows)
    best_rows = plot_best_wer_by_model(best_wer_by_model, rows)
    write_tsv(best_wer_tsv, best_rows)
    print(f"wrote {summary_tsv}")
    print(f"wrote {peak_lr_wer}")
    print(f"wrote {peak_lr_logprob}")
    print(f"wrote {logprob_wer}")
    print(f"wrote {best_wer_tsv}")
    print(f"wrote {best_wer_by_model}")
    print(f"points {len(rows)}")


if __name__ == "__main__":
    main()
