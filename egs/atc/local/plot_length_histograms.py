#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_SPLITS = (
    ("ATC validation", Path("egs/atc/data/validation.jsonl")),
    ("ATCOSIM test", Path("egs/atc/data/atcosim/test.jsonl")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ATC utterance length histograms.")
    parser.add_argument("--out", type=Path, default=Path("egs/atc/exp/utterance_length_histograms.png"))
    parser.add_argument("--stats-tsv", type=Path, default=Path("egs/atc/exp/utterance_length_histograms.tsv"))
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(name: str, token_lengths: list[int], durations: list[float]) -> dict[str, Any]:
    return {
        "split": name,
        "utterances": len(token_lengths),
        "token_min": min(token_lengths),
        "token_p50": percentile([float(value) for value in token_lengths], 50),
        "token_mean": statistics.fmean(token_lengths),
        "token_p95": percentile([float(value) for value in token_lengths], 95),
        "token_max": max(token_lengths),
        "seconds_min": min(durations),
        "seconds_p50": percentile(durations, 50),
        "seconds_mean": statistics.fmean(durations),
        "seconds_p95": percentile(durations, 95),
        "seconds_max": max(durations),
        "total_seconds": sum(durations),
        "total_hours": sum(durations) / 3600.0,
    }


def write_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(out: Path, datasets: list[tuple[str, list[int], list[float]]], bins: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ("#376996", "#d95f02")

    for column, (name, token_lengths, durations) in enumerate(datasets):
        ax = axes[0][column]
        ax.hist(token_lengths, bins=bins, color=colors[column], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(statistics.fmean(token_lengths), color="black", linestyle="--", linewidth=1.0, label="mean")
        ax.axvline(statistics.median(token_lengths), color="black", linestyle=":", linewidth=1.0, label="median")
        ax.set_title(f"{name}: tokens")
        ax.set_xlabel("input_ids per utterance")
        ax.set_ylabel("utterances")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1][column]
        ax.hist(durations, bins=bins, color=colors[column], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(statistics.fmean(durations), color="black", linestyle="--", linewidth=1.0, label="mean")
        ax.axvline(statistics.median(durations), color="black", linestyle=":", linewidth=1.0, label="median")
        ax.set_title(f"{name}: seconds")
        ax.set_xlabel("duration seconds")
        ax.set_ylabel("utterances")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("ATC Utterance Length Distributions")
    fig.savefig(out, dpi=180)


def main() -> None:
    args = parse_args()
    datasets = []
    stats = []
    for name, path in DEFAULT_SPLITS:
        rows = read_jsonl(path)
        token_lengths = [len(row.get("input_ids", [])) for row in rows]
        durations = [float(row["duration"]) for row in rows]
        if not token_lengths or not durations:
            raise SystemExit(f"No length data found in {path}")
        datasets.append((name, token_lengths, durations))
        stats.append(summarize(name, token_lengths, durations))

    make_plot(args.out, datasets, args.bins)
    write_stats(args.stats_tsv, stats)
    print(f"wrote {args.out}")
    print(f"wrote {args.stats_tsv}")
    for row in stats:
        print(
            f"{row['split']}: n={row['utterances']} "
            f"tokens mean={row['token_mean']:.2f} p95={row['token_p95']:.1f} max={row['token_max']} "
            f"seconds mean={row['seconds_mean']:.2f} p95={row['seconds_p95']:.2f} max={row['seconds_max']:.2f}"
        )


if __name__ == "__main__":
    main()
