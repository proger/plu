#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic ATC train subsamples.")
    parser.add_argument("--input", type=Path, default=Path("egs/atc/data/train.jsonl"))
    parser.add_argument("--out-prefix", type=Path, default=Path("egs/atc/data/train_subsample"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}pct"
    return f"{percent:g}pct".replace(".", "p")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows found in {args.input}")
    for fraction in args.fractions:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"fractions must be in (0, 1], got {fraction}")

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    manifest = {
        "input": str(args.input),
        "source_rows": len(rows),
        "seed": args.seed,
        "subsamples": [],
    }

    for fraction in sorted(args.fractions):
        count = max(1, round(len(rows) * fraction))
        selected = sorted(indices[:count])
        subset = [dict(rows[index], split=f"train_subsample_{pct_label(fraction)}") for index in selected]
        out_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{pct_label(fraction)}.jsonl")
        write_jsonl(out_path, subset)
        manifest["subsamples"].append(
            {
                "fraction": fraction,
                "rows": len(subset),
                "jsonl": str(out_path),
            }
        )
        print(f"wrote {out_path} ({len(subset)} rows)")

    manifest_path = args.out_prefix.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
