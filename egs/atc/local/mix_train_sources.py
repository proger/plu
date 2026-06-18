#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed ATC train JSONL from prepared source JSONL files.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used for relative paths.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wav-list", type=Path, default=None)
    parser.add_argument("--refs-tsv", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--order",
        choices=("hash", "input"),
        default="hash",
        help="Use stable hash order by default so sources are mixed without RNG state.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path, root)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_sort_key(row: dict[str, Any]) -> str:
    source = str(row.get("source_dataset", ""))
    split = str(row.get("split", ""))
    utt_id = str(row.get("id", ""))
    return hashlib.sha256(f"{source}\0{split}\0{utt_id}".encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_path = args.out if args.out.is_absolute() else root / args.out
    wav_list_path = args.wav_list if args.wav_list is not None else out_path.with_suffix(".wav.list")
    refs_path = args.refs_tsv if args.refs_tsv is not None else out_path.with_suffix(".refs.tsv")
    manifest_path = args.manifest if args.manifest is not None else out_path.with_suffix(".manifest.json")
    wav_list_path = wav_list_path if wav_list_path.is_absolute() else root / wav_list_path
    refs_path = refs_path if refs_path.is_absolute() else root / refs_path
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path

    rows: list[dict[str, Any]] = []
    input_manifests: list[dict[str, Any]] = []
    for input_index, input_path_arg in enumerate(args.inputs):
        input_path = input_path_arg if input_path_arg.is_absolute() else root / input_path_arg
        source_rows = read_jsonl(input_path)
        for source_row_index, row in enumerate(source_rows):
            row = dict(row)
            row["mix_input_index"] = input_index
            row["mix_source_row_index"] = source_row_index
            rows.append(row)
        input_manifests.append({"path": rel(input_path, root), "rows": len(source_rows)})

    if args.order == "hash":
        rows.sort(key=lambda row: (row_sort_key(row), int(row["mix_input_index"]), int(row["mix_source_row_index"])))

    seen_keys: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    source_audio_seconds: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        source = str(row.get("source_dataset", ""))
        key = (source, str(row.get("id", "")))
        if key in seen_keys:
            raise ValueError(f"duplicate source/id in mixed data: {key}")
        seen_keys.add(key)
        source_counts[source] += 1
        source_audio_seconds[source] += float(row.get("duration", 0.0))

    write_jsonl(out_path, rows)
    wav_list_path.parent.mkdir(parents=True, exist_ok=True)
    wav_list_path.write_text("\n".join(str(row["path"]) for row in rows) + "\n", encoding="utf-8")
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text(
        "\n".join(f"{row['id']}\t{row['path']}\t{row['text']}" for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": input_manifests,
        "order": args.order,
        "rows": len(rows),
        "audio_seconds": sum(source_audio_seconds.values()),
        "audio_hours": sum(source_audio_seconds.values()) / 3600.0,
        "sources": [
            {
                "source_dataset": source,
                "rows": source_counts[source],
                "audio_seconds": source_audio_seconds[source],
                "audio_hours": source_audio_seconds[source] / 3600.0,
            }
            for source in sorted(source_counts)
        ],
        "jsonl": rel(out_path, root),
        "wav_list": rel(wav_list_path, root),
        "refs_tsv": rel(refs_path, root),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
