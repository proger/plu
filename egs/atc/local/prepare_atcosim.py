#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset

from prepare_atc import labels_for_text, safe_id, wav_info, write_jsonl
from plu.tokenizer import get_tokenizer


DATASET_NAME = "Jzuluaga/atcosim_corpus"
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally prepare ATCOSIM data for the ATC recipe.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used for relative paths.")
    parser.add_argument("--out-dir", type=Path, default=Path("egs/atc/data/atcosim"))
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-languages", type=int, default=100)
    parser.add_argument("--model-max-target-positions", type=int, default=448)
    parser.add_argument("--max-duration", type=float, default=30.0, help="Skip rows longer than this many seconds; 0 disables filtering.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Keep source transcript case. By default transcripts are uppercased to match egs/atc/data.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path, root)


def normalize_text(text: str, preserve_case: bool) -> str:
    text = " ".join(text.split())
    return text if preserve_case else text.upper()


def prepare_split(args: argparse.Namespace, split: str, root: Path, out_dir: Path, tokenizer: Any) -> dict[str, Any]:
    dataset = load_dataset(args.dataset, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    wav_dir = out_dir / "wav" / split
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    wav_list: list[str] = []
    refs: list[str] = []
    stats: Counter[str] = Counter()
    duration_seconds = 0.0
    max_duration = 0.0
    max_label_tokens = 0
    non_16k_mono_examples: list[dict[str, Any]] = []

    for index, example in enumerate(dataset):
        stats["rows_seen"] += 1
        utt_id = str(example["id"])
        text = normalize_text(str(example["text"]), args.preserve_case)
        audio = example["audio"]
        payload = audio.get("bytes") if isinstance(audio, dict) else None
        if not payload:
            raise ValueError(f"{split}:{index}: audio bytes are missing for {utt_id}")

        sample_rate, channels, duration = wav_info(payload)
        if args.max_duration > 0 and duration > args.max_duration:
            stats["rows_too_long"] += 1
            continue
        if sample_rate != 16000 or channels != 1:
            stats["non_16k_mono"] += 1
            if len(non_16k_mono_examples) < 20:
                non_16k_mono_examples.append({"id": utt_id, "sample_rate": sample_rate, "channels": channels})

        wav_path = wav_dir / f"{index:05d}_{safe_id(utt_id)}.wav"
        wav_path.write_bytes(payload)
        input_ids = labels_for_text(tokenizer, text)
        if any(token >= tokenizer.timestamp_begin for token in input_ids):
            raise ValueError(f"{split}:{index}: timestamp token leaked into no-timestamp labels for {utt_id}")
        if len(input_ids) > args.model_max_target_positions:
            stats["rows_too_long"] += 1
            raise ValueError(
                f"{split}:{index}: {len(input_ids)} label tokens exceed model max target positions "
                f"{args.model_max_target_positions}: {utt_id}"
            )

        path = rel(wav_path, root)
        start = float(example.get("segment_start_time") or 0.0)
        source_duration = float(example.get("duration") or duration)
        end = float(example.get("segment_end_time") or (start + source_duration))
        row = {
            "id": utt_id,
            "source_dataset": args.dataset,
            "split": split,
            "path": path,
            "audio_path": path,
            "text": text,
            "language": args.language,
            "start": 0.0,
            "duration": round(duration, 6),
            "end": round(duration, 6),
            "source_segment_start_time": round(start, 6),
            "source_segment_end_time": round(end, 6),
            "source_duration": round(source_duration, 6),
            "input_ids": input_ids,
        }
        rows.append(row)
        wav_list.append(path)
        refs.append(f"{utt_id}\t{path}\t{text}")
        duration_seconds += duration
        max_duration = max(max_duration, duration)
        max_label_tokens = max(max_label_tokens, len(input_ids))
        stats["rows_written"] += 1

    write_jsonl(out_dir / f"{split}.jsonl", rows)
    (out_dir / f"{split}.wav.list").write_text("\n".join(wav_list) + "\n", encoding="utf-8")
    (out_dir / f"{split}.refs.tsv").write_text("\n".join(refs) + "\n", encoding="utf-8")

    return {
        "split": split,
        "rows": len(rows),
        "audio_seconds": duration_seconds,
        "audio_hours": duration_seconds / 3600.0,
        "max_duration_seconds": max_duration,
        "max_label_tokens": max_label_tokens,
        "non_16k_mono_count": stats["non_16k_mono"],
        "non_16k_mono_examples": non_16k_mono_examples,
        "stats": dict(stats),
        "jsonl": rel(out_dir / f"{split}.jsonl", root),
        "wav_list": rel(out_dir / f"{split}.wav.list", root),
        "refs_tsv": rel(out_dir / f"{split}.refs.tsv", root),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(multilingual=True, language=args.language, task="transcribe", num_languages=args.num_languages)
    split_manifests = [prepare_split(args, split, root, out_dir, tokenizer) for split in args.splits]
    manifest = {
        "dataset": args.dataset,
        "dataset_url": f"https://huggingface.co/datasets/{args.dataset}",
        "language": args.language,
        "num_languages": args.num_languages,
        "label_format": "SOT, language, transcribe, notimestamps, text tokens, EOT",
        "uses_timestamp_tokens": False,
        "text_case": "source" if args.preserve_case else "upper",
        "splits": split_manifests,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
