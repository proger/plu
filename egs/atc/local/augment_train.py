#!/usr/bin/env python3

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from audiomentations import AddGaussianNoise, BandPassFilter, Gain, PitchShift, SomeOf, TimeStretch
from tqdm import tqdm


TARGET_SR = 16000
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 4x ATC train set with offline audio augmentations.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used for relative paths.")
    parser.add_argument("--input", type=Path, default=Path("egs/atc/data/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("egs/atc/data/train_aug4.jsonl"))
    parser.add_argument("--wav-dir", type=Path, default=Path("egs/atc/data/wav/train_aug4"))
    parser.add_argument("--wav-list", type=Path, default=None)
    parser.add_argument("--refs-tsv", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--augment-per-row", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true", help="Rewrite existing augmented WAVs.")
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_augmenter() -> SomeOf:
    return SomeOf(
        (2, 3),
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.003, p=1.0),
            BandPassFilter(min_center_freq=400.0, max_center_freq=3000.0, p=1.0),
            Gain(min_gain_db=-3.0, max_gain_db=3.0, p=1.0),
            TimeStretch(min_rate=0.97, max_rate=1.03, p=0.5),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
        ],
        p=1.0,
    )


def augmented_row(source: dict[str, Any], path: str, aug_index: int, duration: float) -> dict[str, Any]:
    row = dict(source)
    row["id"] = f"{source['id']}-aug{aug_index}"
    row["path"] = path
    row["audio_path"] = path
    row["split"] = "train_aug4"
    row["duration"] = round(duration, 6)
    row["end"] = round(duration, 6)
    row["augmentation"] = {
        "source_id": source["id"],
        "source_path": source["path"],
        "index": aug_index,
        "recipe": "SomeOf(2,3): AddGaussianNoise, BandPassFilter, Gain, TimeStretch, PitchShift",
    }
    return row


def original_row(source: dict[str, Any]) -> dict[str, Any]:
    row = dict(source)
    row["split"] = "train_aug4"
    row["augmentation"] = {"source_id": source["id"], "index": 0, "recipe": "original"}
    return row


def main() -> None:
    args = parse_args()
    if args.augment_per_row < 0:
        raise ValueError("--augment-per-row must be non-negative")
    root = args.root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    out_path = args.out if args.out.is_absolute() else root / args.out
    wav_dir = args.wav_dir if args.wav_dir.is_absolute() else root / args.wav_dir
    wav_list_path = args.wav_list or out_path.with_suffix(".wav.list")
    refs_path = args.refs_tsv or out_path.with_suffix(".refs.tsv")
    manifest_path = args.manifest or out_path.with_suffix(".manifest.json")
    wav_list_path = wav_list_path if wav_list_path.is_absolute() else root / wav_list_path
    refs_path = refs_path if refs_path.is_absolute() else root / refs_path
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path

    random.seed(args.seed)
    np.random.seed(args.seed)
    augmenter = make_augmenter()
    source_rows = read_jsonl(input_path)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for source_index, source in enumerate(tqdm(source_rows, desc="augmenting ATC train")):
        rows.append(original_row(source))
        stats["original_rows"] += 1

        audio_path = root / source["path"]
        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        for aug_index in range(1, args.augment_per_row + 1):
            aug_audio = augmenter(samples=audio, sample_rate=TARGET_SR)
            aug_audio = np.asarray(aug_audio, dtype=np.float32)
            aug_name = f"{source_index:05d}_{source['id']}_aug{aug_index}.wav"
            aug_path = wav_dir / aug_name
            if args.force or not aug_path.exists():
                sf.write(aug_path, aug_audio, TARGET_SR, subtype="PCM_16")
            duration = float(len(aug_audio)) / TARGET_SR
            rows.append(augmented_row(source, rel(aug_path, root), aug_index, duration))
            stats["augmented_rows"] += 1

    write_jsonl(out_path, rows)
    wav_list_path.parent.mkdir(parents=True, exist_ok=True)
    wav_list_path.write_text("\n".join(str(row["path"]) for row in rows) + "\n", encoding="utf-8")
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text("\n".join(f"{row['id']}\t{row['path']}\t{row['text']}" for row in rows) + "\n", encoding="utf-8")

    audio_seconds = sum(float(row.get("duration", 0.0)) for row in rows)
    manifest = {
        "input": rel(input_path, root),
        "rows": len(rows),
        "source_rows": len(source_rows),
        "augment_per_row": args.augment_per_row,
        "sample_multiplier": len(rows) / max(1, len(source_rows)),
        "audio_seconds": audio_seconds,
        "audio_hours": audio_seconds / 3600.0,
        "seed": args.seed,
        "target_sample_rate": TARGET_SR,
        "augmentation_source": "https://raw.githubusercontent.com/jack-tol/atc-asr-dataset-preparation-toolkit/refs/heads/main/utils/offline_data_augmentation.py",
        "augmentation_recipe": {
            "SomeOf": [2, 3],
            "transforms": [
                "AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.003, p=1.0)",
                "BandPassFilter(min_center_freq=400.0, max_center_freq=3000.0, p=1.0)",
                "Gain(min_gain_db=-3.0, max_gain_db=3.0, p=1.0)",
                "TimeStretch(min_rate=0.97, max_rate=1.03, p=0.5)",
                "PitchShift(min_semitones=-1, max_semitones=1, p=0.3)",
            ],
        },
        "stats": dict(stats),
        "jsonl": rel(out_path, root),
        "wav_list": rel(wav_list_path, root),
        "refs_tsv": rel(refs_path, root),
        "wav_dir": rel(wav_dir, root),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
