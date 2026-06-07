from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from plu.tokenizer import get_tokenizer


SEGMENT_RE = re.compile(r"-(\d{7})-(\d{7})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small UK1E2 PLU JSONL subset.")
    parser.add_argument("--wav-scp", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Repository root used to resolve relative wav.scp paths.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--id-prefix", default="N", help="Select utterances whose second id field starts with this prefix.")
    parser.add_argument("--language", default="uk")
    parser.add_argument("--num-languages", type=int, default=100)
    return parser.parse_args()


def display_path(path: Path, root: Path) -> str:
    if path.is_absolute():
        return os.path.relpath(path, root)
    return path.as_posix()


def load_wavs(path: Path, root: Path) -> dict[str, tuple[str, Path]]:
    wavs = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2:
                continue
            utt_id, wav_path = fields
            wav = Path(wav_path)
            resolved = wav if wav.is_absolute() else root / wav
            wavs[utt_id] = (display_path(wav, root), resolved)
    return wavs


def load_text(path: Path) -> dict[str, str]:
    texts = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2:
                continue
            utt_id, text = fields[0], fields[1].strip()
            if text:
                texts[utt_id] = text
    return texts


def segment_duration(utt_id: str) -> float | None:
    match = SEGMENT_RE.search(utt_id)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return max(0.0, (end - start) / 100.0)


def input_ids_for_text(text: str, language: str, num_languages: int) -> list[int]:
    tokenizer = get_tokenizer(multilingual=True, language=language, task="transcribe", num_languages=num_languages)
    tokens = tokenizer.encode(text)
    return [*tokenizer.sot_sequence_including_notimestamps, *tokens, tokenizer.eot]


def matches_id_prefix(utt_id: str, prefix: str) -> bool:
    if not prefix:
        return True
    fields = utt_id.split("-")
    return len(fields) > 1 and fields[1].startswith(prefix)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    wavs = load_wavs(args.wav_scp, root)
    texts = load_text(args.text)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for utt_id, (wav_path, resolved_wav) in wavs.items():
        if not matches_id_prefix(utt_id, args.id_prefix):
            continue
        text = texts.get(utt_id)
        if text is None or not resolved_wav.exists():
            continue
        rows.append(
            {
                "id": utt_id,
                "path": wav_path,
                "text": text,
                "language": args.language,
                "duration": segment_duration(utt_id),
                "input_ids": input_ids_for_text(text, args.language, args.num_languages),
            }
        )
        if len(rows) >= args.limit:
            break

    if len(rows) < args.limit:
        raise RuntimeError(f"only prepared {len(rows)} rows, requested {args.limit}")

    subset_jsonl = args.out_dir / "subset.jsonl"
    wav_list = args.out_dir / "wav.list"
    refs_tsv = args.out_dir / "refs.tsv"

    with subset_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with wav_list.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["path"] + "\n")

    with refs_tsv.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']}\t{row['path']}\t{row['text']}\n")

    print(json.dumps({"subset_jsonl": str(subset_jsonl), "wav_list": str(wav_list), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
