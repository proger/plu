#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from plu.tokenizer import get_tokenizer


DEFAULT_DECODE = Path("egs/uk1e2/exp/context_sample_b8_multisent_last_taildyn_20260608_113521/decode.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UK1E2 pseudo-labels from selected decode JSONL rows.")
    parser.add_argument("--decode", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root used to resolve and write relative paths.")
    parser.add_argument("--out-dir", type=Path, default=Path("egs/uk1e2/data"))
    parser.add_argument("--prefix", default="best_decoded")
    parser.add_argument("--language", default=None, help="Language code. Defaults to decode.meta.json or row language.")
    parser.add_argument("--num-languages", type=int, default=100)
    parser.add_argument("--max-label-tokens", type=int, default=448)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--align", type=Path, default=None, help="Alignment JSONL keyed by decode_id; non-overlap rows are skipped.")
    parser.add_argument("--allow-rejected", action="store_true", help="Keep selected rows even when sample_accepted is false.")
    parser.add_argument("--allow-empty", action="store_true", help="Keep selected rows with no text tokens.")
    parser.add_argument("--allow-missing-alignment", action="store_true", help="Keep selected rows missing from --align.")
    parser.add_argument("--allow-alignment-failed", action="store_true", help="Keep selected rows whose alignment match_kind is not overlap.")
    parser.add_argument("--allow-rank1-fallback", action="store_true", help="Use sample_rank == 1 when sample_selected is absent.")
    parser.add_argument("--keep-context-prompt", action="store_true", help="Keep previous-utterance conditioning tokens in labels.")
    parser.add_argument("--text-only", action="store_true", help="Retokenize text without timestamps instead of using decoded token ids.")
    return parser.parse_args()


def read_meta(decode: Path) -> dict[str, Any]:
    meta_path = decode.with_name("decode.meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_alignments(path: Path | None, root: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else root / path
    alignments: dict[str, dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            decode_id = row.get("decode_id") or row.get("id")
            if not decode_id:
                raise ValueError(f"{resolved}:{line_number}: alignment row has no decode_id")
            alignments[str(decode_id)] = row
    return alignments


def display_path(path: Path, root: Path) -> str:
    if path.is_absolute():
        return os.path.relpath(path, root)
    return path.as_posix()


def window_command(path: str, start: float, duration: float) -> str:
    return (
        f"ffmpeg -nostdin -ss {start:.2f} -t {duration:.2f} -i {shlex.quote(path)} "
        "-f wav -acodec pcm_s16le -ar 16000 -ac 1 - |"
    )


def clean_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("text_no_timestamps") or row.get("text") or "").split())


def base_prompt(tokenizer: Any, *, timestamps: bool) -> list[int]:
    if timestamps:
        return list(tokenizer.sot_sequence)
    return list(tokenizer.sot_sequence_including_notimestamps)


def labels_for_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    timestamps: bool,
    keep_context_prompt: bool,
    text_only: bool,
) -> list[int]:
    if text_only:
        return [*base_prompt(tokenizer, timestamps=False), *tokenizer.encode(clean_text(row)), tokenizer.eot]

    input_ids = [int(token) for token in row.get("input_ids") or []]
    if keep_context_prompt:
        return input_ids

    prompt_length = int(row.get("prompt_length") or 0)
    generated = input_ids[prompt_length:] if prompt_length > 0 else input_ids
    return [*base_prompt(tokenizer, timestamps=timestamps), *generated]


def selected_row(row: dict[str, Any], *, allow_rank1_fallback: bool) -> bool:
    if "sample_selected" in row:
        return row.get("sample_selected") is True
    return allow_rank1_fallback and row.get("sample_rank") == 1


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    decode = args.decode if args.decode.is_absolute() else root / args.decode
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta(decode)
    language = args.language or meta.get("language") or "uk"
    timestamps = bool(meta.get("timestamps_after_sentence_end")) and not args.text_only
    tokenizer = get_tokenizer(multilingual=True, language=language, task="transcribe", num_languages=args.num_languages)
    alignments = load_alignments(args.align, root)

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    recordings: set[str] = set()
    total_audio_seconds = 0.0

    with decode.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["decode_rows"] += 1
            row = json.loads(line)
            if not selected_row(row, allow_rank1_fallback=args.allow_rank1_fallback):
                stats["skipped_not_selected"] += 1
                continue
            stats["selected_rows"] += 1
            if not args.allow_rejected and row.get("sample_accepted") is not True:
                stats["skipped_rejected"] += 1
                continue

            decode_id = str(row.get("id") or "")
            alignment = alignments.get(decode_id) if alignments else None
            if alignments:
                if alignment is None:
                    stats["skipped_missing_alignment"] += 1
                    if not args.allow_missing_alignment:
                        continue
                elif alignment.get("match_kind") != "overlap":
                    stats["skipped_alignment_failed"] += 1
                    if not args.allow_alignment_failed:
                        continue

            text = clean_text(row)
            text_token_ids = row.get("text_token_ids") or []
            if not args.allow_empty and (not text or not text_token_ids):
                stats["skipped_empty"] += 1
                continue

            source_path = Path(str(row.get("path") or row.get("source") or ""))
            if not source_path.as_posix():
                stats["skipped_missing_path"] += 1
                continue
            resolved_path = source_path if source_path.is_absolute() else root / source_path
            if not resolved_path.exists():
                stats["skipped_missing_path"] += 1
                continue

            start = float(row.get("start") or 0.0)
            duration = float(row.get("duration") or max(0.0, float(row.get("end") or start) - start))
            if duration <= 0.0:
                stats["skipped_nonpositive_duration"] += 1
                continue

            input_ids = labels_for_row(
                row,
                tokenizer,
                timestamps=timestamps,
                keep_context_prompt=args.keep_context_prompt,
                text_only=args.text_only,
            )
            if not input_ids:
                stats["skipped_empty_labels"] += 1
                continue
            if args.max_label_tokens > 0 and len(input_ids) > args.max_label_tokens:
                stats["skipped_too_long"] += 1
                continue

            path = display_path(source_path, root)
            recording_id = str(row.get("recording_id") or "")
            if recording_id:
                recordings.add(recording_id)
            total_audio_seconds += duration
            stats["written_rows"] += 1
            stats["context_prompt_stripped"] += int(not args.keep_context_prompt and int(row.get("context_token_count") or 0) > 0)

            rows.append(
                {
                    "id": row.get("utt_id") or row.get("id") or f"decode-{line_number}",
                    "decode_id": row.get("id"),
                    "recording_id": recording_id,
                    "window_index": row.get("window_index"),
                    "path": path,
                    "start": round(start, 2),
                    "end": round(start + duration, 2),
                    "duration": round(duration, 2),
                    "text": text,
                    "timestamped_text": row.get("text") or "",
                    "language": language,
                    "input_ids": input_ids,
                    "source_decode": display_path(args.decode, root),
                    "alternative": row.get("alternative"),
                    "sample_score": row.get("sample_score"),
                    "avg_logprob": row.get("avg_logprob"),
                    "sample_penalty": row.get("sample_penalty"),
                    "sample_accepted": row.get("sample_accepted"),
                    "sample_reject_reasons": row.get("sample_reject_reasons") or [],
                    "alignment_match_kind": alignment.get("match_kind") if alignment else None,
                    "alignment_coverage": alignment.get("coverage") if alignment else None,
                    "reference_counterpart": alignment.get("reference_counterpart") if alignment else None,
                    "reference_ids": alignment.get("reference_ids") if alignment else [],
                    "text_token_count": len(text_token_ids),
                    "label_token_count": len(input_ids),
                }
            )

    train_jsonl = out_dir / f"{args.prefix}.jsonl"
    eval_jsonl = out_dir / f"{args.prefix}_eval.jsonl"
    refs_tsv = out_dir / f"{args.prefix}.refs.tsv"
    wav_scp = out_dir / f"{args.prefix}.wav.scp"
    manifest_json = out_dir / f"{args.prefix}.manifest.json"

    with train_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with eval_jsonl.open("w", encoding="utf-8") as f:
        for row in rows[: args.eval_limit]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with refs_tsv.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']}\t{row['path']}\t{row['start']:.2f}\t{row['duration']:.2f}\t{row['text']}\n")

    with wav_scp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']} {window_command(str(row['path']), float(row['start']), float(row['duration']))}\n")

    manifest = {
        "decode": display_path(args.decode, root),
        "decode_meta": meta,
        "train_jsonl": display_path(train_jsonl, root),
        "eval_jsonl": display_path(eval_jsonl, root),
        "refs_tsv": display_path(refs_tsv, root),
        "wav_scp": display_path(wav_scp, root),
        "align": display_path(args.align, root) if args.align is not None else None,
        "alignments": len(alignments),
        "language": language,
        "timestamps_in_labels": timestamps,
        "text_only": args.text_only,
        "keep_context_prompt": args.keep_context_prompt,
        "max_label_tokens": args.max_label_tokens,
        "eval_limit": args.eval_limit,
        "recordings": len(recordings),
        "total_audio_seconds": round(total_audio_seconds, 3),
        "total_audio_hours": round(total_audio_seconds / 3600.0, 3),
        "stats": dict(stats),
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
