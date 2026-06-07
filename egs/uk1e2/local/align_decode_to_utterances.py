#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align streaming decode JSONL rows to human utterance references by recording and time overlap."
    )
    parser.add_argument("--decode", type=Path, required=True, help="Input decode JSONL to read and optionally follow.")
    parser.add_argument("--refs", type=Path, required=True, help="Human utterance JSONL with recording_id/start/end/text.")
    parser.add_argument("--output", type=Path, required=True, help="Aligned JSONL output.")
    parser.add_argument("--follow", action="store_true", help="Keep polling for new decode rows after reaching EOF.")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval for --follow.")
    parser.add_argument(
        "--until-summary",
        type=Path,
        default=None,
        help="When following, exit after EOF once this summary file exists.",
    )
    parser.add_argument(
        "--nearest-gap-seconds",
        type=float,
        default=1.0,
        help="If no reference overlaps, pair the nearest reference only when the gap is within this threshold.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional decode-row limit for checks.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(text.split())


def strip_timestamps(text: str) -> str:
    return clean_text(TIMESTAMP_RE.sub("", text))


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    chars = []
    for char in text:
        if char.isalnum() or char == "'":
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def word_units(text: str) -> tuple[list[str], list[tuple[str, int]]]:
    words = text.split()
    units: list[tuple[str, int]] = []
    for word_index, word in enumerate(words):
        for unit in normalize_for_match(word).split():
            units.append((unit, word_index))
    return words, units


def meaningful_unit(unit: str) -> bool:
    return len(unit) > 2 or any(char.isdigit() for char in unit)


def normalized_word_units(word: str) -> set[str]:
    return set(normalize_for_match(word).split())


def expand_span_edges(words: list[str], hyp_units: list[str], start_word: int, end_word: int) -> tuple[int, int, bool]:
    if not words or not hyp_units:
        return start_word, end_word, False
    leading_units = {unit for unit in hyp_units[:4] if meaningful_unit(unit)}
    trailing_units = {unit for unit in hyp_units[-4:] if meaningful_unit(unit)}
    expanded = False

    for word_index in range(max(0, start_word - 4), start_word):
        if normalized_word_units(words[word_index]) & leading_units:
            start_word = word_index
            expanded = True
            break

    for word_index in range(end_word, min(len(words), end_word + 4)):
        if normalized_word_units(words[word_index]) & trailing_units:
            end_word = word_index + 1
            expanded = True

    return start_word, end_word, expanded


def best_local_ref_span(hyp_text: str, ref_text: str) -> tuple[str, dict[str, int] | None]:
    hyp_units = normalize_for_match(hyp_text).split()
    ref_words, ref_units = word_units(ref_text)
    if not hyp_units or not ref_units:
        return "", None

    width = len(ref_units) + 1
    previous_scores = [0] * width
    previous_starts = [0] * width
    best_score = 0
    best_start_unit = 0
    best_end_unit = 0
    best_span_delta = 10**9

    for hyp_index, hyp_unit in enumerate(hyp_units, start=1):
        current_scores = [0] * width
        current_starts = [0] * width
        for ref_index, (ref_unit, _) in enumerate(ref_units, start=1):
            match_score = 2 if hyp_unit == ref_unit else -2
            diagonal_score = previous_scores[ref_index - 1] + match_score
            diagonal_start = previous_starts[ref_index - 1] if previous_scores[ref_index - 1] > 0 else ref_index - 1
            up_score = previous_scores[ref_index] - 1
            up_start = previous_starts[ref_index]
            left_score = current_scores[ref_index - 1] - 1
            left_start = current_starts[ref_index - 1]

            score = 0
            start = ref_index - 1
            for candidate_score, candidate_start in (
                (diagonal_score, diagonal_start),
                (up_score, up_start),
                (left_score, left_start),
            ):
                if candidate_score > score:
                    score = candidate_score
                    start = candidate_start

            if score <= 0:
                continue
            current_scores[ref_index] = score
            current_starts[ref_index] = start
            span_delta = abs((ref_index - start) - len(hyp_units))
            if score > best_score or (score == best_score and span_delta < best_span_delta):
                best_score = score
                best_start_unit = start
                best_end_unit = ref_index - 1
                best_span_delta = span_delta
        previous_scores = current_scores
        previous_starts = current_starts

    minimum_score = max(2, min(4, len(hyp_units)))
    if best_score < minimum_score:
        return "", None
    start_word = ref_units[best_start_unit][1]
    end_word = ref_units[best_end_unit][1] + 1
    start_word, end_word, expanded = expand_span_edges(ref_words, hyp_units, start_word, end_word)
    return clean_text(" ".join(ref_words[start_word:end_word])), {
        "score": best_score,
        "ref_word_start": start_word,
        "ref_word_end": end_word,
        "hyp_words": len(hyp_units),
        "expanded_edges": int(expanded),
    }


def best_full_hyp_ref_span(hyp_text: str, ref_text: str) -> tuple[str, dict[str, float | int] | None]:
    hyp_units = normalize_for_match(hyp_text).split()
    ref_words, ref_units = word_units(ref_text)
    if not hyp_units or not ref_units:
        return "", None

    width = len(ref_units) + 1
    previous_distances = [0] * width
    previous_starts = list(range(width))
    previous_matches = [0] * width

    for hyp_index, hyp_unit in enumerate(hyp_units, start=1):
        current_distances = [hyp_index] + [0] * len(ref_units)
        current_starts = [0] * width
        current_matches = [0] * width
        for ref_index, (ref_unit, _) in enumerate(ref_units, start=1):
            is_match = hyp_unit == ref_unit
            candidates = [
                (
                    previous_distances[ref_index - 1] + (0 if is_match else 1),
                    previous_starts[ref_index - 1],
                    previous_matches[ref_index - 1] + (1 if is_match else 0),
                ),
                (previous_distances[ref_index] + 1, previous_starts[ref_index], previous_matches[ref_index]),
                (current_distances[ref_index - 1] + 1, current_starts[ref_index - 1], current_matches[ref_index - 1]),
            ]
            distance, start, matches = min(candidates, key=lambda item: (item[0], -item[2], abs((ref_index - item[1]) - len(hyp_units))))
            current_distances[ref_index] = distance
            current_starts[ref_index] = start
            current_matches[ref_index] = matches
        previous_distances = current_distances
        previous_starts = current_starts
        previous_matches = current_matches

    best_end = 0
    best_key: tuple[float, int, int] | None = None
    for ref_index in range(1, width):
        start = previous_starts[ref_index]
        span_units = max(1, ref_index - start)
        distance = previous_distances[ref_index]
        matches = previous_matches[ref_index]
        error_rate = distance / max(len(hyp_units), span_units)
        key = (error_rate, -matches, abs(span_units - len(hyp_units)))
        if best_key is None or key < best_key:
            best_key = key
            best_end = ref_index

    start_unit = previous_starts[best_end]
    span_units = max(1, best_end - start_unit)
    distance = previous_distances[best_end]
    matches = previous_matches[best_end]
    error_rate = distance / max(len(hyp_units), span_units)
    max_error_rate = 0.8 if len(hyp_units) <= 4 else 0.7
    if matches == 0 or error_rate > max_error_rate:
        return "", None
    start_word = ref_units[start_unit][1]
    end_word = ref_units[best_end - 1][1] + 1
    start_word, end_word, expanded = expand_span_edges(ref_words, hyp_units, start_word, end_word)
    return clean_text(" ".join(ref_words[start_word:end_word])), {
        "distance": distance,
        "error_rate": round(error_rate, 4),
        "matches": matches,
        "ref_word_start": start_word,
        "ref_word_end": end_word,
        "hyp_words": len(hyp_units),
        "expanded_edges": int(expanded),
    }


def proportional_text_slice(text: str, ref_start: float, ref_end: float, span_start: float, span_end: float) -> str:
    words = text.split()
    if not words:
        return ""
    duration = max(ref_end - ref_start, 1e-6)
    start_ratio = (max(span_start, ref_start) - ref_start) / duration
    end_ratio = (min(span_end, ref_end) - ref_start) / duration
    start_index = max(0, min(len(words) - 1, int(start_ratio * len(words))))
    end_index = max(start_index + 1, min(len(words), math.ceil(end_ratio * len(words))))
    return clean_text(" ".join(words[start_index:end_index]))


def counterpart_from_refs(
    hyp_text: str,
    reference_rows: list[dict[str, Any]],
    speech_start: float,
    speech_end: float,
    field: str,
) -> tuple[str, str, dict[str, float | int] | None]:
    reference_text = clean_text(" ".join(str(ref.get(field) or "") for ref in reference_rows))
    counterpart, metadata = best_full_hyp_ref_span(hyp_text, reference_text)
    if counterpart:
        return counterpart, "sequence_alignment", metadata

    counterpart, metadata = best_local_ref_span(hyp_text, reference_text)
    if counterpart:
        return counterpart, "local_alignment", metadata

    pieces = []
    for ref in reference_rows:
        ref_start = float(ref["start"])
        ref_end = float(ref["end"])
        if ref_end <= speech_start or ref_start >= speech_end:
            continue
        piece = proportional_text_slice(str(ref.get(field) or ""), ref_start, ref_end, speech_start, speech_end)
        if piece:
            pieces.append(piece)
    if pieces:
        return clean_text(" ".join(pieces)), "time_proportion", None
    return reference_text, "full_reference", None


def load_refs(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            recording_id = row.get("recording_id")
            if not recording_id:
                continue
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: reference row needs numeric start/end") from exc
            by_recording[recording_id].append(
                {
                    "id": row.get("id"),
                    "recording_id": recording_id,
                    "start": start,
                    "end": end,
                    "text": clean_text(str(row.get("normalized_text") or row.get("text") or "")),
                    "raw_text": clean_text(str(row.get("text") or "")),
                    "speaker_id": row.get("speaker_id"),
                    "utterance_id": row.get("utterance_id"),
                    "domain": row.get("domain"),
                    "source": row.get("source"),
                    "utterance_url": row.get("utterance_url"),
                }
            )
    for rows in by_recording.values():
        rows.sort(key=lambda row: (row["start"], row["end"]))
    return dict(by_recording)


def load_processed_ids(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            decode_id = row.get("decode_id") or row.get("id")
            if decode_id:
                processed.add(str(decode_id))
    return processed


def decode_speech_span(row: dict[str, Any]) -> tuple[float, float, list[float], str]:
    window_start = float(row.get("start", 0.0))
    window_end = float(row.get("end", window_start))
    text = str(row.get("text") or "")
    offsets = [float(match.group(1)) for match in TIMESTAMP_RE.finditer(text)]
    if len(offsets) >= 2:
        speech_start = window_start + offsets[0]
        speech_end = window_start + offsets[-1]
        source = "decode_timestamps"
    elif offsets:
        speech_start = window_start + offsets[0]
        speech_end = float(row.get("next_start") or window_end)
        source = "decode_start_timestamp"
    else:
        speech_start = window_start
        speech_end = float(row.get("next_start") or window_end)
        source = "decode_window"

    speech_start = max(window_start, min(speech_start, window_end))
    speech_end = max(window_start, min(speech_end, window_end))
    if speech_end <= speech_start:
        speech_end = min(window_end, max(speech_start, float(row.get("next_start") or window_end)))
    if speech_end <= speech_start:
        speech_end = window_end
        speech_start = window_start
        source = "decode_window"
    return round(speech_start, 2), round(speech_end, 2), offsets, source


def overlap_seconds(start: float, end: float, ref: dict[str, Any]) -> float:
    return max(0.0, min(end, float(ref["end"])) - max(start, float(ref["start"])))


def nearest_ref(start: float, end: float, refs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not refs:
        return None, None
    midpoint = (start + end) / 2.0
    best: dict[str, Any] | None = None
    best_gap: float | None = None
    for ref in refs:
        if midpoint < ref["start"]:
            gap = ref["start"] - midpoint
        elif midpoint > ref["end"]:
            gap = midpoint - ref["end"]
        else:
            gap = 0.0
        if best_gap is None or gap < best_gap:
            best = ref
            best_gap = gap
    return best, best_gap


def reference_view(ref: dict[str, Any], overlap: float, gap: float | None = None) -> dict[str, Any]:
    view = {
        "id": ref.get("id"),
        "start": round(float(ref["start"]), 2),
        "end": round(float(ref["end"]), 2),
        "overlap_seconds": round(overlap, 2),
        "text": ref.get("text") or "",
        "raw_text": ref.get("raw_text") or "",
        "speaker_id": ref.get("speaker_id"),
        "utterance_id": ref.get("utterance_id"),
        "domain": ref.get("domain"),
        "source": ref.get("source"),
        "utterance_url": ref.get("utterance_url"),
    }
    if gap is not None:
        view["gap_seconds"] = round(gap, 2)
    return view


def align_row(
    row: dict[str, Any],
    refs_by_recording: dict[str, list[dict[str, Any]]],
    nearest_gap_seconds: float,
) -> dict[str, Any]:
    recording_id = str(row.get("recording_id") or "")
    refs = refs_by_recording.get(recording_id, [])
    speech_start, speech_end, timestamp_offsets, span_source = decode_speech_span(row)
    span_duration = max(0.0, speech_end - speech_start)

    matched: list[tuple[dict[str, Any], float]] = []
    for ref in refs:
        if ref["end"] <= speech_start:
            continue
        if ref["start"] >= speech_end:
            break
        overlap = overlap_seconds(speech_start, speech_end, ref)
        if overlap > 0.0:
            matched.append((ref, overlap))

    match_kind = "overlap" if matched else "unmatched"
    gap_seconds: float | None = None
    if not matched and nearest_gap_seconds >= 0.0:
        near, gap_seconds = nearest_ref(speech_start, speech_end, refs)
        if near is not None and gap_seconds is not None and gap_seconds <= nearest_gap_seconds:
            matched.append((near, 0.0))
            match_kind = "nearest"

    reference_rows = [
        reference_view(ref, overlap, gap_seconds if match_kind == "nearest" else None)
        for ref, overlap in matched
    ]
    primary = max(reference_rows, key=lambda item: item["overlap_seconds"], default=None)
    reference_text = clean_text(" ".join(ref["text"] for ref in reference_rows if ref.get("text")))
    reference_raw_text = clean_text(" ".join(ref["raw_text"] for ref in reference_rows if ref.get("raw_text")))
    total_overlap = sum(ref["overlap_seconds"] for ref in reference_rows)
    hyp_text = clean_text(str(row.get("text_no_timestamps") or strip_timestamps(str(row.get("text") or ""))))
    reference_counterpart, counterpart_method, counterpart_metadata = counterpart_from_refs(
        hyp_text,
        reference_rows,
        speech_start,
        speech_end,
        "text",
    )
    reference_raw_counterpart, _, _ = counterpart_from_refs(
        hyp_text,
        reference_rows,
        speech_start,
        speech_end,
        "raw_text",
    )

    return {
        "decode_id": row.get("id"),
        "utt_id": row.get("utt_id"),
        "recording_id": recording_id,
        "window_index": row.get("window_index"),
        "alternative": row.get("alternative", row.get("i")),
        "path": row.get("path"),
        "source": row.get("source"),
        "decode_start": row.get("start"),
        "decode_end": row.get("end"),
        "speech_start": speech_start,
        "speech_end": speech_end,
        "speech_span_source": span_source,
        "timestamp_offsets": timestamp_offsets,
        "hyp_text": hyp_text,
        "avg_logprob": row.get("avg_logprob"),
        "no_speech_prob": row.get("no_speech_prob"),
        "next_start": row.get("next_start"),
        "reference_text": reference_text,
        "reference_counterpart": reference_counterpart,
        "reference_counterpart_method": counterpart_method,
        "reference_counterpart_metadata": counterpart_metadata,
        "reference_raw_text": reference_raw_text,
        "reference_raw_counterpart": reference_raw_counterpart,
        "reference_source_field": "normalized_text",
        "reference_ids": [ref["id"] for ref in reference_rows],
        "primary_reference_id": primary["id"] if primary else None,
        "match_kind": match_kind,
        "overlap_seconds": round(total_overlap, 2),
        "coverage": round(total_overlap / span_duration, 4) if span_duration else 0.0,
        "references": reference_rows,
    }


def process_decode_line(
    line: str,
    *,
    refs_by_recording: dict[str, list[dict[str, Any]]],
    out_f,
    processed_ids: set[str],
    nearest_gap_seconds: float,
) -> bool:
    row = json.loads(line)
    decode_id = row.get("id")
    if decode_id and str(decode_id) in processed_ids:
        return False
    aligned = align_row(row, refs_by_recording, nearest_gap_seconds)
    out_f.write(json.dumps(aligned, ensure_ascii=False) + "\n")
    out_f.flush()
    if decode_id:
        processed_ids.add(str(decode_id))
    return True


def main() -> None:
    args = parse_args()
    refs_by_recording = load_refs(args.refs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(args.output)

    written = 0
    with args.decode.open("r", encoding="utf-8") as in_f, args.output.open("a", encoding="utf-8") as out_f:
        while True:
            position = in_f.tell()
            line = in_f.readline()
            if not line:
                if not args.follow:
                    break
                if args.until_summary is not None and args.until_summary.exists():
                    break
                time.sleep(args.poll_seconds)
                continue
            if args.follow and not line.endswith("\n"):
                in_f.seek(position)
                time.sleep(args.poll_seconds)
                continue
            if not line.strip():
                continue
            if process_decode_line(
                line,
                refs_by_recording=refs_by_recording,
                out_f=out_f,
                processed_ids=processed_ids,
                nearest_gap_seconds=args.nearest_gap_seconds,
            ):
                written += 1
            if args.limit is not None and written >= args.limit:
                break


if __name__ == "__main__":
    main()
