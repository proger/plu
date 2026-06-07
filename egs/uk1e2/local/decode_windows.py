#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plu.test as plu_test
from benchmarks.bench_end_to_end import pack_mx_model
from plu.train_data import SAMPLE_RATE, _decode_pcm, _resample, load_audio, log_mel_spectrogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a wav.scp in fixed windows with PLU.")
    parser.add_argument("--wav-scp", type=Path, default=Path("data/local/wav.scp"))
    parser.add_argument("--out-dir", type=Path, default=Path("egs/uk1e2/exp/decode"))
    parser.add_argument(
        "--exp",
        type=Path,
        default=Path("egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8/train"),
        help="PLU model directory to decode with.",
    )
    parser.add_argument("--language", default="uk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--hop-seconds", type=float, default=15.0)
    parser.add_argument("--min-window-seconds", type=float, default=1.0)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument(
        "--timestamps-after-sentence-end",
        action="store_true",
        help="Force a Whisper timestamp token after '.', '!', or '?' in each decoded sequence.",
    )
    parser.add_argument(
        "--sentence-hop",
        action="store_true",
        help="Advance each 30s window by one discovered sentence-end timestamp instead of a fixed hop.",
    )
    parser.add_argument(
        "--condition-on-previous-utterance",
        action="store_true",
        help="Prefix each decode with the previous utterance text using Whisper's <|startofprev|> prompt format.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional source-recording limit for smoke tests.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_wav_scp(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recording_id, source = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: expected '<recording-id> <source>'") from exc
            rows.append((recording_id, source))
    return rows


def decode_wav_bytes(payload: bytes, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    audio = _decode_pcm(frames, sample_width)
    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)
    return _resample(audio, source_rate, sample_rate)


def command_argv(source: str) -> list[str] | None:
    source = source.strip()
    if not source.endswith("|"):
        return None
    return shlex.split(source[:-1].strip())


def load_scp_audio(source: str) -> torch.Tensor:
    argv = command_argv(source)
    if argv is None:
        return load_audio(source)
    process = subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return decode_wav_bytes(process.stdout)


def source_media_path(source: str) -> str:
    argv = command_argv(source)
    if argv is None:
        return source
    for index, arg in enumerate(argv[:-1]):
        if arg == "-i":
            return argv[index + 1]
    return source


def format_utt_id(file_index: int, recording_id: str, window_index: int, start_sample: int, end_sample: int) -> str:
    start_cs = round(start_sample * 100 / SAMPLE_RATE)
    end_cs = round(end_sample * 100 / SAMPLE_RATE)
    return f"S{file_index:05d}-{recording_id}-U{window_index:07d}-{start_cs:07d}-{end_cs:07d}"


def window_command(media_path: str, start_seconds: float, duration_seconds: float) -> str:
    quoted = shlex.quote(media_path)
    return (
        f"ffmpeg -nostdin -ss {start_seconds:.2f} -t {duration_seconds:.2f} -i {quoted} "
        "-f wav -acodec pcm_s16le -ar 16000 -ac 1 - |"
    )


def completed_utterances(path: Path, batch_size: int) -> set[str]:
    return set(completed_window_info(path, batch_size))


def completed_window_info(path: Path, batch_size: int) -> dict[str, float | None]:
    counts: Counter[str] = Counter()
    next_starts: dict[str, float | None] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            utt_id = row.get("utt_id")
            if utt_id:
                counts[utt_id] += 1
                if utt_id not in next_starts:
                    next_starts[utt_id] = row.get("next_start")
    return {utt_id: next_starts.get(utt_id) for utt_id, count in counts.items() if count >= batch_size}


def completed_context_rows(path: Path) -> dict[str, dict[str, object]]:
    selected_rows: dict[str, dict[str, object]] = {}
    fallback_rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return selected_rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            utt_id = row.get("utt_id")
            if not utt_id:
                continue
            if row.get("smc_selected"):
                selected_rows[str(utt_id)] = row
            elif int(row.get("alternative", 0)) == 0:
                fallback_rows[str(utt_id)] = row
    return {**fallback_rows, **selected_rows}


def suppress_tokens(tokenizer, eos_token_id: int, *, allow_timestamps: bool) -> list[int]:
    tokens = []
    for token in tokenizer.special_tokens.values():
        if token == eos_token_id:
            continue
        if allow_timestamps and token >= tokenizer.timestamp_begin:
            continue
        tokens.append(token)
    return tokens


def decode_timestamped_text(tokenizer, generated: list[int]) -> str:
    special_ids = set(tokenizer.special_tokens.values()) | {tokenizer.pad_token_id}
    token_ids = [token for token in generated if token >= tokenizer.timestamp_begin or token not in special_ids]
    return tokenizer.decode_with_timestamps(token_ids).strip()


def text_token_ids(tokenizer, generated: list[int]) -> list[int]:
    special_ids = set(tokenizer.special_tokens.values()) | {tokenizer.pad_token_id}
    return [token for token in generated if token < tokenizer.timestamp_begin and token not in special_ids]


def token_ends_sentence(tokenizer, token: int) -> bool:
    if token >= tokenizer.timestamp_begin or token == tokenizer.eot:
        return False
    return tokenizer.decode([token]).rstrip().endswith((".", "!", "?"))


def sentence_end_timestamps(tokenizer, generated: list[int], time_precision: float) -> list[dict[str, float | int]]:
    timestamps = []
    pending_sentence_end = False
    for token in generated:
        if token >= tokenizer.timestamp_begin:
            if pending_sentence_end:
                offset = (token - tokenizer.timestamp_begin) * time_precision
                timestamps.append({"token": token, "offset": round(offset, 2)})
            pending_sentence_end = False
            continue
        pending_sentence_end = token_ends_sentence(tokenizer, token)
    return timestamps


def max_ngram_count(tokens: list[int], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return max(counts.values(), default=0)


def text_tokens_since_last_timestamp(tokenizer, generated: list[int]) -> int:
    special_ids = set(tokenizer.special_tokens.values()) | {tokenizer.pad_token_id}
    count = 0
    for token in generated:
        if token >= tokenizer.timestamp_begin:
            count = 0
        elif token not in special_ids:
            count += 1
    return count


def smc_particle_quality(tokenizer, generated: list[int], text_tokens: list[int], logprobs: list[float], no_speech_prob: float) -> dict[str, object]:
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
    repeated_2gram_max = max_ngram_count(text_tokens, 2)
    repeated_3gram_max = max_ngram_count(text_tokens, 3)
    repeated_4gram_max = max_ngram_count(text_tokens, 4)
    distinct_ratio = len(set(text_tokens)) / len(text_tokens) if text_tokens else 1.0
    tail_tokens = text_tokens_since_last_timestamp(tokenizer, generated)

    repeated_2gram_excess = max(0, repeated_2gram_max - 10)
    repeated_3gram_excess = max(0, repeated_3gram_max - 6)
    repeated_4gram_excess = max(0, repeated_4gram_max - 4)
    tail_excess = max(0, tail_tokens - 96)
    low_diversity_excess = max(0.0, 0.10 - distinct_ratio) if len(text_tokens) >= 48 else 0.0
    penalty = (
        0.12 * repeated_2gram_excess
        + 0.35 * repeated_3gram_excess
        + 0.60 * repeated_4gram_excess
        + 0.01 * tail_excess
        + 5.0 * low_diversity_excess
    )

    reject_reasons: list[str] = []
    if repeated_2gram_max >= 32:
        reject_reasons.append("repeated_2gram")
    if repeated_3gram_max >= 10:
        reject_reasons.append("repeated_3gram")
    if repeated_4gram_max >= 8:
        reject_reasons.append("repeated_4gram")
    if len(text_tokens) >= 48 and distinct_ratio < 0.06:
        reject_reasons.append("low_distinct_token_ratio")
    if tail_tokens > 192:
        reject_reasons.append("long_text_without_timestamp")
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        reject_reasons.append("no_speech_low_logprob")

    return {
        "avg_logprob_raw": avg_logprob,
        "smc_score": avg_logprob - penalty,
        "smc_penalty": penalty,
        "smc_accepted": not reject_reasons,
        "smc_reject_reasons": reject_reasons,
        "repeated_2gram_max": repeated_2gram_max,
        "repeated_3gram_max": repeated_3gram_max,
        "repeated_4gram_max": repeated_4gram_max,
        "distinct_token_ratio": distinct_ratio,
        "text_tokens_since_timestamp": tail_tokens,
    }


def select_smc_particle(qualities: list[dict[str, object]]) -> int:
    accepted = [index for index, quality in enumerate(qualities) if quality["smc_accepted"]]
    candidates = accepted if accepted else list(range(len(qualities)))
    return max(candidates, key=lambda index: float(qualities[index]["smc_score"]))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.sentence_hop and not args.timestamps_after_sentence_end:
        raise ValueError("--sentence-hop requires --timestamps-after-sentence-end")
    args.language = plu_test.normalize_language(args.language)
    device = plu_test.resolve_device(args.device)
    dtype = plu_test.parse_dtype(args.dtype)
    model_path = args.exp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decode_jsonl = args.out_dir / "decode.jsonl"
    wav_scp_out = args.out_dir / "wav.scp"
    windows_tsv = args.out_dir / "windows.tsv"
    meta_json = args.out_dir / "decode.meta.json"

    model = plu_test.WhisperForConditionalGeneration.from_pretrained(model_path, map_location="cpu")
    model.eval().to(device=device, dtype=dtype)
    tokenizer = plu_test.configure_tokenizer(model_path, model, args.language)
    packed, stats = pack_mx_model(model, "mxfp8")
    base_prompt = plu_test.prompt_ids(tokenizer, args.language, without_timestamps=not args.timestamps_after_sentence_end)
    sampler = plu_test.Mxfp8KvCacheCudaGraphSampler(
        model,
        tokenizer,
        packed,
        base_prompt,
        suppress_tokens(tokenizer, model.config.eos_token_id, allow_timestamps=args.timestamps_after_sentence_end),
        decode_batch_size=args.decode_batch_size,
        timestamps_after_sentence_end=args.timestamps_after_sentence_end,
    )

    rows = read_wav_scp(args.wav_scp)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed_info = completed_window_info(decode_jsonl, args.decode_batch_size) if args.resume else {}
    completed_rows = completed_context_rows(decode_jsonl) if args.resume and args.condition_on_previous_utterance else {}
    completed = set(completed_info)
    mode = "a" if args.resume and decode_jsonl.exists() else "w"
    window_samples = round(args.window_seconds * SAMPLE_RATE)
    hop_samples = round(args.hop_seconds * SAMPLE_RATE)
    min_window_samples = round(args.min_window_seconds * SAMPLE_RATE)
    time_precision = 30.0 / model.config.max_source_positions
    max_previous_tokens = model.config.max_target_positions // 2 - 1
    sample_first_sentence_only = args.sentence_hop and args.decode_batch_size > 1
    start_time = time.perf_counter()

    meta_json.write_text(
        json.dumps(
            {
                "wav_scp": str(args.wav_scp),
                "model": str(model_path),
                "language": args.language,
                "device": args.device,
                "dtype": args.dtype,
                "window_seconds": args.window_seconds,
                "hop_seconds": args.hop_seconds,
                "min_window_seconds": args.min_window_seconds,
                "decode_batch_size": args.decode_batch_size,
                "max_new_tokens": args.max_new_tokens,
                "timestamps_after_sentence_end": args.timestamps_after_sentence_end,
                "sentence_hop": args.sentence_hop,
                "condition_on_previous_utterance": args.condition_on_previous_utterance,
                "decode_strategy": "greedy" if args.decode_batch_size == 1 else "smc_best_particle",
                "sample_first_sentence_only": sample_first_sentence_only,
                "sampler_smc_page_size": getattr(sampler, "smc_page_size", None),
                "smc_quality": {
                    "repeated_2gram_max_reject": 32,
                    "repeated_3gram_max_reject": 10,
                    "repeated_4gram_max_reject": 8,
                    "distinct_token_ratio_reject_below": 0.06,
                    "long_text_without_timestamp_reject_above": 192,
                },
                "max_previous_tokens": max_previous_tokens,
                "time_precision": time_precision,
                "mxfp8_packed_linear_count": stats["mx_packed_linear_count"],
                "cuda_graph_capture_ms": sampler.capture_ms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with decode_jsonl.open(mode, encoding="utf-8") as decode_f, wav_scp_out.open("w", encoding="utf-8") as wav_f, windows_tsv.open(
        "w", encoding="utf-8"
    ) as windows_f:
        windows_f.write("utt_id\trecording_id\twindow_index\tstart\tend\tduration\tsource\n")
        total_windows = 0
        decoded_windows = 0
        total_audio_seconds = 0.0
        total_decode_seconds = 0.0
        total_generated_tokens = 0

        for file_index, (recording_id, source) in enumerate(rows):
            media_path = source_media_path(source)
            audio = load_scp_audio(source).flatten().float()
            duration = audio.numel() / SAMPLE_RATE
            total_audio_seconds += duration
            window_index = 0
            start_sample = 0
            previous_text_tokens: list[int] = []

            while start_sample < audio.numel() or (audio.numel() == 0 and window_index == 0):
                end_sample = min(start_sample + window_samples, audio.numel())
                utt_id = format_utt_id(file_index, recording_id, window_index, start_sample, end_sample)
                start_seconds = start_sample / SAMPLE_RATE
                end_seconds = end_sample / SAMPLE_RATE
                segment_duration = end_seconds - start_seconds
                next_start_sample = min(start_sample + hop_samples, audio.numel())
                hop_source = "fixed"
                wav_f.write(f"{utt_id} {window_command(media_path, start_seconds, segment_duration)}\n")
                windows_f.write(
                    f"{utt_id}\t{recording_id}\t{window_index}\t{start_seconds:.2f}\t{end_seconds:.2f}\t{segment_duration:.2f}\t{source}\n"
                )
                total_windows += 1
                context_tokens = previous_text_tokens[-max_previous_tokens:] if args.condition_on_previous_utterance else []
                context_text = tokenizer.decode(context_tokens).strip() if context_tokens else ""
                current_prompt = plu_test.prompt_ids(
                    tokenizer,
                    args.language,
                    without_timestamps=not args.timestamps_after_sentence_end,
                    previous_tokens=context_tokens if context_tokens else None,
                    max_previous_tokens=max_previous_tokens,
                )
                prompt_len = min(len(current_prompt), model.config.max_target_positions)
                next_previous_text_tokens: list[int] | None = None

                if utt_id not in completed:
                    segment = audio[start_sample:end_sample]
                    features = log_mel_spectrogram(segment, model.config.num_mel_bins).unsqueeze(0).to(device=device, dtype=dtype)
                    encoder_hidden_states = plu_test.encode_packed_mxfp8(model, packed, features)
                    torch.cuda.synchronize()
                    decode_start = time.perf_counter()
                    samples = sampler.sample_many(
                        encoder_hidden_states,
                        args.max_new_tokens,
                        prompt=current_prompt,
                        stop_after_first_sentence=sample_first_sentence_only,
                    )
                    torch.cuda.synchronize()
                    decode_seconds = time.perf_counter() - decode_start
                    total_decode_seconds += decode_seconds
                    decoded_windows += 1
                    sample_records: list[dict[str, object]] = []

                    for alternative, (token_ids, logprobs, no_speech_prob) in enumerate(samples):
                        generated = token_ids[prompt_len:]
                        sample_text_token_ids = text_token_ids(tokenizer, generated)
                        text_no_timestamps = tokenizer.decode(sample_text_token_ids).strip()
                        text = decode_timestamped_text(tokenizer, generated) if args.timestamps_after_sentence_end else text_no_timestamps
                        sample_sentence_timestamps = (
                            sentence_end_timestamps(tokenizer, generated, time_precision) if args.timestamps_after_sentence_end else []
                        )
                        total_generated_tokens += len(logprobs)
                        quality = smc_particle_quality(tokenizer, generated, sample_text_token_ids, logprobs, no_speech_prob)
                        token_logprobs = [round(float(logprob), 6) for logprob in logprobs]
                        sample_records.append(
                            {
                                "alternative": alternative,
                                "token_ids": token_ids,
                                "generated": generated,
                                "logprobs": logprobs,
                                "token_logprobs": token_logprobs,
                                "no_speech_prob": no_speech_prob,
                                "text_token_ids": sample_text_token_ids,
                                "text_no_timestamps": text_no_timestamps,
                                "text": text,
                                "sentence_end_timestamps": sample_sentence_timestamps,
                                "quality": quality,
                            }
                        )

                    selected_alternative = select_smc_particle([record["quality"] for record in sample_records])
                    selected_record = sample_records[selected_alternative]
                    selected_quality = selected_record["quality"]
                    selected_accepted = bool(selected_quality["smc_accepted"])
                    next_previous_text_tokens = selected_record["text_token_ids"] if selected_accepted else []
                    if args.sentence_hop:
                        next_start_sample = min(end_sample, audio.numel())
                        hop_source = "window_end" if selected_accepted else "smc_rejected_window_end"
                        if selected_accepted:
                            for timestamp in selected_record["sentence_end_timestamps"]:
                                candidate = start_sample + round(float(timestamp["offset"]) * SAMPLE_RATE)
                                if start_sample < candidate <= end_sample:
                                    next_start_sample = candidate
                                    hop_source = "sentence_timestamp"
                                    break

                    ranked_alternatives = sorted(
                        range(len(sample_records)),
                        key=lambda index: float(sample_records[index]["quality"]["smc_score"]),
                        reverse=True,
                    )
                    rank_by_alternative = {alternative: rank + 1 for rank, alternative in enumerate(ranked_alternatives)}

                    for record in sample_records:
                        alternative = int(record["alternative"])
                        quality = record["quality"]
                        logprobs = record["logprobs"]
                        row = {
                            "id": f"{utt_id}-A{alternative:02d}",
                            "utt_id": utt_id,
                            "recording_id": recording_id,
                            "alternative": alternative,
                            "i": alternative,
                            "window_index": window_index,
                            "start": round(start_seconds, 2),
                            "end": round(end_seconds, 2),
                            "duration": round(segment_duration, 2),
                            "text": record["text"],
                            "conf": None,
                            "avg_logprob": round(float(quality["avg_logprob_raw"]), 3),
                            "cumulative_logprob": round(sum(logprobs), 6),
                            "decode_strategy": "greedy" if alternative == 0 else "smc_sample",
                            "smc_selected": alternative == selected_alternative,
                            "smc_selected_for_context": alternative == selected_alternative and selected_accepted,
                            "smc_selected_alternative": selected_alternative,
                            "smc_rank": rank_by_alternative[alternative],
                            "smc_score": round(float(quality["smc_score"]), 6),
                            "sampler_page_resample_count": getattr(sampler, "last_page_resample_count", 0),
                            "smc_penalty": round(float(quality["smc_penalty"]), 6),
                            "smc_accepted": bool(quality["smc_accepted"]),
                            "smc_reject_reasons": quality["smc_reject_reasons"],
                            "repeated_2gram_max": quality["repeated_2gram_max"],
                            "repeated_3gram_max": quality["repeated_3gram_max"],
                            "repeated_4gram_max": quality["repeated_4gram_max"],
                            "distinct_token_ratio": round(float(quality["distinct_token_ratio"]), 6),
                            "text_tokens_since_timestamp": quality["text_tokens_since_timestamp"],
                            "no_speech_prob": round(float(record["no_speech_prob"]), 3),
                            "path": media_path,
                            "source": source,
                            "language": args.language or "",
                            "langprob": 1.0 if args.language else 0.0,
                            "input_ids": record["token_ids"],
                            "prompt_length": prompt_len,
                            "token_logprobs": record["token_logprobs"],
                            "context_text": context_text,
                            "context_token_count": len(context_tokens),
                            "text_token_ids": record["text_token_ids"],
                            "decode_seconds": round(decode_seconds, 6),
                            "next_start": round(next_start_sample / SAMPLE_RATE, 2),
                            "hop_source": hop_source,
                        }
                        if args.timestamps_after_sentence_end:
                            row["text_no_timestamps"] = record["text_no_timestamps"]
                            row["timestamp_ids"] = [token for token in record["generated"] if token >= tokenizer.timestamp_begin]
                            row["sentence_end_timestamps"] = record["sentence_end_timestamps"]
                        decode_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    decode_f.flush()
                    if args.condition_on_previous_utterance and next_previous_text_tokens is not None:
                        previous_text_tokens = next_previous_text_tokens
                elif args.sentence_hop:
                    saved_next_start = completed_info.get(utt_id)
                    if saved_next_start is None:
                        next_start_sample = min(start_sample + window_samples, audio.numel())
                    else:
                        next_start_sample = min(round(float(saved_next_start) * SAMPLE_RATE), audio.numel())
                    if args.condition_on_previous_utterance:
                        completed_row = completed_rows.get(utt_id)
                        if completed_row is not None:
                            saved_text_tokens = completed_row.get("text_token_ids") or []
                            previous_text_tokens = [int(token) for token in saved_text_tokens]
                elif args.condition_on_previous_utterance:
                    completed_row = completed_rows.get(utt_id)
                    if completed_row is not None:
                        saved_text_tokens = completed_row.get("text_token_ids") or []
                        previous_text_tokens = [int(token) for token in saved_text_tokens]

                window_index += 1
                if args.sentence_hop:
                    if next_start_sample <= start_sample:
                        next_start_sample = min(start_sample + window_samples, audio.numel())
                    if next_start_sample >= audio.numel() or audio.numel() - next_start_sample < min_window_samples:
                        break
                    start_sample = next_start_sample
                    continue
                if start_sample + hop_samples >= audio.numel():
                    break
                start_sample += hop_samples

            elapsed = time.perf_counter() - start_time
            print(
                json.dumps(
                    {
                        "recording": recording_id,
                        "file_index": file_index,
                        "duration": round(duration, 2),
                        "total_windows": total_windows,
                        "decoded_windows": decoded_windows,
                        "elapsed_seconds": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "recordings": len(rows),
        "windows": total_windows,
        "decoded_windows": decoded_windows,
        "output_rows": decoded_windows * args.decode_batch_size,
        "total_audio_seconds": total_audio_seconds,
        "total_decode_seconds": total_decode_seconds,
        "wall_decode_rtf": total_decode_seconds / total_audio_seconds if total_audio_seconds else 0.0,
        "effective_decode_rtf": total_decode_seconds / (total_audio_seconds * args.decode_batch_size) if total_audio_seconds else 0.0,
        "tokens_per_second": total_generated_tokens / total_decode_seconds if total_decode_seconds else 0.0,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
