#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plu.test as plu_test
from benchmarks.bench_end_to_end import pack_mx_model
from egs.uk1e2.local.decode_windows import (
    decode_timestamped_text,
    format_utt_id,
    load_scp_audio,
    read_wav_scp,
    sample_particle_quality,
    select_sample,
    sentence_end_timestamps,
    sentence_segments,
    source_media_path,
    suppress_tokens,
    text_token_ids,
    window_command,
)
from plu.train_data import HOP_LENGTH, N_FFT, N_SAMPLES, SAMPLE_RATE, mel_filters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode wav.scp windows with cross-recording batching.")
    parser.add_argument("--wav-scp", type=Path, default=Path("data/local/wav.scp"))
    parser.add_argument("--out-dir", type=Path, default=Path("egs/uk1e2/exp/decode_batched"))
    parser.add_argument(
        "--exp",
        type=Path,
        default=Path("egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8/train"),
    )
    parser.add_argument("--language", default="uk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--hop-seconds", type=float, default=15.0)
    parser.add_argument("--min-window-seconds", type=float, default=1.0)
    parser.add_argument("--decode-batch-size", type=int, default=8, help="Alternatives per window.")
    parser.add_argument("--window-batch-size", type=int, default=4, help="Independent windows decoded per sampler call.")
    parser.add_argument("--audio-load-workers", type=int, default=1, help="Parallel audio preload workers. Runtime is included in elapsed time.")
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--timestamps-after-sentence-end", action="store_true")
    parser.add_argument("--sentence-hop", action="store_true")
    parser.add_argument("--sentence-hop-mode", choices=["first", "last"], default="last")
    parser.add_argument("--condition-on-previous-utterance", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def make_state(file_index: int, recording_id: str, source: str) -> dict[str, object]:
    audio = load_scp_audio(source).flatten().float()
    return {
        "file_index": file_index,
        "recording_id": recording_id,
        "source": source,
        "media_path": source_media_path(source),
        "audio": audio,
        "duration": audio.numel() / SAMPLE_RATE,
        "window_index": 0,
        "start_sample": 0,
        "previous_text_tokens": [],
    }


class AsyncStateLoader:
    def __init__(self, rows: list[tuple[str, str]], workers: int):
        self.rows = rows
        self.wait_seconds = 0.0
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.futures = (
            [
                self.executor.submit(make_state, index, recording_id, source)
                for index, (recording_id, source) in enumerate(rows)
            ]
            if self.executor is not None
            else []
        )

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int) -> dict[str, object]:
        start = time.perf_counter()
        if self.executor is None:
            recording_id, source = self.rows[index]
            state = make_state(index, recording_id, source)
        else:
            state = self.futures[index].result()
        self.wait_seconds += time.perf_counter() - start
        return state

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)


def batched_log_mel_spectrogram(
    segments: list[torch.Tensor],
    n_mels: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    window: torch.Tensor | None = None,
    mel_basis: torch.Tensor | None = None,
    audio_buffer: torch.Tensor | None = None,
    device_audio_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    if audio_buffer is None:
        audio = torch.empty((len(segments), N_SAMPLES), dtype=torch.float32)
    else:
        if audio_buffer.shape[0] < len(segments) or audio_buffer.shape[1] < N_SAMPLES:
            raise ValueError(f"audio_buffer is too small for {len(segments)} x {N_SAMPLES}")
        audio = audio_buffer[: len(segments), :N_SAMPLES]
    for index, segment in enumerate(segments):
        flat = segment.flatten().float()
        copy_length = min(flat.numel(), N_SAMPLES)
        if copy_length:
            audio[index, :copy_length].copy_(flat[:copy_length])
        if copy_length < N_SAMPLES:
            audio[index, copy_length:N_SAMPLES].zero_()
    if device_audio_buffer is None:
        audio = audio.to(device=device, non_blocking=True)
    else:
        if device_audio_buffer.shape[0] < len(segments) or device_audio_buffer.shape[1] < N_SAMPLES:
            raise ValueError(f"device_audio_buffer is too small for {len(segments)} x {N_SAMPLES}")
        device_audio = device_audio_buffer[: len(segments), :N_SAMPLES]
        device_audio.copy_(audio, non_blocking=True)
        audio = device_audio
    if window is None:
        window = torch.hann_window(N_FFT, device=device)
    if mel_basis is None:
        mel_basis = mel_filters(n_mels, device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = torch.matmul(mel_basis, magnitudes)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    return ((log_spec + 4.0) / 4.0).to(dtype=dtype)


def encode_window_features(model, packed: dict[int, object], features: torch.Tensor) -> torch.Tensor:
    encoder_batch_size = 1
    if features.shape[0] <= encoder_batch_size:
        return plu_test.encode_packed_mxfp8(model, packed, features)
    chunks = [
        plu_test.encode_packed_mxfp8(model, packed, features[start : start + encoder_batch_size])
        for start in range(0, features.shape[0], encoder_batch_size)
    ]
    return torch.cat(chunks, dim=0)


def build_sample_records(tokenizer, samples, prompt_len: int, time_precision: float, timestamps: bool) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    generated_token_count = 0
    for alternative, (token_ids, logprobs, no_speech_prob) in enumerate(samples):
        generated = token_ids[prompt_len:]
        sample_text_token_ids = text_token_ids(tokenizer, generated)
        text_no_timestamps = tokenizer.decode(sample_text_token_ids).strip()
        text = decode_timestamped_text(tokenizer, generated) if timestamps else text_no_timestamps
        sample_sentence_timestamps = sentence_end_timestamps(tokenizer, generated, time_precision) if timestamps else []
        sample_sentence_segments = sentence_segments(tokenizer, generated, logprobs, time_precision) if timestamps else []
        quality = sample_particle_quality(tokenizer, generated, sample_text_token_ids, logprobs, no_speech_prob)
        generated_token_count += len(logprobs)
        records.append(
            {
                "alternative": alternative,
                "token_ids": token_ids,
                "generated": generated,
                "logprobs": logprobs,
                "token_logprobs": [round(float(logprob), 6) for logprob in logprobs],
                "no_speech_prob": no_speech_prob,
                "text_token_ids": sample_text_token_ids,
                "text_no_timestamps": text_no_timestamps,
                "text": text,
                "sentence_end_timestamps": sample_sentence_timestamps,
                "sentence_segments": sample_sentence_segments,
                "quality": quality,
            }
        )
    return records, generated_token_count


def select_next_start(
    *,
    selected_record: dict[str, object],
    selected_accepted: bool,
    sentence_hop: bool,
    sentence_hop_mode: str,
    start_sample: int,
    end_sample: int,
    audio_samples: int,
) -> tuple[int, str]:
    if not sentence_hop:
        return end_sample, "fixed"
    next_start_sample = min(end_sample, audio_samples)
    hop_source = "window_end" if selected_accepted else "sample_rejected_window_end"
    if selected_accepted:
        timestamps = list(selected_record["sentence_end_timestamps"])
        if sentence_hop_mode == "last":
            timestamps = list(reversed(timestamps))
        for timestamp in timestamps:
            candidate = start_sample + round(float(timestamp["offset"]) * SAMPLE_RATE)
            if start_sample < candidate <= end_sample:
                return candidate, f"{sentence_hop_mode}_sentence_timestamp"
    return next_start_sample, hop_source


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.window_batch_size < 1:
        raise ValueError("--window-batch-size must be at least 1")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be at least 1")
    if args.sentence_hop and not args.timestamps_after_sentence_end:
        raise ValueError("--sentence-hop requires --timestamps-after-sentence-end")

    args.language = plu_test.normalize_language(args.language)
    device = plu_test.resolve_device(args.device)
    dtype = plu_test.parse_dtype(args.dtype)
    model_path = args.exp
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = plu_test.WhisperForConditionalGeneration.from_pretrained(model_path, map_location="cpu")
    model.eval().to(device=device, dtype=dtype)
    tokenizer = plu_test.configure_tokenizer(model_path, model, args.language)
    packed, stats = pack_mx_model(model, "mxfp8")
    base_prompt = plu_test.prompt_ids(tokenizer, args.language, without_timestamps=not args.timestamps_after_sentence_end)
    suppress = suppress_tokens(tokenizer, model.config.eos_token_id, allow_timestamps=args.timestamps_after_sentence_end)
    sampler = plu_test.Mxfp8KvCacheCudaGraphSampler(
        model,
        tokenizer,
        packed,
        base_prompt,
        suppress,
        decode_batch_size=args.decode_batch_size * args.window_batch_size,
        timestamps_after_sentence_end=args.timestamps_after_sentence_end,
        cross_cache_batch_size=args.window_batch_size,
    )
    prefill_sampler = plu_test.Mxfp8KvCacheCudaGraphSampler(
        model,
        tokenizer,
        packed,
        base_prompt,
        suppress,
        decode_batch_size=args.window_batch_size,
        timestamps_after_sentence_end=args.timestamps_after_sentence_end,
        cross_cache_batch_size=args.window_batch_size,
        cross_key_caches=sampler.cross_key_caches,
        cross_value_caches=sampler.cross_value_caches,
    )

    rows = read_wav_scp(args.wav_scp)
    if args.limit is not None:
        rows = rows[: args.limit]

    decode_jsonl = args.out_dir / "decode.jsonl"
    wav_scp_out = args.out_dir / "wav.scp"
    windows_tsv = args.out_dir / "windows.tsv"
    meta_json = args.out_dir / "decode.meta.json"
    window_samples = round(args.window_seconds * SAMPLE_RATE)
    hop_samples = round(args.hop_seconds * SAMPLE_RATE)
    min_window_samples = round(args.min_window_seconds * SAMPLE_RATE)
    time_precision = 30.0 / model.config.max_source_positions
    max_previous_tokens = model.config.max_target_positions // 2 - 1
    sample_first_sentence_only = args.sentence_hop and args.sentence_hop_mode == "first" and args.decode_batch_size > 1

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
                "window_batch_size": args.window_batch_size,
                "audio_load_workers": args.audio_load_workers,
                "sampler_decode_batch_size": args.decode_batch_size * args.window_batch_size,
                "sampler_cross_cache_batch_size": args.window_batch_size,
                "encoder_window_batch_size": 1,
                "compact_prompt_prefill": True,
                "max_new_tokens": args.max_new_tokens,
                "timestamps_after_sentence_end": args.timestamps_after_sentence_end,
                "sentence_hop": args.sentence_hop,
                "sentence_hop_mode": args.sentence_hop_mode,
                "condition_on_previous_utterance": args.condition_on_previous_utterance,
                "decode_strategy": "greedy" if args.decode_batch_size == 1 else "sample_best",
                "sample_first_sentence_only": sample_first_sentence_only,
                "max_previous_tokens": max_previous_tokens,
                "time_precision": time_precision,
                "mxfp8_packed_linear_count": stats["mx_packed_linear_count"],
                "cuda_graph_capture_ms": sampler.capture_ms + prefill_sampler.capture_ms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    start_time = time.perf_counter()
    state_loader = AsyncStateLoader(rows, args.audio_load_workers)
    total_audio_load_seconds = 0.0
    row_cursor = 0
    active: list[dict[str, object]] = []
    total_audio_seconds = 0.0
    total_windows = 0
    decoded_windows = 0
    total_batch_prep_seconds = 0.0
    total_mel_seconds = 0.0
    total_encoder_seconds = 0.0
    total_sample_seconds = 0.0
    total_model_seconds = 0.0
    total_postprocess_seconds = 0.0
    total_generated_tokens = 0
    mel_window = torch.hann_window(N_FFT, device=device)
    mel_basis = mel_filters(model.config.num_mel_bins, device)
    audio_buffer = torch.empty(
        (args.window_batch_size, N_SAMPLES),
        dtype=torch.float32,
        pin_memory=device.type == "cuda",
    )
    device_audio_buffer = torch.empty((args.window_batch_size, N_SAMPLES), device=device, dtype=torch.float32)
    stage_events = None
    if device.type == "cuda":
        stage_events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(6))

    def fill_active() -> None:
        nonlocal row_cursor, total_audio_seconds, total_audio_load_seconds
        while len(active) < args.window_batch_size and row_cursor < len(state_loader):
            state = state_loader.get(row_cursor)
            total_audio_load_seconds = state_loader.wait_seconds
            total_audio_seconds += float(state["duration"])
            active.append(state)
            row_cursor += 1

    fill_active()

    with decode_jsonl.open("w", encoding="utf-8") as decode_f, wav_scp_out.open("w", encoding="utf-8") as wav_f, windows_tsv.open(
        "w", encoding="utf-8"
    ) as windows_f:
        windows_f.write("utt_id\trecording_id\twindow_index\tstart\tend\tduration\tsource\n")
        last_flushed_decoded_windows = 0

        while active:
            batch_prep_start = time.perf_counter()
            batch = list(active)
            window_records: list[dict[str, object]] = []
            audio_segments: list[torch.Tensor] = []
            prompts: list[list[int]] = []

            for state in batch:
                audio = state["audio"]
                start_sample = int(state["start_sample"])
                end_sample = min(start_sample + window_samples, audio.numel())
                file_index = int(state["file_index"])
                recording_id = str(state["recording_id"])
                source = str(state["source"])
                media_path = str(state["media_path"])
                window_index = int(state["window_index"])
                utt_id = format_utt_id(file_index, recording_id, window_index, start_sample, end_sample)
                start_seconds = start_sample / SAMPLE_RATE
                end_seconds = end_sample / SAMPLE_RATE
                segment_duration = end_seconds - start_seconds
                context_tokens = (
                    list(state["previous_text_tokens"])[-max_previous_tokens:] if args.condition_on_previous_utterance else []
                )
                context_text = tokenizer.decode(context_tokens).strip() if context_tokens else ""
                prompt = plu_test.prompt_ids(
                    tokenizer,
                    args.language,
                    without_timestamps=not args.timestamps_after_sentence_end,
                    previous_tokens=context_tokens if context_tokens else None,
                    max_previous_tokens=max_previous_tokens,
                )
                segment = audio[start_sample:end_sample]
                audio_segments.append(segment)
                prompts.append(prompt)
                wav_f.write(f"{utt_id} {window_command(media_path, start_seconds, segment_duration)}\n")
                windows_f.write(
                    f"{utt_id}\t{recording_id}\t{window_index}\t{start_seconds:.2f}\t{end_seconds:.2f}\t{segment_duration:.2f}\t{source}\n"
                )
                window_records.append(
                    {
                        "state": state,
                        "utt_id": utt_id,
                        "recording_id": recording_id,
                        "source": source,
                        "media_path": media_path,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "segment_duration": segment_duration,
                        "context_tokens": context_tokens,
                        "context_text": context_text,
                        "prompt_len": min(len(prompt), model.config.max_target_positions),
                    }
                )
                total_windows += 1

            while len(prompts) < args.window_batch_size:
                prompts.append(prompts[-1])
                audio_segments.append(audio_segments[-1])
            total_batch_prep_seconds += time.perf_counter() - batch_prep_start

            model_start = time.perf_counter()
            mel_start = time.perf_counter()
            if stage_events is not None:
                stage_events[0].record()
            features = batched_log_mel_spectrogram(
                audio_segments,
                model.config.num_mel_bins,
                device,
                dtype,
                window=mel_window,
                mel_basis=mel_basis,
                audio_buffer=audio_buffer,
                device_audio_buffer=device_audio_buffer,
            )
            if stage_events is not None:
                stage_events[1].record()
            else:
                total_mel_seconds += time.perf_counter() - mel_start
            encoder_start = time.perf_counter()
            if stage_events is not None:
                stage_events[2].record()
            encoder_hidden_states = encode_window_features(model, packed, features)
            if stage_events is not None:
                stage_events[3].record()
            else:
                total_encoder_seconds += time.perf_counter() - encoder_start
            sample_start = time.perf_counter()
            if stage_events is not None:
                stage_events[4].record()
            grouped_samples = sampler.sample_many_grouped_compact_prefill(
                prefill_sampler,
                encoder_hidden_states,
                args.max_new_tokens,
                prompts=prompts,
                alternatives_per_prompt=args.decode_batch_size,
                stop_after_first_sentence=sample_first_sentence_only,
            )
            if stage_events is not None:
                stage_events[5].record()
                stage_events[5].synchronize()
                mel_seconds = stage_events[0].elapsed_time(stage_events[1]) / 1000.0
                encoder_seconds = stage_events[2].elapsed_time(stage_events[3]) / 1000.0
                sample_seconds = stage_events[4].elapsed_time(stage_events[5]) / 1000.0
                total_mel_seconds += mel_seconds
                total_encoder_seconds += encoder_seconds
                total_sample_seconds += sample_seconds
            else:
                sample_seconds = time.perf_counter() - sample_start
                total_sample_seconds += sample_seconds
            model_seconds = time.perf_counter() - model_start
            total_model_seconds += model_seconds
            real_window_count = len(window_records)
            decoded_windows += real_window_count
            per_window_decode_seconds = model_seconds / real_window_count

            postprocess_start = time.perf_counter()
            finished_states: list[dict[str, object]] = []
            for batch_index, window in enumerate(window_records):
                sample_records, generated_tokens = build_sample_records(
                    tokenizer,
                    grouped_samples[batch_index],
                    int(window["prompt_len"]),
                    time_precision,
                    args.timestamps_after_sentence_end,
                )
                total_generated_tokens += generated_tokens
                selected_alternative = select_sample([record["quality"] for record in sample_records])
                selected_record = sample_records[selected_alternative]
                selected_quality = selected_record["quality"]
                selected_accepted = bool(selected_quality["sample_accepted"])
                next_previous_text_tokens = selected_record["text_token_ids"] if selected_accepted else []
                state = window["state"]
                next_start_sample, hop_source = select_next_start(
                    selected_record=selected_record,
                    selected_accepted=selected_accepted,
                    sentence_hop=args.sentence_hop,
                    sentence_hop_mode=args.sentence_hop_mode,
                    start_sample=int(window["start_sample"]),
                    end_sample=int(window["end_sample"]),
                    audio_samples=state["audio"].numel(),
                )
                if not args.sentence_hop:
                    next_start_sample = min(int(window["start_sample"]) + hop_samples, state["audio"].numel())

                ranked_alternatives = sorted(
                    range(len(sample_records)),
                    key=lambda index: float(sample_records[index]["quality"]["sample_score"]),
                    reverse=True,
                )
                rank_by_alternative = {alternative: rank + 1 for rank, alternative in enumerate(ranked_alternatives)}

                for record in sample_records:
                    alternative = int(record["alternative"])
                    quality = record["quality"]
                    logprobs = record["logprobs"]
                    row = {
                        "id": f"{window['utt_id']}-A{alternative:02d}",
                        "utt_id": window["utt_id"],
                        "recording_id": window["recording_id"],
                        "alternative": alternative,
                        "i": alternative,
                        "window_index": int(state["window_index"]),
                        "start": round(float(window["start_seconds"]), 2),
                        "end": round(float(window["end_seconds"]), 2),
                        "duration": round(float(window["segment_duration"]), 2),
                        "text": record["text"],
                        "conf": None,
                        "avg_logprob": round(float(quality["avg_logprob_raw"]), 3),
                        "cumulative_logprob": round(sum(logprobs), 6),
                        "decode_strategy": "greedy" if alternative == 0 else "sample",
                        "sample_selected": alternative == selected_alternative,
                        "sample_selected_for_context": alternative == selected_alternative and selected_accepted,
                        "sample_selected_alternative": selected_alternative,
                        "sample_rank": rank_by_alternative[alternative],
                        "sample_score": round(float(quality["sample_score"]), 6),
                        "sample_penalty": round(float(quality["sample_penalty"]), 6),
                        "sample_accepted": bool(quality["sample_accepted"]),
                        "sample_reject_reasons": quality["sample_reject_reasons"],
                        "repeated_2gram_max": quality["repeated_2gram_max"],
                        "repeated_3gram_max": quality["repeated_3gram_max"],
                        "repeated_4gram_max": quality["repeated_4gram_max"],
                        "distinct_token_ratio": round(float(quality["distinct_token_ratio"]), 6),
                        "text_tokens_since_timestamp": quality["text_tokens_since_timestamp"],
                        "no_speech_prob": round(float(record["no_speech_prob"]), 3),
                        "path": window["media_path"],
                        "source": window["source"],
                        "language": args.language or "",
                        "langprob": 1.0 if args.language else 0.0,
                        "input_ids": record["token_ids"],
                        "prompt_length": int(window["prompt_len"]),
                        "token_logprobs": record["token_logprobs"],
                        "context_text": window["context_text"],
                        "context_token_count": len(window["context_tokens"]),
                        "text_token_ids": record["text_token_ids"],
                        "decode_seconds": round(per_window_decode_seconds, 6),
                        "decode_batch_seconds": round(model_seconds, 6),
                        "sample_batch_seconds": round(sample_seconds, 6),
                        "window_batch_size": real_window_count,
                        "next_start": round(next_start_sample / SAMPLE_RATE, 2),
                        "hop_source": hop_source,
                    }
                    if args.timestamps_after_sentence_end:
                        row["text_no_timestamps"] = record["text_no_timestamps"]
                        row["timestamp_ids"] = [token for token in record["generated"] if token >= tokenizer.timestamp_begin]
                        row["sentence_end_timestamps"] = record["sentence_end_timestamps"]
                        row["sentence_segments"] = [
                            {
                                **segment_row,
                                "absolute_start": round(float(window["start_seconds"]) + float(segment_row["start"]), 2),
                                "absolute_end": round(float(window["start_seconds"]) + float(segment_row["end"]), 2),
                            }
                            for segment_row in record["sentence_segments"]
                        ]
                    decode_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                state["previous_text_tokens"] = next_previous_text_tokens
                state["window_index"] = int(state["window_index"]) + 1
                if next_start_sample <= int(window["start_sample"]):
                    next_start_sample = min(int(window["start_sample"]) + window_samples, state["audio"].numel())
                if next_start_sample >= state["audio"].numel() or state["audio"].numel() - next_start_sample < min_window_samples:
                    finished_states.append(state)
                else:
                    state["start_sample"] = next_start_sample

            total_postprocess_seconds += time.perf_counter() - postprocess_start
            finished_state_ids = {id(state) for state in finished_states}
            active = [state for state in active if id(state) not in finished_state_ids]
            if finished_states or decoded_windows - last_flushed_decoded_windows >= 128:
                decode_f.flush()
                last_flushed_decoded_windows = decoded_windows
            for state in finished_states:
                elapsed = time.perf_counter() - start_time
                print(
                    json.dumps(
                        {
                            "recording": state["recording_id"],
                            "file_index": state["file_index"],
                            "duration": round(float(state["duration"]), 2),
                            "total_windows": total_windows,
                            "decoded_windows": decoded_windows,
                            "elapsed_seconds": round(elapsed, 2),
                            "stage_seconds": {
                                "audio_load": round(total_audio_load_seconds, 3),
                                "batch_prep": round(total_batch_prep_seconds, 3),
                                "mel": round(total_mel_seconds, 3),
                                "encoder": round(total_encoder_seconds, 3),
                                "sample": round(total_sample_seconds, 3),
                                "model": round(total_model_seconds, 3),
                                "postprocess": round(total_postprocess_seconds, 3),
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            fill_active()

    state_loader.close()
    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "recordings": len(rows),
        "windows": total_windows,
        "decoded_windows": decoded_windows,
        "output_rows": decoded_windows * args.decode_batch_size,
        "total_audio_seconds": total_audio_seconds,
        "total_audio_load_seconds": total_audio_load_seconds,
        "total_batch_prep_seconds": total_batch_prep_seconds,
        "total_mel_seconds": total_mel_seconds,
        "total_encoder_seconds": total_encoder_seconds,
        "total_sample_seconds": total_sample_seconds,
        "total_postprocess_seconds": total_postprocess_seconds,
        "total_model_seconds": total_model_seconds,
        "total_decode_seconds": total_sample_seconds,
        "wall_decode_rtf": total_sample_seconds / total_audio_seconds if total_audio_seconds else 0.0,
        "wall_model_rtf": total_model_seconds / total_audio_seconds if total_audio_seconds else 0.0,
        "effective_decode_rtf": total_sample_seconds / (total_audio_seconds * args.decode_batch_size) if total_audio_seconds else 0.0,
        "effective_model_rtf": total_model_seconds / (total_audio_seconds * args.decode_batch_size) if total_audio_seconds else 0.0,
        "tokens_per_second": total_generated_tokens / total_sample_seconds if total_sample_seconds else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_rtf": elapsed_seconds / total_audio_seconds if total_audio_seconds else 0.0,
        "stage_seconds": {
            "audio_load": total_audio_load_seconds,
            "batch_prep": total_batch_prep_seconds,
            "mel": total_mel_seconds,
            "encoder": total_encoder_seconds,
            "sample": total_sample_seconds,
            "model": total_model_seconds,
            "postprocess": total_postprocess_seconds,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
