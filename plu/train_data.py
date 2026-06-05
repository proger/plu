from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from plu.whisper_tokenizer import WhisperTokenizer


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
N_FRAMES = N_SAMPLES // HOP_LENGTH


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = data[:, 0].astype(np.int32) | (data[:, 1].astype(np.int32) << 8) | (data[:, 2].astype(np.int32) << 16)
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    tensor = torch.from_numpy(audio.astype(np.float32)).view(1, 1, -1)
    resampled = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return resampled.view(-1).numpy().astype(np.float32)


def _load_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    audio = _decode_pcm(frames, sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return _resample(audio, source_rate, sample_rate)


def _load_with_ffmpeg(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    process = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(process.stdout, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        with np.load(path) as data:
            first_key = sorted(data.files)[0]
            return data[first_key].astype(np.float32)
    if suffix == ".wav":
        return _load_wav(path, sample_rate)
    return _load_with_ffmpeg(path, sample_rate)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length].astype(np.float32, copy=False)
    return np.pad(audio.astype(np.float32, copy=False), (0, length - len(audio)))


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies)
    mels = frequencies / 200.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / 200.0
    logstep = math.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels)
    frequencies = 200.0 * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / 200.0
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return frequencies


_MEL_FILTER_CACHE: dict[int, torch.Tensor] = {}


def mel_filters(n_mels: int, device: torch.device) -> torch.Tensor:
    cached = _MEL_FILTER_CACHE.get(n_mels)
    if cached is None:
        mel_min = _hz_to_mel(np.array([0.0]))[0]
        mel_max = _hz_to_mel(np.array([SAMPLE_RATE / 2]))[0]
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = _mel_to_hz(mel_points)
        fft_frequencies = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
        ramps = hz_points[:, None] - fft_frequencies[None, :]
        fdiff = np.diff(hz_points)
        lower = -ramps[:-2] / fdiff[:-1, None]
        upper = ramps[2:] / fdiff[1:, None]
        weights = np.maximum(0.0, np.minimum(lower, upper))
        weights *= (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]))[:, None]
        cached = torch.from_numpy(weights.astype(np.float32))
        _MEL_FILTER_CACHE[n_mels] = cached
    return cached.to(device)


def log_mel_spectrogram(audio: np.ndarray | torch.Tensor, n_mels: int) -> torch.Tensor:
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(pad_or_trim(audio))
    else:
        audio = audio.float()
        if audio.numel() != N_SAMPLES:
            audio = torch.from_numpy(pad_or_trim(audio.cpu().numpy()))

    window = torch.hann_window(N_FFT, device=audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters(n_mels, audio.device) @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


class JsonlAudioDataset(Dataset):
    def __init__(self, path_or_paths: str | list[str], n_mels: int):
        paths = [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)
        self.n_mels = n_mels
        self.examples: list[dict[str, Any]] = []
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)
                    duration = example.get("duration")
                    if duration is not None and float(duration) > CHUNK_LENGTH:
                        continue
                    self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if "input_features" in example:
            input_features = torch.tensor(example["input_features"], dtype=torch.float32)
        else:
            audio_path = example.get("path") or example.get("audio")
            if audio_path is None:
                raise ValueError("dataset example must contain path, audio, or input_features")
            input_features = log_mel_spectrogram(load_audio(audio_path), self.n_mels)

        labels = example.get("input_ids") or example.get("labels")
        if labels is None:
            raise ValueError("dataset example must contain input_ids or labels")
        labels = [int(token) for token in labels]
        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": example.get("text"),
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    pad_token_id: int
    sot_token_id: int = WhisperTokenizer.sot

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_frames = max(feature["input_features"].shape[-1] for feature in features)
        n_mels = features[0]["input_features"].shape[0]
        input_features = torch.zeros(len(features), n_mels, max_frames, dtype=torch.float32)
        for i, feature in enumerate(features):
            current = feature["input_features"].float()
            input_features[i, :, : current.shape[-1]] = current

        max_label_len = max(feature["labels"].numel() for feature in features)
        labels = torch.full((len(features), max_label_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(labels)
        for i, feature in enumerate(features):
            current = feature["labels"].long()
            labels[i, : current.numel()] = current
            attention_mask[i, : current.numel()] = 1

        labels = labels.masked_fill(attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and bool((labels[:, 0] == self.sot_token_id).all().item()):
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
            "texts": [feature.get("text") for feature in features],
        }


class Corpus:
    def __init__(self, args: argparse.Namespace, tokenizer: WhisperTokenizer, n_mels: int):
        self.args = args
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer.pad_token_id)

    def load_dataset(self, path_or_paths: str | list[str]) -> JsonlAudioDataset:
        dataset = JsonlAudioDataset(path_or_paths, self.n_mels)
        if len(dataset) == 0:
            raise ValueError(f"dataset is empty after filtering: {path_or_paths}")
        return dataset

    def make_train_dataloader(self, dataset: Dataset) -> DataLoader:
        args = self.args
        return DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            shuffle=False,
        )

    def make_eval_dataloader(self, dataset: Dataset) -> DataLoader:
        args = self.args
        return DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            shuffle=False,
        )


def register_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", type=str, nargs="+", help="jsonl filename for training data, can be multiple files", required=False)
    parser.add_argument("--eval", type=str, help="jsonl filename for evaluation data", required=True)
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Kept for CLI compatibility; JSONL loading is local.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--language", type=str, help="Language code retained for CLI compatibility.", default="Russian")
    parser.add_argument("--task", type=str, default="transcribe", help="Task retained for CLI compatibility.", required=False)
    parser.add_argument("--dataloader_pin_memory", type=str_to_bool, default=True, help="Whether or not to pin memory for the DataLoader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses to use for data loading.")
