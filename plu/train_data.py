from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
import subprocess
import struct
import sys
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from plu.tokenizer import SOT, WhisperTokenizer


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


def _decode_pcm(raw: bytes, sample_width: int) -> torch.Tensor:
    if sample_width == 1:
        return (torch.frombuffer(bytearray(raw), dtype=torch.uint8).float() - 128.0) / 128.0
    if sample_width == 2:
        return torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    if sample_width == 3:
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, 3).int()
        values = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        values = torch.where((values & 0x800000).ne(0), values | ~0xFFFFFF, values)
        return values.float() / 8388608.0
    if sample_width == 4:
        return torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _resample(audio: torch.Tensor, source_rate: int, target_rate: int = SAMPLE_RATE) -> torch.Tensor:
    if source_rate == target_rate:
        return audio.float()
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    tensor = audio.float().view(1, 1, -1)
    resampled = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return resampled.view(-1).float()


def _load_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    audio = _decode_pcm(frames, sample_width)
    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)
    return _resample(audio, source_rate, sample_rate)


def _load_with_ffmpeg(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
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
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.int16).flatten().float() / 32768.0


def _npy_dtype(descr: str) -> torch.dtype:
    if descr in {"<f4", "|f4"}:
        return torch.float32
    if descr in {"<f8", "|f8"}:
        return torch.float64
    if descr in {"<i2", "|i2"}:
        return torch.int16
    if descr in {"<i4", "|i4"}:
        return torch.int32
    if descr in {"<i8", "|i8"}:
        return torch.int64
    if descr in {"|u1", "<u1"}:
        return torch.uint8
    if descr in {"|i1", "<i1"}:
        return torch.int8
    raise ValueError(f"unsupported npy dtype: {descr}")


def _load_npy_bytes(payload: bytes) -> torch.Tensor:
    if not payload.startswith(b"\x93NUMPY"):
        raise ValueError("not an npy payload")
    major = payload[6]
    if major == 1:
        header_len = struct.unpack("<H", payload[8:10])[0]
        data_offset = 10 + header_len
        header_start = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", payload[8:12])[0]
        data_offset = 12 + header_len
        header_start = 12
    else:
        raise ValueError(f"unsupported npy version: {major}")

    header = ast.literal_eval(payload[header_start:data_offset].decode("latin1"))
    if header.get("fortran_order"):
        raise ValueError("Fortran-order npy arrays are not supported")
    shape = tuple(int(dim) for dim in header["shape"])
    dtype = _npy_dtype(str(header["descr"]))
    count = math.prod(shape) if shape else 1
    tensor = torch.frombuffer(bytearray(payload[data_offset:]), dtype=dtype)[:count]
    return tensor.reshape(shape).float()


def _load_npy(path: Path) -> torch.Tensor:
    return _load_npy_bytes(path.read_bytes())


def _load_npz(path: Path, key: str | None = None) -> torch.Tensor:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".npy"))
        if not names:
            raise ValueError(f"{path} does not contain any npy arrays")
        selected = f"{key}.npy" if key is not None else names[0]
        if selected not in names:
            raise KeyError(f"{path} does not contain {selected}")
        return _load_npy_bytes(archive.read(selected))


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy(path)
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".wav":
        return _load_wav(path, sample_rate)
    return _load_with_ffmpeg(path, sample_rate)


def pad_or_trim(audio: torch.Tensor, length: int = N_SAMPLES) -> torch.Tensor:
    audio = audio.flatten().float()
    if audio.numel() >= length:
        return audio[:length]
    return F.pad(audio, (0, length - audio.numel()))


def _hz_to_mel(frequencies: torch.Tensor) -> torch.Tensor:
    frequencies = frequencies.float()
    mels = frequencies / 200.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / 200.0
    logstep = math.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    log_mels = min_log_mel + torch.log(torch.clamp(frequencies, min=min_log_hz) / min_log_hz) / logstep
    return torch.where(log_region, log_mels, mels)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    mels = mels.float()
    frequencies = 200.0 * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / 200.0
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    log_frequencies = min_log_hz * torch.exp(logstep * (mels - min_log_mel))
    return torch.where(log_region, log_frequencies, frequencies)


_MEL_FILTER_CACHE: dict[int, torch.Tensor] = {}


def _mel_filter_asset_candidates() -> list[Path]:
    filename = "mel_filters.npz"
    candidates = []
    if "PLU_AUDIO_ASSETS" in os.environ:
        candidates.append(Path(os.environ["PLU_AUDIO_ASSETS"]).expanduser() / filename)
    candidates.append(Path(__file__).resolve().parent / "assets" / filename)
    for site_path in sys.path:
        if site_path:
            candidates.append(Path(site_path) / "whisper" / "assets" / filename)
    candidates.append(Path(os.environ.get("PLU_AUDIO_CACHE", "~/.cache/plu/audio")).expanduser() / filename)
    return candidates


def _mel_filter_asset_path() -> Path | None:
    candidates = _mel_filter_asset_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate

    destination = candidates[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as f:
            shutil.copyfileobj(response, f)
        return destination
    except OSError:
        return None


def _analytic_mel_filters(n_mels: int) -> torch.Tensor:
    mel_min = _hz_to_mel(torch.tensor([0.0]))[0]
    mel_max = _hz_to_mel(torch.tensor([SAMPLE_RATE / 2]))[0]
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = torch.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    ramps = hz_points[:, None] - fft_frequencies[None, :]
    fdiff = hz_points[1:] - hz_points[:-1]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    weights *= (2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels]))[:, None]
    return weights.float()


def mel_filters(n_mels: int, device: torch.device) -> torch.Tensor:
    cached = _MEL_FILTER_CACHE.get(n_mels)
    if cached is None:
        asset_path = _mel_filter_asset_path() if n_mels in {80, 128} else None
        if asset_path is not None:
            cached = _load_npz(asset_path, f"mel_{n_mels}")
        else:
            cached = _analytic_mel_filters(n_mels)
        _MEL_FILTER_CACHE[n_mels] = cached
    return cached.to(device)


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int) -> torch.Tensor:
    audio = pad_or_trim(audio.float())

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
            "duration": example.get("duration"),
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    pad_token_id: int
    sot_token_id: int = SOT

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
            labels = labels[:, 1:].contiguous()

        return {
            "input_features": input_features,
            "labels": labels,
            "texts": [feature.get("text") for feature in features],
            "durations": torch.tensor(
                [
                    float(feature["duration"]) if feature.get("duration") is not None else feature["input_features"].shape[-1] / (SAMPLE_RATE / HOP_LENGTH)
                    for feature in features
                ],
                dtype=torch.float32,
            ),
        }


class Corpus:
    def __init__(self, args: argparse.Namespace, tokenizer: WhisperTokenizer, n_mels: int):
        self.args = args
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(tokenizer.pad_token_id, tokenizer.sot)

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
