from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egs.uk1e2.local.decode_windows_batched import batched_log_mel_spectrogram
from plu.train_data import N_FFT, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, mel_filters


def test_batched_log_mel_spectrogram_matches_per_window_cpu():
    generator = torch.Generator().manual_seed(0)
    segments = [
        torch.randn(SAMPLE_RATE // 2, generator=generator),
        torch.randn(SAMPLE_RATE * 3 + 17, generator=generator),
        torch.randn(SAMPLE_RATE * 30 + 256, generator=generator),
    ]
    device = torch.device("cpu")
    actual = batched_log_mel_spectrogram(
        segments,
        80,
        device,
        torch.float32,
        window=torch.hann_window(N_FFT, device=device),
        mel_basis=mel_filters(80, device),
        audio_buffer=torch.empty((len(segments), N_SAMPLES), dtype=torch.float32),
        device_audio_buffer=torch.empty((len(segments), N_SAMPLES), device=device, dtype=torch.float32),
    )
    expected = torch.stack([log_mel_spectrogram(segment, 80) for segment in segments])
    torch.testing.assert_close(actual, expected)
