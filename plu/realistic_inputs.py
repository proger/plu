from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from plu.train_data import load_audio, log_mel_spectrogram


DEFAULT_WAV = "data/segments/wav/S01496-N00113931549-U0000002-0004842-0007981.wav"


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def collect_realistic_inputs(model_name_or_path: str, wav_path: str | Path = DEFAULT_WAV, device: str = "cuda") -> dict[str, Any]:
    os.environ["PLU_OPS_BACKEND"] = "ref"
    sys.modules.pop("plu.whisper", None)
    from plu.whisper import WhisperAttention, WhisperEncoderLayer, WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def keep(name: str, tensor: torch.Tensor) -> None:
        if name not in captured:
            captured[name] = tensor.detach().cpu()

    def attention_pre_hook(module, args, kwargs):
        hidden_states = args[0]
        key_value_states = kwargs.get("key_value_states")
        if key_value_states is None and len(args) > 1:
            key_value_states = args[1]
        source = hidden_states if key_value_states is None else key_value_states
        keep("qkv_proj.x", hidden_states)
        keep("qkv_proj.source", source)

    def encoder_layer_pre_hook(module, args):
        keep("residual_add.residual", args[0])

    def final_layer_norm_hook(module, args, output):
        keep("gelu_mlp.x", output)
        keep("layer_norm.out", output)

    hooks = []
    for module in model.modules():
        if isinstance(module, WhisperAttention):
            hooks.append(module.register_forward_pre_hook(attention_pre_hook, with_kwargs=True))
        if isinstance(module, WhisperEncoderLayer):
            hooks.append(module.register_forward_pre_hook(encoder_layer_pre_hook))
            hooks.append(module.final_layer_norm.register_forward_hook(final_layer_norm_hook))

    audio = load_audio(wav_path)
    features = log_mel_spectrogram(audio, model.config.num_mel_bins).unsqueeze(0).to(device)
    decoder_input_ids = torch.full((1, 4), model.config.decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(features, decoder_input_ids=decoder_input_ids)
    keep("input_features", features)
    keep("logits", output.logits)

    for hook in hooks:
        hook.remove()

    return {"summaries": {name: _summary(tensor) for name, tensor in sorted(captured.items())}, "tensors": captured}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect realistic operation inputs from the reference Whisper model.")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("realistic_inputs.pt"))
    args = parser.parse_args()

    payload = collect_realistic_inputs(args.model, args.wav, args.device)
    torch.save(payload["tensors"], args.out)
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
