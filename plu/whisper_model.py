from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class WhisperConfig:
    vocab_size: int
    num_mel_bins: int = 80
    d_model: int = 384
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_ffn_dim: int = 1536
    max_source_positions: int = 1500
    max_target_positions: int = 448
    pad_token_id: int = 50257
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    decoder_start_token_id: int = 50258

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhisperConfig":
        if "dims" in data:
            data = data["dims"]
        if "n_vocab" in data:
            return cls(
                vocab_size=int(data["n_vocab"]),
                num_mel_bins=int(data.get("n_mels", 80)),
                d_model=int(data["n_audio_state"]),
                encoder_layers=int(data["n_audio_layer"]),
                encoder_attention_heads=int(data["n_audio_head"]),
                decoder_layers=int(data["n_text_layer"]),
                decoder_attention_heads=int(data["n_text_head"]),
                encoder_ffn_dim=int(data.get("n_audio_state", 384)) * 4,
                decoder_ffn_dim=int(data.get("n_text_state", 384)) * 4,
                max_source_positions=int(data["n_audio_ctx"]),
                max_target_positions=int(data["n_text_ctx"]),
            )

        return cls(
            vocab_size=int(data["vocab_size"]),
            num_mel_bins=int(data.get("num_mel_bins", data.get("n_mels", 80))),
            d_model=int(data["d_model"]),
            encoder_layers=int(data["encoder_layers"]),
            encoder_attention_heads=int(data["encoder_attention_heads"]),
            decoder_layers=int(data["decoder_layers"]),
            decoder_attention_heads=int(data["decoder_attention_heads"]),
            encoder_ffn_dim=int(data.get("encoder_ffn_dim", data["d_model"] * 4)),
            decoder_ffn_dim=int(data.get("decoder_ffn_dim", data["d_model"] * 4)),
            max_source_positions=int(data.get("max_source_positions", 1500)),
            max_target_positions=int(data.get("max_target_positions", 448)),
            pad_token_id=int(data.get("pad_token_id", data.get("eos_token_id", 50257))),
            bos_token_id=int(data.get("bos_token_id", 50257)),
            eos_token_id=int(data.get("eos_token_id", 50257)),
            decoder_start_token_id=int(data.get("decoder_start_token_id", 50258)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_type"] = "whisper"
        return data


@dataclass
class Seq2SeqOutput:
    loss: Tensor | None
    logits: Tensor


class CastLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class CastConv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class CastLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> Tensor:
    if channels % 2:
        raise ValueError("sinusoidal embeddings require an even channel count")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
    scaled_time = torch.arange(length, dtype=torch.float32)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class WhisperAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_proj = CastLinear(embed_dim, embed_dim)
        self.k_proj = CastLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = CastLinear(embed_dim, embed_dim)
        self.out_proj = CastLinear(embed_dim, embed_dim)

    def _shape(self, tensor: Tensor) -> Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: Tensor, key_value_states: Tensor | None = None, causal_mask: Tensor | None = None) -> Tensor:
        query = self._shape(self.q_proj(hidden_states))
        source = hidden_states if key_value_states is None else key_value_states
        key = self._shape(self.k_proj(source))
        value = self._shape(self.v_proj(source))

        weights = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if causal_mask is not None:
            weights = weights + causal_mask[: weights.shape[-2], : weights.shape[-1]].to(weights.device)
        weights = F.softmax(weights.float(), dim=-1).to(query.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.embed_dim)
        return self.out_proj(attended)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.self_attn = WhisperAttention(config.d_model, config.encoder_attention_heads)
        self.self_attn_layer_norm = CastLayerNorm(config.d_model)
        self.fc1 = CastLinear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = CastLinear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = CastLayerNorm(config.d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        return residual + hidden_states


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.self_attn = WhisperAttention(config.d_model, config.decoder_attention_heads)
        self.self_attn_layer_norm = CastLayerNorm(config.d_model)
        self.encoder_attn = WhisperAttention(config.d_model, config.decoder_attention_heads)
        self.encoder_attn_layer_norm = CastLayerNorm(config.d_model)
        self.fc1 = CastLinear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = CastLinear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = CastLayerNorm(config.d_model)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, causal_mask: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask=causal_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, key_value_states=encoder_hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.conv1 = CastConv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = CastConv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.embed_positions.weight.requires_grad = False
        with torch.no_grad():
            self.embed_positions.weight.copy_(sinusoids(config.max_source_positions, config.d_model))
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = CastLayerNorm(config.d_model)

    def forward(self, input_features: Tensor) -> Tensor:
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        if hidden_states.shape[1] > self.config.max_source_positions:
            raise ValueError(f"input features are too long: {hidden_states.shape[1]} > {self.config.max_source_positions}")
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        hidden_states = hidden_states + self.embed_positions(positions).to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)
        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = CastLayerNorm(config.d_model)
        mask = torch.empty(config.max_target_positions, config.max_target_positions).fill_(-float("inf")).triu_(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        if input_ids.shape[1] > self.config.max_target_positions:
            input_ids = input_ids[:, -self.config.max_target_positions :]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        hidden_states = hidden_states.to(encoder_hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, self.causal_mask)
        return self.layer_norm(hidden_states)


class WhisperModel(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


def shift_tokens_right(labels: Tensor, pad_token_id: int, decoder_start_token_id: int) -> Tensor:
    decoder_input_ids = labels.new_full(labels.shape, pad_token_id)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = decoder_start_token_id
    decoder_input_ids.masked_fill_(decoder_input_ids == -100, pad_token_id)
    return decoder_input_ids


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.model = WhisperModel(config)
        self.proj_out = CastLinear(config.d_model, config.vocab_size, bias=False)
        self.proj_out.weight = self.model.decoder.embed_tokens.weight

    def forward(self, input_features: Tensor, labels: Tensor | None = None, decoder_input_ids: Tensor | None = None) -> Seq2SeqOutput:
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("labels or decoder_input_ids are required")
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        encoder_hidden_states = self.model.encoder(input_features)
        decoder_hidden_states = self.model.decoder(decoder_input_ids, encoder_hidden_states)
        logits = self.proj_out(decoder_hidden_states).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        return Seq2SeqOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_features: Tensor, max_new_tokens: int = 255, eos_token_id: int | None = None) -> Tensor:
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        was_training = self.training
        self.eval()
        encoder_hidden_states = self.model.encoder(input_features)
        tokens = torch.full(
            (input_features.shape[0], 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_features.device,
        )
        finished = torch.zeros(input_features.shape[0], dtype=torch.bool, device=input_features.device)
        for _ in range(max_new_tokens):
            decoder_hidden_states = self.model.decoder(tokens, encoder_hidden_states)
            next_token_logits = self.proj_out(decoder_hidden_states[:, -1]).float()
            next_tokens = next_token_logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_token_id), next_tokens)
            tokens = torch.cat([tokens, next_tokens[:, None]], dim=1)
            finished |= next_tokens == eos_token_id
            if bool(finished.all()):
                break
        if was_training:
            self.train()
        return tokens

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | os.PathLike[str], map_location: str | torch.device = "cpu") -> "WhisperForConditionalGeneration":
        model_path = resolve_model_path(model_name_or_path)
        config = load_config(model_path)
        model = cls(config)
        state_dict = load_pretrained_state_dict(model_path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        tolerated_missing = {"proj_out.weight"}
        missing = [key for key in missing if key not in tolerated_missing]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint did not match Whisper model; missing={missing[:20]}, unexpected={unexpected[:20]}")
        return model

    def save_pretrained(self, output_dir: str | os.PathLike[str]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), output_path / "pytorch_model.bin")


def _hf_cache_root() -> Path:
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


def _cached_snapshot(repo_id: str) -> Path | None:
    snapshots = _hf_cache_root() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, path.open("wb") as f:
            f.write(response.read())
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def _download_hf_repo(repo_id: str) -> Path:
    output_dir = Path(os.environ.get("PLU_MODEL_CACHE", "~/.cache/plu/models")).expanduser() / repo_id.replace("/", "--")
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    if not (output_dir / "config.json").exists():
        _download(f"{base_url}/config.json", output_dir / "config.json")

    for filename in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
        target = output_dir / filename
        if not target.exists():
            _download(f"{base_url}/{filename}", target)

    if (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists():
        return output_dir

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = output_dir / index_name
        if not index_path.exists() and not _download(f"{base_url}/{index_name}", index_path):
            continue
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        for shard in sorted(set(index.get("weight_map", {}).values())):
            shard_path = output_dir / shard
            if not shard_path.exists():
                _download(f"{base_url}/{shard}", shard_path)
        return output_dir

    for filename in ("model.safetensors", "pytorch_model.bin"):
        if _download(f"{base_url}/{filename}", output_dir / filename):
            return output_dir

    raise FileNotFoundError(f"could not find model weights for {repo_id}")


def resolve_model_path(model_name_or_path: str | os.PathLike[str]) -> Path:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return path
    repo_id = str(model_name_or_path)
    cached = _cached_snapshot(repo_id)
    if cached is not None:
        return cached
    return _download_hf_repo(repo_id)


def load_config(model_path: str | os.PathLike[str]) -> WhisperConfig:
    path = Path(model_path)
    if path.is_file():
        if path.suffix in {".pt", ".bin"}:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "dims" in checkpoint:
                return WhisperConfig.from_dict({"dims": checkpoint["dims"].__dict__ if hasattr(checkpoint["dims"], "__dict__") else checkpoint["dims"]})
        path = path.parent
    with (path / "config.json").open("r", encoding="utf-8") as f:
        return WhisperConfig.from_dict(json.load(f))


_SAFETENSOR_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    raw = Path(path).read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    data_offset = 8 + header_size
    tensors: dict[str, Tensor] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFETENSOR_DTYPES[metadata["dtype"]]
        start, end = metadata["data_offsets"]
        buffer = memoryview(raw)[data_offset + start : data_offset + end]
        tensors[name] = torch.frombuffer(buffer, dtype=dtype).reshape(metadata["shape"]).clone()
    return tensors


def _load_sharded_state_dict(model_path: Path, index_name: str, map_location: str | torch.device) -> dict[str, Tensor]:
    with (model_path / index_name).open("r", encoding="utf-8") as f:
        index = json.load(f)
    state: dict[str, Tensor] = {}
    for shard in sorted(set(index.get("weight_map", {}).values())):
        shard_path = model_path / shard
        if shard_path.suffix == ".safetensors":
            state.update(load_safetensors(shard_path))
        else:
            state.update(torch.load(shard_path, map_location=map_location))
    return state


def load_pretrained_state_dict(model_path: str | os.PathLike[str], map_location: str | torch.device = "cpu") -> dict[str, Tensor]:
    path = Path(model_path)
    if path.is_file():
        if path.suffix == ".safetensors":
            return load_safetensors(path)
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            return loaded["model_state_dict"]
        if isinstance(loaded, dict) and "model_state" in loaded:
            return loaded["model_state"]
        if isinstance(loaded, dict) and "state_dict" in loaded:
            return loaded["state_dict"]
        if isinstance(loaded, dict) and "model_state_dict" not in loaded:
            return loaded
        raise RuntimeError(f"unsupported checkpoint format: {path}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / index_name).exists():
            return _load_sharded_state_dict(path, index_name, map_location)

    if (path / "model.safetensors").exists():
        return load_safetensors(path / "model.safetensors")
    if (path / "pytorch_model.bin").exists():
        return torch.load(path / "pytorch_model.bin", map_location=map_location, weights_only=False)
    raise FileNotFoundError(f"no supported model weights found in {path}")
