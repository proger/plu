from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path

import torch

from plu.triton.sampling import make_suppress_mask, sample_next_token, update_decode_control_
from plu.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, WhisperTokenizer
from plu.train_data import SAMPLE_RATE, load_audio, log_mel_spectrogram
from plu.whisper import WhisperForConditionalGeneration, resolve_model_path


logger = getLogger(__name__)

MODEL_ALIASES = {
    "tiny": "openai/whisper-tiny",
    "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base",
    "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small",
    "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium",
    "medium.en": "openai/whisper-medium.en",
    "large": "openai/whisper-large",
    "large-v1": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a PLU Whisper model with PLU decoding.")
    parser.add_argument(
        "--exp",
        type=Path,
        help="Experiment directory containing a PLU model checkpoint.",
        default=Path("exp/1"),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Skip --exp and test this PLU model directory, HF model ID, or alias such as large-v3-turbo.",
        default=None,
    )
    parser.add_argument(
        "--download_root",
        type=Path,
        help="Directory where PLU should cache downloaded Hugging Face model files.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for PLU inference: cuda or auto. MXFP8 decoding requires CUDA.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16"],
        help="Activation dtype for PLU inference.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional Whisper language code for the transcription prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=255,
        help="Maximum number of generated tokens per file.",
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=1,
        help="Number of decoder alternatives to run for each encoded file. B=1 is greedy; B>1 samples at temperature 1.",
    )
    parser.add_argument(
        "filenames",
        type=Path,
        nargs="*",
        help="Path to the audio file(s) to transcribe.",
    )
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type != "cuda":
        raise RuntimeError("PLU +test stores linear weights as MXFP8 and requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("PLU +test requires CUDA, but torch.cuda.is_available() is false")
    return device


def canonical_model_name(model_name_or_path: str) -> str:
    return MODEL_ALIASES.get(model_name_or_path, model_name_or_path)


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.lower()
    if normalized not in LANGUAGES and normalized in TO_LANGUAGE_CODE:
        normalized = TO_LANGUAGE_CODE[normalized]
    if normalized not in LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    return normalized


def resolve_requested_model(args: argparse.Namespace) -> Path:
    if args.download_root is not None:
        os.environ["PLU_MODEL_CACHE"] = str(args.download_root.expanduser())

    requested = args.model if args.model is not None else str(args.exp)
    requested = canonical_model_name(requested)
    path = Path(requested).expanduser()
    if path.exists():
        return path

    return resolve_model_path(requested)


def configure_tokenizer(model_path: Path, model: WhisperForConditionalGeneration, language: str | None) -> WhisperTokenizer:
    tokenizer = WhisperTokenizer.from_pretrained(model_path, pad_token_id=model.config.pad_token_id)
    return WhisperTokenizer(
        encoding=tokenizer.encoding,
        num_languages=tokenizer.num_languages,
        language=language,
        task="transcribe",
        source_dir=tokenizer.source_dir,
        pad_token_id=model.config.pad_token_id,
    )


def prompt_ids(
    tokenizer: WhisperTokenizer,
    language: str | None,
    *,
    without_timestamps: bool = True,
    previous_tokens: Sequence[int] | None = None,
    max_previous_tokens: int | None = None,
) -> list[int]:
    prompt = [tokenizer.sot]
    if language:
        prompt.append(tokenizer.to_language_token(language))
    prompt.append(tokenizer.transcribe)
    if without_timestamps:
        prompt.append(tokenizer.no_timestamps)
    if previous_tokens:
        context = [int(token) for token in previous_tokens]
        if max_previous_tokens is not None:
            context = context[-max_previous_tokens:]
        prompt = [tokenizer.sot_prev, *context, *prompt]
    return prompt


def load_input_features(filename: Path, n_mels: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    audio = load_audio(filename)
    if audio.ndim == 2 and audio.shape[0] == n_mels:
        features = audio.float()
        duration = float(features.shape[-1]) / 100.0
    else:
        audio = audio.flatten().float()
        duration = float(audio.numel()) / SAMPLE_RATE
        features = log_mel_spectrogram(audio, n_mels)
    return features.unsqueeze(0).to(device=device, dtype=dtype), duration


class Mxfp8KvCacheCudaGraphSampler:
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        tokenizer: WhisperTokenizer,
        packed: dict[int, object],
        prompt: list[int],
        suppress_tokens: list[int],
        decode_batch_size: int = 1,
        timestamps_after_sentence_end: bool = False,
        cross_cache_batch_size: int | None = None,
        cross_key_caches: list[torch.Tensor] | None = None,
        cross_value_caches: list[torch.Tensor] | None = None,
    ):
        if decode_batch_size < 1:
            raise ValueError("--decode_batch_size must be at least 1")
        if cross_cache_batch_size is None:
            cross_cache_batch_size = decode_batch_size
        if cross_cache_batch_size < 1:
            raise ValueError("cross_cache_batch_size must be at least 1")
        if (cross_key_caches is None) != (cross_value_caches is None):
            raise ValueError("cross_key_caches and cross_value_caches must be provided together")
        self.model = model
        self.tokenizer = tokenizer
        self.packed = packed
        self.prompt = prompt
        self.decode_batch_size = decode_batch_size
        self.cross_cache_batch_size = cross_cache_batch_size
        self.timestamps_after_sentence_end = timestamps_after_sentence_end
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.max_target_positions = model.config.max_target_positions
        self.encoder_positions = model.config.max_source_positions
        self.hidden_size = model.config.d_model
        self.eos_token_id = model.config.eos_token_id
        self.no_speech_token = tokenizer.no_speech
        self.timestamp_begin = tokenizer.timestamp_begin
        self.suppress_tokens = torch.tensor(
            [token for token in suppress_tokens if 0 <= token < model.config.vocab_size],
            device=self.device,
            dtype=torch.long,
        )
        self.suppress_mask = make_suppress_mask(model.config.vocab_size, self.suppress_tokens, device=self.device)
        sentence_end_token_mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
        for token in range(min(self.timestamp_begin, model.config.vocab_size)):
            if token != self.eos_token_id and tokenizer.decode([token]).rstrip().endswith((".", "!", "?")):
                sentence_end_token_mask[token] = True
        self.sentence_end_token_mask = sentence_end_token_mask.to(device=self.device)
        self.static_current_token = torch.full((self.decode_batch_size,), self.eos_token_id, device=self.device, dtype=torch.long)
        self.static_position = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.long)
        self.static_force_timestamp = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.bool)
        self.static_pair_timestamp = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.bool)
        self.static_pair_timestamp_token = torch.full(
            (self.decode_batch_size,),
            self.timestamp_begin,
            device=self.device,
            dtype=torch.long,
        )
        self.greedy_batch_mask = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.bool)
        self.greedy_batch_mask[0] = True
        self.static_min_timestamp = torch.full(
            (self.decode_batch_size,),
            self.timestamp_begin,
            device=self.device,
            dtype=torch.long,
        )
        self.static_sampling_seed = torch.zeros((), device=self.device, dtype=torch.int64)
        self._identity_cross_cache_index = torch.arange(self.decode_batch_size, device=self.device, dtype=torch.long)
        self.static_cross_cache_index = self._identity_cross_cache_index % self.cross_cache_batch_size
        self._grouped_cross_cache_index_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._sampling_seed_counter = 0
        self.generated_token_buffer = torch.empty(
            (self.decode_batch_size, self.max_target_positions),
            device=self.device,
            dtype=torch.long,
        )
        self.generated_logprob_buffer = torch.empty(
            (self.decode_batch_size, self.max_target_positions),
            device=self.device,
            dtype=torch.float32,
        )
        self.generated_lengths = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.long)
        self.finished = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.bool)
        self.no_speech_probs = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.float32)
        self.all_finished = torch.zeros((), device=self.device, dtype=torch.bool)

        decoder = model.model.decoder
        self.self_key_caches = []
        self.self_value_caches = []
        self.cross_key_caches = []
        self.cross_value_caches = []
        if cross_key_caches is not None and len(cross_key_caches) != len(decoder.layers):
            raise ValueError(f"expected {len(decoder.layers)} shared cross key caches, got {len(cross_key_caches)}")
        if cross_value_caches is not None and len(cross_value_caches) != len(decoder.layers):
            raise ValueError(f"expected {len(decoder.layers)} shared cross value caches, got {len(cross_value_caches)}")
        for layer_index, layer in enumerate(decoder.layers):
            heads = layer.self_attn.num_heads
            head_dim = layer.self_attn.head_dim
            self.self_key_caches.append(
                torch.empty((self.decode_batch_size, heads, self.max_target_positions, head_dim), device=self.device, dtype=self.dtype)
            )
            self.self_value_caches.append(
                torch.empty((self.decode_batch_size, heads, self.max_target_positions, head_dim), device=self.device, dtype=self.dtype)
            )
            expected_cross_shape = (self.cross_cache_batch_size, heads, self.encoder_positions, head_dim)
            if cross_key_caches is None:
                self.cross_key_caches.append(torch.empty(expected_cross_shape, device=self.device, dtype=self.dtype))
                self.cross_value_caches.append(torch.empty(expected_cross_shape, device=self.device, dtype=self.dtype))
            else:
                key_cache = cross_key_caches[layer_index]
                value_cache = cross_value_caches[layer_index]
                if key_cache.shape != expected_cross_shape or value_cache.shape != expected_cross_shape:
                    raise ValueError(
                        "shared cross caches have incompatible shapes "
                        f"at layer {layer_index}: expected {expected_cross_shape}, got {tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
                    )
                if key_cache.device != self.device or value_cache.device != self.device:
                    raise ValueError("shared cross caches must be on the sampler device")
                if key_cache.dtype != self.dtype or value_cache.dtype != self.dtype:
                    raise ValueError("shared cross caches must match the sampler dtype")
                self.cross_key_caches.append(key_cache)
                self.cross_value_caches.append(value_cache)

        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_next_token: torch.Tensor | None = None
        self.graph_selected_logprob: torch.Tensor | None = None
        self.graph_no_speech_prob: torch.Tensor | None = None
        self.prompt_graph: torch.cuda.CUDAGraph | None = None
        self.prompt_graph_next_token: torch.Tensor | None = None
        self.prompt_graph_selected_logprob: torch.Tensor | None = None
        self.prompt_graph_no_speech_prob: torch.Tensor | None = None
        self.capture_ms = 0.0
        self._capture()

    def _grouped_cross_cache_index(self, prompt_count: int, alternatives_per_prompt: int) -> torch.Tensor:
        key = (prompt_count, alternatives_per_prompt)
        cached = self._grouped_cross_cache_index_cache.get(key)
        if cached is None:
            cached = torch.arange(prompt_count, device=self.device, dtype=torch.long).repeat_interleave(alternatives_per_prompt)
            self._grouped_cross_cache_index_cache[key] = cached
        return cached

    def _project_heads(self, module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        from plu.benchmarks.bench_end_to_end import _packed_linear

        projected = _packed_linear(self.packed, module, hidden_states, master_weight_grads=False)
        batch, seq_len, features = projected.shape
        heads = self.model.config.decoder_attention_heads
        head_dim = features // heads
        return projected.reshape(batch, seq_len, heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = hidden_states.shape
        return hidden_states.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, heads * head_dim)

    def _token_embedding(self) -> torch.Tensor:
        decoder = self.model.model.decoder
        token_embedding = decoder.embed_tokens(self.static_current_token).to(dtype=self.dtype)
        position_embedding = decoder.embed_positions(self.static_position).to(dtype=self.dtype)
        return (token_embedding + position_embedding).view(self.decode_batch_size, 1, self.hidden_size)

    @torch.no_grad()
    def prepare_encoder(self, encoder_hidden_states: torch.Tensor) -> None:
        from plu.benchmarks.bench_end_to_end import _packed_qkv

        decoder = self.model.model.decoder
        for index, layer in enumerate(decoder.layers):
            key = _packed_qkv(self.packed, layer.encoder_attn.k_proj, encoder_hidden_states, layer.encoder_attn.num_heads, master_weight_grads=False)
            value = _packed_qkv(self.packed, layer.encoder_attn.v_proj, encoder_hidden_states, layer.encoder_attn.num_heads, master_weight_grads=False)
            key_cache = self.cross_key_caches[index]
            value_cache = self.cross_value_caches[index]
            if key.shape[0] == 1 and self.cross_cache_batch_size > 1:
                key = key.expand_as(key_cache)
                value = value.expand_as(value_cache)
            elif key.shape[0] != self.cross_cache_batch_size and self.cross_cache_batch_size % key.shape[0] == 0:
                repeats = self.cross_cache_batch_size // key.shape[0]
                key = key.repeat_interleave(repeats, dim=0)
                value = value.repeat_interleave(repeats, dim=0)
            if key.shape != key_cache.shape or value.shape != value_cache.shape:
                raise ValueError(
                    "KV-cache sampler requires static encoder cache shapes "
                    f"{tuple(key_cache.shape)}, got {tuple(key.shape)}"
                )
            key_cache.copy_(key)
            value_cache.copy_(value)

    @torch.no_grad()
    def _decoder_step(self, *, compute_no_speech: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from plu.benchmarks.bench_end_to_end import _packed_gelu_mlp, _packed_linear
        from plu.triton.decode_attention import cross_kv_cache_attention, self_kv_cache_attention
        from plu.triton.layer_norm import layer_norm
        from plu.triton.residual_add import residual_add

        hidden_states = self._token_embedding()
        decoder = self.model.model.decoder
        for index, layer in enumerate(decoder.layers):
            residual = hidden_states
            normed = layer_norm(hidden_states, layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias, layer.self_attn_layer_norm.eps)
            query = self._project_heads(layer.self_attn.q_proj, normed)
            key = self._project_heads(layer.self_attn.k_proj, normed)
            value = self._project_heads(layer.self_attn.v_proj, normed)
            attended = self_kv_cache_attention(
                query,
                key,
                value,
                self.self_key_caches[index],
                self.self_value_caches[index],
                self.static_position,
            )
            hidden_states = _packed_linear(self.packed, layer.self_attn.out_proj, self._merge_heads(attended), master_weight_grads=False)
            hidden_states = residual_add(residual, hidden_states)

            residual = hidden_states
            normed = layer_norm(hidden_states, layer.encoder_attn_layer_norm.weight, layer.encoder_attn_layer_norm.bias, layer.encoder_attn_layer_norm.eps)
            query = self._project_heads(layer.encoder_attn.q_proj, normed)
            cross_cache_index = self.static_cross_cache_index if self.cross_cache_batch_size != self.decode_batch_size else None
            attended = cross_kv_cache_attention(query, self.cross_key_caches[index], self.cross_value_caches[index], cross_cache_index)
            hidden_states = _packed_linear(self.packed, layer.encoder_attn.out_proj, self._merge_heads(attended), master_weight_grads=False)
            hidden_states = residual_add(residual, hidden_states)

            residual = hidden_states
            normed = layer_norm(hidden_states, layer.final_layer_norm.weight, layer.final_layer_norm.bias, layer.final_layer_norm.eps)
            hidden_states = _packed_gelu_mlp(self.packed, layer.fc1, layer.fc2, normed, master_weight_grads=False)
            hidden_states = residual_add(residual, hidden_states)

        decoder_hidden_states = layer_norm(hidden_states, decoder.layer_norm.weight, decoder.layer_norm.bias, decoder.layer_norm.eps)
        logits = _packed_linear(self.packed, self.model.proj_out, decoder_hidden_states, master_weight_grads=False).float().squeeze(1)
        if compute_no_speech:
            raw_logprobs = logits.log_softmax(dim=-1)
            no_speech_prob = raw_logprobs[:, self.no_speech_token].exp()
        else:
            no_speech_prob = logits.new_zeros((self.decode_batch_size,), dtype=torch.float32)
        next_token, selected_logprob = sample_next_token(
            logits,
            timestamp_begin=self.timestamp_begin,
            suppress_mask=self.suppress_mask,
            timestamps_after_sentence_end=self.timestamps_after_sentence_end,
            min_timestamp_tokens=self.static_min_timestamp,
            force_timestamp=self.static_force_timestamp,
            pair_timestamp=self.static_pair_timestamp,
            pair_timestamp_tokens=self.static_pair_timestamp_token,
            greedy_batch_mask=None if self.decode_batch_size == 1 else self.greedy_batch_mask,
            seed=self.static_sampling_seed,
        )
        return next_token, selected_logprob, no_speech_prob

    def _capture(self) -> None:
        self.static_current_token.fill_(self.prompt[0])
        self.static_position.zero_()
        for key_cache, value_cache in zip(self.self_key_caches, self.self_value_caches):
            key_cache.zero_()
            value_cache.zero_()
        for key_cache, value_cache in zip(self.cross_key_caches, self.cross_value_caches):
            key_cache.zero_()
            value_cache.zero_()
        for _ in range(3):
            self._decoder_step()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        prompt_graph = torch.cuda.CUDAGraph()
        capture_start = torch.cuda.Event(enable_timing=True)
        capture_end = torch.cuda.Event(enable_timing=True)
        capture_start.record()
        with torch.cuda.graph(graph):
            self.graph_next_token, self.graph_selected_logprob, self.graph_no_speech_prob = self._decoder_step(compute_no_speech=False)
        with torch.cuda.graph(prompt_graph):
            self.prompt_graph_next_token, self.prompt_graph_selected_logprob, self.prompt_graph_no_speech_prob = self._decoder_step(
                compute_no_speech=True
            )
        capture_end.record()
        torch.cuda.synchronize()
        self.graph = graph
        self.prompt_graph = prompt_graph
        self.capture_ms = capture_start.elapsed_time(capture_end)

    def _set_replay_inputs(
        self,
        tokens: int | list[int] | torch.Tensor,
        position: int | list[int] | torch.Tensor,
        *,
        force_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
        min_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
    ) -> None:
        self._set_replay_token_position(tokens, position)
        self._set_replay_control(
            force_timestamps=force_timestamps,
            pair_timestamps=pair_timestamps,
            pair_timestamp_tokens=pair_timestamp_tokens,
            min_timestamp_tokens=min_timestamp_tokens,
        )

    def _set_replay_token_position(
        self,
        tokens: int | list[int] | torch.Tensor,
        position: int | list[int] | torch.Tensor,
    ) -> None:
        if isinstance(tokens, int):
            self.static_current_token.fill_(tokens)
        elif isinstance(tokens, torch.Tensor):
            self.static_current_token.copy_(tokens.to(device=self.device, dtype=torch.long))
        else:
            if len(tokens) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} tokens, got {len(tokens)}")
            self.static_current_token.copy_(torch.tensor(tokens, device=self.device, dtype=torch.long))
        if isinstance(position, int):
            self.static_position.fill_(position)
        elif isinstance(position, torch.Tensor):
            self.static_position.copy_(position.to(device=self.device, dtype=torch.long))
        else:
            if len(position) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} positions, got {len(position)}")
            self.static_position.copy_(torch.tensor(position, device=self.device, dtype=torch.long))

    def _set_replay_control(
        self,
        *,
        force_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
        min_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
    ) -> None:
        if isinstance(force_timestamps, bool):
            self.static_force_timestamp.fill_(force_timestamps)
        elif isinstance(force_timestamps, torch.Tensor):
            self.static_force_timestamp.copy_(force_timestamps.to(device=self.device, dtype=torch.bool))
        else:
            if len(force_timestamps) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} timestamp-force flags, got {len(force_timestamps)}")
            self.static_force_timestamp.copy_(torch.tensor(force_timestamps, device=self.device, dtype=torch.bool))
        if isinstance(pair_timestamps, bool):
            self.static_pair_timestamp.fill_(pair_timestamps)
        elif isinstance(pair_timestamps, torch.Tensor):
            self.static_pair_timestamp.copy_(pair_timestamps.to(device=self.device, dtype=torch.bool))
        else:
            if len(pair_timestamps) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} timestamp-pair flags, got {len(pair_timestamps)}")
            self.static_pair_timestamp.copy_(torch.tensor(pair_timestamps, device=self.device, dtype=torch.bool))
        if pair_timestamp_tokens is None:
            self.static_pair_timestamp_token.fill_(self.timestamp_begin)
        elif isinstance(pair_timestamp_tokens, int):
            self.static_pair_timestamp_token.fill_(pair_timestamp_tokens)
        elif isinstance(pair_timestamp_tokens, torch.Tensor):
            self.static_pair_timestamp_token.copy_(pair_timestamp_tokens.to(device=self.device, dtype=torch.long))
        else:
            if len(pair_timestamp_tokens) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} timestamp-pair tokens, got {len(pair_timestamp_tokens)}")
            self.static_pair_timestamp_token.copy_(torch.tensor(pair_timestamp_tokens, device=self.device, dtype=torch.long))
        if min_timestamp_tokens is None:
            self.static_min_timestamp.fill_(self.timestamp_begin)
        elif isinstance(min_timestamp_tokens, int):
            self.static_min_timestamp.fill_(min_timestamp_tokens)
        elif isinstance(min_timestamp_tokens, torch.Tensor):
            self.static_min_timestamp.copy_(min_timestamp_tokens.to(device=self.device, dtype=torch.long))
        else:
            if len(min_timestamp_tokens) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} minimum timestamp tokens, got {len(min_timestamp_tokens)}")
            self.static_min_timestamp.copy_(torch.tensor(min_timestamp_tokens, device=self.device, dtype=torch.long))

    def _replay_captured_graph(self, *, compute_no_speech: bool) -> None:
        if compute_no_speech:
            assert self.prompt_graph is not None
            assert self.prompt_graph_next_token is not None
            assert self.prompt_graph_selected_logprob is not None
            assert self.prompt_graph_no_speech_prob is not None
            self.prompt_graph.replay()
            self.graph_next_token.copy_(self.prompt_graph_next_token)
            self.graph_selected_logprob.copy_(self.prompt_graph_selected_logprob)
            self.graph_no_speech_prob.copy_(self.prompt_graph_no_speech_prob)
        else:
            self.graph.replay()

    def _replay_device(
        self,
        tokens: int | list[int] | torch.Tensor,
        position: int | list[int] | torch.Tensor,
        *,
        force_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
        min_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
        compute_no_speech: bool = False,
    ) -> None:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if compute_no_speech and (
            self.prompt_graph is None
            or self.prompt_graph_next_token is None
            or self.prompt_graph_selected_logprob is None
            or self.prompt_graph_no_speech_prob is None
        ):
            raise RuntimeError("prompt CUDA graph sampler has not been captured")
        self._set_replay_inputs(
            tokens,
            position,
            force_timestamps=force_timestamps,
            pair_timestamps=pair_timestamps,
            pair_timestamp_tokens=pair_timestamp_tokens,
            min_timestamp_tokens=min_timestamp_tokens,
        )
        self._sampling_seed_counter = (self._sampling_seed_counter + 1) & 0x7FFFFFFF
        self.static_sampling_seed.fill_(self._sampling_seed_counter)
        self._replay_captured_graph(compute_no_speech=compute_no_speech)

    def _replay_device_static_control(
        self,
        tokens: int | list[int] | torch.Tensor,
        position: int | list[int] | torch.Tensor,
        *,
        compute_no_speech: bool = False,
    ) -> None:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if compute_no_speech and (
            self.prompt_graph is None
            or self.prompt_graph_next_token is None
            or self.prompt_graph_selected_logprob is None
            or self.prompt_graph_no_speech_prob is None
        ):
            raise RuntimeError("prompt CUDA graph sampler has not been captured")
        self._set_replay_token_position(tokens, position)
        self._sampling_seed_counter = (self._sampling_seed_counter + 1) & 0x7FFFFFFF
        self.static_sampling_seed.fill_(self._sampling_seed_counter)
        self._replay_captured_graph(compute_no_speech=compute_no_speech)

    def _replay(
        self,
        tokens: int | list[int] | torch.Tensor,
        position: int | list[int] | torch.Tensor,
        *,
        force_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamps: bool | list[bool] | torch.Tensor = False,
        pair_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
        min_timestamp_tokens: int | list[int] | torch.Tensor | None = None,
    ) -> tuple[list[int], list[float], list[float]]:
        if self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        self._replay_device(
            tokens,
            position,
            force_timestamps=force_timestamps,
            pair_timestamps=pair_timestamps,
            pair_timestamp_tokens=pair_timestamp_tokens,
            min_timestamp_tokens=min_timestamp_tokens,
        )
        next_tokens = self.graph_next_token.detach().cpu().tolist()
        selected_logprobs = self.graph_selected_logprob.detach().cpu().tolist()
        no_speech_probs = self.graph_no_speech_prob.detach().cpu().tolist()
        return next_tokens, selected_logprobs, no_speech_probs

    def sample_many(
        self,
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int,
        prompt: list[int] | None = None,
        stop_after_first_sentence: bool = False,
    ) -> list[tuple[list[int], list[float], float]]:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if encoder_hidden_states.shape[0] == 1:
            self.static_cross_cache_index.zero_()
        elif encoder_hidden_states.shape[0] == self.decode_batch_size and self.cross_cache_batch_size == self.decode_batch_size:
            self.static_cross_cache_index.copy_(self._identity_cross_cache_index)
        else:
            raise ValueError(
                "sample_many expects one encoder batch shared by all rows, or one encoder batch per row "
                f"with full cross cache; got encoder batch {encoder_hidden_states.shape[0]} and cross cache batch {self.cross_cache_batch_size}"
            )
        self.prepare_encoder(encoder_hidden_states)

        prompt = self.prompt if prompt is None else prompt
        prompt_len = min(len(prompt), self.max_target_positions)
        self.generated_lengths.zero_()
        self.finished.zero_()
        self.no_speech_probs.zero_()
        self.all_finished.fill_(False)
        self.greedy_batch_mask.zero_()
        self.greedy_batch_mask[0] = True
        self.static_current_token.fill_(self.eos_token_id)
        self.static_position.zero_()
        self.static_force_timestamp.zero_()
        self.static_pair_timestamp.zero_()
        self.static_pair_timestamp_token.fill_(self.timestamp_begin)
        self.static_min_timestamp.fill_(self.timestamp_begin)

        for position, token in enumerate(prompt[: max(prompt_len - 1, 0)]):
            self._replay_device_static_control(token, position)
        if prompt_len:
            final_position = prompt_len - 1
            force_initial_timestamp = self.timestamps_after_sentence_end
            self._replay_device(
                prompt[final_position],
                final_position,
                force_timestamps=force_initial_timestamp,
                pair_timestamps=False,
                pair_timestamp_tokens=self.timestamp_begin,
                min_timestamp_tokens=self.timestamp_begin,
                compute_no_speech=True,
            )
            self.no_speech_probs.copy_(self.graph_no_speech_prob)

        max_steps = min(max_new_tokens, self.generated_token_buffer.shape[1])
        if prompt_len > 0 and max_steps > 0:
            finish_check_interval = 8
            self.static_position.fill_(prompt_len - 1)
            for step in range(max_steps):
                update_decode_control_(
                    self.graph_next_token,
                    self.graph_selected_logprob,
                    self.generated_token_buffer,
                    self.generated_logprob_buffer,
                    self.generated_lengths,
                    self.finished,
                    self.static_current_token,
                    self.static_force_timestamp,
                    self.static_pair_timestamp,
                    self.static_pair_timestamp_token,
                    self.static_min_timestamp,
                    self.static_position,
                    self.static_sampling_seed,
                    self.all_finished,
                    self.sentence_end_token_mask,
                    eos_token_id=self.eos_token_id,
                    timestamp_begin=self.timestamp_begin,
                    timestamps_after_sentence_end=self.timestamps_after_sentence_end,
                    stop_after_first_sentence=stop_after_first_sentence,
                )
                must_stop_for_length = step + 1 >= max_steps or prompt_len + step >= self.max_target_positions
                check_finished = must_stop_for_length or (step + 1) % finish_check_interval == 0
                if check_finished and bool(self.all_finished.detach().cpu().item()):
                    break
                if must_stop_for_length:
                    break
                self.graph.replay()

        prompt = prompt[:prompt_len]
        lengths = self.generated_lengths.detach().cpu().tolist()
        generated_tokens = self.generated_token_buffer.detach().cpu()
        generated_logprobs = self.generated_logprob_buffer.detach().cpu()
        no_speech_probs = self.no_speech_probs.detach().cpu().tolist()
        self._sampling_seed_counter = int(self.static_sampling_seed.detach().cpu().item())
        return [
            (
                prompt + generated_tokens[index, :length].tolist(),
                generated_logprobs[index, :length].tolist(),
                no_speech_probs[index],
            )
            for index, length in enumerate(lengths)
        ]

    def prefill_prompt_caches(self, encoder_hidden_states: torch.Tensor, prompts: list[list[int]]) -> torch.Tensor:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if not prompts:
            raise ValueError("prompts must not be empty")
        if len(prompts) != self.decode_batch_size:
            raise ValueError(f"expected {self.decode_batch_size} prompts, got {len(prompts)}")
        if encoder_hidden_states.shape[0] != len(prompts):
            raise ValueError(f"expected {len(prompts)} encoder batches, got {encoder_hidden_states.shape[0]}")
        if self.cross_cache_batch_size < len(prompts):
            raise ValueError(f"cross cache batch {self.cross_cache_batch_size} is smaller than prompt batch {len(prompts)}")
        self.static_cross_cache_index.copy_(self._identity_cross_cache_index[: len(prompts)])
        self.prepare_encoder(encoder_hidden_states)

        prompts = [prompt[: self.max_target_positions] for prompt in prompts]
        if any(not prompt for prompt in prompts):
            raise ValueError("prompts must not be empty")
        prompt_lens = torch.tensor([len(prompt) for prompt in prompts], device=self.device, dtype=torch.long)
        max_nonfinal_len = max(max(len(prompt) - 1, 0) for prompt in prompts)
        if max_nonfinal_len == 0:
            return prompt_lens

        prompt_token_rows: list[list[int]] = []
        prompt_position_rows: list[list[int]] = []
        for position in range(max_nonfinal_len):
            tokens_row: list[int] = []
            positions_row: list[int] = []
            for prompt in prompts:
                prompt_position = min(position, max(len(prompt) - 2, 0))
                tokens_row.append(int(prompt[prompt_position]))
                positions_row.append(prompt_position)
            prompt_token_rows.append(tokens_row)
            prompt_position_rows.append(positions_row)

        prompt_tokens = torch.tensor(prompt_token_rows, device=self.device, dtype=torch.long)
        prompt_positions = torch.tensor(prompt_position_rows, device=self.device, dtype=torch.long)
        self.static_force_timestamp.zero_()
        self.static_pair_timestamp.zero_()
        self.static_pair_timestamp_token.fill_(self.timestamp_begin)
        self.static_min_timestamp.fill_(self.timestamp_begin)
        for position in range(max_nonfinal_len):
            self._replay_device_static_control(
                prompt_tokens[position],
                prompt_positions[position],
            )
        return prompt_lens

    def sample_many_grouped_compact_prefill(
        self,
        prefill_sampler: "Mxfp8KvCacheCudaGraphSampler",
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int,
        prompts: list[list[int]],
        alternatives_per_prompt: int,
        stop_after_first_sentence: bool = False,
    ) -> list[list[tuple[list[int], list[float], float]]]:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if alternatives_per_prompt < 1:
            raise ValueError("alternatives_per_prompt must be at least 1")
        if not prompts:
            return []
        prompts = [prompt[: self.max_target_positions] for prompt in prompts]
        if any(not prompt for prompt in prompts):
            raise ValueError("prompts must not be empty")
        if len(prompts) * alternatives_per_prompt != self.decode_batch_size:
            raise ValueError(
                "grouped sampling requires decode_batch_size == len(prompts) * alternatives_per_prompt, "
                f"got {self.decode_batch_size}, {len(prompts)}, {alternatives_per_prompt}"
            )
        if prefill_sampler.decode_batch_size != len(prompts):
            raise ValueError(f"prefill sampler batch must be {len(prompts)}, got {prefill_sampler.decode_batch_size}")
        if encoder_hidden_states.shape[0] != len(prompts):
            raise ValueError(f"expected {len(prompts)} encoder batches, got {encoder_hidden_states.shape[0]}")
        if self.cross_cache_batch_size < len(prompts):
            raise ValueError(f"cross cache batch {self.cross_cache_batch_size} is smaller than prompt batch {len(prompts)}")

        self.static_cross_cache_index.copy_(self._grouped_cross_cache_index(len(prompts), alternatives_per_prompt))
        prefill_prompt_lens = prefill_sampler.prefill_prompt_caches(encoder_hidden_states, prompts)
        cross_caches_shared = all(
            key_cache.data_ptr() == source_key_cache.data_ptr() and value_cache.data_ptr() == source_value_cache.data_ptr()
            for key_cache, value_cache, source_key_cache, source_value_cache in zip(
                self.cross_key_caches,
                self.cross_value_caches,
                prefill_sampler.cross_key_caches,
                prefill_sampler.cross_value_caches,
            )
        )
        if not cross_caches_shared:
            for key_cache, value_cache, source_key_cache, source_value_cache in zip(
                self.cross_key_caches,
                self.cross_value_caches,
                prefill_sampler.cross_key_caches,
                prefill_sampler.cross_value_caches,
            ):
                key_cache.copy_(source_key_cache)
                value_cache.copy_(source_value_cache)
        prompt_lens = prefill_prompt_lens.repeat_interleave(alternatives_per_prompt)
        max_prompt_len = int(prompt_lens.max().item())

        self.generated_lengths.zero_()
        self.finished.zero_()
        self.no_speech_probs.zero_()
        self.all_finished.fill_(False)
        self.greedy_batch_mask.zero_()
        self.greedy_batch_mask[::alternatives_per_prompt] = True
        self.static_current_token.fill_(self.eos_token_id)
        self.static_position.zero_()
        self.static_force_timestamp.zero_()
        self.static_pair_timestamp.zero_()
        self.static_pair_timestamp_token.fill_(self.timestamp_begin)
        self.static_min_timestamp.fill_(self.timestamp_begin)

        max_nonfinal_len = max(max(len(prompt) - 1, 0) for prompt in prompts)
        for layer_index, (key_cache, value_cache) in enumerate(zip(self.self_key_caches, self.self_value_caches)):
            source_key_cache = prefill_sampler.self_key_caches[layer_index]
            source_value_cache = prefill_sampler.self_value_caches[layer_index]
            if max_nonfinal_len == 0:
                continue
            for prompt_index, prompt in enumerate(prompts):
                nonfinal_len = max(len(prompt) - 1, 0)
                if nonfinal_len == 0:
                    continue
                row_start = prompt_index * alternatives_per_prompt
                row_end = row_start + alternatives_per_prompt
                key_cache[row_start:row_end, :, :nonfinal_len, :].copy_(
                    source_key_cache[prompt_index : prompt_index + 1, :, :nonfinal_len, :].expand(alternatives_per_prompt, -1, -1, -1)
                )
                value_cache[row_start:row_end, :, :nonfinal_len, :].copy_(
                    source_value_cache[prompt_index : prompt_index + 1, :, :nonfinal_len, :].expand(alternatives_per_prompt, -1, -1, -1)
                )

        final_tokens = torch.tensor(
            [int(prompt[-1]) for prompt in prompts for _ in range(alternatives_per_prompt)],
            device=self.device,
            dtype=torch.long,
        )
        final_positions = prompt_lens - 1
        force_timestamps = torch.full(
            (self.decode_batch_size,),
            self.timestamps_after_sentence_end,
            device=self.device,
            dtype=torch.bool,
        )
        self._replay_device(
            final_tokens,
            final_positions,
            force_timestamps=force_timestamps,
            pair_timestamps=False,
            pair_timestamp_tokens=self.timestamp_begin,
            min_timestamp_tokens=self.timestamp_begin,
            compute_no_speech=True,
        )
        self.no_speech_probs.copy_(self.graph_no_speech_prob)

        max_steps = min(max_new_tokens, self.generated_token_buffer.shape[1])
        if max_steps > 0:
            finish_check_interval = 8
            self.static_position.copy_(final_positions)
            for step in range(max_steps):
                update_decode_control_(
                    self.graph_next_token,
                    self.graph_selected_logprob,
                    self.generated_token_buffer,
                    self.generated_logprob_buffer,
                    self.generated_lengths,
                    self.finished,
                    self.static_current_token,
                    self.static_force_timestamp,
                    self.static_pair_timestamp,
                    self.static_pair_timestamp_token,
                    self.static_min_timestamp,
                    self.static_position,
                    self.static_sampling_seed,
                    self.all_finished,
                    self.sentence_end_token_mask,
                    eos_token_id=self.eos_token_id,
                    timestamp_begin=self.timestamp_begin,
                    timestamps_after_sentence_end=self.timestamps_after_sentence_end,
                    stop_after_first_sentence=stop_after_first_sentence,
                )
                must_stop_for_length = step + 1 >= max_steps or max_prompt_len + step >= self.max_target_positions
                check_finished = must_stop_for_length or (step + 1) % finish_check_interval == 0
                if check_finished and bool(self.all_finished.detach().cpu().item()):
                    break
                if must_stop_for_length:
                    break
                self.graph.replay()

        lengths = self.generated_lengths.detach().cpu().tolist()
        generated_tokens = self.generated_token_buffer.detach().cpu()
        generated_logprobs = self.generated_logprob_buffer.detach().cpu()
        no_speech_probs = self.no_speech_probs.detach().cpu().tolist()
        self._sampling_seed_counter = int(self.static_sampling_seed.detach().cpu().item())

        grouped: list[list[tuple[list[int], list[float], float]]] = []
        for prompt_index, prompt in enumerate(prompts):
            group: list[tuple[list[int], list[float], float]] = []
            row_start = prompt_index * alternatives_per_prompt
            for row in range(row_start, row_start + alternatives_per_prompt):
                length = lengths[row]
                group.append(
                    (
                        prompt + generated_tokens[row, :length].tolist(),
                        generated_logprobs[row, :length].tolist(),
                        no_speech_probs[row],
                    )
                )
            grouped.append(group)
        return grouped

@torch.no_grad()
def encode_packed_mxfp8(model: WhisperForConditionalGeneration, packed: dict[int, object], input_features: torch.Tensor) -> torch.Tensor:
    from plu.benchmarks.bench_end_to_end import _packed_encoder_layer
    from plu.triton.conv1d_gelu import conv1d_gelu
    from plu.triton.embedding import encoder_position_embedding
    from plu.triton.layer_norm import layer_norm

    encoder = model.model.encoder
    hidden_states = conv1d_gelu(input_features, encoder.conv1.weight, encoder.conv1.bias, encoder.conv1.stride[0], encoder.conv1.padding[0])
    hidden_states = conv1d_gelu(hidden_states, encoder.conv2.weight, encoder.conv2.bias, encoder.conv2.stride[0], encoder.conv2.padding[0])
    hidden_states = hidden_states.transpose(1, 2)
    if hidden_states.shape[1] > encoder.config.max_source_positions:
        raise ValueError(f"input features are too long: {hidden_states.shape[1]} > {encoder.config.max_source_positions}")
    hidden_states = encoder_position_embedding(hidden_states, encoder.embed_positions.weight)
    for layer in encoder.layers:
        hidden_states = _packed_encoder_layer(packed, layer, hidden_states, master_weight_grads=False)
    return layer_norm(hidden_states, encoder.layer_norm.weight, encoder.layer_norm.bias, encoder.layer_norm.eps)


def recognize(
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperTokenizer,
    packed: dict[int, object],
    sampler: Mxfp8KvCacheCudaGraphSampler,
    filename: Path,
    *,
    language: str | None,
    dtype: torch.dtype,
    device: torch.device,
    max_new_tokens: int,
):
    logger.debug("recognize %s", filename)
    try:
        features, duration = load_input_features(filename, model.config.num_mel_bins, device, dtype)
        prompt = prompt_ids(tokenizer, language)
        encoder_hidden_states = encode_packed_mxfp8(model, packed, features)
        samples = sampler.sample_many(encoder_hidden_states, max_new_tokens)
    except Exception:
        logger.exception("failed to recognize %s", filename)
        return

    for index, (token_ids, logprobs, no_speech_prob) in enumerate(samples):
        generated = token_ids[len(prompt) :]
        text = tokenizer.batch_decode([generated], skip_special_tokens=True)[0].strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

        yield {
            "i": index,
            "start": 0.0,
            "end": round(duration, 2),
            "text": text,
            "conf": None,
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(no_speech_prob, 3),
            "path": str(filename),
            "language": language or "",
            "langprob": 1.0 if language else 0.0,
            "input_ids": token_ids,
        }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    args.language = normalize_language(args.language)
    model_path = resolve_requested_model(args)

    model = WhisperForConditionalGeneration.from_pretrained(model_path, map_location="cpu")
    model.eval().to(device=device, dtype=dtype)
    tokenizer = configure_tokenizer(model_path, model, args.language)

    from plu.benchmarks.bench_end_to_end import pack_mx_model

    packed, stats = pack_mx_model(model, "mxfp8")
    logger.info("Packed %s PLU linear layers as MXFP8 for +test.", stats["mx_packed_linear_count"])
    prompt = prompt_ids(tokenizer, args.language)
    sampler = Mxfp8KvCacheCudaGraphSampler(
        model,
        tokenizer,
        packed,
        prompt,
        [token for token in tokenizer.special_tokens.values() if token != model.config.eos_token_id],
        decode_batch_size=args.decode_batch_size,
    )
    logger.info("Captured CUDA graph for PLU sampling in %.3f ms.", sampler.capture_ms)

    for filename in args.filenames:
        for seg in recognize(
            model,
            tokenizer,
            packed,
            sampler,
            filename,
            language=args.language,
            dtype=dtype,
            device=device,
            max_new_tokens=args.max_new_tokens,
        ):
            print(json.dumps(seg, ensure_ascii=False))


if __name__ == "__main__":
    main()
