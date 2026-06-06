from __future__ import annotations

import argparse
import json
import os
from logging import getLogger
from pathlib import Path

import torch

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
        choices=["bf16", "fp16"],
        help="Activation dtype for PLU inference.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="Retained for legacy recipe compatibility; ignored by PLU inference.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="Retained for legacy recipe compatibility; ignored by PLU inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Retained for legacy recipe compatibility; ignored by PLU inference.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Retained for legacy recipe compatibility; PLU resolution uses local paths first.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Retained for legacy conversion compatibility; ignored by PLU inference.",
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
    if name == "fp16":
        return torch.float16
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

    if args.local_files_only:
        raise FileNotFoundError(f"{requested} is not a local PLU model directory")

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


def prompt_ids(tokenizer: WhisperTokenizer, language: str | None) -> list[int]:
    prompt = [tokenizer.sot]
    if language:
        prompt.append(tokenizer.to_language_token(language))
    prompt.extend([tokenizer.transcribe, tokenizer.no_timestamps])
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


class Mxfp8CudaGraphSampler:
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        tokenizer: WhisperTokenizer,
        packed: dict[int, object],
        prompt: list[int],
        suppress_tokens: list[int],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.packed = packed
        self.prompt = prompt
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.max_target_positions = model.config.max_target_positions
        self.hidden_size = model.config.d_model
        self.encoder_positions = model.config.max_source_positions
        self.eos_token_id = model.config.eos_token_id
        self.no_speech_token = tokenizer.no_speech
        self.timestamp_begin = tokenizer.timestamp_begin
        self.suppress_tokens = torch.tensor(
            [token for token in suppress_tokens if 0 <= token < model.config.vocab_size],
            device=self.device,
            dtype=torch.long,
        )

        self.static_encoder_hidden_states = torch.zeros(
            1,
            self.encoder_positions,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.prompt_tensor = torch.tensor(prompt, device=self.device, dtype=torch.long)

        self.bucket_states = []
        self.capture_ms = 0.0
        self._capture()

    @torch.no_grad()
    def _graph_step(self, decoder_ids: torch.Tensor, sample_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from benchmarks.bench_end_to_end import _packed_decoder_layer, _packed_linear
        from plu.triton.embedding import decoder_embedding
        from plu.triton.layer_norm import layer_norm

        decoder = self.model.model.decoder
        hidden_states = decoder_embedding(
            decoder_ids,
            decoder.embed_tokens.weight,
            decoder.embed_positions.weight,
            self.static_encoder_hidden_states.dtype,
        )
        for layer in decoder.layers:
            hidden_states = _packed_decoder_layer(
                self.packed,
                layer,
                hidden_states,
                self.static_encoder_hidden_states,
                decoder.causal_mask,
                master_weight_grads=False,
            )
        decoder_hidden_states = layer_norm(hidden_states, decoder.layer_norm.weight, decoder.layer_norm.bias, decoder.layer_norm.eps)
        sample_index_expanded = sample_index.view(1, 1, 1).expand(1, 1, self.hidden_size)
        sampled_hidden = torch.gather(decoder_hidden_states, 1, sample_index_expanded)
        logits = _packed_linear(self.packed, self.model.proj_out, sampled_hidden, master_weight_grads=False).float().squeeze(1)
        raw_logprobs = logits.log_softmax(dim=-1)
        no_speech_prob = raw_logprobs[:, self.no_speech_token].exp()
        if self.suppress_tokens.numel():
            logits.index_fill_(1, self.suppress_tokens, -float("inf"))
        logits[:, self.timestamp_begin :].fill_(-float("inf"))
        logprobs = logits.log_softmax(dim=-1)
        next_token = logits.argmax(dim=-1)
        selected_logprob = logprobs.gather(1, next_token[:, None]).squeeze(1)
        return next_token, selected_logprob, no_speech_prob

    def _bucket_sizes(self) -> list[int]:
        sizes = []
        size = 8
        while size < self.max_target_positions:
            if size >= len(self.prompt):
                sizes.append(size)
            size *= 2
        if not sizes or sizes[-1] != self.max_target_positions:
            sizes.append(self.max_target_positions)
        return sizes

    def _make_bucket_state(self, size: int) -> dict[str, object]:
        return {
            "size": size,
            "decoder_ids": torch.full((1, size), self.eos_token_id, device=self.device, dtype=torch.long),
            "sample_index": torch.zeros(1, device=self.device, dtype=torch.long),
            "graph": None,
            "next_token": None,
            "selected_logprob": None,
            "no_speech_prob": None,
        }

    def _reset_bucket(self, state: dict[str, object]) -> None:
        decoder_ids = state["decoder_ids"]
        assert isinstance(decoder_ids, torch.Tensor)
        decoder_ids.fill_(self.eos_token_id)
        prompt_len = min(len(self.prompt), decoder_ids.shape[1])
        decoder_ids[0, :prompt_len].copy_(self.prompt_tensor[:prompt_len])
        sample_index = state["sample_index"]
        assert isinstance(sample_index, torch.Tensor)
        sample_index.fill_(max(0, prompt_len - 1))

    def _reset_decoder_ids(self) -> None:
        for state in self.bucket_states:
            self._reset_bucket(state)

    def _capture(self) -> None:
        total_capture_ms = 0.0
        self.bucket_states = [self._make_bucket_state(size) for size in self._bucket_sizes()]
        for state in self.bucket_states:
            self._reset_bucket(state)
            decoder_ids = state["decoder_ids"]
            sample_index = state["sample_index"]
            assert isinstance(decoder_ids, torch.Tensor)
            assert isinstance(sample_index, torch.Tensor)
            for _ in range(3):
                self._graph_step(decoder_ids, sample_index)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_start = torch.cuda.Event(enable_timing=True)
            capture_end = torch.cuda.Event(enable_timing=True)
            capture_start.record()
            with torch.cuda.graph(graph):
                next_token, selected_logprob, no_speech_prob = self._graph_step(decoder_ids, sample_index)
            capture_end.record()
            torch.cuda.synchronize()

            state["graph"] = graph
            state["next_token"] = next_token
            state["selected_logprob"] = selected_logprob
            state["no_speech_prob"] = no_speech_prob
            total_capture_ms += capture_start.elapsed_time(capture_end)
        self.capture_ms = total_capture_ms

    def copy_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> None:
        if encoder_hidden_states.shape != self.static_encoder_hidden_states.shape:
            raise ValueError(
                "CUDA graph sampler requires static encoder states "
                f"{tuple(self.static_encoder_hidden_states.shape)}, got {tuple(encoder_hidden_states.shape)}"
            )
        self.static_encoder_hidden_states.copy_(encoder_hidden_states)

    def _state_for_token_count(self, token_count: int) -> dict[str, object]:
        window_tokens = min(token_count, self.max_target_positions)
        for state in self.bucket_states:
            size = state["size"]
            assert isinstance(size, int)
            if size >= window_tokens:
                return state
        return self.bucket_states[-1]

    def _write_generated_token(self, write_position: int, next_token: torch.Tensor) -> None:
        if write_position < self.max_target_positions:
            for state in self.bucket_states:
                size = state["size"]
                decoder_ids = state["decoder_ids"]
                assert isinstance(size, int)
                assert isinstance(decoder_ids, torch.Tensor)
                if write_position < size:
                    decoder_ids[0, write_position].copy_(next_token)
        else:
            state = self.bucket_states[-1]
            decoder_ids = state["decoder_ids"]
            assert isinstance(decoder_ids, torch.Tensor)
            decoder_ids[:, :-1].copy_(decoder_ids[:, 1:].clone())
            decoder_ids[0, -1].copy_(next_token)

    def sample(self, encoder_hidden_states: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[float], float]:
        if not self.bucket_states:
            raise RuntimeError("CUDA graph sampler has not been captured")

        self.copy_encoder_hidden_states(encoder_hidden_states)
        self._reset_decoder_ids()
        generated: list[int] = []
        selected_logprobs: list[float] = []
        no_speech_prob = 0.0

        for step in range(max_new_tokens):
            token_count = len(self.prompt) + len(generated)
            state = self._state_for_token_count(token_count)
            size = state["size"]
            sample_index = state["sample_index"]
            graph = state["graph"]
            next_token_tensor = state["next_token"]
            selected_logprob_tensor = state["selected_logprob"]
            no_speech_prob_tensor = state["no_speech_prob"]
            assert isinstance(size, int)
            assert isinstance(sample_index, torch.Tensor)
            assert isinstance(graph, torch.cuda.CUDAGraph)
            assert isinstance(next_token_tensor, torch.Tensor)
            assert isinstance(selected_logprob_tensor, torch.Tensor)
            assert isinstance(no_speech_prob_tensor, torch.Tensor)

            sample_position = min(token_count, size) - 1
            sample_index.fill_(sample_position)
            graph.replay()

            next_token_gpu = next_token_tensor[0]
            if step == 0:
                no_speech_prob = float(no_speech_prob_tensor[0].detach().cpu())
            next_token = int(next_token_gpu.detach().cpu())
            selected_logprobs.append(float(selected_logprob_tensor[0].detach().cpu()))

            self._write_generated_token(token_count, next_token_gpu)

            generated.append(next_token)
            if next_token == self.eos_token_id:
                break

        return self.prompt + generated, selected_logprobs, no_speech_prob


class Mxfp8KvCacheCudaGraphSampler:
    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        tokenizer: WhisperTokenizer,
        packed: dict[int, object],
        prompt: list[int],
        suppress_tokens: list[int],
        decode_batch_size: int = 1,
    ):
        if decode_batch_size < 1:
            raise ValueError("--decode_batch_size must be at least 1")
        self.model = model
        self.tokenizer = tokenizer
        self.packed = packed
        self.prompt = prompt
        self.decode_batch_size = decode_batch_size
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
        self.static_current_token = torch.full((self.decode_batch_size,), self.eos_token_id, device=self.device, dtype=torch.long)
        self.static_position = torch.zeros(self.decode_batch_size, device=self.device, dtype=torch.long)

        decoder = model.model.decoder
        self.self_key_caches = []
        self.self_value_caches = []
        self.cross_key_caches = []
        self.cross_value_caches = []
        for layer in decoder.layers:
            heads = layer.self_attn.num_heads
            head_dim = layer.self_attn.head_dim
            self.self_key_caches.append(
                torch.empty((self.decode_batch_size, heads, self.max_target_positions, head_dim), device=self.device, dtype=self.dtype)
            )
            self.self_value_caches.append(
                torch.empty((self.decode_batch_size, heads, self.max_target_positions, head_dim), device=self.device, dtype=self.dtype)
            )
            self.cross_key_caches.append(
                torch.empty((self.decode_batch_size, heads, self.encoder_positions, head_dim), device=self.device, dtype=self.dtype)
            )
            self.cross_value_caches.append(
                torch.empty((self.decode_batch_size, heads, self.encoder_positions, head_dim), device=self.device, dtype=self.dtype)
            )

        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_next_token: torch.Tensor | None = None
        self.graph_selected_logprob: torch.Tensor | None = None
        self.graph_no_speech_prob: torch.Tensor | None = None
        self.capture_ms = 0.0
        self._capture()

    def _project_heads(self, module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        from benchmarks.bench_end_to_end import _packed_linear

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
        from benchmarks.bench_end_to_end import _packed_qkv

        decoder = self.model.model.decoder
        for index, layer in enumerate(decoder.layers):
            key = _packed_qkv(self.packed, layer.encoder_attn.k_proj, encoder_hidden_states, layer.encoder_attn.num_heads, master_weight_grads=False)
            value = _packed_qkv(self.packed, layer.encoder_attn.v_proj, encoder_hidden_states, layer.encoder_attn.num_heads, master_weight_grads=False)
            key_cache = self.cross_key_caches[index]
            value_cache = self.cross_value_caches[index]
            if key.shape[0] == 1 and self.decode_batch_size > 1:
                key = key.expand_as(key_cache)
                value = value.expand_as(value_cache)
            if key.shape != key_cache.shape or value.shape != value_cache.shape:
                raise ValueError(
                    "KV-cache sampler requires static encoder cache shapes "
                    f"{tuple(key_cache.shape)}, got {tuple(key.shape)}"
                )
            key_cache.copy_(key)
            value_cache.copy_(value)

    @torch.no_grad()
    def _decoder_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from benchmarks.bench_end_to_end import _packed_gelu_mlp, _packed_linear
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
            attended = cross_kv_cache_attention(query, self.cross_key_caches[index], self.cross_value_caches[index])
            hidden_states = _packed_linear(self.packed, layer.encoder_attn.out_proj, self._merge_heads(attended), master_weight_grads=False)
            hidden_states = residual_add(residual, hidden_states)

            residual = hidden_states
            normed = layer_norm(hidden_states, layer.final_layer_norm.weight, layer.final_layer_norm.bias, layer.final_layer_norm.eps)
            hidden_states = _packed_gelu_mlp(self.packed, layer.fc1, layer.fc2, normed, master_weight_grads=False)
            hidden_states = residual_add(residual, hidden_states)

        decoder_hidden_states = layer_norm(hidden_states, decoder.layer_norm.weight, decoder.layer_norm.bias, decoder.layer_norm.eps)
        logits = _packed_linear(self.packed, self.model.proj_out, decoder_hidden_states, master_weight_grads=False).float().squeeze(1)
        raw_logprobs = logits.log_softmax(dim=-1)
        no_speech_prob = raw_logprobs[:, self.no_speech_token].exp()
        if self.suppress_tokens.numel():
            logits.index_fill_(1, self.suppress_tokens, -float("inf"))
        logits[:, self.timestamp_begin :].fill_(-float("inf"))
        logprobs = logits.log_softmax(dim=-1)
        if self.decode_batch_size == 1:
            next_token = logits.argmax(dim=-1)
        else:
            uniform = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(uniform.clamp_(min=1e-6, max=1.0 - 1e-6)))
            next_token = (logits + gumbel).argmax(dim=-1)
        selected_logprob = logprobs.gather(1, next_token[:, None]).squeeze(1)
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
        capture_start = torch.cuda.Event(enable_timing=True)
        capture_end = torch.cuda.Event(enable_timing=True)
        capture_start.record()
        with torch.cuda.graph(graph):
            self.graph_next_token, self.graph_selected_logprob, self.graph_no_speech_prob = self._decoder_step()
        capture_end.record()
        torch.cuda.synchronize()
        self.graph = graph
        self.capture_ms = capture_start.elapsed_time(capture_end)

    def _replay(self, tokens: int | list[int] | torch.Tensor, position: int) -> tuple[list[int], list[float], list[float]]:
        if self.graph is None or self.graph_next_token is None or self.graph_selected_logprob is None or self.graph_no_speech_prob is None:
            raise RuntimeError("CUDA graph sampler has not been captured")
        if isinstance(tokens, int):
            self.static_current_token.fill_(tokens)
        elif isinstance(tokens, torch.Tensor):
            self.static_current_token.copy_(tokens.to(device=self.device, dtype=torch.long))
        else:
            if len(tokens) != self.decode_batch_size:
                raise ValueError(f"expected {self.decode_batch_size} tokens, got {len(tokens)}")
            self.static_current_token.copy_(torch.tensor(tokens, device=self.device, dtype=torch.long))
        self.static_position.fill_(position)
        self.graph.replay()
        next_tokens = self.graph_next_token.detach().cpu().tolist()
        selected_logprobs = self.graph_selected_logprob.detach().cpu().tolist()
        no_speech_probs = self.graph_no_speech_prob.detach().cpu().tolist()
        return next_tokens, selected_logprobs, no_speech_probs

    def sample_many(self, encoder_hidden_states: torch.Tensor, max_new_tokens: int) -> list[tuple[list[int], list[float], float]]:
        self.prepare_encoder(encoder_hidden_states)
        for key_cache, value_cache in zip(self.self_key_caches, self.self_value_caches):
            key_cache.zero_()
            value_cache.zero_()

        selected_logprobs: list[list[float]] = [[] for _ in range(self.decode_batch_size)]
        generated: list[list[int]] = [[] for _ in range(self.decode_batch_size)]
        finished = [False] * self.decode_batch_size
        no_speech_probs = [0.0] * self.decode_batch_size

        next_tokens = [self.eos_token_id] * self.decode_batch_size
        next_logprobs = [0.0] * self.decode_batch_size
        prompt_len = min(len(self.prompt), self.max_target_positions)
        for position, token in enumerate(self.prompt[:prompt_len]):
            next_tokens, next_logprobs, step_no_speech_probs = self._replay(token, position)
            if position == prompt_len - 1:
                no_speech_probs = step_no_speech_probs

        while max((len(tokens) for tokens in generated), default=0) < max_new_tokens:
            for index, (next_token, next_logprob) in enumerate(zip(next_tokens, next_logprobs)):
                if finished[index]:
                    continue
                generated[index].append(next_token)
                selected_logprobs[index].append(next_logprob)
                if next_token == self.eos_token_id:
                    finished[index] = True
            if all(finished):
                break
            if max(len(tokens) for tokens in generated) >= max_new_tokens:
                break
            position = prompt_len + max(len(tokens) for tokens in generated) - 1
            if position >= self.max_target_positions:
                break
            replay_tokens = [tokens[-1] if tokens and not done else self.eos_token_id for tokens, done in zip(generated, finished)]
            next_tokens, next_logprobs, _ = self._replay(replay_tokens, position)

        prompt = self.prompt[:prompt_len]
        return [
            (prompt + generated_tokens, logprobs, no_speech_prob)
            for generated_tokens, logprobs, no_speech_prob in zip(generated, selected_logprobs, no_speech_probs)
        ]

    def sample(self, encoder_hidden_states: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[float], float]:
        return self.sample_many(encoder_hidden_states, max_new_tokens)[0]


@torch.no_grad()
def encode_packed_mxfp8(model: WhisperForConditionalGeneration, packed: dict[int, object], input_features: torch.Tensor) -> torch.Tensor:
    from benchmarks.bench_end_to_end import _packed_encoder_layer
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


@torch.no_grad()
def decode_packed_mxfp8(
    model: WhisperForConditionalGeneration,
    packed: dict[int, object],
    encoder_hidden_states: torch.Tensor,
    prompt: list[int],
    max_new_tokens: int,
    timestamp_begin: int,
    no_speech_token: int,
    suppress_tokens: list[int],
) -> tuple[list[int], list[float], float]:
    from benchmarks.bench_end_to_end import _packed_decoder_layer, _packed_linear
    from plu.triton.embedding import decoder_embedding
    from plu.triton.layer_norm import layer_norm

    decoder = model.model.decoder
    tokens = torch.tensor([prompt], device=encoder_hidden_states.device, dtype=torch.long)
    selected_logprobs: list[float] = []
    no_speech_prob = 0.0

    for step in range(max_new_tokens):
        decoder_input_ids = tokens[:, -decoder.config.max_target_positions :]
        hidden_states = decoder_embedding(
            decoder_input_ids,
            decoder.embed_tokens.weight,
            decoder.embed_positions.weight,
            encoder_hidden_states.dtype,
        )
        for layer in decoder.layers:
            hidden_states = _packed_decoder_layer(
                packed,
                layer,
                hidden_states,
                encoder_hidden_states,
                decoder.causal_mask,
                master_weight_grads=False,
            )
        decoder_hidden_states = layer_norm(hidden_states, decoder.layer_norm.weight, decoder.layer_norm.bias, decoder.layer_norm.eps)
        logits = _packed_linear(packed, model.proj_out, decoder_hidden_states[:, -1:], master_weight_grads=False).float().squeeze(1)
        raw_logprobs = logits.log_softmax(dim=-1)
        if step == 0 and 0 <= no_speech_token < raw_logprobs.shape[-1]:
            no_speech_prob = float(raw_logprobs[0, no_speech_token].exp().detach().cpu())
        if suppress_tokens:
            valid_suppress_tokens = [token for token in suppress_tokens if 0 <= token < logits.shape[-1]]
            logits[:, valid_suppress_tokens] = -float("inf")
        logits[:, timestamp_begin:] = -float("inf")
        logprobs = logits.log_softmax(dim=-1)
        next_token = logits.argmax(dim=-1)
        selected_logprobs.append(float(logprobs.gather(1, next_token[:, None]).squeeze().detach().cpu()))
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        if int(next_token.item()) == model.config.eos_token_id:
            break

    return tokens[0].detach().cpu().tolist(), selected_logprobs, no_speech_prob


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

    from benchmarks.bench_end_to_end import pack_mx_model

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
