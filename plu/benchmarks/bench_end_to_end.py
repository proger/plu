from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from plu.train_data import HOP_LENGTH, SAMPLE_RATE


DEFAULT_FEATURE_FPS = SAMPLE_RATE / HOP_LENGTH


def set_tf32(enabled: bool) -> None:
    precision = "tf32" if enabled else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = precision
        torch.backends.cuda.matmul.fp32_precision = precision
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = enabled
        try:
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        except AttributeError:
            pass


def parse_dtype(name: str) -> torch.dtype:
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def encoder_tokens_after_conv(input_feature_frames: int) -> int:
    return (input_feature_frames + 1) // 2


def audio_seconds(input_feature_frames: int, feature_fps: float) -> float:
    return input_feature_frames / feature_fps


def extend_encoder_positions(model: torch.nn.Module, positions: int) -> bool:
    from plu.whisper import sinusoids

    encoder = model.model.encoder
    if encoder.config.max_source_positions >= positions and encoder.embed_positions.weight.shape[0] >= positions:
        return False

    encoder.config.max_source_positions = positions
    old_weight = encoder.embed_positions.weight
    embedding = torch.nn.Embedding(
        positions,
        model.config.d_model,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    embedding.weight.requires_grad = False
    with torch.no_grad():
        embedding.weight.copy_(sinusoids(positions, model.config.d_model).to(device=old_weight.device, dtype=old_weight.dtype))
    encoder.embed_positions = embedding
    return True


def encoder_backward_start(total_layers: int, backward_layers: int | None) -> int:
    if backward_layers is None:
        return 0
    if backward_layers < 0:
        raise ValueError("--encoder-backward-layers must be non-negative")
    if backward_layers > total_layers:
        raise ValueError(f"--encoder-backward-layers={backward_layers} exceeds encoder layer count {total_layers}")
    return total_layers - backward_layers


@dataclass(frozen=True)
class PackedMxLinear:
    packed_weight: torch.Tensor
    scale_codes: torch.Tensor
    in_features: int
    input_grad_packed_weight: torch.Tensor
    input_grad_scale_codes: torch.Tensor
    input_grad_in_features: int
    master_weight: torch.Tensor
    bias: torch.Tensor | None
    format: str
    bf16_storage_bytes: int
    forward_packed_storage_bytes: int
    input_grad_packed_storage_bytes: int
    packed_storage_bytes: int

    def __call__(self, x: torch.Tensor, master_weight_grads: bool) -> torch.Tensor:
        if self.format == "mxfp8":
            from plu.triton.mx_linear import mxfp8_linear, mxfp8_linear_with_master_weight

            if master_weight_grads:
                return mxfp8_linear_with_master_weight(
                    x,
                    self.packed_weight,
                    self.scale_codes,
                    self.in_features,
                    self.master_weight,
                    self.bias,
                    self.input_grad_packed_weight,
                    self.input_grad_scale_codes,
                    self.input_grad_in_features,
                )
            return mxfp8_linear(
                x,
                self.packed_weight,
                self.scale_codes,
                self.in_features,
                self.bias,
                self.input_grad_packed_weight,
                self.input_grad_scale_codes,
                self.input_grad_in_features,
            )
        if self.format == "mxfp4":
            from plu.triton.mx_linear import mxfp4_linear, mxfp4_linear_with_master_weight

            if master_weight_grads:
                return mxfp4_linear_with_master_weight(
                    x,
                    self.packed_weight,
                    self.scale_codes,
                    self.in_features,
                    self.master_weight,
                    self.bias,
                    self.input_grad_packed_weight,
                    self.input_grad_scale_codes,
                    self.input_grad_in_features,
                )
            return mxfp4_linear(
                x,
                self.packed_weight,
                self.scale_codes,
                self.in_features,
                self.bias,
                self.input_grad_packed_weight,
                self.input_grad_scale_codes,
                self.input_grad_in_features,
            )
        if self.format == "nvfp4":
            from plu.triton.mx_linear import nvfp4_linear, nvfp4_linear_with_master_weight

            if master_weight_grads:
                return nvfp4_linear_with_master_weight(
                    x,
                    self.packed_weight,
                    self.scale_codes,
                    self.in_features,
                    self.master_weight,
                    self.bias,
                    self.input_grad_packed_weight,
                    self.input_grad_scale_codes,
                    self.input_grad_in_features,
                )
            return nvfp4_linear(
                x,
                self.packed_weight,
                self.scale_codes,
                self.in_features,
                self.bias,
                self.input_grad_packed_weight,
                self.input_grad_scale_codes,
                self.input_grad_in_features,
            )
        raise RuntimeError(f"unsupported packed MX format: {self.format}")

    def heads(self, x: torch.Tensor, num_heads: int, master_weight_grads: bool) -> torch.Tensor:
        if self.format == "mxfp8":
            from plu.triton.mx_linear import mxfp8_linear_heads, mxfp8_linear_heads_with_master_weight

            if master_weight_grads:
                return mxfp8_linear_heads_with_master_weight(
                    x,
                    self.packed_weight,
                    self.scale_codes,
                    self.in_features,
                    num_heads,
                    self.master_weight,
                    self.bias,
                    self.input_grad_packed_weight,
                    self.input_grad_scale_codes,
                    self.input_grad_in_features,
                )
            return mxfp8_linear_heads(
                x,
                self.packed_weight,
                self.scale_codes,
                self.in_features,
                num_heads,
                self.bias,
                self.input_grad_packed_weight,
                self.input_grad_scale_codes,
                self.input_grad_in_features,
            )
        projected = self(x, master_weight_grads)
        batch, seq_len, features = projected.shape
        head_dim = features // num_heads
        return projected.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    def from_heads(self, x: torch.Tensor, master_weight_grads: bool) -> torch.Tensor:
        if self.format == "mxfp8":
            from plu.triton.mx_linear import mxfp8_linear_from_heads, mxfp8_linear_from_heads_with_master_weight

            if master_weight_grads:
                return mxfp8_linear_from_heads_with_master_weight(
                    x,
                    self.packed_weight,
                    self.scale_codes,
                    self.in_features,
                    self.master_weight,
                    self.bias,
                    self.input_grad_packed_weight,
                    self.input_grad_scale_codes,
                    self.input_grad_in_features,
                )
            return mxfp8_linear_from_heads(
                x,
                self.packed_weight,
                self.scale_codes,
                self.in_features,
                self.bias,
                self.input_grad_packed_weight,
                self.input_grad_scale_codes,
                self.input_grad_in_features,
            )
        batch, heads, seq_len, head_dim = x.shape
        merged = x.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, heads * head_dim)
        return self(merged, master_weight_grads)


def pack_mx_linear(module: torch.nn.Linear, format: str) -> PackedMxLinear:
    from plu.triton.mx_linear import pack_mxfp4_weight, pack_mxfp8_weight, pack_nvfp4_weight

    weight = module.weight.detach().contiguous()
    bias = module.bias
    if format == "mxfp8":
        packed_weight, scale_codes, in_features = pack_mxfp8_weight(weight)
        input_grad_packed_weight, input_grad_scale_codes, input_grad_in_features = pack_mxfp8_weight(weight.T.contiguous())
    elif format == "mxfp4":
        packed_weight, scale_codes, in_features = pack_mxfp4_weight(weight)
        input_grad_packed_weight, input_grad_scale_codes, input_grad_in_features = pack_mxfp4_weight(weight.T.contiguous())
    elif format == "nvfp4":
        packed_weight, scale_codes, in_features = pack_nvfp4_weight(weight)
        input_grad_packed_weight, input_grad_scale_codes, input_grad_in_features = pack_nvfp4_weight(weight.T.contiguous())
    else:
        raise RuntimeError(f"unsupported MX format: {format}")
    bf16_storage_bytes = weight.numel() * weight.element_size()
    forward_packed_storage_bytes = packed_weight.numel() * packed_weight.element_size() + scale_codes.numel() * scale_codes.element_size()
    input_grad_packed_storage_bytes = (
        input_grad_packed_weight.numel() * input_grad_packed_weight.element_size()
        + input_grad_scale_codes.numel() * input_grad_scale_codes.element_size()
    )
    packed_storage_bytes = (
        forward_packed_storage_bytes
        + input_grad_packed_storage_bytes
    )
    return PackedMxLinear(
        packed_weight,
        scale_codes,
        in_features,
        input_grad_packed_weight,
        input_grad_scale_codes,
        input_grad_in_features,
        module.weight,
        bias,
        format,
        bf16_storage_bytes,
        forward_packed_storage_bytes,
        input_grad_packed_storage_bytes,
        packed_storage_bytes,
    )


def pack_mx_model(
    model: torch.nn.Module,
    format: str,
    module_filter: Any | None = None,
) -> tuple[dict[int, PackedMxLinear], dict[str, Any]]:
    pack_start = time.perf_counter()
    packed: dict[int, PackedMxLinear] = {}
    for module in model.modules():
        if module.__class__.__name__ == "CastLinear" and (module_filter is None or module_filter(module)):
            packed[id(module)] = pack_mx_linear(module, format)
    pack_ms = (time.perf_counter() - pack_start) * 1000.0
    bf16_bytes = sum(linear.bf16_storage_bytes for linear in packed.values())
    forward_packed_bytes = sum(linear.forward_packed_storage_bytes for linear in packed.values())
    input_grad_packed_bytes = sum(linear.input_grad_packed_storage_bytes for linear in packed.values())
    packed_bytes = sum(linear.packed_storage_bytes for linear in packed.values())
    stats = {
        "mx_format": format,
        "mx_pack_ms": pack_ms,
        "mx_packed_linear_count": len(packed),
        "mx_bf16_linear_weight_storage_bytes": bf16_bytes,
        "mx_forward_packed_linear_weight_storage_bytes": forward_packed_bytes,
        "mx_input_grad_packed_linear_weight_storage_bytes": input_grad_packed_bytes,
        "mx_total_packed_linear_weight_storage_bytes": packed_bytes,
        "mx_packed_linear_weight_storage_bytes": packed_bytes,
        "mx_linear_weight_compression_vs_bf16": bf16_bytes / packed_bytes if packed_bytes else None,
        "mx_forward_weight_transfer_roofline_vs_bf16": bf16_bytes / forward_packed_bytes if forward_packed_bytes else None,
        "mx_train_weight_transfer_roofline_vs_bf16": (2 * bf16_bytes) / packed_bytes if packed_bytes else None,
        "mx_packed_weight_grads": True,
        "mx_bias_grads": True,
    }
    return packed, stats


def _packed_linear(
    packed: dict[int, PackedMxLinear],
    module: torch.nn.Module,
    x: torch.Tensor,
    master_weight_grads: bool,
) -> torch.Tensor:
    packed_linear = packed.get(id(module))
    if packed_linear is None:
        return module(x)
    return packed_linear(x, master_weight_grads)


def _packed_qkv(
    packed: dict[int, PackedMxLinear],
    module: torch.nn.Module,
    x: torch.Tensor,
    num_heads: int,
    master_weight_grads: bool,
) -> torch.Tensor:
    packed_linear = packed.get(id(module))
    if packed_linear is None:
        from plu.whisper import qkv_proj

        return qkv_proj(x, module.weight, module.bias, num_heads)
    return packed_linear.heads(x, num_heads, master_weight_grads)


def _packed_attention(
    packed: dict[int, PackedMxLinear],
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    causal_mask: torch.Tensor | None = None,
    master_weight_grads: bool = False,
) -> torch.Tensor:
    from plu.triton.flash_attention import flash_attention

    source = hidden_states if key_value_states is None else key_value_states
    query = _packed_qkv(packed, module.q_proj, hidden_states, module.num_heads, master_weight_grads)
    key = _packed_qkv(packed, module.k_proj, source, module.num_heads, master_weight_grads)
    value = _packed_qkv(packed, module.v_proj, source, module.num_heads, master_weight_grads)
    attended = flash_attention(query, key, value, causal_mask)
    packed_out = packed.get(id(module.out_proj))
    if packed_out is None:
        from plu.whisper import c_proj

        return c_proj(attended, module.out_proj.weight, module.out_proj.bias)
    return packed_out.from_heads(attended, master_weight_grads)


class _PackedGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, preact: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(preact)
        from plu.triton.gelu_mlp.backward import gelu_forward

        return gelu_forward(preact)

    @staticmethod
    def backward(ctx, grad_hidden: torch.Tensor) -> tuple[torch.Tensor]:
        (preact,) = ctx.saved_tensors
        from plu.triton.gelu_mlp.backward import gelu_backward

        return (gelu_backward(preact, grad_hidden.contiguous()),)


def _packed_gelu(preact: torch.Tensor) -> torch.Tensor:
    return _PackedGelu.apply(preact)


def _packed_gelu_mlp(
    packed: dict[int, PackedMxLinear],
    fc1: torch.nn.Module,
    fc2: torch.nn.Module,
    hidden_states: torch.Tensor,
    master_weight_grads: bool,
) -> torch.Tensor:
    if id(fc1) not in packed and id(fc2) not in packed:
        from plu.whisper import gelu_mlp

        return gelu_mlp(hidden_states, fc1.weight, fc1.bias, fc2.weight, fc2.bias)
    preact = _packed_linear(packed, fc1, hidden_states, master_weight_grads)
    hidden = _packed_gelu(preact)
    return _packed_linear(packed, fc2, hidden, master_weight_grads)


def _packed_encoder_layer(
    packed: dict[int, PackedMxLinear],
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    master_weight_grads: bool,
) -> torch.Tensor:
    from plu.triton.layer_norm import layer_norm
    from plu.triton.residual_add import residual_add

    residual = hidden_states
    hidden_states = layer_norm(hidden_states, layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias, layer.self_attn_layer_norm.eps)
    hidden_states = _packed_attention(packed, layer.self_attn, hidden_states, master_weight_grads=master_weight_grads)
    hidden_states = residual_add(residual, hidden_states)

    residual = hidden_states
    hidden_states = layer_norm(hidden_states, layer.final_layer_norm.weight, layer.final_layer_norm.bias, layer.final_layer_norm.eps)
    hidden_states = _packed_gelu_mlp(packed, layer.fc1, layer.fc2, hidden_states, master_weight_grads)
    return residual_add(residual, hidden_states)


def _packed_decoder_layer(
    packed: dict[int, PackedMxLinear],
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    causal_mask: torch.Tensor,
    master_weight_grads: bool,
) -> torch.Tensor:
    from plu.triton.layer_norm import layer_norm
    from plu.triton.residual_add import residual_add

    residual = hidden_states
    hidden_states = layer_norm(hidden_states, layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias, layer.self_attn_layer_norm.eps)
    hidden_states = _packed_attention(packed, layer.self_attn, hidden_states, causal_mask=causal_mask, master_weight_grads=master_weight_grads)
    hidden_states = residual_add(residual, hidden_states)

    residual = hidden_states
    hidden_states = layer_norm(hidden_states, layer.encoder_attn_layer_norm.weight, layer.encoder_attn_layer_norm.bias, layer.encoder_attn_layer_norm.eps)
    hidden_states = _packed_attention(
        packed,
        layer.encoder_attn,
        hidden_states,
        key_value_states=encoder_hidden_states,
        master_weight_grads=master_weight_grads,
    )
    hidden_states = residual_add(residual, hidden_states)

    residual = hidden_states
    hidden_states = layer_norm(hidden_states, layer.final_layer_norm.weight, layer.final_layer_norm.bias, layer.final_layer_norm.eps)
    hidden_states = _packed_gelu_mlp(packed, layer.fc1, layer.fc2, hidden_states, master_weight_grads)
    return residual_add(residual, hidden_states)


def packed_mx_forward(
    model: torch.nn.Module,
    packed: dict[int, PackedMxLinear],
    input_features: torch.Tensor,
    labels: torch.Tensor,
    master_weight_grads: bool,
    encoder_backward_layers: int | None = None,
    fused_unembedding_ce: bool = False,
) -> Any:
    from plu.triton.conv1d_gelu import conv1d_gelu
    from plu.triton.cross_entropy import cross_entropy
    from plu.triton.embedding import decoder_embedding, encoder_position_embedding
    from plu.triton.layer_norm import layer_norm
    from plu.whisper import Seq2SeqOutput, shift_tokens_right

    encoder = model.model.encoder
    decoder = model.model.decoder
    hidden_states = conv1d_gelu(input_features, encoder.conv1.weight, encoder.conv1.bias, encoder.conv1.stride[0], encoder.conv1.padding[0])
    hidden_states = conv1d_gelu(hidden_states, encoder.conv2.weight, encoder.conv2.bias, encoder.conv2.stride[0], encoder.conv2.padding[0])
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = encoder_position_embedding(hidden_states, encoder.embed_positions.weight)
    backward_start = encoder_backward_start(len(encoder.layers), encoder_backward_layers) if master_weight_grads else 0
    for layer in encoder.layers[:backward_start]:
        with torch.no_grad():
            hidden_states = _packed_encoder_layer(packed, layer, hidden_states, master_weight_grads=False)
    if backward_start:
        hidden_states = hidden_states.detach()
    for layer in encoder.layers[backward_start:]:
        hidden_states = _packed_encoder_layer(packed, layer, hidden_states, master_weight_grads)
    encoder_hidden_states = layer_norm(hidden_states, encoder.layer_norm.weight, encoder.layer_norm.bias, encoder.layer_norm.eps)

    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    if decoder_input_ids.shape[1] > decoder.config.max_target_positions:
        decoder_input_ids = decoder_input_ids[:, -decoder.config.max_target_positions :]
    hidden_states = decoder_embedding(decoder_input_ids, decoder.embed_tokens.weight, decoder.embed_positions.weight, encoder_hidden_states.dtype)
    for layer in decoder.layers:
        hidden_states = _packed_decoder_layer(packed, layer, hidden_states, encoder_hidden_states, decoder.causal_mask, master_weight_grads)
    decoder_hidden_states = layer_norm(hidden_states, decoder.layer_norm.weight, decoder.layer_norm.bias, decoder.layer_norm.eps)

    if fused_unembedding_ce:
        from plu.triton.unembedding_cross_entropy import unembedding_cross_entropy

        loss = unembedding_cross_entropy(decoder_hidden_states, model.proj_out.weight, labels, ignore_index=-100)
        logits = decoder_hidden_states.new_empty(0)
    else:
        logits = _packed_linear(packed, model.proj_out, decoder_hidden_states, master_weight_grads).float()
        loss = cross_entropy(logits, labels, ignore_index=-100)
    return Seq2SeqOutput(loss=loss, logits=logits)


def limited_model_forward(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    labels: torch.Tensor,
    encoder_backward_layers: int | None = None,
    fused_unembedding_ce: bool = False,
) -> Any:
    from plu.whisper import Seq2SeqOutput, cross_entropy, encoder_position_embedding, shift_tokens_right

    encoder = model.model.encoder
    decoder = model.model.decoder
    hidden_states = encoder.conv1(input_features)
    hidden_states = encoder.conv2(hidden_states)
    hidden_states = hidden_states.transpose(1, 2)
    if hidden_states.shape[1] > encoder.config.max_source_positions:
        raise ValueError(f"input features are too long: {hidden_states.shape[1]} > {encoder.config.max_source_positions}")
    hidden_states = encoder_position_embedding(hidden_states, encoder.embed_positions.weight)
    backward_start = encoder_backward_start(len(encoder.layers), encoder_backward_layers)
    for layer in encoder.layers[:backward_start]:
        with torch.no_grad():
            hidden_states = layer(hidden_states)
    if backward_start:
        hidden_states = hidden_states.detach()
    for layer in encoder.layers[backward_start:]:
        hidden_states = layer(hidden_states)
    encoder_hidden_states = encoder.layer_norm(hidden_states)

    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    decoder_hidden_states = decoder(decoder_input_ids, encoder_hidden_states)
    if fused_unembedding_ce:
        from plu.triton.unembedding_cross_entropy import unembedding_cross_entropy

        loss = unembedding_cross_entropy(decoder_hidden_states, model.proj_out.weight, labels, ignore_index=-100)
        logits = decoder_hidden_states.new_empty(0)
    else:
        logits = model.proj_out(decoder_hidden_states).float()
        loss = cross_entropy(logits, labels, ignore_index=-100)
    return Seq2SeqOutput(loss=loss, logits=logits)


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PLU_OPS_BACKEND"] = args.child_backend
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for end-to-end benchmarks")
    if args.cuda_graph and not args.device.startswith("cuda"):
        raise RuntimeError("CUDA graph capture requires a CUDA device")

    set_tf32(args.tf32)
    torch.manual_seed(args.seed)

    from plu.whisper import WhisperForConditionalGeneration

    dtype = parse_dtype(args.dtype)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device=args.device, dtype=dtype)
    model.train()
    if args.mode == "forward":
        model.eval()

    encoder_tokens = encoder_tokens_after_conv(args.input_features)
    extended_positions = False
    if encoder_tokens > model.config.max_source_positions:
        if not args.extend_encoder_positions:
            raise ValueError(
                f"input_features={args.input_features} becomes {encoder_tokens} encoder tokens, "
                f"but model supports only {model.config.max_source_positions}"
            )
        extended_positions = extend_encoder_positions(model, encoder_tokens)

    input_features = torch.randn(
        args.batch,
        model.config.num_mel_bins,
        args.input_features,
        device=args.device,
        dtype=dtype,
    )
    labels = torch.randint(model.config.vocab_size, (args.batch, args.decoder_len), device=args.device)
    packed_mx: dict[int, PackedMxLinear] | None = None
    mx_stats: dict[str, Any] = {}
    if args.child_backend in {"mxfp8", "mxfp4", "nvfp4"}:
        packed_mx, mx_stats = pack_mx_model(model, args.child_backend)
    total_encoder_layers = len(model.model.encoder.layers)
    backward_encoder_layer_start = encoder_backward_start(total_encoder_layers, args.encoder_backward_layers)

    def step() -> tuple[torch.Tensor, torch.Size]:
        if args.mode == "forward":
            with torch.no_grad():
                if packed_mx is not None:
                    output = packed_mx_forward(
                        model,
                        packed_mx,
                        input_features,
                        labels,
                        master_weight_grads=False,
                        encoder_backward_layers=None,
                        fused_unembedding_ce=args.fused_unembedding_ce,
                    )
                else:
                    if args.fused_unembedding_ce:
                        output = limited_model_forward(model, input_features, labels, None, args.fused_unembedding_ce)
                    else:
                        output = model(input_features, labels=labels)
        else:
            model.zero_grad(set_to_none=True)
            if packed_mx is not None:
                output = packed_mx_forward(
                    model,
                    packed_mx,
                    input_features,
                    labels,
                    master_weight_grads=True,
                    encoder_backward_layers=args.encoder_backward_layers,
                    fused_unembedding_ce=args.fused_unembedding_ce,
                )
            else:
                if args.encoder_backward_layers is None and not args.fused_unembedding_ce:
                    output = model(input_features, labels=labels)
                else:
                    output = limited_model_forward(model, input_features, labels, args.encoder_backward_layers, args.fused_unembedding_ce)
        if output.loss is None:
            raise RuntimeError("expected loss for end-to-end benchmark")
        if args.mode == "forward_backward":
            output.loss.backward()
        if output.logits.numel() == 0:
            logits_shape = torch.Size((labels.shape[0], labels.shape[1], model.config.vocab_size))
        else:
            logits_shape = output.logits.shape
        return output.loss.detach(), logits_shape

    times_ms: list[float] = []
    loss_value = 0.0
    logits_shape: torch.Size | None = None
    cuda_graph_capture_ms: float | None = None

    for _ in range(args.warmup):
        loss, logits_shape = step()
        loss_value = float(loss.cpu())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if args.cuda_graph:
        graph = torch.cuda.CUDAGraph()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        capture_start = time.perf_counter()
        with torch.cuda.graph(graph):
            static_loss, logits_shape = step()
        torch.cuda.synchronize()
        cuda_graph_capture_ms = (time.perf_counter() - capture_start) * 1000.0

        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        loss_value = float(static_loss.detach().cpu())
    else:
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss, logits_shape = step()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
            loss_value = float(loss.cpu())

    mean_ms = sum(times_ms) / len(times_ms)
    audio_secs_per_sample = audio_seconds(args.input_features, args.feature_fps)
    processed_audio_secs = args.batch * audio_secs_per_sample
    mean_seconds = mean_ms / 1000.0
    return {
        "op": "end_to_end",
        "target": args.child_backend,
        "mode": f"{args.mode}_cuda_graph" if args.cuda_graph else args.mode,
        "cuda_graph": args.cuda_graph,
        "model": args.model,
        "dtype": str(dtype).replace("torch.", ""),
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "batch": args.batch,
        "input_features": args.input_features,
        "feature_fps": args.feature_fps,
        "audio_seconds": processed_audio_secs,
        "audio_seconds_per_sample": audio_secs_per_sample,
        "processed_audio_seconds": processed_audio_secs,
        "encoder_tokens_after_conv": encoder_tokens,
        "encoder_layers": total_encoder_layers,
        "encoder_backward_layers": args.encoder_backward_layers if args.mode == "forward_backward" else None,
        "encoder_backward_layer_start": backward_encoder_layer_start if args.mode == "forward_backward" else None,
        "decoder_tokens": args.decoder_len,
        "input_features_shape": list(input_features.shape),
        "labels_shape": list(labels.shape),
        "logits_shape": list(logits_shape) if logits_shape is not None else None,
        "extended_encoder_positions": extended_positions,
        "tf32": args.tf32,
        "fused_unembedding_ce": args.fused_unembedding_ce,
        "warmup": args.warmup,
        "iters": args.iters,
        "times_ms": times_ms,
        "mean_ms": mean_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "cuda_graph_capture_ms": cuda_graph_capture_ms,
        "x_real_time": processed_audio_secs / mean_seconds,
        "real_time_factor": mean_seconds / processed_audio_secs,
        "loss": loss_value,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        **mx_stats,
    }


def child_argv(args: argparse.Namespace, backend: str) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-backend",
        backend,
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch",
        str(args.batch),
        "--input-features",
        str(args.input_features),
        "--feature-fps",
        str(args.feature_fps),
        "--decoder-len",
        str(args.decoder_len),
        "--encoder-backward-layers",
        "none" if args.encoder_backward_layers is None else str(args.encoder_backward_layers),
        "--mode",
        args.mode,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
    ]
    argv.append("--tf32" if args.tf32 else "--no-tf32")
    argv.append("--extend-encoder-positions" if args.extend_encoder_positions else "--no-extend-encoder-positions")
    argv.append("--cuda-graph" if args.cuda_graph else "--no-cuda-graph")
    argv.append("--fused-unembedding-ce" if args.fused_unembedding_ce else "--no-fused-unembedding-ce")
    return argv


def run_parent(args: argparse.Namespace) -> None:
    if args.backend == "all":
        backends = ["ref", "triton", "mxfp8", "mxfp4", "nvfp4"]
    else:
        backends = [args.backend]
    rows: list[dict[str, Any]] = []
    for backend in backends:
        env = os.environ.copy()
        env["PLU_OPS_BACKEND"] = backend
        completed = subprocess.run(child_argv(args, backend), env=env, text=True, capture_output=True)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            raise SystemExit(completed.returncode)

        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            print(json.dumps(row), flush=True)

    by_target = {row["target"]: row for row in rows}
    if "ref" in by_target and "triton" in by_target:
        ref_ms = by_target["ref"]["mean_ms"]
        triton_ms = by_target["triton"]["mean_ms"]
        summary = {
            "op": "end_to_end",
            "target": "summary",
            "mode": f"{args.mode}_cuda_graph" if args.cuda_graph else args.mode,
            "cuda_graph": args.cuda_graph,
            "model": args.model,
            "dtype": by_target["triton"]["dtype"],
            "batch": args.batch,
            "input_features": args.input_features,
            "feature_fps": by_target["triton"]["feature_fps"],
            "audio_seconds": by_target["triton"]["audio_seconds"],
            "audio_seconds_per_sample": by_target["triton"]["audio_seconds_per_sample"],
            "processed_audio_seconds": by_target["triton"]["processed_audio_seconds"],
            "encoder_tokens_after_conv": by_target["triton"]["encoder_tokens_after_conv"],
            "encoder_layers": by_target["triton"]["encoder_layers"],
            "encoder_backward_layers": by_target["triton"]["encoder_backward_layers"],
            "encoder_backward_layer_start": by_target["triton"]["encoder_backward_layer_start"],
            "decoder_tokens": args.decoder_len,
            "fused_unembedding_ce": by_target["triton"]["fused_unembedding_ce"],
            "ref_mean_ms": ref_ms,
            "triton_mean_ms": triton_ms,
            "speedup": ref_ms / triton_ms,
            "wall_time_reduction_pct": (1.0 - triton_ms / ref_ms) * 100.0,
            "ref_x_real_time": by_target["ref"]["x_real_time"],
            "triton_x_real_time": by_target["triton"]["x_real_time"],
            "ref_real_time_factor": by_target["ref"]["real_time_factor"],
            "triton_real_time_factor": by_target["triton"]["real_time_factor"],
            "real_time_speedup": by_target["triton"]["x_real_time"] / by_target["ref"]["x_real_time"],
            "ref_peak_allocated_gib": by_target["ref"]["peak_allocated_gib"],
            "triton_peak_allocated_gib": by_target["triton"]["peak_allocated_gib"],
        }
        print(json.dumps(summary), flush=True)
    baseline = by_target.get("triton") or by_target.get("ref")
    if baseline is not None:
        for target in ("mxfp8", "mxfp4", "nvfp4"):
            if target not in by_target:
                continue
            target_ms = by_target[target]["mean_ms"]
            if args.mode == "forward_backward":
                roofline_speedup = by_target[target].get("mx_train_weight_transfer_roofline_vs_bf16")
            else:
                roofline_speedup = by_target[target].get("mx_forward_weight_transfer_roofline_vs_bf16")
            actual_speedup = baseline["mean_ms"] / target_ms
            summary = {
                "op": "end_to_end",
                "target": f"summary_{target}",
                "baseline": baseline["target"],
                "mode": f"{args.mode}_cuda_graph" if args.cuda_graph else args.mode,
                "cuda_graph": args.cuda_graph,
                "model": args.model,
                "dtype": by_target[target]["dtype"],
                "batch": args.batch,
                "input_features": args.input_features,
                "audio_seconds": by_target[target]["audio_seconds"],
                "audio_seconds_per_sample": by_target[target]["audio_seconds_per_sample"],
                "processed_audio_seconds": by_target[target]["processed_audio_seconds"],
                "encoder_layers": by_target[target]["encoder_layers"],
                "encoder_backward_layers": by_target[target]["encoder_backward_layers"],
                "encoder_backward_layer_start": by_target[target]["encoder_backward_layer_start"],
                "decoder_tokens": args.decoder_len,
                "fused_unembedding_ce": by_target[target]["fused_unembedding_ce"],
                "baseline_mean_ms": baseline["mean_ms"],
                "target_mean_ms": target_ms,
                "speedup_vs_baseline": actual_speedup,
                "baseline_x_real_time": baseline["x_real_time"],
                "target_x_real_time": by_target[target]["x_real_time"],
                "real_time_speedup_vs_baseline": by_target[target]["x_real_time"] / baseline["x_real_time"],
                "baseline_peak_allocated_gib": baseline["peak_allocated_gib"],
                "target_peak_allocated_gib": by_target[target]["peak_allocated_gib"],
                "weight_transfer_roofline_speedup": roofline_speedup,
                "weight_transfer_roofline_fraction": actual_speedup / roofline_speedup if roofline_speedup else None,
            }
            print(json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full PLU Whisper forward+backward passes.")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo", help="Local model path or Hugging Face repo id.")
    parser.add_argument("--backend", choices=["all", "ref", "triton", "mxfp8", "mxfp4", "nvfp4"], default="all")
    parser.add_argument("--child-backend", choices=["ref", "triton", "mxfp8", "mxfp4", "nvfp4"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "fp32", "bfloat16", "bf16"], default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-features", type=int, default=3000, help="Input feature frames before Whisper conv downsampling.")
    parser.add_argument("--feature-fps", type=float, default=DEFAULT_FEATURE_FPS, help="Input feature frames per second of audio.")
    parser.add_argument("--decoder-len", type=int, default=448)
    parser.add_argument(
        "--encoder-backward-layers",
        default=None,
        help="For forward_backward, run full encoder forward but backprop through only the last N encoder layers. Use 'none' for all layers.",
    )
    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward_backward")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extend-encoder-positions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-unembedding-ce", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.feature_fps <= 0:
        raise ValueError("--feature-fps must be positive")
    if args.encoder_backward_layers is not None:
        if str(args.encoder_backward_layers).lower() == "none":
            args.encoder_backward_layers = None
        else:
            args.encoder_backward_layers = int(args.encoder_backward_layers)
    return args


def main() -> None:
    args = parse_args()
    if args.child_backend is not None:
        print(json.dumps(run_child(args)), flush=True)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
