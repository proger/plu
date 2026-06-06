from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from plu.realistic_inputs import DEFAULT_WAV
from plu.train_data import load_audio, log_mel_spectrogram


def _set_tf32(enabled: bool) -> None:
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


def _load_ref_model(model_name_or_path: str, device: str):
    os.environ["PLU_OPS_BACKEND"] = "ref"
    sys.modules.pop("plu.whisper", None)
    from plu.whisper import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return model


def _clone_arg(arg: Any) -> Any:
    if not torch.is_tensor(arg):
        return arg
    cloned = arg.detach().clone()
    if cloned.is_floating_point():
        cloned.requires_grad_(True)
    return cloned


def _grad_source(output: Tensor, mode: str) -> tuple[Tensor, Tensor | None]:
    if output.ndim == 0:
        return output, None
    if mode == "sum":
        return output.float().sum(), None
    grad = torch.randn_like(output)
    return output, grad


def _summary(tensor: Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
    }


def _max_error(actual: Tensor, expected: Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return float((actual.detach().float() - expected.detach().float()).abs().max().cpu())


def _compare_tensor_op(
    name: str,
    ref_fn: Callable[..., Tensor],
    triton_fn: Callable[..., Tensor],
    args: tuple[Any, ...],
    *,
    grad_mode: str = "random",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    grad_atol: float = 1e-3,
    grad_rtol: float = 1e-3,
) -> dict[str, Any]:
    ref_args = tuple(_clone_arg(arg) for arg in args)
    triton_args = tuple(_clone_arg(arg) for arg in args)
    ref_out = ref_fn(*ref_args)
    triton_out = triton_fn(*triton_args)
    torch.testing.assert_close(triton_out, ref_out, atol=atol, rtol=rtol)

    grad_names: list[str] = []
    if any(torch.is_tensor(arg) and arg.requires_grad for arg in ref_args):
        if ref_out.ndim == 0:
            ref_out.backward()
            triton_out.backward()
        else:
            ref_scalar, grad = _grad_source(ref_out, grad_mode)
            if grad is None:
                triton_out.float().sum().backward()
                ref_scalar.backward()
            else:
                ref_out.backward(grad)
                triton_out.backward(grad)
        for index, (ref_arg, triton_arg) in enumerate(zip(ref_args, triton_args)):
            if torch.is_tensor(ref_arg) and ref_arg.requires_grad:
                torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=grad_atol, rtol=grad_rtol, msg=f"{name}.arg{index}")
                grad_names.append(f"arg{index}")

    return {
        "op": name,
        "output": _summary(ref_out),
        "max_abs_error": _max_error(triton_out, ref_out),
        "checked_grads": grad_names,
    }


def _compare_matmul_top1(ref_fn, triton_fn, x: Tensor, weight: Tensor, bias: Tensor | None) -> dict[str, Any]:
    ref_args = (_clone_arg(x), _clone_arg(weight), _clone_arg(bias) if bias is not None else None)
    triton_args = (_clone_arg(x), _clone_arg(weight), _clone_arg(bias) if bias is not None else None)
    ref_values, ref_indices = ref_fn(*ref_args)
    triton_values, triton_indices = triton_fn(*triton_args)
    torch.testing.assert_close(triton_values, ref_values, atol=2e-1, rtol=5e-3)
    torch.testing.assert_close(triton_indices, ref_indices)
    grad = torch.randn_like(ref_values)
    ref_values.backward(grad)
    triton_values.backward(grad)
    for index, (ref_arg, triton_arg) in enumerate(zip(ref_args, triton_args)):
        if torch.is_tensor(ref_arg) and ref_arg.requires_grad:
            torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=5e-3, rtol=1e-3, msg=f"matmul_top1.arg{index}")
    return {
        "op": "matmul_top1",
        "output": _summary(ref_values),
        "max_abs_error": _max_error(triton_values, ref_values),
        "indices_match": True,
        "checked_grads": ["arg0", "arg1"] + (["arg2"] if bias is not None else []),
    }


def build_realistic_cases(model_name_or_path: str, wav_path: str | Path, device: str) -> dict[str, Any]:
    from plu.ref.c_proj import c_proj as ref_c_proj
    from plu.ref.conv1d_gelu import conv1d_gelu as ref_conv1d_gelu
    from plu.ref.embedding import decoder_embedding as ref_decoder_embedding
    from plu.ref.embedding import encoder_position_embedding as ref_encoder_position_embedding
    from plu.ref.flash_attention import flash_attention as ref_flash_attention
    from plu.ref.layer_norm import layer_norm as ref_layer_norm
    from plu.ref.qkv_proj import qkv_proj as ref_qkv_proj
    from plu.ref.residual_add import residual_add as ref_residual_add

    model = _load_ref_model(model_name_or_path, device)
    audio = load_audio(wav_path)
    features = log_mel_spectrogram(audio, model.config.num_mel_bins).unsqueeze(0).to(device)
    decoder_input_ids = torch.full((1, 4), model.config.decoder_start_token_id, dtype=torch.long, device=device)

    encoder = model.model.encoder
    decoder = model.model.decoder
    enc_layer = encoder.layers[0]

    with torch.no_grad():
        conv1 = ref_conv1d_gelu(features, encoder.conv1.weight, encoder.conv1.bias, encoder.conv1.stride[0], encoder.conv1.padding[0])
        conv2 = ref_conv1d_gelu(conv1, encoder.conv2.weight, encoder.conv2.bias, encoder.conv2.stride[0], encoder.conv2.padding[0])
        encoder_input = conv2.transpose(1, 2).contiguous()
        positioned = ref_encoder_position_embedding(encoder_input, encoder.embed_positions.weight)
        attn_norm = ref_layer_norm(positioned, enc_layer.self_attn_layer_norm.weight, enc_layer.self_attn_layer_norm.bias, enc_layer.self_attn_layer_norm.eps)
        query = ref_qkv_proj(attn_norm, enc_layer.self_attn.q_proj.weight, enc_layer.self_attn.q_proj.bias, enc_layer.self_attn.num_heads)
        key = ref_qkv_proj(attn_norm, enc_layer.self_attn.k_proj.weight, enc_layer.self_attn.k_proj.bias, enc_layer.self_attn.num_heads)
        value = ref_qkv_proj(attn_norm, enc_layer.self_attn.v_proj.weight, enc_layer.self_attn.v_proj.bias, enc_layer.self_attn.num_heads)
        attended = ref_flash_attention(query, key, value, None)
        projected = ref_c_proj(attended, enc_layer.self_attn.out_proj.weight, enc_layer.self_attn.out_proj.bias)
        residual = ref_residual_add(positioned, projected)
        mlp_input = ref_layer_norm(residual, enc_layer.final_layer_norm.weight, enc_layer.final_layer_norm.bias, enc_layer.final_layer_norm.eps)
        encoder_hidden = model.model.encoder(features)
        decoder_hidden = model.model.decoder(decoder_input_ids, encoder_hidden)
        logits = model.proj_out(decoder_hidden).float()

    lora_rank = 8
    return {
        "model": model,
        "features": features.detach(),
        "conv1": (features.detach(), encoder.conv1.weight.detach(), encoder.conv1.bias.detach(), encoder.conv1.stride[0], encoder.conv1.padding[0]),
        "conv2": (conv1.detach(), encoder.conv2.weight.detach(), encoder.conv2.bias.detach(), encoder.conv2.stride[0], encoder.conv2.padding[0]),
        "encoder_position_embedding": (encoder_input.detach(), encoder.embed_positions.weight.detach()),
        "decoder_embedding": (decoder_input_ids.detach(), decoder.embed_tokens.weight.detach(), decoder.embed_positions.weight.detach(), encoder_hidden.dtype),
        "layer_norm": (positioned.detach(), enc_layer.self_attn_layer_norm.weight.detach(), enc_layer.self_attn_layer_norm.bias.detach(), enc_layer.self_attn_layer_norm.eps),
        "qkv_proj": (attn_norm.detach(), enc_layer.self_attn.q_proj.weight.detach(), enc_layer.self_attn.q_proj.bias.detach(), enc_layer.self_attn.num_heads),
        "flash_attention": (query.detach(), key.detach(), value.detach(), None),
        "c_proj": (attended.detach(), enc_layer.self_attn.out_proj.weight.detach(), enc_layer.self_attn.out_proj.bias.detach()),
        "residual_add": (positioned.detach(), projected.detach()),
        "gelu_mlp": (mlp_input.detach(), enc_layer.fc1.weight.detach(), enc_layer.fc1.bias.detach(), enc_layer.fc2.weight.detach(), enc_layer.fc2.bias.detach()),
        "linear": (mlp_input.detach(), enc_layer.fc1.weight.detach(), enc_layer.fc1.bias.detach()),
        "matmul_top1": (decoder_hidden[:, -1].detach(), model.proj_out.weight.detach(), model.proj_out.bias.detach() if model.proj_out.bias is not None else None),
        "cross_entropy": (logits.detach(), logits.argmax(dim=-1).detach()),
        "lora": (
            mlp_input.detach(),
            mlp_input.detach(),
            enc_layer.fc1.weight.detach(),
            enc_layer.fc1.bias.detach(),
            enc_layer.fc1.weight[:lora_rank].detach(),
            enc_layer.fc1.weight[:, :lora_rank].detach(),
            1.0,
        ),
    }


def validate_realistic_ops(model_name_or_path: str, wav_path: str | Path = DEFAULT_WAV, device: str = "cuda") -> list[dict[str, Any]]:
    from plu.ref.c_proj import c_proj as ref_c_proj
    from plu.ref.conv1d_gelu import conv1d_gelu as ref_conv1d_gelu
    from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
    from plu.ref.embedding import decoder_embedding as ref_decoder_embedding
    from plu.ref.embedding import encoder_position_embedding as ref_encoder_position_embedding
    from plu.ref.flash_attention import flash_attention as ref_flash_attention
    from plu.ref.gelu_mlp import gelu_mlp as ref_gelu_mlp
    from plu.ref.layer_norm import layer_norm as ref_layer_norm
    from plu.ref.linear import linear as ref_linear
    from plu.ref.lora import lora_linear as ref_lora_linear
    from plu.ref.matmul_top1 import matmul_top1 as ref_matmul_top1
    from plu.ref.qkv_proj import qkv_proj as ref_qkv_proj
    from plu.ref.residual_add import residual_add as ref_residual_add
    from plu.triton.c_proj import c_proj as triton_c_proj
    from plu.triton.conv1d_gelu import conv1d_gelu as triton_conv1d_gelu
    from plu.triton.cross_entropy import cross_entropy as triton_cross_entropy
    from plu.triton.embedding import decoder_embedding as triton_decoder_embedding
    from plu.triton.embedding import encoder_position_embedding as triton_encoder_position_embedding
    from plu.triton.flash_attention import flash_attention as triton_flash_attention
    from plu.triton.gelu_mlp import gelu_mlp as triton_gelu_mlp
    from plu.triton.layer_norm import layer_norm as triton_layer_norm
    from plu.triton.linear import linear as triton_linear
    from plu.triton.lora import lora_linear as triton_lora_linear
    from plu.triton.matmul_top1 import matmul_top1 as triton_matmul_top1
    from plu.triton.qkv_proj import qkv_proj as triton_qkv_proj
    from plu.triton.residual_add import residual_add as triton_residual_add

    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Triton realistic numerics")
    _set_tf32(True)
    torch.manual_seed(0)
    cases = build_realistic_cases(model_name_or_path, wav_path, device)
    results = [
        _compare_tensor_op(
            "conv1d_gelu.conv1",
            lambda x, weight, bias, stride, padding: ref_conv1d_gelu(x, weight, bias, stride, padding),
            lambda x, weight, bias, stride, padding: triton_conv1d_gelu(x, weight, bias, stride, padding),
            cases["conv1"],
            grad_mode="sum",
            atol=2e-2,
            rtol=5e-3,
            grad_atol=3.0,
            grad_rtol=3e-2,
        ),
        _compare_tensor_op(
            "conv1d_gelu.conv2",
            lambda x, weight, bias, stride, padding: ref_conv1d_gelu(x, weight, bias, stride, padding),
            lambda x, weight, bias, stride, padding: triton_conv1d_gelu(x, weight, bias, stride, padding),
            cases["conv2"],
            grad_mode="sum",
            atol=2e-2,
            rtol=5e-3,
            grad_atol=3.0,
            grad_rtol=3e-2,
        ),
        _compare_tensor_op("encoder_position_embedding", ref_encoder_position_embedding, triton_encoder_position_embedding, cases["encoder_position_embedding"], atol=0, rtol=0, grad_atol=0, grad_rtol=0),
        _compare_tensor_op("decoder_embedding", ref_decoder_embedding, triton_decoder_embedding, cases["decoder_embedding"], atol=0, rtol=0, grad_atol=1e-6, grad_rtol=1e-4),
        _compare_tensor_op("layer_norm", ref_layer_norm, triton_layer_norm, cases["layer_norm"], atol=2e-5, rtol=2e-5, grad_atol=2e-4, grad_rtol=2e-4),
        _compare_tensor_op("qkv_proj", ref_qkv_proj, triton_qkv_proj, cases["qkv_proj"], grad_mode="sum", atol=5e-2, rtol=5e-3, grad_atol=2e-1, grad_rtol=5e-3),
        _compare_tensor_op("flash_attention", ref_flash_attention, triton_flash_attention, cases["flash_attention"], grad_mode="sum", atol=2e-3, rtol=1e-2, grad_atol=1e-2, grad_rtol=5e-2),
        _compare_tensor_op("c_proj", ref_c_proj, triton_c_proj, cases["c_proj"], grad_mode="sum", atol=5e-2, rtol=5e-3, grad_atol=2e-1, grad_rtol=5e-3),
        _compare_tensor_op("residual_add", ref_residual_add, triton_residual_add, cases["residual_add"], atol=0, rtol=0, grad_atol=0, grad_rtol=0),
        _compare_tensor_op("gelu_mlp", ref_gelu_mlp, triton_gelu_mlp, cases["gelu_mlp"], grad_mode="sum", atol=1e-1, rtol=7e-3, grad_atol=1.0, grad_rtol=7e-3),
        _compare_tensor_op("linear", ref_linear, triton_linear, cases["linear"], grad_mode="sum", atol=5e-2, rtol=5e-3, grad_atol=2e-1, grad_rtol=5e-3),
        _compare_matmul_top1(ref_matmul_top1, triton_matmul_top1, *cases["matmul_top1"]),
        _compare_tensor_op("cross_entropy", ref_cross_entropy, triton_cross_entropy, cases["cross_entropy"], atol=1e-4, rtol=1e-4, grad_atol=1e-4, grad_rtol=1e-4),
        _compare_tensor_op("lora", ref_lora_linear, triton_lora_linear, cases["lora"], grad_mode="sum", atol=1e-1, rtol=7e-3, grad_atol=1.0, grad_rtol=7e-3),
    ]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Triton ops against ref ops on distributions captured from a reference Whisper wav pass.")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = validate_realistic_ops(args.model, args.wav, args.device)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
