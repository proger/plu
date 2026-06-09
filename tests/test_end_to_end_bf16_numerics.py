from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "plu" / "assets" / "realistic_inputs.pt"
DEFAULT_MODEL = (
    Path(os.environ.get("PLU_MODEL_CACHE", "~/.cache/plu/models")).expanduser()
    / "openai--whisper-large-v3-turbo"
)
GRAD_SLICE = 128
DECODER_TOKENS = 4
ENCODER_BACKWARD_LAYERS = 24


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


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _required_file(path: Path, reason: str) -> Path:
    if not path.exists():
        pytest.skip(reason)
    return path


def _realistic_inputs_path() -> Path:
    return _required_file(
        Path(os.environ.get("PLU_REALISTIC_INPUTS", str(DEFAULT_INPUTS))).expanduser(),
        "realistic input payload is required; generate plu/assets/realistic_inputs.pt first",
    )


def _model_path() -> Path:
    return _required_file(
        Path(os.environ.get("PLU_E2E_MODEL", str(DEFAULT_MODEL))).expanduser(),
        "cached openai/whisper-large-v3-turbo model is required for end-to-end numerics",
    )


def _slice_grad(parameter: torch.nn.Parameter) -> torch.Tensor:
    if parameter.grad is None:
        raise RuntimeError("expected gradient to be populated")
    grad = parameter.grad.detach().float().cpu()
    if grad.ndim == 1:
        return grad[:GRAD_SLICE].contiguous()
    if grad.ndim == 2:
        return grad[:GRAD_SLICE, :GRAD_SLICE].contiguous()
    if grad.ndim == 3:
        return grad[:GRAD_SLICE, :GRAD_SLICE, :].contiguous()
    raise RuntimeError(f"unsupported gradient rank: {grad.ndim}")


def _labels_from_payload(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    if "labels" in payload:
        labels = payload["labels"]
    elif "logits" in payload:
        labels = payload["logits"].argmax(dim=-1)
    else:
        labels = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    return labels[:, :DECODER_TOKENS].long().contiguous()


def _keep_layer_norm_parameters_high_precision(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(dtype=torch.float32)


def _layer_norm_parameter_dtypes(model: torch.nn.Module) -> list[str]:
    dtypes = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            dtypes.append(str(module.weight.dtype))
            if module.bias is not None:
                dtypes.append(str(module.bias.dtype))
    return dtypes


def _pack_frozen_encoder_layers(model: torch.nn.Module, frozen_layers: int):
    from plu.benchmarks.bench_end_to_end import pack_mx_linear

    packed = {}
    for layer in model.model.encoder.layers[:frozen_layers]:
        for module in layer.modules():
            if module.__class__.__name__ == "CastLinear":
                packed[id(module)] = pack_mx_linear(module, "mxfp8")
    return packed


def _limited_forward(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    frozen_encoder_mxfp8: bool,
):
    from plu.benchmarks.bench_end_to_end import _packed_encoder_layer, encoder_backward_start
    from plu.whisper import Seq2SeqOutput, cross_entropy, encoder_position_embedding, shift_tokens_right

    encoder = model.model.encoder
    hidden_states = encoder.conv1(input_features)
    hidden_states = encoder.conv2(hidden_states)
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = encoder_position_embedding(hidden_states, encoder.embed_positions.weight)

    backward_start = encoder_backward_start(len(encoder.layers), ENCODER_BACKWARD_LAYERS)
    packed = _pack_frozen_encoder_layers(model, backward_start) if frozen_encoder_mxfp8 and backward_start else None
    for layer in encoder.layers[:backward_start]:
        with torch.no_grad():
            if packed is None:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = _packed_encoder_layer(packed, layer, hidden_states, master_weight_grads=False)
    if backward_start:
        hidden_states = hidden_states.detach()
    for layer in encoder.layers[backward_start:]:
        hidden_states = layer(hidden_states)
    encoder_hidden_states = encoder.layer_norm(hidden_states)

    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    decoder_hidden_states = model.model.decoder(decoder_input_ids, encoder_hidden_states)
    logits = model.proj_out(decoder_hidden_states).float()
    loss = cross_entropy(logits, labels, ignore_index=-100)
    return Seq2SeqOutput(loss=loss, logits=logits), backward_start, len(packed) if packed is not None else 0


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ref", "triton"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-encoder-mxfp8", action="store_true")
    args = parser.parse_args()

    os.environ["PLU_OPS_BACKEND"] = args.backend
    _set_tf32(True)
    torch.manual_seed(0)

    from plu.whisper import WhisperForConditionalGeneration

    device = torch.device("cuda")
    payload = _torch_load(Path(args.inputs))
    input_features = payload["input_features"].to(device=device, dtype=torch.bfloat16).contiguous()
    labels = _labels_from_payload(payload).to(device=device)

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device=device, dtype=torch.bfloat16)
    _keep_layer_norm_parameters_high_precision(model)
    model.train()
    model.zero_grad(set_to_none=True)

    activations: dict[str, torch.Tensor] = {}

    def save_activation(name: str):
        def hook(_module, _inputs, output):
            activations[name] = output.detach().float().cpu()

        return hook

    handles = [
        model.model.encoder.register_forward_hook(save_activation("encoder_hidden")),
        model.model.decoder.register_forward_hook(save_activation("decoder_hidden")),
    ]
    try:
        output, encoder_backward_layer_start, frozen_encoder_packed_linear_count = _limited_forward(
            model,
            input_features,
            labels,
            frozen_encoder_mxfp8=args.frozen_encoder_mxfp8,
        )
        assert output.loss is not None
        output.loss.backward()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    frozen_encoder_layer = model.model.encoder.layers[0]
    encoder_layer = model.model.encoder.layers[encoder_backward_layer_start]
    decoder_layer = model.model.decoder.layers[0]
    result = {
        "backend": args.backend,
        "weight_dtype": str(next(model.parameters()).dtype),
        "input_dtype": str(input_features.dtype),
        "layer_norm_parameter_dtypes": _layer_norm_parameter_dtypes(model),
        "encoder_backward_layers": ENCODER_BACKWARD_LAYERS,
        "encoder_backward_layer_start": encoder_backward_layer_start,
        "frozen_encoder_weight_format": "mxfp8" if args.frozen_encoder_mxfp8 else "bf16",
        "frozen_encoder_packed_linear_count": frozen_encoder_packed_linear_count,
        "frozen_encoder_self_q_weight_grad_is_none": frozen_encoder_layer.self_attn.q_proj.weight.grad is None,
        "encoder_conv1_weight_grad_is_none": model.model.encoder.conv1.weight.grad is None,
        "labels": labels.detach().cpu(),
        "loss": output.loss.detach().float().cpu(),
        "logits": output.logits.detach().float().cpu(),
        "activations": activations,
        "grads": {
            "encoder_train_self_q_weight": _slice_grad(encoder_layer.self_attn.q_proj.weight),
            "encoder_train_mlp_fc1_weight": _slice_grad(encoder_layer.fc1.weight),
            "decoder_self_q_weight": _slice_grad(decoder_layer.self_attn.q_proj.weight),
            "decoder_cross_q_weight": _slice_grad(decoder_layer.encoder_attn.q_proj.weight),
            "decoder_proj_out_weight": _slice_grad(model.proj_out.weight),
        },
    }
    torch.save(result, args.out)


def _run_backend(backend: str, model: Path, inputs: Path, out: Path) -> None:
    env = os.environ.copy()
    env["PLU_OPS_BACKEND"] = backend
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        backend,
        "--model",
        str(model),
        "--inputs",
        str(inputs),
        "--out",
        str(out),
    ]
    if backend == "triton":
        command.append("--frozen-encoder-mxfp8")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("PLU_E2E_NUMERICS_TIMEOUT", "180")),
    )
    assert completed.returncode == 0, (
        f"{backend} subprocess failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    diff = (actual.detach().float() - expected.detach().float()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=f"{name}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}",
    )


def _assert_rel_l2(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_rel_l2: float,
    min_cosine: float,
) -> None:
    actual_flat = actual.detach().float().flatten()
    expected_flat = expected.detach().float().flatten()
    diff = actual_flat - expected_flat
    rel_l2 = float(diff.norm() / expected_flat.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(actual_flat, expected_flat, dim=0))
    assert rel_l2 <= max_rel_l2 and cosine >= min_cosine, (
        f"{name}: rel_l2={rel_l2:.6g} > {max_rel_l2:.6g} or cosine={cosine:.6g} < {min_cosine:.6g}"
    )


def _assert_grad_sanity(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_norm = float(actual.detach().float().norm())
    expected_norm = float(expected.detach().float().norm())
    assert torch.isfinite(actual.detach().float()).all(), f"{name}: Triton gradient has non-finite values"
    assert torch.isfinite(expected.detach().float()).all(), f"{name}: reference gradient has non-finite values"
    assert actual_norm > 0.0, f"{name}: Triton gradient is zero"
    assert expected_norm > 0.0, f"{name}: reference gradient is zero"
    ratio = actual_norm / expected_norm
    assert 1e-3 <= ratio <= 10.0, f"{name}: gradient norm ratio {ratio:.6g} is outside sanity bounds"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton end-to-end numerics")
def test_whisper_turbo_real_inputs_bf16_weights_end_to_end_numerics(tmp_path: Path) -> None:
    pytest.importorskip("triton")
    inputs = _realistic_inputs_path()
    model = _model_path()
    ref_path = tmp_path / "ref.pt"
    triton_path = tmp_path / "triton.pt"

    _run_backend("ref", model, inputs, ref_path)
    _run_backend("triton", model, inputs, triton_path)

    ref = _torch_load(ref_path)
    triton = _torch_load(triton_path)
    assert ref["weight_dtype"] == "torch.bfloat16"
    assert triton["weight_dtype"] == "torch.bfloat16"
    assert ref["input_dtype"] == "torch.bfloat16"
    assert triton["input_dtype"] == "torch.bfloat16"
    assert set(ref["layer_norm_parameter_dtypes"]) == {"torch.float32"}
    assert set(triton["layer_norm_parameter_dtypes"]) == {"torch.float32"}
    assert ref["encoder_backward_layers"] == ENCODER_BACKWARD_LAYERS
    assert triton["encoder_backward_layers"] == ENCODER_BACKWARD_LAYERS
    assert ref["encoder_backward_layer_start"] == 8
    assert triton["encoder_backward_layer_start"] == 8
    assert ref["frozen_encoder_weight_format"] == "bf16"
    assert triton["frozen_encoder_weight_format"] == "mxfp8"
    assert triton["frozen_encoder_packed_linear_count"] == 48
    assert ref["frozen_encoder_self_q_weight_grad_is_none"]
    assert triton["frozen_encoder_self_q_weight_grad_is_none"]
    assert ref["encoder_conv1_weight_grad_is_none"]
    assert triton["encoder_conv1_weight_grad_is_none"]
    torch.testing.assert_close(triton["labels"], ref["labels"], atol=0, rtol=0)

    _assert_close("loss", triton["loss"], ref["loss"], atol=1.2e-1, rtol=2e-2)
    _assert_close("logits", triton["logits"], ref["logits"], atol=5e-1, rtol=3e-2)
    _assert_rel_l2(
        "encoder_hidden",
        triton["activations"]["encoder_hidden"],
        ref["activations"]["encoder_hidden"],
        max_rel_l2=1.5e-1,
        min_cosine=9.9e-1,
    )
    _assert_rel_l2(
        "decoder_hidden",
        triton["activations"]["decoder_hidden"],
        ref["activations"]["decoder_hidden"],
        max_rel_l2=6e-2,
        min_cosine=9.98e-1,
    )

    for name in ("encoder_train_self_q_weight", "encoder_train_mlp_fc1_weight"):
        _assert_grad_sanity(f"{name}.grad", triton["grads"][name], ref["grads"][name])

    _assert_close(
        "decoder_self_q_weight.grad",
        triton["grads"]["decoder_self_q_weight"],
        ref["grads"]["decoder_self_q_weight"],
        atol=1e-4,
        rtol=1e-1,
    )
    _assert_close(
        "decoder_cross_q_weight.grad",
        triton["grads"]["decoder_cross_q_weight"],
        ref["grads"]["decoder_cross_q_weight"],
        atol=2e-3,
        rtol=2e-1,
    )
    _assert_close(
        "decoder_proj_out_weight.grad",
        triton["grads"]["decoder_proj_out_weight"],
        ref["grads"]["decoder_proj_out_weight"],
        atol=2e-3,
        rtol=2e-1,
    )


if __name__ == "__main__":
    _child_main()
