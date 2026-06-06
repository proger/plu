import argparse
import concurrent.futures
import contextlib
import gc
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from plu.train_data import Corpus, register_data_args
from plu.wer import word_error_rate
from plu.tokenizer import NO_SPEECH, WhisperTokenizer
from plu.whisper import Seq2SeqOutput, WhisperForConditionalGeneration, cross_entropy, encoder_position_embedding, resolve_model_path, shift_tokens_right


logger = logging.getLogger(__name__)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to a pretrained model directory or model identifier from huggingface.co/models.",
        default="openai/whisper-large-v3",
    )
    register_data_args(parser)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=5000, help="Total number of optimization steps to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of forward/backward passes to accumulate before each optimizer update.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use.",
    )
    parser.add_argument("--num_warmup_steps", type=int, default=25, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--exp", type=str, default="exp/1", help="Where to store checkpoints and model files.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Retained for CLI compatibility; metrics are written as JSONL.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every n optimizer steps.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint directory.")
    parser.add_argument("--eval_at_init", action="store_true", help="Evaluate the model at initialization")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16", help="Autocast precision.")
    parser.add_argument("--device", type=str, default=None, help="Training device. Defaults to cuda when available, otherwise cpu.")

    parser.add_argument(
        "--encoder_backward_layers",
        default="auto",
        help="Run the full encoder forward pass but backpropagate through only the last N encoder layers. Default auto uses up to 24; use 'none' for all layers.",
    )
    parser.add_argument("--cuda_graph", action=argparse.BooleanOptionalAction, default=True, help="Capture and replay the training forward/backward pass.")
    parser.add_argument("--cuda_graph_warmup_steps", type=int, default=3, help="Warmup forward/backward passes before CUDA graph capture.")
    parser.add_argument("--static_input_features", type=int, default=3000, help="Static feature-frame length used for CUDA graph training.")
    parser.add_argument("--static_decoder_len", type=int, default=448, help="Static decoder-label length used for CUDA graph training.")
    parser.add_argument(
        "--frozen_encoder_linear_format",
        choices=["bf16", "mxfp8"],
        default="mxfp8",
        help="Weight format for linear layers in frozen encoder layers.",
    )

    args = parser.parse_args()
    if args.encoder_backward_layers is not None:
        encoder_backward_layers = str(args.encoder_backward_layers).lower()
        if encoder_backward_layers == "none":
            args.encoder_backward_layers = None
        elif encoder_backward_layers == "auto":
            args.encoder_backward_layers = "auto"
        else:
            args.encoder_backward_layers = int(args.encoder_backward_layers)
    assert args.train is not None, "need a training dataset, use --train file.jsonl"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def encoder_backward_start(total_layers: int, backward_layers: int | None) -> int:
    if backward_layers is None:
        return 0
    if backward_layers < 0:
        raise ValueError("--encoder_backward_layers must be non-negative")
    if backward_layers > total_layers:
        raise ValueError(f"--encoder_backward_layers={backward_layers} exceeds encoder layer count {total_layers}")
    return total_layers - backward_layers


def default_encoder_backward_layers(total_layers: int) -> int:
    return min(24, total_layers)


def freeze_encoder_prefix(model: WhisperForConditionalGeneration, backward_layers: int | None) -> int:
    encoder = model.model.encoder
    start = encoder_backward_start(len(encoder.layers), backward_layers)
    if start:
        for module in (encoder.conv1, encoder.conv2):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for layer in encoder.layers[:start]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    return start


def trainable_parameter_summary(model: torch.nn.Module) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.2f}"


def pack_mxfp8_linears(model: WhisperForConditionalGeneration) -> tuple[dict[int, Any], dict[str, Any]]:
    from benchmarks.bench_end_to_end import pack_mx_model

    return pack_mx_model(model, "mxfp8")


def accelerated_forward(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor,
    labels: torch.Tensor,
    encoder_backward_layers: int | None,
    frozen_encoder_linear_format: str,
    frozen_encoder_linears: dict[int, Any],
) -> Seq2SeqOutput:
    if frozen_encoder_linear_format == "mxfp8":
        from benchmarks.bench_end_to_end import packed_mx_forward

        return packed_mx_forward(
            model,
            frozen_encoder_linears,
            input_features,
            labels,
            master_weight_grads=True,
            encoder_backward_layers=encoder_backward_layers,
            fused_unembedding_ce=False,
        )

    encoder = model.model.encoder
    decoder = model.model.decoder
    backward_start = encoder_backward_start(len(encoder.layers), encoder_backward_layers)

    hidden_states = encoder.conv1(input_features)
    hidden_states = encoder.conv2(hidden_states)
    hidden_states = hidden_states.transpose(1, 2)
    if hidden_states.shape[1] > encoder.config.max_source_positions:
        raise ValueError(f"input features are too long: {hidden_states.shape[1]} > {encoder.config.max_source_positions}")
    hidden_states = encoder_position_embedding(hidden_states, encoder.embed_positions.weight)

    if backward_start:
        with torch.no_grad():
            for layer in encoder.layers[:backward_start]:
                hidden_states = layer(hidden_states)
        hidden_states = hidden_states.detach()

    for layer in encoder.layers[backward_start:]:
        hidden_states = layer(hidden_states)
    encoder_hidden_states = encoder.layer_norm(hidden_states)

    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    if decoder_input_ids.shape[1] > decoder.config.max_target_positions:
        decoder_input_ids = decoder_input_ids[:, -decoder.config.max_target_positions :]
    decoder_hidden_states = decoder(decoder_input_ids, encoder_hidden_states)
    logits = model.proj_out(decoder_hidden_states).float()
    loss = cross_entropy(logits, labels, ignore_index=-100)
    return Seq2SeqOutput(loss=loss, logits=logits)


def copy_static_batch(batch: dict[str, Any], static_features: torch.Tensor, static_labels: torch.Tensor) -> float:
    features = batch["input_features"].to(device=static_features.device, dtype=static_features.dtype, non_blocking=True)
    labels = batch["labels"].to(device=static_labels.device, non_blocking=True)
    if features.shape[0] != static_features.shape[0]:
        raise ValueError(f"CUDA graph batch size changed: {features.shape[0]} != {static_features.shape[0]}")
    if features.shape[1] != static_features.shape[1]:
        raise ValueError(f"feature mel bins changed: {features.shape[1]} != {static_features.shape[1]}")

    feature_frames = min(features.shape[-1], static_features.shape[-1])
    label_tokens = min(labels.shape[-1], static_labels.shape[-1])
    if feature_frames < static_features.shape[-1]:
        static_features.zero_()
    static_labels.fill_(-100)
    static_features[:, :, :feature_frames].copy_(features[:, :, :feature_frames])
    static_labels[:, :label_tokens].copy_(labels[:, :label_tokens])
    durations = batch.get("durations")
    if torch.is_tensor(durations):
        return float(durations.sum().item())
    return float(features.shape[0] * feature_frames / 100.0)


def get_lr_multiplier(name: str, step: int, warmup_steps: int, total_steps: int) -> float:
    if name == "constant":
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    if name == "constant_with_warmup":
        return 1.0

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if name == "linear":
        return max(0.0, 1.0 - progress)
    if name == "cosine":
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if name == "cosine_with_restarts":
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress % 1.0)))))
    if name == "polynomial":
        return max(0.0, (1.0 - progress))
    raise ValueError(f"unknown scheduler: {name}")


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_multiplier(args.lr_scheduler_type, step, args.num_warmup_steps, args.max_train_steps),
    )


def make_optimizer(parameters, args: argparse.Namespace, device: torch.device) -> tuple[torch.optim.Optimizer, str]:
    params = list(parameters)
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, fused=True), "fused_adamw"
        except TypeError:
            logger.warning("fused AdamW is not available in this PyTorch build; falling back to AdamW")
    return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay), "adamw"


def make_grad_scaler(device: torch.device, mixed_precision: str):
    enabled = device.type == "cuda" and mixed_precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "args": vars(args),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "training_state.pt",
    )


def load_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> int:
    state = torch.load(Path(checkpoint_dir) / "training_state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state.get("scaler_state_dict", {}))
    torch.set_rng_state(state["torch_rng_state"].cpu())
    random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state.get("step", 0))


def log_metric(exp: str, payload: dict[str, Any], step: int) -> None:
    Path(exp).mkdir(parents=True, exist_ok=True)
    row = {"step": step, **payload}
    with (Path(exp) / "train_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluation_loop(
    model: WhisperForConditionalGeneration,
    eval_dataloader,
    tokenizer: WhisperTokenizer,
    device: torch.device,
    mixed_precision: str,
    output_filename: str,
) -> dict[str, float]:
    model.eval()
    predictions = []
    references = []
    for batch in eval_dataloader:
        batch = move_batch(batch, device)
        if device.type == "cuda" and mixed_precision in {"bf16", "fp16"}:
            dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
            batch["input_features"] = batch["input_features"].to(dtype=dtype)
        with autocast_context(device, mixed_precision):
            generated_tokens = model.generate(batch["input_features"], max_new_tokens=255).cpu()
        labels = batch["labels"].detach().cpu()
        labels = labels.masked_fill(labels.eq(-100), tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i, text in enumerate(batch.get("texts") or []):
            if text:
                decoded_labels[i] = text

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)
        del generated_tokens, labels, batch
        gc.collect()

    try:
        wer = 100 * word_error_rate(predictions=predictions, references=references)
    except Exception:
        logger.exception("failed to compute WER")
        wer = 112
    eval_metrics = {"eval/wer": wer}

    for i, (hyp, ref) in enumerate(zip(predictions, references)):
        print(f"{i} ref", ref, sep="\t")
        print(f"{i} hyp", hyp, sep="\t")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({"metrics": eval_metrics, "hyp": predictions, "ref": references}, f, ensure_ascii=False)

    return eval_metrics


def next_training_batch(iterator, corpus: Corpus, args: argparse.Namespace):
    while True:
        try:
            return next(iterator), iterator
        except StopIteration:
            logger.info("training dataloader reached the end; resetting")
            iterator = iter(corpus.make_train_dataloader(corpus.load_dataset(args.train)))
        except Exception as exc:
            logger.error("data error while reading a training batch: %s; skipping", exc)


class TrainingBatchPrefetcher:
    def __init__(self, iterator, corpus: Corpus, args: argparse.Namespace):
        self.iterator = iterator
        self.corpus = corpus
        self.args = args
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None

    def _submit(self) -> None:
        if self.future is None:
            self.future = self.executor.submit(next_training_batch, self.iterator, self.corpus, self.args)

    def next(self, *, prefetch_next: bool = True):
        self._submit()
        batch, self.iterator = self.future.result()
        self.future = None
        if prefetch_next:
            self._submit()
        return batch

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z", level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)

    exp_path = Path(args.exp)
    exp_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_dir = resolve_model_path(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_dir, pad_token_id=model.config.pad_token_id)
    if args.encoder_backward_layers == "auto":
        args.encoder_backward_layers = default_encoder_backward_layers(len(model.model.encoder.layers))
    use_accelerated_forward = args.encoder_backward_layers is not None or args.frozen_encoder_linear_format != "bf16"
    if args.cuda_graph:
        if device.type != "cuda":
            raise RuntimeError("--cuda_graph requires a CUDA device")
        if args.mixed_precision != "bf16":
            raise RuntimeError("--cuda_graph training currently matches the benchmark BF16 path; use --mixed_precision bf16")
        if args.gradient_accumulation_steps != 1:
            raise RuntimeError("--cuda_graph training requires --gradient_accumulation_steps 1")
    if args.mixed_precision == "bf16" and device.type == "cuda":
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device)

    frozen_encoder_layers = freeze_encoder_prefix(model, args.encoder_backward_layers)
    mx_stats: dict[str, Any] = {}
    frozen_encoder_linears = {}
    if args.frozen_encoder_linear_format == "mxfp8":
        frozen_encoder_linears, mx_stats = pack_mxfp8_linears(model)
        logger.info(
            "Packed %s model linear layers as MXFP8; frozen encoder prefix layers also use the packed path.",
            len(frozen_encoder_linears),
        )

    corpus = Corpus(args, tokenizer=tokenizer, n_mels=model.config.num_mel_bins)
    train_dataloader = corpus.make_train_dataloader(corpus.load_dataset(args.train))
    eval_dataloader = corpus.make_eval_dataloader(corpus.load_dataset(args.eval))

    if args.eval_at_init:
        evaluation_loop(model, eval_dataloader, tokenizer, device, args.mixed_precision, os.path.join(args.exp, "init_results.json"))

    optimizer, optimizer_name = make_optimizer((parameter for parameter in model.parameters() if parameter.requires_grad), args, device)
    lr_scheduler = make_scheduler(optimizer, args)
    scaler = make_grad_scaler(device, args.mixed_precision)

    initial_step = 0
    if args.resume_from_checkpoint:
        initial_step = load_checkpoint(args.resume_from_checkpoint, model, optimizer, lr_scheduler, scaler, device)

    train_iterator = iter(train_dataloader)
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    logger.info("Instantaneous batch size per device = %s", args.per_device_train_batch_size)
    logger.info("Total train batch size with accumulation = %s", total_batch_size)
    logger.info("Gradient accumulation steps = %s", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %s", args.max_train_steps)
    logger.info("Optimizer = %s", optimizer_name)
    logger.info(trainable_parameter_summary(model))
    if args.encoder_backward_layers is not None:
        logger.info(
            "Encoder layers = %s; backward layers = %s; frozen prefix layers = %s.",
            len(model.model.encoder.layers),
            args.encoder_backward_layers,
            frozen_encoder_layers,
        )

    model.train()
    running_loss = 0.0
    tic = time.time()
    global_step = initial_step
    throughput_summary: dict[str, float] = {}

    if args.cuda_graph:
        static_features = torch.zeros(
            args.per_device_train_batch_size,
            model.config.num_mel_bins,
            args.static_input_features,
            device=device,
            dtype=torch.bfloat16,
        )
        static_labels = torch.full(
            (args.per_device_train_batch_size, args.static_decoder_len),
            -100,
            device=device,
            dtype=torch.long,
        )

        capture_batch, train_iterator = next_training_batch(train_iterator, corpus, args)
        copy_static_batch(capture_batch, static_features, static_labels)

        def graph_step() -> torch.Tensor:
            model.zero_grad(set_to_none=True)
            outputs = accelerated_forward(
                model,
                static_features,
                static_labels,
                args.encoder_backward_layers,
                args.frozen_encoder_linear_format,
                frozen_encoder_linears,
            )
            if outputs.loss is None:
                raise RuntimeError("model did not return a loss")
            outputs.loss.backward()
            return outputs.loss.detach()

        for _ in range(args.cuda_graph_warmup_steps):
            graph_step()
        torch.cuda.synchronize()

        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        with torch.cuda.graph(graph):
            static_loss = graph_step()
        torch.cuda.synchronize()
        cuda_graph_capture_ms = (time.perf_counter() - capture_start) * 1000.0
        logger.info("CUDA graph capture took %.3f ms.", cuda_graph_capture_ms)

        train_iterator = iter(train_dataloader)
        prefetcher = TrainingBatchPrefetcher(train_iterator, corpus, args)
        total_audio_seconds = 0.0
        total_gpu_ms = 0.0
        total_wall_seconds = 0.0
        window_audio_seconds = 0.0
        window_gpu_ms = 0.0
        window_wall_seconds = 0.0
        window_steps = 0

        try:
            while global_step < args.max_train_steps:
                step_wall_start = time.perf_counter()
                batch = prefetcher.next(prefetch_next=global_step + 1 < args.max_train_steps)
                audio_seconds = copy_static_batch(batch, static_features, static_labels)
                optimizer.zero_grad(set_to_none=False)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                graph.replay()
                end_event.record()

                optimizer.step()
                lr_scheduler.step()
                torch.cuda.synchronize()

                gpu_ms = start_event.elapsed_time(end_event)
                wall_seconds = time.perf_counter() - step_wall_start
                step_loss = float(static_loss.detach().cpu())

                total_audio_seconds += audio_seconds
                total_gpu_ms += gpu_ms
                total_wall_seconds += wall_seconds
                window_audio_seconds += audio_seconds
                window_gpu_ms += gpu_ms
                window_wall_seconds += wall_seconds
                window_steps += 1
                running_loss += step_loss

                if global_step % args.logging_steps == 0:
                    remaining_time = (time.time() - tic) / max(1, global_step - initial_step + 1) * (args.max_train_steps - global_step)
                    remaining_time_hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                    average_loss = running_loss / max(1, window_steps)
                    gpu_seconds = window_gpu_ms / 1000.0
                    metrics = {
                        "train/running_loss": average_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "train/audio_seconds": window_audio_seconds,
                        "train/mean_gpu_ms": window_gpu_ms / max(1, window_steps),
                        "train/mean_wall_ms": 1000.0 * window_wall_seconds / max(1, window_steps),
                        "train/x_real_time_gpu": window_audio_seconds / gpu_seconds if gpu_seconds > 0 else 0.0,
                        "train/real_time_factor_gpu": gpu_seconds / window_audio_seconds if window_audio_seconds > 0 else 0.0,
                        "train/x_real_time_wall": window_audio_seconds / window_wall_seconds if window_wall_seconds > 0 else 0.0,
                        "train/real_time_factor_wall": window_wall_seconds / window_audio_seconds if window_audio_seconds > 0 else 0.0,
                    }
                    logger.info(
                        "At step %s loss is %.3f. GPU RTF %.5f, wall RTF %.5f. Remaining time is %s.",
                        global_step,
                        average_loss,
                        metrics["train/real_time_factor_gpu"],
                        metrics["train/real_time_factor_wall"],
                        remaining_time_hh_mm_ss,
                    )
                    log_metric(args.exp, metrics, global_step)
                    running_loss = 0.0
                    window_audio_seconds = 0.0
                    window_gpu_ms = 0.0
                    window_wall_seconds = 0.0
                    window_steps = 0

                global_step += 1
        finally:
            prefetcher.close()

        total_gpu_seconds = total_gpu_ms / 1000.0
        throughput_summary = {
            "train/total_audio_seconds": total_audio_seconds,
            "train/total_gpu_seconds": total_gpu_seconds,
            "train/total_wall_seconds": total_wall_seconds,
            "train/x_real_time_gpu": total_audio_seconds / total_gpu_seconds if total_gpu_seconds > 0 else 0.0,
            "train/real_time_factor_gpu": total_gpu_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
            "train/x_real_time_wall": total_audio_seconds / total_wall_seconds if total_wall_seconds > 0 else 0.0,
            "train/real_time_factor_wall": total_wall_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
            "train/cuda_graph_capture_ms": cuda_graph_capture_ms,
            "train/mx_packed_linear_count": float(mx_stats.get("mx_packed_linear_count", 0)),
        }
        logger.info("Training throughput summary: %s", throughput_summary)
        log_metric(args.exp, throughput_summary, global_step)
    else:
        while global_step < args.max_train_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step_loss = 0.0
            last_batch = None
            last_outputs = None

            while accumulated < args.gradient_accumulation_steps:
                batch, train_iterator = next_training_batch(train_iterator, corpus, args)
                batch = move_batch(batch, device)
                with autocast_context(device, args.mixed_precision):
                    if use_accelerated_forward:
                        outputs = accelerated_forward(
                            model,
                            batch["input_features"],
                            batch["labels"],
                            args.encoder_backward_layers,
                            args.frozen_encoder_linear_format,
                            frozen_encoder_linears,
                        )
                    else:
                        outputs = model(input_features=batch["input_features"], labels=batch["labels"])
                    loss = outputs.loss / args.gradient_accumulation_steps
                if outputs.loss is None:
                    raise RuntimeError("model did not return a loss")
                scaler.scale(loss).backward()
                step_loss += float(outputs.loss.detach().cpu())
                accumulated += 1
                last_batch = batch
                last_outputs = outputs

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            step_loss /= max(1, accumulated)
            running_loss += step_loss

            if global_step % args.logging_steps == 0 and last_batch is not None and last_outputs is not None:
                labels = last_batch["labels"][:, 0]
                tokens = last_outputs.logits[:, 0].argmax(-1)
                acc = int((tokens == labels).sum().detach().cpu())
                probs = last_outputs.logits[:, 0].softmax(dim=-1)
                probs_ = [round(p, 2) for p in probs.max(-1).values.detach().cpu().tolist()]
                labelprobs = [round(p, 2) for p in probs[:, NO_SPEECH].detach().cpu().tolist()] if probs.shape[-1] > NO_SPEECH else []
                remaining_time = (time.time() - tic) / max(1, global_step - initial_step + 1) * (args.max_train_steps - global_step)
                remaining_time_hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                average_loss = running_loss / max(1, args.logging_steps)
                logger.info(
                    "At step %s loss is %.3f. First token accuracy is %s/%s. Probabilities of the first token are %s, <|nospeech|> probabilities are %s. Remaining time is %s.",
                    global_step,
                    average_loss,
                    acc,
                    len(labels),
                    probs_,
                    labelprobs,
                    remaining_time_hh_mm_ss,
                )
                log_metric(args.exp, {"train/running_loss": average_loss, "lr": lr_scheduler.get_last_lr()[0]}, global_step)
                running_loss = 0.0

            global_step += 1

    checkpoint_dir = os.path.join(args.exp, f"step_{global_step}")
    save_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, scaler, global_step, args)

    eval_metrics = evaluation_loop(model, eval_dataloader, tokenizer, device, args.mixed_precision, os.path.join(args.exp, "results.json"))
    logger.info("Step %s eval metrics: %s", global_step, eval_metrics)
    log_metric(args.exp, eval_metrics, global_step)

    model.save_pretrained(args.exp)
    tokenizer.save_pretrained(args.exp)


if __name__ == "__main__":
    main()
