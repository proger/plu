import argparse
import concurrent.futures
import json
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from plu.train_data import Corpus, register_data_args
from plu.tokenizer import WhisperTokenizer
from plu.whisper import WhisperForConditionalGeneration, resolve_model_path


logger = logging.getLogger(__name__)
TRAIN_DTYPE = torch.bfloat16
LOSS_SKIP_THRESHOLD = 1.0
LOGGING_STEPS = 100
STATIC_INPUT_FEATURES = 3000
STATIC_DECODER_LEN = 448
ENCODER_BACKWARD_LAYERS = 24
TRAINING_STATE_FILENAME = "training_state.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper")
    parser.add_argument(
        "--init",
        type=str,
        help="Model id/path for a fresh run, or a training checkpoint directory/file to resume from.",
        default="openai/whisper-large-v3-turbo",
    )
    register_data_args(parser)
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        choices=["linear", "constant"],
        help="The scheduler type to use.",
    )
    parser.add_argument("--exp", type=str, default="exp/1", help="Where to store checkpoints and model files.")
    parser.add_argument("--cuda_graph_warmup_steps", type=int, default=3, help="Warmup forward/backward passes before CUDA graph capture.")

    args = parser.parse_args()
    assert args.train is not None, "need a training dataset, use --train file.jsonl"
    return args


def checkpoint_state_path(init: str | Path) -> Path | None:
    path = Path(init).expanduser()
    if path.is_file() and path.name == TRAINING_STATE_FILENAME:
        return path
    if path.is_dir() and (path / TRAINING_STATE_FILENAME).exists():
        return path / TRAINING_STATE_FILENAME
    return None


def has_model_files(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(
        (path / filename).exists()
        for filename in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )


def model_source_from_init(init: str | Path) -> str | Path:
    state_path = checkpoint_state_path(init)
    if state_path is None:
        return resolve_model_path(init)

    for candidate in (state_path.parent, state_path.parent.parent):
        if has_model_files(candidate):
            return candidate

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_args = state.get("args", {}) if isinstance(state, dict) else {}
    saved_init = saved_args.get("init") or saved_args.get("model_name_or_path")
    if saved_init:
        return resolve_model_path(saved_init)
    raise FileNotFoundError(f"{state_path} is a training checkpoint, but no model files were found beside it or in its parent")


def freeze_encoder_prefix(model: WhisperForConditionalGeneration, backward_layers: int) -> int:
    encoder = model.model.encoder
    frozen_layers = max(0, len(encoder.layers) - backward_layers)
    if frozen_layers == 0:
        return 0
    for module in (encoder.conv1, encoder.conv2):
        for parameter in module.parameters():
            parameter.requires_grad = False
    for layer in encoder.layers[:frozen_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return frozen_layers


def trainable_parameter_summary(model: torch.nn.Module) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.2f}"


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
    return float(batch["durations"].sum().item())


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def real_time_metrics(audio_seconds: float, gpu_seconds: float, wall_seconds: float, prefix: str = "") -> dict[str, float]:
    return {
        f"train/{prefix}x_real_time_gpu": ratio(audio_seconds, gpu_seconds),
        f"train/{prefix}real_time_factor_gpu": ratio(gpu_seconds, audio_seconds),
        f"train/{prefix}x_real_time_wall": ratio(audio_seconds, wall_seconds),
        f"train/{prefix}real_time_factor_wall": ratio(wall_seconds, audio_seconds),
    }


def get_lr_multiplier(name: str, step: int, total_steps: int) -> float:
    if name == "linear":
        progress = step / max(1, total_steps)
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, 1.0 - progress)
    return 1.0


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr_multiplier(args.lr_scheduler_type, step, total_steps),
    )


def make_optimizer(parameters, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(list(parameters), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
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
            "step": step,
            "args": vars(args),
        },
        path / "training_state.pt",
    )


def load_checkpoint(
    init: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> int:
    state_path = checkpoint_state_path(init)
    if state_path is None:
        raise ValueError(f"{init} is not a training checkpoint")
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state.get("step", 0))


def log_metric(exp: str, payload: dict[str, Any], step: int) -> None:
    Path(exp).mkdir(parents=True, exist_ok=True)
    row = {"step": step, **payload}
    with (Path(exp) / "train_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TrainingBatchPrefetcher:
    def __init__(self, iterator):
        self.iterator = iterator
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None

    def _submit(self) -> None:
        if self.future is None:
            self.future = self.executor.submit(next, self.iterator)

    def next(self, *, prefetch_next: bool = True):
        self._submit()
        batch = self.future.result()
        self.future = None
        if prefetch_next:
            self._submit()
        return batch

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z", level=logging.INFO)
    logger.info("Invocation command: %s", shlex.join([sys.executable, *sys.argv]))

    exp_path = Path(args.exp)
    exp_path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("+train requires CUDA")
    device = torch.device("cuda")

    init_state_path = checkpoint_state_path(args.init)
    model_dir = model_source_from_init(args.init)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_dir, pad_token_id=model.config.pad_token_id)
    backward_layers = min(ENCODER_BACKWARD_LAYERS, len(model.model.encoder.layers))
    model.to(device=device, dtype=TRAIN_DTYPE)

    frozen_encoder_layers = freeze_encoder_prefix(model, backward_layers)
    from plu.benchmarks.bench_end_to_end import pack_mx_model, packed_mx_forward

    frozen_encoder_linears, mx_stats = pack_mx_model(model, "mxfp8")
    logger.info(
        "Packed %s model linear layers as MXFP8; frozen encoder prefix layers also use the packed path.",
        len(frozen_encoder_linears),
    )

    corpus = Corpus(args, tokenizer=tokenizer, n_mels=model.config.num_mel_bins)
    train_dataset = corpus.load_dataset(args.train)
    total_train_steps = len(train_dataset)
    if total_train_steps == 0:
        raise ValueError("train dataset is empty")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = make_optimizer(trainable_parameters, args)
    lr_scheduler = make_scheduler(optimizer, args, total_train_steps)

    initial_step = 0
    if init_state_path is not None:
        initial_step = load_checkpoint(init_state_path, model, optimizer, lr_scheduler, device)
    if initial_step > total_train_steps:
        raise ValueError(f"checkpoint step {initial_step} exceeds train dataset length {total_train_steps}")

    remaining_dataset = train_dataset if initial_step == 0 else Subset(train_dataset, range(initial_step, total_train_steps))
    train_dataloader = corpus.make_train_dataloader(remaining_dataset)
    remaining_steps = len(train_dataloader)
    logger.info("Train batch size = 1")
    logger.info("Total optimization steps = %s", total_train_steps)
    logger.info("Initial step = %s", initial_step)
    logger.info("Remaining optimization steps = %s", remaining_steps)
    logger.info("Optimizer = fused_adamw")
    logger.info(trainable_parameter_summary(model))
    logger.info(
        "Encoder layers = %s; backward layers = %s; frozen prefix layers = %s.",
        len(model.model.encoder.layers),
        backward_layers,
        frozen_encoder_layers,
    )
    if remaining_steps == 0:
        logger.info("No remaining optimization steps; writing model files to %s.", args.exp)
        model.save_pretrained(args.exp)
        tokenizer.save_pretrained(args.exp)
        return

    model.train()
    running_loss = 0.0
    tic = time.time()
    global_step = initial_step
    from plu.triton.linear.backward import collect_linear_grad_norms, summarize_linear_grad_norms

    static_features = torch.zeros(
        1,
        model.config.num_mel_bins,
        STATIC_INPUT_FEATURES,
        device=device,
        dtype=TRAIN_DTYPE,
    )
    static_labels = torch.full(
        (1, STATIC_DECODER_LEN),
        -100,
        device=device,
        dtype=torch.long,
    )

    train_iterator = iter(train_dataloader)
    capture_batch = next(train_iterator)
    copy_static_batch(capture_batch, static_features, static_labels)

    def graph_step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        outputs = packed_mx_forward(
            model,
            frozen_encoder_linears,
            static_features,
            static_labels,
            master_weight_grads=True,
            encoder_backward_layers=backward_layers,
            fused_unembedding_ce=False,
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
    with collect_linear_grad_norms() as graph_linear_grad_norm_records:
        with torch.cuda.graph(graph):
            static_loss = graph_step()
    torch.cuda.synchronize()
    cuda_graph_capture_ms = (time.perf_counter() - capture_start) * 1000.0
    logger.info("CUDA graph capture took %.3f ms.", cuda_graph_capture_ms)

    train_iterator = iter(train_dataloader)
    prefetcher = TrainingBatchPrefetcher(train_iterator)
    total_audio_seconds = 0.0
    total_gpu_ms = 0.0
    total_wall_seconds = 0.0
    window_audio_seconds = 0.0
    window_gpu_ms = 0.0
    window_wall_seconds = 0.0
    window_steps = 0
    window_update_count = 0
    window_skipped_updates = 0

    try:
        while global_step < total_train_steps:
            step_wall_start = time.perf_counter()
            batch = prefetcher.next(prefetch_next=global_step + 1 < total_train_steps)
            audio_seconds = copy_static_batch(batch, static_features, static_labels)
            optimizer.zero_grad(set_to_none=False)
            should_log = global_step % LOGGING_STEPS == 0

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()

            torch.cuda.synchronize()
            gpu_ms = start_event.elapsed_time(end_event)
            step_loss = float(static_loss.detach().cpu())
            grad_norm_metrics = summarize_linear_grad_norms(graph_linear_grad_norm_records)
            update_skipped = step_loss > LOSS_SKIP_THRESHOLD
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if update_skipped:
                optimizer.zero_grad(set_to_none=False)
            else:
                optimizer.step()
                lr_scheduler.step()
                torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - step_wall_start

            total_audio_seconds += audio_seconds
            total_gpu_ms += gpu_ms
            total_wall_seconds += wall_seconds
            window_audio_seconds += audio_seconds
            window_gpu_ms += gpu_ms
            window_wall_seconds += wall_seconds
            window_steps += 1
            if update_skipped:
                window_skipped_updates += 1
            else:
                running_loss += step_loss
                window_update_count += 1
            step_gpu_seconds = gpu_ms / 1000.0
            step_metrics = {
                "train/loss": step_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "train/update_skipped": update_skipped,
                "train/loss_skip_threshold": LOSS_SKIP_THRESHOLD,
                "train/step_audio_seconds": audio_seconds,
                "train/step_gpu_ms": gpu_ms,
                "train/step_wall_ms": 1000.0 * wall_seconds,
                "train/grad_norm/pre_clip_total": float(pre_clip_grad_norm.detach().cpu()),
                "train/grad_norm/clip_max": 1.0,
                "train/text_labels": batch.get("texts") or [],
            }
            step_metrics.update(real_time_metrics(audio_seconds, step_gpu_seconds, wall_seconds, prefix="step_"))
            step_metrics.update(grad_norm_metrics)

            if should_log:
                remaining_time = (time.time() - tic) / max(1, global_step - initial_step + 1) * (total_train_steps - global_step)
                remaining_time_hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                average_loss = running_loss / max(1, window_update_count)
                gpu_seconds = window_gpu_ms / 1000.0
                metrics = {
                    "train/running_loss": average_loss,
                    "train/update_count": float(window_update_count),
                    "train/skipped_updates": float(window_skipped_updates),
                    "lr": step_metrics["lr"],
                    "train/audio_seconds": window_audio_seconds,
                    "train/mean_gpu_ms": window_gpu_ms / max(1, window_steps),
                    "train/mean_wall_ms": 1000.0 * window_wall_seconds / max(1, window_steps),
                }
                metrics.update(real_time_metrics(window_audio_seconds, gpu_seconds, window_wall_seconds))
                metrics.update(grad_norm_metrics)
                step_metrics.update(metrics)
                logger.info(
                    "At step %s loss is %.3f over %s updates; skipped %s. Linear grad norm %.3f. GPU RTF %.5f, wall RTF %.5f. Remaining time is %s.",
                    global_step,
                    average_loss,
                    window_update_count,
                    window_skipped_updates,
                    metrics.get("train/grad_norm/linear_total", 0.0),
                    metrics["train/real_time_factor_gpu"],
                    metrics["train/real_time_factor_wall"],
                    remaining_time_hh_mm_ss,
                )
                running_loss = 0.0
                window_audio_seconds = 0.0
                window_gpu_ms = 0.0
                window_wall_seconds = 0.0
                window_steps = 0
                window_update_count = 0
                window_skipped_updates = 0
            log_metric(args.exp, step_metrics, global_step)

            global_step += 1
    finally:
        prefetcher.close()

    total_gpu_seconds = total_gpu_ms / 1000.0
    throughput_summary = {
        "train/total_audio_seconds": total_audio_seconds,
        "train/total_gpu_seconds": total_gpu_seconds,
        "train/total_wall_seconds": total_wall_seconds,
        "train/cuda_graph_capture_ms": cuda_graph_capture_ms,
        "train/mx_packed_linear_count": float(mx_stats.get("mx_packed_linear_count", 0)),
    }
    throughput_summary.update(real_time_metrics(total_audio_seconds, total_gpu_seconds, total_wall_seconds))
    logger.info("Training throughput summary: %s", throughput_summary)
    log_metric(args.exp, throughput_summary, global_step)
    checkpoint_dir = Path(args.exp) / f"step_{global_step}"
    save_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, global_step, args)

    model.save_pretrained(args.exp)
    tokenizer.save_pretrained(args.exp)


if __name__ == "__main__":
    main()
