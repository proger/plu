import argparse
import concurrent.futures
import json
import logging
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Subset

from plu.train_data import Corpus, register_data_args
from plu.tokenizer import WhisperTokenizer
from plu.whisper import WhisperForConditionalGeneration, resolve_model_path


logger = logging.getLogger(__name__)
TRAIN_DTYPE = torch.bfloat16
LOSS_SKIP_THRESHOLD = 1.0
LOGGING_STEPS = 100
ROWS_PER_UPDATE = 1
STATIC_INPUT_FEATURES = 3000
DECODER_TOKEN_BUCKETS = (64, 96, 128, 160, 192, 224, 256, 320, 448)
ENCODER_BACKWARD_LAYERS = 24
TRAINING_STATE_FILENAME = "training_state.pt"


@dataclass
class ClipGradNorms:
    total: Tensor
    linear: Tensor
    non_linear: Tensor


@dataclass
class OptimizerGraph:
    graph: torch.cuda.CUDAGraph
    lr: Tensor
    grad_norms: ClipGradNorms
    capture_ms: float


@dataclass
class TrainingGraph:
    decoder_tokens: int
    graph: torch.cuda.CUDAGraph
    static_features: Tensor
    static_labels: Tensor
    static_loss: Tensor
    grad_norm_records: list[Any]
    grad_norm_metadata: dict[str, float]
    grad_buffers: list[Tensor | None]
    optimizer_graph: OptimizerGraph | None
    capture_ms: float


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper")
    parser.add_argument(
        "--init",
        type=str,
        help="Model id/path for a fresh run, or a training checkpoint directory/file to resume from.",
        default="openai/whisper-large-v3-turbo",
    )
    register_data_args(parser)
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--beta1", "--adam_beta1", dest="beta1", type=float, default=0, help="AdamW beta1.")
    parser.add_argument("--beta2", "--adam_beta2", dest="beta2", type=float, default=0.9999, help="AdamW beta2.")
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


def decoder_token_bucket(label_tokens: int) -> int:
    for bucket in DECODER_TOKEN_BUCKETS:
        if label_tokens <= bucket:
            return bucket
    raise ValueError(f"label length {label_tokens} exceeds largest decoder bucket {DECODER_TOKEN_BUCKETS[-1]}")


def dataset_examples(dataset: Any) -> list[dict[str, Any]] | None:
    if hasattr(dataset, "examples"):
        return dataset.examples
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "examples"):
        return [dataset.dataset.examples[index] for index in dataset.indices]
    return None


def example_label_tokens(example: dict[str, Any], sot_token_id: int) -> int:
    labels = example.get("input_ids") or example.get("labels") or []
    tokens = len(labels)
    if tokens > 0 and int(labels[0]) == sot_token_id:
        tokens -= 1
    return tokens


def active_decoder_buckets(dataset: Any, sot_token_id: int) -> tuple[int, ...]:
    examples = dataset_examples(dataset)
    if examples is None:
        return DECODER_TOKEN_BUCKETS
    return tuple(sorted({decoder_token_bucket(example_label_tokens(example, sot_token_id)) for example in examples}))


def copy_static_batch(
    batch: dict[str, Any],
    static_features: torch.Tensor,
    static_labels: torch.Tensor,
    *,
    allow_label_truncation: bool = False,
) -> float:
    features = batch["input_features"]
    labels = batch["labels"]
    if features.shape[0] > static_features.shape[0]:
        raise ValueError(f"CUDA graph batch size exceeded: {features.shape[0]} > {static_features.shape[0]}")
    if features.shape[1] != static_features.shape[1]:
        raise ValueError(f"feature mel bins changed: {features.shape[1]} != {static_features.shape[1]}")
    if labels.shape[-1] > static_labels.shape[-1] and not allow_label_truncation:
        raise ValueError(f"label length {labels.shape[-1]} exceeds CUDA graph decoder bucket {static_labels.shape[-1]}")

    feature_frames = min(features.shape[-1], static_features.shape[-1])
    label_tokens = min(labels.shape[-1], static_labels.shape[-1])
    batch_size = features.shape[0]
    if batch_size < static_features.shape[0] or feature_frames < static_features.shape[-1]:
        static_features.zero_()
    static_labels.fill_(-100)
    static_features[:batch_size, :, :feature_frames].copy_(features[:, :, :feature_frames], non_blocking=True)
    static_labels[:batch_size, :label_tokens].copy_(labels[:, :label_tokens], non_blocking=True)
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


def make_optimizer(parameters, args: argparse.Namespace, device: torch.device) -> tuple[torch.optim.Optimizer, Tensor]:
    lr = torch.tensor(args.learning_rate, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        list(parameters),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=True,
        capturable=True,
    )
    return optimizer, lr


def set_optimizer_lr(lr: Tensor, args: argparse.Namespace, step: int, total_steps: int) -> None:
    lr.fill_(args.learning_rate * get_lr_multiplier(args.lr_scheduler_type, step, total_steps))


def attach_grad_buffers(parameters: list[torch.nn.Parameter], grad_buffers: list[Tensor | None]) -> None:
    for parameter, grad in zip(parameters, grad_buffers, strict=True):
        parameter.grad = grad


def linear_parameter_ids(model: torch.nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if module.__class__.__name__ != "CastLinear":
            continue
        ids.add(id(module.weight))
    return ids


def linear_grad_norm_metadata(records: list[Any]) -> dict[str, float]:
    return {
        "train/grad_norm/linear_calls": float(len(records)),
        "train/grad_norm/linear_weight_elements": float(sum(record.weight_elements for record in records)),
        "train/grad_norm/linear_bias_elements": float(sum(record.bias_elements for record in records)),
    }


def linear_grad_norm_sq(records: list[Any], device: torch.device) -> Tensor:
    if not records:
        return torch.zeros((), device=device, dtype=torch.float32)
    parts = [record.weight_norm_sq.reshape(()) for record in records]
    parts.extend(record.bias_norm_sq.reshape(()) for record in records if record.bias_norm_sq is not None)
    return torch.stack(parts).sum()


def parameter_grad_norm_sq(parameters: list[torch.nn.Parameter], device: torch.device) -> Tensor:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.zeros((), device=device, dtype=torch.float32)
    grad_norms = torch._foreach_norm(grads, 2.0)
    return torch.stack([norm.float() * norm.float() for norm in grad_norms]).sum()


def update_clip_grad_scale(
    *,
    scale: Tensor,
    linear_records: list[Any],
    non_linear_parameters: list[torch.nn.Parameter],
    device: torch.device,
) -> ClipGradNorms:
    linear_norm_sq = linear_grad_norm_sq(linear_records, device)
    non_linear_norm_sq = parameter_grad_norm_sq(non_linear_parameters, device)
    total_norm_sq = linear_norm_sq + non_linear_norm_sq
    total_norm = total_norm_sq.sqrt()
    scale.copy_(total_norm.clamp_min(1.0))
    return ClipGradNorms(total=total_norm, linear=linear_norm_sq.sqrt(), non_linear=non_linear_norm_sq.sqrt())


def capture_training_graph(
    *,
    model: WhisperForConditionalGeneration,
    packed_mx_forward: Any,
    frozen_encoder_linears: dict[int, Any],
    trainable_parameters: list[torch.nn.Parameter],
    capture_batch: dict[str, Any],
    decoder_tokens: int,
    backward_layers: int,
    warmup_steps: int,
    device: torch.device,
) -> TrainingGraph:
    static_features = torch.zeros(
        capture_batch["input_features"].shape[0],
        model.config.num_mel_bins,
        STATIC_INPUT_FEATURES,
        device=device,
        dtype=TRAIN_DTYPE,
    )
    static_labels = torch.full(
        (capture_batch["input_features"].shape[0], decoder_tokens),
        -100,
        device=device,
        dtype=torch.long,
    )
    copy_static_batch(capture_batch, static_features, static_labels, allow_label_truncation=True)

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

    for _ in range(warmup_steps):
        graph_step()
    torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter()
    from plu.triton.linear.backward import collect_linear_grad_norms

    with collect_linear_grad_norms() as grad_norm_records:
        with torch.cuda.graph(graph):
            static_loss = graph_step()
    torch.cuda.synchronize()
    return TrainingGraph(
        decoder_tokens=decoder_tokens,
        graph=graph,
        static_features=static_features,
        static_labels=static_labels,
        static_loss=static_loss,
        grad_norm_records=grad_norm_records,
        grad_norm_metadata=linear_grad_norm_metadata(grad_norm_records),
        grad_buffers=[parameter.grad for parameter in trainable_parameters],
        optimizer_graph=None,
        capture_ms=(time.perf_counter() - capture_start) * 1000.0,
    )


def initialize_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            if "step" not in state:
                state["step"] = torch.zeros((), device=parameter.device, dtype=torch.float32)
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)


def capture_optimizer_graph(
    optimizer: torch.optim.Optimizer,
    lr: Tensor,
    *,
    linear_records: list[Any],
    non_linear_parameters: list[torch.nn.Parameter],
    device: torch.device,
    restore_state: bool = False,
) -> OptimizerGraph:
    optimizer.zero_grad(set_to_none=False)
    previous_lr = lr.detach().clone()
    previous_grad_scale = optimizer.grad_scale.detach().clone()
    state_backup: list[tuple[dict[str, Any], dict[str, Tensor]]] = []
    if restore_state:
        for state in optimizer.state.values():
            tensors = {
                key: value.detach().clone()
                for key, value in state.items()
                if key in {"step", "exp_avg", "exp_avg_sq"} and torch.is_tensor(value)
            }
            state_backup.append((state, tensors))
    lr.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter()
    with torch.cuda.graph(graph):
        grad_norms = update_clip_grad_scale(
            scale=optimizer.grad_scale,
            linear_records=linear_records,
            non_linear_parameters=non_linear_parameters,
            device=device,
        )
        optimizer.step()
    torch.cuda.synchronize()
    if restore_state:
        for state, tensors in state_backup:
            for key, value in tensors.items():
                state[key].copy_(value)
    else:
        for state in optimizer.state.values():
            step = state.get("step")
            if torch.is_tensor(step):
                step.zero_()
    lr.copy_(previous_lr)
    optimizer.grad_scale.copy_(previous_grad_scale)
    return OptimizerGraph(graph=graph, lr=lr, grad_norms=grad_norms, capture_ms=(time.perf_counter() - capture_start) * 1000.0)


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path / "training_state.pt",
    )


def load_checkpoint(
    init: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    state_path = checkpoint_state_path(init)
    if state_path is None:
        raise ValueError(f"{init} is not a training checkpoint")
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
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
    if len(train_dataset) == 0:
        raise ValueError("train dataset is empty")
    total_train_steps = (len(train_dataset) + ROWS_PER_UPDATE - 1) // ROWS_PER_UPDATE

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    linear_ids = linear_parameter_ids(model)
    non_linear_parameters = [parameter for parameter in trainable_parameters if id(parameter) not in linear_ids]
    optimizer, optimizer_lr = make_optimizer(trainable_parameters, args, device)
    optimizer.grad_scale = torch.ones((), device=device, dtype=torch.float32)

    initial_step = 0
    if init_state_path is not None:
        initial_step = load_checkpoint(init_state_path, model, optimizer, device)
    if initial_step > total_train_steps:
        raise ValueError(f"checkpoint step {initial_step} exceeds train dataset length {total_train_steps}")

    initial_row = min(len(train_dataset), initial_step * ROWS_PER_UPDATE)
    remaining_dataset = train_dataset if initial_row == 0 else Subset(train_dataset, range(initial_row, len(train_dataset)))
    train_dataloader = corpus.make_train_dataloader(remaining_dataset)
    remaining_steps = len(train_dataloader)
    decoder_buckets = active_decoder_buckets(remaining_dataset, tokenizer.sot)
    logger.info("Train rows per update = %s", ROWS_PER_UPDATE)
    logger.info("Initial row = %s", initial_row)
    logger.info("Total optimization steps = %s", total_train_steps)
    logger.info("Initial step = %s", initial_step)
    logger.info("Remaining optimization steps = %s", remaining_steps)
    logger.info("Decoder token buckets = %s", decoder_buckets)
    logger.info("Optimizer = fused_adamw; betas = (%s, %s); weight_decay = %s", args.beta1, args.beta2, args.weight_decay)
    logger.info(trainable_parameter_summary(model))
    logger.info(
        "Gradient clipping uses fused AdamW grad_scale from %s linear parameters and %s non-linear parameters.",
        len(trainable_parameters) - len(non_linear_parameters),
        len(non_linear_parameters),
    )
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

    train_iterator = iter(train_dataloader)
    capture_batch = next(train_iterator)
    training_graphs: dict[int, TrainingGraph] = {}
    for bucket in decoder_buckets:
        training_graph = capture_training_graph(
            model=model,
            packed_mx_forward=packed_mx_forward,
            frozen_encoder_linears=frozen_encoder_linears,
            trainable_parameters=trainable_parameters,
            capture_batch=capture_batch,
            decoder_tokens=bucket,
            backward_layers=backward_layers,
            warmup_steps=args.cuda_graph_warmup_steps,
            device=device,
        )
        training_graphs[bucket] = training_graph
        logger.info("CUDA graph capture for decoder bucket %s took %.3f ms.", bucket, training_graph.capture_ms)
    optimizer_capture_ms = 0.0
    initialize_optimizer_state(optimizer)
    for bucket, training_graph in training_graphs.items():
        attach_grad_buffers(trainable_parameters, training_graph.grad_buffers)
        optimizer_graph = capture_optimizer_graph(
            optimizer,
            optimizer_lr,
            linear_records=training_graph.grad_norm_records,
            non_linear_parameters=non_linear_parameters,
            device=device,
            restore_state=initial_step > 0,
        )
        training_graph.optimizer_graph = optimizer_graph
        optimizer_capture_ms += optimizer_graph.capture_ms
        logger.info("CUDA graph capture for optimizer bucket %s took %.3f ms.", bucket, optimizer_graph.capture_ms)
    cuda_graph_capture_ms = sum(training_graph.capture_ms for training_graph in training_graphs.values()) + optimizer_capture_ms

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
            label_tokens = int(batch["labels"].shape[-1])
            decoder_bucket = decoder_token_bucket(label_tokens)
            training_graph = training_graphs[decoder_bucket]
            audio_seconds = copy_static_batch(batch, training_graph.static_features, training_graph.static_labels)
            should_log = global_step % LOGGING_STEPS == 0

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            training_graph.graph.replay()
            end_event.record()

            torch.cuda.synchronize()
            gpu_ms = start_event.elapsed_time(end_event)
            step_loss = float(training_graph.static_loss.detach().cpu())
            attach_grad_buffers(trainable_parameters, training_graph.grad_buffers)
            update_skipped = step_loss > LOSS_SKIP_THRESHOLD
            if update_skipped:
                clip_grad_norms = update_clip_grad_scale(
                    scale=optimizer.grad_scale,
                    linear_records=training_graph.grad_norm_records,
                    non_linear_parameters=non_linear_parameters,
                    device=device,
                )
                torch.cuda.synchronize()
            else:
                if training_graph.optimizer_graph is None:
                    raise RuntimeError(f"optimizer graph for decoder bucket {decoder_bucket} was not captured")
                set_optimizer_lr(optimizer_lr, args, global_step, total_train_steps)
                training_graph.optimizer_graph.graph.replay()
                torch.cuda.synchronize()
                clip_grad_norms = training_graph.optimizer_graph.grad_norms
            wall_seconds = time.perf_counter() - step_wall_start
            grad_norm_metrics = {
                **training_graph.grad_norm_metadata,
                "train/grad_norm/linear_total": float(clip_grad_norms.linear.detach().cpu()),
                "train/grad_norm/non_linear_total": float(clip_grad_norms.non_linear.detach().cpu()),
            }

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
                "lr": float(optimizer_lr.detach().cpu()),
                "train/update_skipped": update_skipped,
                "train/loss_skip_threshold": LOSS_SKIP_THRESHOLD,
                "train/label_tokens": float(label_tokens),
                "train/decoder_bucket_tokens": float(decoder_bucket),
                "train/step_audio_seconds": audio_seconds,
                "train/step_gpu_ms": gpu_ms,
                "train/step_wall_ms": 1000.0 * wall_seconds,
                "train/grad_norm/pre_clip_total": float(clip_grad_norms.total.detach().cpu()),
                "train/grad_norm/clip_max": 1.0,
                "train/grad_norm/optimizer_grad_scale": float(optimizer.grad_scale.detach().cpu()),
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
                    "At step %s loss is %.3f over %s updates; skipped %s. Grad norm %.3f. GPU RTF %.5f, wall RTF %.5f. Remaining time is %s.",
                    global_step,
                    average_loss,
                    window_update_count,
                    window_skipped_updates,
                    step_metrics["train/grad_norm/pre_clip_total"],
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
    save_checkpoint(checkpoint_dir, model, optimizer, global_step, args)

    model.save_pretrained(args.exp)
    tokenizer.save_pretrained(args.exp)


if __name__ == "__main__":
    main()
