import argparse
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

import numpy as np
import torch

from plu.lora import LoraConfig, apply_lora, print_trainable_parameters, save_lora_adapters
from plu.train_data import Corpus, register_data_args
from plu.wer import word_error_rate
from plu.tokenizer import NO_SPEECH, WhisperTokenizer
from plu.whisper import WhisperForConditionalGeneration, resolve_model_path


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
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="fp16", help="Autocast precision.")
    parser.add_argument("--device", type=str, default=None, help="Training device. Defaults to cuda when available, otherwise cpu.")

    parser.add_argument("--use_peft", type=str_to_bool, default=True, help="Whether to use LoRA adapters.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    args = parser.parse_args()
    assert args.train is not None, "need a training dataset, use --train file.jsonl"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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
            "numpy_rng_state": np.random.get_state(),
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
    np.random.set_state(state["numpy_rng_state"])
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
        with autocast_context(device, mixed_precision):
            generated_tokens = model.generate(batch["input_features"], max_new_tokens=255).cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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
    model.to(device)

    corpus = Corpus(args, tokenizer=tokenizer, n_mels=model.config.num_mel_bins)
    train_dataloader = corpus.make_train_dataloader(corpus.load_dataset(args.train))
    eval_dataloader = corpus.make_eval_dataloader(corpus.load_dataset(args.eval))

    if args.eval_at_init:
        evaluation_loop(model, eval_dataloader, tokenizer, device, args.mixed_precision, os.path.join(args.exp, "init_results.json"))

    lora_config = None
    if args.use_peft:
        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=("q_proj", "v_proj"),
            lora_dropout=args.lora_dropout,
        )
        model = apply_lora(model, lora_config)
        model.to(device)
        logger.info(print_trainable_parameters(model))

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
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

    model.train()
    running_loss = 0.0
    tic = time.time()
    global_step = initial_step

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

    if lora_config is not None:
        save_lora_adapters(model, args.exp, lora_config, base_model_name_or_path=args.model_name_or_path)
        with (Path(args.exp) / "config.json").open("w", encoding="utf-8") as f:
            json.dump(model.config.to_dict(), f, indent=2)
    else:
        model.save_pretrained(args.exp)
    tokenizer.save_pretrained(args.exp)


if __name__ == "__main__":
    main()
