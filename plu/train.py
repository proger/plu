# seeded from https://github.com/huggingface/peft/tree/main/examples
import argparse
import gc
import json
import logging
import math
import os
import sys

import datasets

import evaluate
import numpy as np
import torch
import transformers

from accelerate import Accelerator, dispatch_model
from accelerate.logging import get_logger

from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    SchedulerType,
    WhisperForConditionalGeneration,
    get_scheduler,
    set_seed,
)
from peft import LoraConfig, PeftModel, get_peft_model

from plu.dataloader import Corpus, register_data_args

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="openai/whisper-large-v3",
    )
    register_data_args(parser)
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5000,
        help="Number of samples to prefetch in the streaming mode.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=25, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--exp", type=str, default="exp/1", help="Where to store the checkpoints and the model files.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--with_tracking",
        type=bool,
        default=True,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `exp`."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Whether the various states should be saved at the end of every n steps, or 0 for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=0,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--eval_at_init", action="store_true", help="Evaluate the model at initialization")

    # lora specific args
    parser.add_argument(
        "--use_peft",
        type=bool,
        default=True,
        help="Whether to use PEFT",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LORA alpha",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="LORA rank",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LORA dropout",
    )

    args = parser.parse_args()

    assert args.train is not None, "need a training dataset, use --train file.jsonl"
    return args



def save_model_hook(models, weights, exp):
    for model in models:
        model.save_pretrained(exp)
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        PeftModel.from_pretrained(model.base_model.model, input_dir)


def evaluation_loop(model, eval_dataloader, processor, metric):
    model.eval()
    predictions = []
    references = []
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"],
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
            del generated_tokens, labels, batch
        gc.collect()
    try:
        wer = 100 * metric.compute(predictions=predictions, references=references)
    except:
        wer = 112
    eval_metrics = {"eval/wer": wer}

    for i, (hyp, ref) in enumerate(zip(predictions, references)):
        print(f'{i} ref', ref, sep='\t')
        print(f'{i} hyp', hyp, sep='\t')

    return eval_metrics, {"hyp": predictions, "ref": references}


def load_peft(model, args):
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    # as Whisper model uses Conv layer in encoder, checkpointing disables grad computation
    # to avoid this, make the inputs trainable
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()

    accelerator_kwargs = {"gradient_accumulation_steps": args.gradient_accumulation_steps}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.exp
    accelerator = Accelerator(**accelerator_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.exp is not None:
            os.makedirs(args.exp, exist_ok=True)
    accelerator.wait_for_everyone()

    # load dataset
    corpus = Corpus(args)
    train_dataloader = corpus.make_train_dataloader(corpus.load_dataset(args.train))
    eval_dataloader = corpus.make_eval_dataloader(corpus.load_dataset(args.eval))

    # metric
    metric = evaluate.load("wer")

    # model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path, quantization_config=quant_config,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if len(set(model.hf_device_map.values()).intersection({"cpu", "disk"})) > 0:
        raise ValueError("Training on CPU or disk is not supported.")
    if len(set(model.hf_device_map.values())) > 1:
        device_map = model.hf_device_map.copy()
        # required because `labels` are on main execution device (0) while the output of `proj_out` is on other device.
        # So, this leads to device mismatch error when calculation cross-entropy between logits and labels.
        # Won't arise during inference as `labels` aren't supplied during that time
        # instead of changing device of one of the tied modules, I have to do this for all tied modules
        # else the execution device of remaining tied modules isn't changed
        device_map["model.decoder.embed_tokens"] = model._hf_hook.execution_device
        device_map["model.decoder.embed_positions"] = model._hf_hook.execution_device
        device_map["proj_out"] = model._hf_hook.execution_device
        dispatch_model(model, device_map=device_map)

    eval_dataloader = accelerator.prepare(eval_dataloader)

    if args.eval_at_init:
        model = accelerator.prepare(model)
        model = accelerator.prepare(model)
        init_eval_metrics, init_eval_outputs = evaluation_loop(
            model, eval_dataloader, corpus.processor, metric
        )
        with open(os.path.join(args.exp, "init_results.json"), "w") as f:
            json.dump({"eval_metrics": init_eval_metrics,
                       "eval_outputs": init_eval_outputs}, f)

    # preparing peft model
    if args.use_peft:
        model = load_peft(model, args)
        model = accelerator.prepare(model)
    else:
        model = accelerator.prepare(model)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything else with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    accelerator.print(model)

    # Note here that the max steps is adjusted by the accelerator's num_processes
    args.max_train_steps = math.ceil(args.max_train_steps / accelerator.num_processes)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            "logs", config={"argv": ' '.join(sys.argv)}, init_kwargs={}
        )

    # saving and loading checkpoints for resuming training
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    starting_epoch = 0
    best_metric = None
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        global_step = resume_step = int(training_difference.replace("step_", ""))
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)

    # We need to adjust the progress bar to the current step
    progress_bar.update(resume_step)
    model.train()
    if args.with_tracking:
        total_loss = 0
        running_loss = 0
    for step, batch in enumerate(accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            global_step += 1
            progress_bar.update(1)

        if args.with_tracking:
            step_loss = accelerator.reduce(loss.detach().clone()).item()
            total_loss += step_loss
            running_loss += step_loss

        do_checkpoint, do_eval = False, False
        if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
            do_checkpoint = True

        if global_step % args.logging_steps == 0:
            if args.with_tracking:
                accelerator.log({"train/running_loss": running_loss / args.logging_steps}, step=global_step)
                logger.info(f"running loss: {running_loss / args.logging_steps}")
                running_loss = 0

        if args.evaluation_steps and global_step % args.evaluation_steps == 0:
            do_eval = True

        do_break = False
        if global_step >= args.max_train_steps:
            do_checkpoint = True
            do_eval = True
            do_break = True

        if do_checkpoint:
            exp = os.path.join(args.exp, f"step_{global_step}")
            accelerator.save_state(exp)

        if do_eval:
            eval_metrics, eval_outputs = evaluation_loop(
                model, eval_dataloader, corpus.processor, metric
            )
            if args.with_tracking:
                logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
                accelerator.log(eval_metrics, step=global_step)
            if best_metric is None or eval_metrics["eval/wer"] < best_metric:
                best_metric = eval_metrics["eval/wer"]
                accelerator.save_state(os.path.join(args.exp, "best_checkpoint"))
            model.train()

        if do_break:
            break

    if not args.checkpointing_steps:
        exp = os.path.join(args.exp, f"step_{global_step}")
        accelerator.save_state(exp)

    if not args.evaluation_steps:
        eval_metrics, eval_outputs = evaluation_loop(
            model, eval_dataloader, corpus.processor, metric
        )
        if args.with_tracking:
            logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
            accelerator.log(eval_metrics, step=global_step)
        if best_metric is None or eval_metrics["eval/wer"] < best_metric:
            best_metric = eval_metrics["eval/wer"]
            accelerator.save_state(os.path.join(args.exp, "best_checkpoint"))

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.exp, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        corpus.processor.tokenizer.save_pretrained(args.exp)

    with open(os.path.join(args.exp, "results.json"), "w") as f:
        json.dump({"eval_metrics": eval_metrics,
                   "eval_outputs": eval_outputs}, f)

if __name__ == "__main__":
    main()
