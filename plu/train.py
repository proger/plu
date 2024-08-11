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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
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


@torch.no_grad()
def evaluation_loop(model, eval_dataloader, processor, metric, output_filename):
    model.eval()
    predictions = []
    references = []
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
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

    with open(output_filename, "w") as f:
        json.dump({"metrics": eval_metrics,
                   "hyp": predictions,
                   "ref": references}, f, ensure_ascii=False)

    return eval_metrics


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
    accelerator_kwargs["log_with"] = args.report_to
    accelerator_kwargs["project_dir"] = args.exp
    accelerator = Accelerator(mixed_precision='fp16', **accelerator_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.INFO,
    )
    # logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_error()
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

    # from openai-whisper
    # ipdb> self.model.dims
    # ModelDimensions(n_mels=128, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_vocab=51866, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.decoder_start_token_id = 50258 # <|startoftranscript|>

    eval_dataloader = accelerator.prepare(eval_dataloader)

    if args.eval_at_init:
        model = accelerator.prepare(model)
        model = accelerator.prepare(model)
        evaluation_loop(model, eval_dataloader, corpus.processor, metric, os.path.join(args.exp, "init_results.json"))

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

    #accelerator.print(model)

    # Note here that the max steps is adjusted by the accelerator's num_processes
    args.max_train_steps = math.ceil(args.max_train_steps / accelerator.num_processes)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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
    logger.info(f"Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        global_step = int(training_difference.replace("step_", ""))

    # We need to adjust the progress bar to the current step
    model.train()
    total_loss = 0
    running_loss = 0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            global_step += 1
            progress_bar.update(1)

        step_loss = accelerator.reduce(loss.detach().clone()).item()
        total_loss += step_loss
        running_loss += step_loss

        if global_step % args.logging_steps == 0:
            accelerator.log({"train/running_loss": running_loss / args.logging_steps}, step=global_step)
            labels = batch['labels'][:,0]
            tokens = outputs.logits[:,0].argmax(-1)
            probs = outputs.logits[:,0].softmax(dim=-1)
            probs_ = [round(p, 2) for p in probs.max(-1).values.detach().tolist()]
            acc = (tokens == labels).sum()
            labels = labels.tolist()
            tokens = tokens.tolist()
            no_speech = 50363
            labelprobs = [round(p, 2) for p in probs[:,no_speech].detach().tolist()]
            logger.info(f"running loss: {running_loss / args.logging_steps} vad accuracy: {acc} first tokens: {tokens} probs: {probs_} labels: {labels} labelprobs: {labelprobs}")
            running_loss = 0

        if global_step >= args.max_train_steps:
            break

    exp = os.path.join(args.exp, f"step_{global_step}")
    accelerator.save_state(exp)

    eval_metrics = evaluation_loop(model, eval_dataloader, corpus.processor, metric, os.path.join(args.exp, "results.json"))
    logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
    accelerator.log(eval_metrics, step=global_step)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.exp, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        corpus.processor.tokenizer.save_pretrained(args.exp)

if __name__ == "__main__":
    main()
