import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import sys

from datasets import Audio, load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from whisper.tokenizer import get_tokenizer


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class Corpus:
    def __init__(self, args):
        self.args = args
        self.language = args.language
        self.task = args.task
        self.processor = WhisperProcessor.from_pretrained(
            getattr(args, "model_name_or_path", "openai/whisper-large-v3"),
            language=self.language,
            task=self.task
        )
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

    def __call__(self, example):
        # load and (possibly) resample audio data to 16kHz
        audio = example["audio"]

        # compute log-Mel input features from input audio array
        example["input_features"] = self.processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # compute input length of audio sample in seconds
        example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # read pretokenized label ids
        example["labels"] = example["input_ids"]
        return example

    def get_decoder_prompt_ids(self):
        return [50258] # sot

    def load_dataset(self, path_or_paths):
        dataset = load_dataset("json", data_files={"train": path_or_paths})["train"]

        if self.args.subsample:
            dataset = dataset.select(range(100))

        def resolve_paths(example):
            example["audio"] = example["path"]
            del example["path"]
            return example

        dataset = dataset.map(resolve_paths, desc="resolving paths")

        print(dataset, file=sys.stderr)

        dataset = dataset.to_iterable_dataset(num_shards=self.args.preprocessing_num_workers)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(self, remove_columns=dataset.features)
        dataset = dataset.with_format("torch")

        def is_audio_in_length_range(input_length):
            return input_length <= 30

        dataset = dataset.filter(is_audio_in_length_range, input_columns=["input_length"])

        return dataset

    def make_train_dataloader(self, dataset):
        args = self.args
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
        return train_dataloader

    def make_eval_dataloader(self, dataset):
        args = self.args
        eval_dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
        )
        return eval_dataloader


def echo_loop(eval_dataloader, corpus):
    tokenizer = get_tokenizer(multilingual=True, language='ru', task='transcribe')

    references = []
    for batch_index, batch in enumerate(eval_dataloader):
        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, corpus.processor.tokenizer.pad_token_id)
        decoded_labels = corpus.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i, label in enumerate(labels):
            xs = tokenizer.encoding.decode_tokens_bytes(label)
            def format(x):
                try:
                    x = x.decode('utf-8')
                except UnicodeDecodeError:
                    x = repr(x)
                if x == '\n':
                    x = '␊'
                return x.replace(' ', '⎽')
            #print(f'{batch_index}:{i}', ' '.join(format(x) for x in xs), sep='\t')

        references.extend(decoded_labels)
        del labels, batch


def register_data_args(parser):
    parser.add_argument("--train", type=str, nargs='+', help="jsonl filename for training data, can be multiple files", required=False)
    parser.add_argument("--eval", type=str, help="jsonl filename for evaluation data", required=True)
    parser.add_argument("--subsample", action="store_true", help="Use a tiny fraction of data for testing")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--language", type=str, help="Language code for WhisperProcessor: 'Hindi' ", default="Russian")
    parser.add_argument(
        "--task", type=str, default="transcribe", help="Task to use for training; e.g., 'transcribe' ", required=False
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=bool,
        default=True,
        help="Whether or not to pin memory for the DataLoader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading.",
    )


def test_data(args):
    corpus = Corpus(args)
    test = corpus.load_dataset(args.eval)
    eval_dataloader = corpus.make_eval_dataloader(test)

    echo_loop(eval_dataloader, corpus)


def cli():
    parser = argparse.ArgumentParser(description="Is my data ok?")
    register_data_args(parser)
    args = parser.parse_args()
    test_data(args)
