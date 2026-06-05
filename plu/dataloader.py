import argparse
import json
import logging
from pathlib import Path

import numpy as np

from plu.tokenizer import WhisperTokenizer, get_tokenizer
from plu.train_data import Corpus, register_data_args
from plu.whisper import load_config


logger = logging.getLogger(__name__)


def render_token_piece(piece: bytes) -> str:
    try:
        text = piece.decode("utf-8")
    except UnicodeDecodeError:
        text = repr(piece)
    if text == "\n":
        text = "␊"
    return text.replace(" ", "⎽")


def load_tokenizer_and_n_mels(args: argparse.Namespace) -> tuple[WhisperTokenizer, int]:
    config = None
    model_path = Path(args.model_name_or_path).expanduser() if args.model_name_or_path else None
    if model_path is not None and model_path.exists():
        config = load_config(model_path)
        tokenizer = WhisperTokenizer.from_pretrained(model_path, pad_token_id=config.pad_token_id)
    else:
        if model_path is not None:
            logger.warning("model path %s does not exist; using built-in tokenizer defaults", model_path)
        tokenizer = get_tokenizer(
            multilingual=True,
            language=args.language,
            task=args.task,
            num_languages=args.num_languages,
        )

    if args.n_mels is not None:
        n_mels = args.n_mels
    elif config is not None:
        n_mels = config.num_mel_bins
    else:
        n_mels = 128

    return tokenizer, n_mels


def batch_summary(batch, tokenizer: WhisperTokenizer, batch_index: int) -> dict:
    labels = batch["labels"].cpu().numpy()
    labels_for_decode = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    token_pieces = []
    for label in labels_for_decode:
        pieces = tokenizer.encoding.decode_tokens_bytes(token_id for token_id in label if token_id != tokenizer.pad_token_id)
        token_pieces.append([render_token_piece(piece) for piece in pieces])

    return {
        "batch": batch_index,
        "input_features_shape": list(batch["input_features"].shape),
        "labels_shape": list(batch["labels"].shape),
        "decoded_labels": decoded_labels,
        "token_pieces": token_pieces,
    }


def test_data(args: argparse.Namespace) -> list[dict]:
    tokenizer, n_mels = load_tokenizer_and_n_mels(args)
    corpus = Corpus(args, tokenizer=tokenizer, n_mels=n_mels)

    if args.split == "train":
        if args.train is None:
            raise ValueError("--train is required when --split train is selected")
        dataset = corpus.load_dataset(args.train)
        dataloader = corpus.make_train_dataloader(dataset)
    else:
        dataset = corpus.load_dataset(args.eval)
        dataloader = corpus.make_eval_dataloader(dataset)

    summaries = []
    for batch_index, batch in enumerate(dataloader):
        summary = batch_summary(batch, tokenizer, batch_index)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
        if args.max_batches is not None and batch_index + 1 >= args.max_batches:
            break
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Inspect local PLU dataloader batches")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Optional local Whisper model directory used for config/tokenizer metadata.",
    )
    parser.add_argument("--n_mels", type=int, default=None, help="Override mel bin count; defaults to model config or 128.")
    parser.add_argument("--num_languages", type=int, default=100, help="Number of Whisper language tokens.")
    parser.add_argument("--split", choices=["eval", "train"], default="eval", help="Dataset split to inspect.")
    parser.add_argument("--max_batches", type=int, default=1, help="Maximum number of batches to print; use -1 for all.")
    register_data_args(parser)
    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches < 0:
        args.max_batches = None

    logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s", level=logging.INFO)
    test_data(args)


if __name__ == "__main__":
    main()
