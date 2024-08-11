import argparse
import json
from logging import getLogger, basicConfig
import os
from pathlib import Path
import subprocess
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from transformers import (
    WhisperForConditionalGeneration,
    AutoTokenizer
)
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge finetuned adapter into the model, convert to CTranslate2 and do a test with faster-whisper")
    parser.add_argument(
        "--exp",
        type=Path,
        help="Path to pretrained model or model identifier from huggingface.co/models. If you pass in the ctranslate2 model directory, skips the merge step.",
        default="exp/1",
    )
    parser.add_argument(
        "filenames",
        type=Path,
        nargs="*",
        help="Path to the audio file(s) to transcribe.",
    )

    args = parser.parse_args()
    return args



#basicConfig(level="DEBUG")
logger = getLogger(__name__)


class MyWhisperModel(WhisperModel):
    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: list[int],
        without_timestamps: bool = False,
        prefix: str | None = None,
        hotwords: str | None = None,
    ) -> list[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend([tokenizer.sot, tokenizer.language, tokenizer.transcribe])

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt



def recognize(model: MyWhisperModel, filename: Path, prefix: str | None = None):
    logger.debug("recognize %s", filename)
    try:
        segments, info = model.transcribe(
            str(filename),
            beam_size=5,
            word_timestamps=False,
            without_timestamps=True, # our training format doesn't have timestamps
            temperature=[0.0],
            prefix=prefix,
            log_prob_threshold=None,
            no_speech_threshold=None,
            compression_ratio_threshold=None,
        )
    except Exception:
        logger.exception("failed to recognize %s", filename)
        return

    tokenizer = None

    for i, segment in enumerate(segments):
        start = round(segment.start, 2)
        end = round(segment.end, 2)
        if segment.words:
            text = "".join(word.word for word in segment.words)
            conf = [round(word.probability, 2) for word in segment.words]
        else:
            text = segment.text
            conf = None
        avg_logprob = round(segment.avg_logprob, 3)
        no_speech_prob = round(segment.no_speech_prob, 3)

        if tokenizer is None or info.language != tokenizer.language:
            tokenizer = Tokenizer(
                model.hf_tokenizer,
                model.model.is_multilingual,
                task='transcribe',
                language=info.language, # assume lid gives us this language, this only affects input_ids output
            )

        prompt_ids = model.get_prompt(tokenizer, previous_tokens=[], without_timestamps=True, prefix=prefix)

        yield dict(
            i=i,
            start=start,
            end=end,
            text=text,
            conf=conf,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            path=str(filename),
            language=info.language,
            langprob=round(info.language_probability, 2),
            input_ids=prompt_ids + segment.tokens,
        )


def load_model(exp):
    model = WhisperForConditionalGeneration.from_pretrained(exp)
    tokenizer = AutoTokenizer.from_pretrained(exp)

    peft_model = PeftModel.from_pretrained(model, exp,)
    peft_model = peft_model.merge_and_unload()

    return peft_model, tokenizer


def merge_and_convert(exp):
    peft_model, tokenizer = load_model(exp)

    merged_model_dir = Path(exp) / "merged"
    peft_model._hf_peft_config_loaded = False # wtf
    peft_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)

    ct_output_dir = Path(exp) / "ct"

    subprocess.run(["ct2-transformers-converter",
                    "--model", str(merged_model_dir), "--output_dir", str(ct_output_dir),
                    "--force", "--quantization", "float16", "--copy", "tokenizer.json", "tokenizer_config.json"])
    (ct_output_dir / "preprocessor_config.json").write_text(json.dumps({
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128, # 
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000
    }))

    return ct_output_dir


def main():
    args = parse_args()
    if (args.exp / 'model.bin').exists():
        model_dir = args.exp
    else:
        if args.exp.exists():
            model_dir = merge_and_convert(args.exp)
        else:
            model_dir = str(args.exp) # probably hub name

    if os.environ.get("LD_LIBRARY_PATH", "").find("cudnn") == -1:
        logger.warning("If this crashes, re-run with env LD_LIBRARY_PATH=/ai/env/lib/python3.10/site-packages/nvidia/cudnn/lib")

    model = MyWhisperModel(
        str(model_dir),
        device='cuda',
        compute_type='float16',
        num_workers=1,
        cpu_threads=1,
    )

    for filename in args.filenames:
        for seg in recognize(model, filename):
            print(json.dumps(seg, ensure_ascii=False))
