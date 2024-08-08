import argparse
import json
from logging import getLogger, basicConfig
from pathlib import Path
import subprocess

from transformers import (
    WhisperForConditionalGeneration,
    AutoTokenizer
)
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge finetuned adapter into the model, convert to CTranslate2 and do a test with faster-whisper")
    parser.add_argument(
        "--exp",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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



basicConfig(level="DEBUG")
logger = getLogger(__name__)


def recognize(model: WhisperModel, filename: Path, prefix: str | None = None):
    logger.debug("recognize %s", filename)
    try:
        segments, info = model.transcribe(
            str(filename),
            beam_size=5,
            word_timestamps=True,
            temperature=[0.0],
            prefix=prefix,
        )
    except Exception:
        logger.exception("failed to recognize %s", filename)
        return

    metadata = dict(
        filename=str(filename),
        lang=info.language,
        langprob=round(info.language_probability, 2),
        duration=round(info.duration, 2),
    )

    for i, segment in enumerate(segments):
        start = round(segment.start, 2)
        end = round(segment.end, 2)
        text = "".join(word.word for word in segment.words)
        conf = [round(word.probability, 2) for word in segment.words]
        avg_logprob = round(segment.avg_logprob, 3)
        no_speech_prob = round(segment.no_speech_prob, 3)
        yield dict(
            i=i,
            start=start,
            end=end,
            text=text,
            conf=conf,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            metadata=metadata,
        )


def merge_and_convert(exp):
    model = WhisperForConditionalGeneration.from_pretrained(exp)
    tokenizer = AutoTokenizer.from_pretrained(exp)

    peft_model = PeftModel.from_pretrained(model, exp,)
    peft_model = peft_model.merge_and_unload()

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
    model_dir = merge_and_convert(args.exp)

    logger.warning("if this crashes, re-run with env LD_LIBRARY_PATH=/ai/env/lib/python3.10/site-packages/nvidia/cudnn/lib")
    from faster_whisper import WhisperModel
    model = WhisperModel(
        str(model_dir),
        device='cuda',
        compute_type='float16',
        num_workers=1,
        cpu_threads=1,
    )

    for seg in recognize(model, args.filenames):
        print(seg)
