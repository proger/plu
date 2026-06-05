import argparse
import json
from logging import getLogger
import os
from pathlib import Path
import shutil
import subprocess

import torch
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

from plu.whisper import WhisperForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a PLU adapter, convert to CTranslate2, and test with faster-whisper")
    parser.add_argument(
        "--exp",
        type=Path,
        help="Experiment directory. If it contains a CTranslate2 model, it is tested directly; if it contains a PLU adapter, it is merged and converted.",
        default=Path("exp/1"),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Skip merge/convert and test this faster-whisper alias, HF CTranslate2 model ID, or local CTranslate2 model directory.",
        default=None,
    )
    parser.add_argument(
        "--download_root",
        type=Path,
        help="Directory where faster-whisper should cache downloaded models.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for faster-whisper: cuda, cpu, or auto.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="CTranslate2 compute type, e.g. float16, int8_float16, int8, default.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="CPU threads for faster-whisper.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Worker count for faster-whisper.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use a locally cached faster-whisper model.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="float16",
        help="CTranslate2 conversion quantization for merged PLU adapters.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code to pass to faster-whisper; skips language detection when set.",
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



def recognize(model: MyWhisperModel, filename: Path, prefix: str | None = None, language: str | None = None):
    logger.debug("recognize %s", filename)
    try:
        segments, info = model.transcribe(
            str(filename),
            beam_size=5,
            word_timestamps=False,
            without_timestamps=True, # our training format doesn't have timestamps
            temperature=[0.0],
            prefix=prefix,
            language=language,
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


def _module_by_name(model, name: str):
    modules = dict(model.named_modules())
    try:
        return modules[name]
    except KeyError as exc:
        raise KeyError(f"LoRA target module not found in base model: {name}") from exc


def merge_lora_into_base(exp: Path) -> Path:
    adapter_config_path = exp / "adapter_config.json"
    adapter_model_path = exp / "adapter_model.bin"
    if not adapter_config_path.exists() or not adapter_model_path.exists():
        raise FileNotFoundError(f"{exp} does not contain adapter_config.json and adapter_model.bin")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"{adapter_config_path} does not define base_model_name_or_path")

    model = WhisperForConditionalGeneration.from_pretrained(base_model, map_location="cpu")
    adapter_state = torch.load(adapter_model_path, map_location="cpu", weights_only=False)
    scaling = float(adapter_config["lora_alpha"]) / int(adapter_config["r"])

    for key, lora_a in adapter_state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module_name = key[: -len(".lora_A.weight")]
        lora_b = adapter_state[f"{module_name}.lora_B.weight"]
        module = _module_by_name(model, module_name)
        update = torch.matmul(lora_b.to(module.weight.dtype), lora_a.to(module.weight.dtype)) * scaling
        module.weight.data.add_(update)

    merged_model_dir = exp / "merged"
    model.save_pretrained(merged_model_dir)
    copy_tokenizer_files(exp, Path(base_model), merged_model_dir)
    write_preprocessor_config(merged_model_dir, model.config.num_mel_bins)
    return merged_model_dir


def copy_tokenizer_files(exp: Path, base_model: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "vocabulary.json"):
        for source_dir in (exp, base_model):
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
                break


def write_preprocessor_config(output_dir: Path, n_mels: int) -> None:
    (output_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "chunk_length": 30,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": n_mels,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "WhisperProcessor",
                "return_attention_mask": False,
                "sampling_rate": 16000,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def convert_to_ct2(model_dir: Path, output_dir: Path, quantization: str) -> Path:
    if not (model_dir / "tokenizer.json").exists() and not (
        (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
    ):
        raise FileNotFoundError(
            f"{model_dir} must contain tokenizer.json or both vocab.json and merges.txt for ct2-transformers-converter"
        )

    copy_files = [
        filename
        for filename in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json", "merges.txt", "vocabulary.json")
        if (model_dir / filename).exists()
    ]
    cmd = [
        "ct2-transformers-converter",
        "--model",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--force",
        "--quantization",
        quantization,
    ]
    if copy_files:
        cmd.extend(["--copy_files", *copy_files])
    subprocess.run(cmd, check=True)
    return output_dir


def merge_and_convert(exp: Path, quantization: str) -> Path:
    ct_output_dir = exp / "ct"
    if (ct_output_dir / "model.bin").exists():
        return ct_output_dir
    merged_model_dir = merge_lora_into_base(exp)
    return convert_to_ct2(merged_model_dir, ct_output_dir, quantization)


def resolve_model_arg(args) -> str:
    if args.model is not None:
        return args.model
    if (args.exp / "model.bin").exists():
        return str(args.exp)
    if args.exp.exists():
        return str(merge_and_convert(args.exp, args.quantization))
    return str(args.exp)


def main():
    args = parse_args()
    model_name_or_path = resolve_model_arg(args)

    if Path(model_name_or_path).exists() and not (Path(model_name_or_path) / "model.bin").exists():
        raise ValueError(
            f"{model_name_or_path} exists but is not a faster-whisper/CTranslate2 model directory. "
            "Convert or merge training outputs separately, then pass the converted directory."
        )

    if args.device == "cuda" and os.environ.get("LD_LIBRARY_PATH", "").find("cudnn") == -1:
        logger.warning("If this crashes, re-run with env LD_LIBRARY_PATH=/ai/env/lib/python3.10/site-packages/nvidia/cudnn/lib")

    model = MyWhisperModel(
        model_name_or_path,
        device=args.device,
        compute_type=args.compute_type,
        num_workers=args.num_workers,
        cpu_threads=args.cpu_threads,
        download_root=str(args.download_root) if args.download_root is not None else None,
        local_files_only=args.local_files_only,
    )

    for filename in args.filenames:
        for seg in recognize(model, filename, language=args.language):
            print(json.dumps(seg, ensure_ascii=False))


if __name__ == "__main__":
    main()
