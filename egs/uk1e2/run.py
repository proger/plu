#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def apply_env_assignments(argv: list[str]) -> list[str]:
    remaining = []
    for arg in argv:
        if not arg.startswith("--") and "=" in arg:
            key, value = arg.split("=", 1)
            os.environ[key] = value
        else:
            remaining.append(arg)
    return remaining


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    argv = apply_env_assignments(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run the UK1E2 PLU end-to-end recipe.")
    parser.add_argument("--stage", type=int, default=env_int("stage", 0))
    parser.add_argument("--stop-stage", type=int, default=env_int("stop_stage", 5))
    return parser.parse_args(argv)


def rel(path: Path, root: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    return os.path.relpath(path, root)


def under_root(path: Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def run(cmd: list[str], *, root: Path, stdout: Path | None = None, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if stdout is None:
        subprocess.run(cmd, cwd=root, env=env, check=True)
    else:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=root, env=env, check=True, stdout=f)


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UK1E2 training; set up a GPU before running stage 3.")


def main() -> None:
    args = parse_args()
    recipe_dir = Path(__file__).resolve().parent
    root = recipe_dir.parents[1]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(root)
    env["PLU_OPS_BACKEND"] = env_str("PLU_OPS_BACKEND", "triton")

    limit = env_int("N", 100)
    language = env_str("LANGUAGE", "uk")
    num_languages = env_int("NUM_LANGUAGES", 100)
    n_mels = env_int("N_MELS", 80)
    id_prefix = env_str("ID_PREFIX", "N")

    exp = under_root(Path(env_str("EXP", "egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8")), root)
    data_dir = exp / "data"
    train_exp = exp / "train"

    wav_scp = Path(env_str("WAV_SCP", "data/segments/wav.scp"))
    text_full = Path(env_str("TEXT_FULL", "data/local/text.full"))

    model_name_or_path = env_str("MODEL_NAME_OR_PATH", "openai/whisper-large-v3-turbo")
    baseline_model = env_str("BASELINE_MODEL", "large-v3-turbo")

    train_steps = env_int("TRAIN_STEPS", 100)
    train_batch_size = env_int("TRAIN_BATCH_SIZE", 1)
    eval_batch_size = env_int("EVAL_BATCH_SIZE", 4)
    train_eval_limit = env_int("TRAIN_EVAL_LIMIT", 4)
    gradient_accumulation_steps = env_int("GRADIENT_ACCUMULATION_STEPS", 1)
    learning_rate = env_str("LEARNING_RATE", "1e-6")
    static_input_features = env_int("STATIC_INPUT_FEATURES", 3000)
    static_decoder_len = env_int("STATIC_DECODER_LEN", 448)

    test_device = env_str("TEST_DEVICE", "cuda")
    test_dtype = env_str("TEST_DTYPE", "bf16")
    test_language = env_str("TEST_LANGUAGE", language)
    download_root = under_root(Path(env_str("DOWNLOAD_ROOT", rel(exp / "models", root))), root)

    exp.mkdir(parents=True, exist_ok=True)

    if args.stage <= 0 <= args.stop_stage:
        run(
            [
                sys.executable,
                "egs/uk1e2/local/prepare_subset.py",
                "--wav-scp",
                rel(wav_scp, root),
                "--text",
                rel(text_full, root),
                "--root",
                ".",
                "--out-dir",
                rel(data_dir, root),
                "--limit",
                str(limit),
                "--id-prefix",
                id_prefix,
                "--language",
                language,
                "--num-languages",
                str(num_languages),
            ],
            root=root,
            env=env,
        )
        (data_dir / "train_eval.jsonl").write_text(
            "".join((data_dir / "subset.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)[:train_eval_limit]),
            encoding="utf-8",
        )

    if (data_dir / "subset.jsonl").exists() and not (data_dir / "train_eval.jsonl").exists():
        (data_dir / "train_eval.jsonl").write_text(
            "".join((data_dir / "subset.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)[:train_eval_limit]),
            encoding="utf-8",
        )

    if args.stage <= 1 <= args.stop_stage:
        run(
            [
                "+dataloader",
                "--eval",
                rel(data_dir / "subset.jsonl", root),
                "--n_mels",
                str(n_mels),
                "--language",
                language,
                "--task",
                "transcribe",
                "--per_device_eval_batch_size",
                str(eval_batch_size),
                "--dataloader_num_workers",
                "0",
                "--dataloader_pin_memory",
                "false",
                "--max_batches",
                "1",
            ],
            root=root,
            stdout=exp / "dataloader.jsonl",
            env=env,
        )

    test_files = (data_dir / "wav.list").read_text(encoding="utf-8").splitlines()

    if args.stage <= 2 <= args.stop_stage:
        run(
            [
                "+test",
                "--model",
                baseline_model,
                "--download_root",
                rel(download_root, root),
                "--device",
                test_device,
                "--dtype",
                test_dtype,
                "--language",
                test_language,
                *test_files,
            ],
            root=root,
            stdout=exp / "before.test.jsonl",
            env=env,
        )
        run(
            [
                sys.executable,
                "egs/uk1e2/local/score_test_jsonl.py",
                "--refs",
                rel(data_dir / "subset.jsonl", root),
                "--hyps",
                rel(exp / "before.test.jsonl", root),
            ],
            root=root,
            stdout=exp / "before.wer.json",
            env=env,
        )

    if args.stage <= 3 <= args.stop_stage:
        require_cuda()
        if train_exp.exists():
            shutil.rmtree(train_exp)
        run(
            [
                "+train",
                "--model_name_or_path",
                model_name_or_path,
                "--train",
                rel(data_dir / "subset.jsonl", root),
                "--eval",
                rel(data_dir / "train_eval.jsonl", root),
                "--exp",
                rel(train_exp, root),
                "--learning_rate",
                learning_rate,
                "--max_train_steps",
                str(train_steps),
                "--per_device_train_batch_size",
                str(train_batch_size),
                "--per_device_eval_batch_size",
                str(eval_batch_size),
                "--gradient_accumulation_steps",
                str(gradient_accumulation_steps),
                "--dataloader_num_workers",
                "0",
                "--dataloader_pin_memory",
                "false",
                "--device",
                "cuda",
                "--static_input_features",
                str(static_input_features),
                "--static_decoder_len",
                str(static_decoder_len),
            ],
            root=root,
            env=env,
        )

    if args.stage <= 4 <= args.stop_stage:
        run(
            [
                "+test",
                "--exp",
                rel(train_exp, root),
                "--download_root",
                rel(download_root, root),
                "--device",
                test_device,
                "--dtype",
                test_dtype,
                "--language",
                test_language,
                *test_files,
            ],
            root=root,
            stdout=exp / "after.test.jsonl",
            env=env,
        )
        run(
            [
                sys.executable,
                "egs/uk1e2/local/score_test_jsonl.py",
                "--refs",
                rel(data_dir / "subset.jsonl", root),
                "--hyps",
                rel(exp / "after.test.jsonl", root),
            ],
            root=root,
            stdout=exp / "after.wer.json",
            env=env,
        )

    if args.stage <= 5 <= args.stop_stage:
        run(
            [
                sys.executable,
                "egs/uk1e2/local/compare_wer.py",
                "--before",
                rel(exp / "before.wer.json", root),
                "--after",
                rel(exp / "after.wer.json", root),
            ],
            root=root,
            stdout=exp / "wer.txt",
            env=env,
        )
        print((exp / "wer.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
