from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print before/after WER summary.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    delta = after["wer_percent"] - before["wer_percent"]
    print(f"before_wer={before['wer_percent']:.2f}%")
    print(f"after_wer={after['wer_percent']:.2f}%")
    print(f"delta={delta:+.2f}%")
    print(f"refs={after['references']} words={after['total_words']}")
    print(f"missing_before={before['missing_hypotheses']} missing_after={after['missing_hypotheses']}")


if __name__ == "__main__":
    main()
