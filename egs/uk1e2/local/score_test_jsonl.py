from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score +test JSONL output against a PLU JSONL reference file.")
    parser.add_argument("--refs", type=Path, required=True, help="Reference JSONL produced by prepare_subset.py.")
    parser.add_argument("--hyps", type=Path, required=True, help="+test JSONL output.")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    chars = []
    for char in text:
        if char.isalnum() or char == "'":
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            cost = previous[j - 1] if ref_word == hyp_word else previous[j - 1] + 1
            current.append(min(cost, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1]


def path_key(path: str) -> str:
    return str(Path(path).resolve())


def load_refs(path: Path) -> list[dict]:
    refs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            refs.append(row)
    return refs


def load_hyps(path: Path) -> dict[str, str]:
    by_path: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_path[path_key(row["path"])].append((int(row.get("i", 0)), row.get("text", "")))
    return {
        key: " ".join(text for _, text in sorted(segments))
        for key, segments in by_path.items()
    }


def main() -> None:
    args = parse_args()
    refs = load_refs(args.refs)
    hyps = load_hyps(args.hyps)

    total_words = 0
    total_edits = 0
    missing = 0
    details = []

    for row in refs:
        key = path_key(row["path"])
        ref = normalize(row["text"])
        hyp = normalize(hyps.get(key, ""))
        if key not in hyps:
            missing += 1
        ref_words = ref.split()
        hyp_words = hyp.split()
        edits = edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_edits += edits
        details.append({"id": row.get("id"), "path": row["path"], "ref": ref, "hyp": hyp, "edits": edits, "words": len(ref_words)})

    wer = total_edits / total_words if total_words else 0.0
    print(
        json.dumps(
            {
                "wer": wer,
                "wer_percent": 100.0 * wer,
                "total_edits": total_edits,
                "total_words": total_words,
                "references": len(refs),
                "hypotheses": len(hyps),
                "missing_hypotheses": missing,
                "normalization": "NFKC + casefold + keep unicode alnum/apostrophe",
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
