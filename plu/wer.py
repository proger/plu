from __future__ import annotations

from typing import Iterable, Sequence


def _edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(prediction, start=1):
            if ref_word == hyp_word:
                cost = previous[j - 1]
            else:
                cost = previous[j - 1] + 1
            current.append(min(cost, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1]


def word_error_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    total_edits = 0
    total_words = 0

    for prediction, reference in zip(predictions, references):
        ref_words = reference.split()
        hyp_words = prediction.split()
        total_edits += _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0 if total_edits == 0 else 1.0
    return total_edits / total_words
