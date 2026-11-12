"""Text helpers shared by the server and the evaluation tools."""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase and strip punctuation, so comparisons see only the words."""
    text = text.lower().replace("’", "'")
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over the reference length."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Total errors over total reference words (how benchmarks report WER)."""
    errors = words = 0.0
    for reference, hypothesis in pairs:
        n = len(normalise(reference).split())
        errors += word_error_rate(reference, hypothesis) * n
        words += n
    return errors / words if words else 0.0
