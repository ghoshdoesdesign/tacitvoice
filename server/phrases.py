"""The user's own vocabulary: a glossary and the prompts they tend to send.

Both are plain text files in ~/.tacitvoice, one entry per line. Matching a
rough transcript against prompts the user has sent before is far easier than
open dictation, which is what makes silent mode usable before any fine-tune.
"""

from __future__ import annotations

from pathlib import Path

from rapidfuzz import fuzz, process

from .text import normalise


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


class Phrasebook:
    def __init__(self, dictionary_file: Path, prompts_file: Path) -> None:
        self._files = (dictionary_file, prompts_file)
        self._mtimes: tuple[float, float] | None = None
        self.glossary: list[str] = []
        self.prompts: list[str] = []
        self._normalised: list[str] = []
        self.reload()

    def reload(self) -> None:
        """Re-read the files if they changed, so edits apply without a restart."""
        mtimes = tuple(f.stat().st_mtime if f.exists() else 0.0 for f in self._files)
        if mtimes == self._mtimes:
            return
        self._mtimes = mtimes  # type: ignore[assignment]
        self.glossary = read_lines(self._files[0])
        self.prompts = read_lines(self._files[1])
        self._normalised = [normalise(p) for p in self.prompts]

    def similar(self, candidates: list[str], limit: int = 5, min_score: float = 55.0) -> list[tuple[str, float]]:
        """Known prompts closest to any candidate, best first, scored 0-100."""
        best: dict[int, float] = {}
        for candidate in candidates:
            query = normalise(candidate)
            if not query:
                continue
            for _, score, i in process.extract(query, self._normalised, scorer=fuzz.WRatio, limit=limit):
                if score >= min_score and score > best.get(i, 0.0):
                    best[i] = float(score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:limit]
        return [(self.prompts[i], score) for i, score in ranked]

    def remember(self, prompt: str) -> None:
        """Add a prompt the user actually sent, so the next one like it is easier."""
        prompt = " ".join(prompt.split())
        if len(prompt.split()) < 3 or normalise(prompt) in self._normalised:
            return
        path = self._files[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(prompt + "\n")
        self.reload()
