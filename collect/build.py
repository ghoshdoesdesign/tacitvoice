"""Build collect/index.html, the recording page to send to other people.

    python collect/build.py              # 60 sentences per person
    python collect/build.py --per-person 30

The page is one self-contained file. Host it anywhere that serves it over
https (GitHub Pages, Netlify Drop, ...) and send people the link. It uploads
nothing: each person downloads a .zip of their recordings and sends it back,
and tools.import_friend adds it to the training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
SENTENCE_FILES = ("calibration.txt", "calibration2.txt")
SECONDS_PER_TAKE = 10  # read, hold, say, release


def sentences() -> list[str]:
    out = []
    for name in SENTENCE_FILES:
        for line in (HERE.parent / "sentences" / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def build(per_person: int = 60) -> str:
    minutes = round(per_person * 3 * SECONDS_PER_TAKE / 60 / 5) * 5
    page = (HERE / "page.html").read_text(encoding="utf-8")
    return (
        page.replace("__SENTENCES__", json.dumps(sentences(), ensure_ascii=False))
        .replace("__PER_PERSON__", str(per_person))
        .replace("__MINUTES__", str(minutes))
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-person", type=int, default=60, help="sentences each person reads, three ways each")
    args = ap.parse_args()
    out = HERE / "index.html"
    out.write_text(build(args.per_person), encoding="utf-8")
    print(f"wrote {out}")
