"""Add another person's recordings to the training data (run on the Mac).

    uv run python -m tools.import_friend ~/Downloads/silent-speech-asha.zip

Takes the .zip that the recording page (collect/index.html) gave them and
unpacks it into data/friends/<name>/, in the layout tools.pack reads. The
file comes from someone else's computer, so only the entries the page is
known to write are read, and nothing in it is trusted as a path.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import cv2

TAKE = re.compile(r"^takes/(\d{4})_(aloud|whisper|silent)\.(webm|mp4)$")
MAX_CLIP_BYTES = 40_000_000
MAX_TAKES = 3000


def slug(name: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "friend")[:30]


def clean_text(text: object) -> str:
    """One line of printable text, however the manifest was written."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in str(text)).split())[:300]


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def import_zip(archive: Path, root: Path, name: str | None = None) -> tuple[Path, int, int]:
    """Unpack one person's zip. Returns (folder, clips kept, clips rejected)."""
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        person = slug(name or str(manifest.get("name", "")))
        folder = root / person
        n = 2
        while folder.exists():
            folder = root / f"{person}-{n}"
            n += 1
        sizes = {info.filename: info.file_size for info in z.infolist()}
        kept = rejected = 0
        counters: dict[str, int] = {}
        for take in list(manifest.get("takes", []))[:MAX_TAKES]:
            file, text = str(take.get("file", "")), clean_text(take.get("text", ""))
            match = TAKE.match(file)
            if not match or file not in sizes or sizes[file] > MAX_CLIP_BYTES or not text:
                rejected += 1
                continue
            mode, ext = match.group(2), match.group(3)
            index = counters.get(mode, 0)
            stem = folder / mode / f"{index:03d}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            video = stem.with_suffix("." + ext)
            video.write_bytes(z.read(file))
            if frame_count(video) < 10:  # not a playable clip
                video.unlink()
                rejected += 1
                continue
            stem.with_suffix(".json").write_text(json.dumps({"text": text, "mode": mode, "speaker": folder.name}))
            counters[mode] = index + 1
            kept += 1
    return folder, kept, rejected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, nargs="+", help="one or more .zip files from the recording page")
    ap.add_argument("--root", type=Path, default=Path("data/friends"))
    ap.add_argument("--name", help="folder name to use instead of the name they typed (one archive only)")
    args = ap.parse_args()
    if args.name and len(args.archive) > 1:
        raise SystemExit("--name works with a single archive.")
    for archive in args.archive:
        folder, kept, rejected = import_zip(archive, args.root, args.name)
        print(f"{archive.name}: {kept} clips into {folder}" + (f", {rejected} rejected" if rejected else ""))
    print("Next: uv run python -m tools.pack")


if __name__ == "__main__":
    main()
