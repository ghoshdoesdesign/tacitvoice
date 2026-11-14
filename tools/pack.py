"""Turn your recordings into one small file for fine-tuning (run on the Mac).

    uv run python -m tools.pack

Reads data/train and data/train2 (recorded with tools.record --out ...) as
training clips and data/baseline as test clips, cuts each down to the 96x96 grey mouth
region the model sees, and writes data/pack.npz. Only your mouth region and
the sentence text go into the file; no audio and no full face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from server import config
from server.lips import MouthCropper
from tools.evaluate import load_clip

MODES = ("aloud", "whisper", "silent")


def collect(root: Path, split: str) -> list[tuple[Path, str, str]]:
    found = []
    for mode in MODES:
        for meta in sorted((root / mode).glob("*.json")):
            found.append((meta.with_suffix(""), mode, split))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--train", type=Path, nargs="+", default=[Path("data/train"), Path("data/train2")],
        help="folders of training recordings; ones that do not exist are skipped",
    )
    ap.add_argument("--test", type=Path, default=Path("data/baseline"))
    ap.add_argument("--out", type=Path, default=Path("data/pack.npz"))
    args = ap.parse_args()

    clips = [c for folder in args.train for c in collect(folder, "train")] + collect(args.test, "test")
    if not any(split == "train" for _, _, split in clips):
        raise SystemExit(f"No recordings under {args.train[0]}. Record with: tools.record ... --out {args.train[0]}")

    crop = MouthCropper(config.AVSR_REPO)
    arrays: dict[str, np.ndarray] = {}
    manifest, skipped, frames = [], 0, {"train": 0, "test": 0}
    for n, (stem, mode, split) in enumerate(clips, 1):
        utterance, text = load_clip(stem)
        patches, face = (None, 0.0)
        if len(utterance.frames) >= 10:
            patches, face = crop(utterance.frames, utterance.timestamps, utterance.fps)
        if patches is None or face < 0.8:
            skipped += 1
            print(f"[{n}/{len(clips)}] skipped {split}/{mode}/{stem.name}: face found in {face:.0%} of frames")
            continue
        key = f"{split}_{stem.parent.parent.name}_{mode}_{stem.name}"  # unique across folders
        arrays[key] = np.asarray(patches, dtype=np.uint8)
        manifest.append({"key": key, "text": text, "mode": mode, "split": split})
        frames[split] += len(patches)
        if n % 25 == 0:
            print(f"[{n}/{len(clips)}] cropped")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, manifest=np.array(json.dumps(manifest)), **arrays)
    size = args.out.stat().st_size / 1e6
    print(f"\n{len(manifest)} clips packed, {skipped} skipped.")
    print(f"training: {frames['train'] / 25 / 60:.0f} min   test: {frames['test'] / 25 / 60:.0f} min")
    print(f"Wrote {args.out} ({size:.0f} MB). Upload it to Google Drive for the Colab notebook.")


if __name__ == "__main__":
    main()
