"""Print how much the mouth moves in each recorded clip (run on the Mac).

    uv run python -m tools.motion

Used to set TACIT_LIPS_MIN_MOTION, the level below which a silent clip is
answered with "No words detected". Every recorded clip contains speech, so
the lowest values here are the floor the cutoff must stay under.
"""

from __future__ import annotations

from pathlib import Path

from server import config
from server.lips import MouthCropper, mouth_motion
from tools.evaluate import load_clip


def main() -> None:
    crop = MouthCropper(config.AVSR_REPO)
    for mode in ("silent", "whisper"):
        values = []
        for meta in sorted((Path("data/baseline") / mode).glob("*.json")):
            utterance, _ = load_clip(meta.with_suffix(""))
            patches, _ = crop(utterance.frames, utterance.timestamps, utterance.fps)
            if patches is not None:
                values.append(round(mouth_motion(patches), 2))
        if values:
            print(f"{mode:8} motion min {min(values):5.2f}  max {max(values):5.2f}   {values}")
    print(f"a silent clip under {config.LIPS_MIN_MOTION} counts as no words")


if __name__ == "__main__":
    main()
