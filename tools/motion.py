"""Check the "no words" detector against your recordings (run on the Mac).

    uv run python -m tools.motion

Every recorded clip starts with a short pause before the sentence, so the
recordings already hold both cases: the first third of a second of each clip
is a still mouth, and the whole clip is a speaking one. This prints the
detector's score for both. The two ranges should not overlap, with the
cutoff (TACIT_LIPS_MIN_MOTION) between them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from server import config
from server.lips import MouthCropper, mouth_motion
from tools.evaluate import load_clip

STILL_FRAMES = 9  # about a third of a second at 25 frames per second


def summary(values: list[float]) -> str:
    v = np.array(values)
    return (f"lowest {v.min():5.2f}  typical {np.median(v):5.2f}  highest {v.max():5.2f}")


def main() -> None:
    crop = MouthCropper(config.AVSR_REPO)
    cutoff = config.LIPS_MIN_MOTION
    still, speaking = [], []
    folders = [Path("data/baseline"), Path("data/train"), Path("data/train2")]
    stems = [m.with_suffix("") for f in folders for m in sorted(f.glob("*/*.json"))]
    for stem in stems[:: max(1, len(stems) // 120)]:  # a sample is enough
        utterance, _ = load_clip(stem)
        if len(utterance.frames) < 40:
            continue
        patches, face = crop(utterance.frames, utterance.timestamps, utterance.fps)
        if patches is None or face < 0.8:
            continue
        still.append(mouth_motion(patches[:STILL_FRAMES]))
        speaking.append(mouth_motion(patches))
    if not speaking:
        raise SystemExit("No recordings found under data/.")

    print(f"{len(speaking)} clips checked. A still mouth scores about 1; the cutoff is {cutoff}.")
    print(f"still (the pause before each sentence):  {summary(still)}")
    print(f"speaking (the whole clip):               {summary(speaking)}")
    wrong_still = sum(v >= cutoff for v in still)
    wrong_speaking = sum(v < cutoff for v in speaking)
    print(f"with the cutoff at {cutoff}: {wrong_still} of {len(still)} pauses would be read as speech, "
          f"{wrong_speaking} of {len(speaking)} sentences would be rejected as no words")


if __name__ == "__main__":
    main()
