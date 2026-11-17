"""Compare ways of telling a still mouth from a speaking one (run on the Mac).

    uv run python -m tools.motion

Every recorded clip has a short pause before and after the sentence, so the
recordings already hold both cases: the first and last third of a second are
a still mouth, and the whole clip is a speaking one. This scores both with
several candidate measures and prints how well each separates them, so the
"No words detected" check can be built on the one that works on real faces.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from server import config
from server.lips import MouthCropper, mouth_motion
from tools.evaluate import load_clip

STILL_FRAMES = 9  # about a third of a second at 25 frames per second


def change(patches: np.ndarray) -> np.ndarray:
    """Per-pixel change between frames two apart, with noise and flicker reduced."""
    smooth = np.stack([cv2.blur(p, (7, 7)) for p in patches]).astype(np.float32)
    smooth -= smooth.mean(axis=(1, 2), keepdims=True)
    return np.abs(smooth[2:] - smooth[:-2])


def amount(patches: np.ndarray) -> float:
    """Mouth movement in grey levels: depends on the camera, but simple."""
    return float(np.percentile(change(patches)[:, 28:84, 12:84].mean(axis=(1, 2)), 90))


def swing(patches: np.ndarray) -> float:
    """Busiest moments against quietest ones in the same clip: speech comes in bursts."""
    mouth = change(patches)[:, 28:84, 12:84].mean(axis=(1, 2))
    return float(np.percentile(mouth, 90) / max(float(np.percentile(mouth, 20)), 0.02))


def centre_vs_edge(patches: np.ndarray) -> float:
    """Lips against the patch's outer border (cheeks and chin edge), not the nose strip."""
    c = change(patches)
    lips = c[:, 34:66, 24:72].mean(axis=(1, 2))
    border = np.concatenate([c[:, :, :8].reshape(len(c), -1), c[:, :, -8:].reshape(len(c), -1)], axis=1).mean(axis=1)
    return float(np.percentile(lips, 90) / max(float(np.median(border)), 0.05))


def nose_ratio(patches: np.ndarray) -> float:
    """Mouth movement against the strip under the nose. Overlapped badly on real clips."""
    c = change(patches)
    mouth, above = c[:, 28:84, 12:84].mean(axis=(1, 2)), c[:, 0:14, 16:80].mean(axis=(1, 2))
    return float(np.percentile(mouth, 90) / max(float(np.median(above)), 0.05))


def opening_by_contrast(patches: np.ndarray) -> float:
    """The measure in use, divided by the picture's own contrast.

    The measure in use is in grey levels, so a dim or washed-out picture
    would score lower for the same mouth. Dividing by how much brightness
    varies across the patch should make one cutoff hold in any lighting.
    Listed here to be checked on other people's recordings.
    """
    contrast = float(np.median([p.std() for p in patches]))
    return 100.0 * mouth_motion(patches) / max(contrast, 1.0)


MEASURES = {
    "mouth opening (in use)": mouth_motion,
    "opening / contrast x100": opening_by_contrast,
    "lips against border": centre_vs_edge,
    "amount of movement": amount,
    "swing within the clip": swing,
    "ratio to nose strip": nose_ratio,
}


def best_cutoff(still: np.ndarray, speaking: np.ndarray) -> tuple[float, float, float]:
    """Cutoff with the fewest mistakes, and the share of each group it gets wrong."""
    best = (2.0, 0.0, 1.0, 1.0)
    for cut in np.unique(np.concatenate([still, speaking])):
        passed, rejected = float(np.mean(still >= cut)), float(np.mean(speaking < cut))
        if passed + rejected < best[0]:
            best = (passed + rejected, float(cut), passed, rejected)
    return best[1], best[2], best[3]


def main() -> None:
    crop = MouthCropper(config.AVSR_REPO)
    folders = [Path("data/baseline"), Path("data/train"), Path("data/train2")]
    stems = [m.with_suffix("") for f in folders for m in sorted(f.glob("*/*.json"))]
    still = {name: [] for name in MEASURES}
    speaking = {name: [] for name in MEASURES}
    clips = 0
    for stem in stems[:: max(1, len(stems) // 150)]:  # a sample is enough
        utterance, _ = load_clip(stem)
        if len(utterance.frames) < 50:
            continue
        patches, face = crop(utterance.frames, utterance.timestamps, utterance.fps)
        if patches is None or face < 0.8:
            continue
        clips += 1
        for name, measure in MEASURES.items():
            still[name] += [measure(patches[:STILL_FRAMES]), measure(patches[-STILL_FRAMES:])]
            speaking[name].append(measure(patches))
    if not clips:
        raise SystemExit("No recordings found under data/.")

    print(f"\n{clips} clips checked. For each measure: the still pauses, the spoken sentences, and the")
    print("cutoff that makes the fewest mistakes. Good separation = both mistake columns near 0%.\n")
    print(f"{'measure':30} {'still: low / typical / high':>28} {'speaking: low / typical / high':>31} "
          f"{'cutoff':>7} {'still read':>11} {'speech lost':>12}")
    for name in MEASURES:
        s, t = np.array(still[name]), np.array(speaking[name])
        cut, passed, rejected = best_cutoff(s, t)
        low = lambda v: np.percentile(v, 10)   # noqa: E731
        high = lambda v: np.percentile(v, 90)  # noqa: E731
        print(f"{name:30} {low(s):8.2f} / {np.median(s):6.2f} / {high(s):6.2f} "
              f"{low(t):11.2f} / {np.median(t):6.2f} / {high(t):6.2f} {cut:7.2f} {passed:10.0%} {rejected:11.0%}")
    print("\nlow and high are the 10th and 90th percentiles. Paste this table back.")


if __name__ == "__main__":
    main()
