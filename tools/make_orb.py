"""Turn a video of a glowing shape on black into the app's listening animation.

    uv run python -m tools.make_orb ~/Downloads/your-video.mp4

Writes transparent PNG frames to ~/.tacitvoice/orb/. The app plays them at the
bottom centre of the screen while the key is held. The video itself is not
part of this repository: use one you have the right to use.

How it works: the black background becomes transparent, the glowing rim keeps
its colours, and the inside of the shape stays mostly opaque so it reads as a
solid bubble on a light page instead of a hollow ring.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from server import config


def silhouette(frame: np.ndarray, rim: int) -> np.ndarray:
    """Filled outline of the shape: 1.0 inside, 0.0 outside, soft at the edge.

    Built from the bright rim only. The faint glow around the shape is as dim
    as its inside, so brightness alone cannot tell the two apart; the hull of
    the rim can.
    """
    bright = (frame.max(axis=2) > rim).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    points = cv2.findNonZero(bright)
    mask = np.zeros(bright.shape, np.uint8)
    if points is not None and len(points) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32) / 255.0


def key_frame(frame: np.ndarray, body: float, rim: int) -> np.ndarray:
    """BGR on black -> BGRA with straight (not premultiplied) alpha."""
    rgb = frame.astype(np.float32) / 255.0
    light = rgb.max(axis=2)  # how much light the pixel adds over black
    glow = np.clip((light - 0.10) * 1.7, 0, 1)  # drops the dim haze, keeps the rim
    alpha = np.maximum(glow, body * silhouette(frame, rim))
    # The source is light over black, i.e. already multiplied by its own
    # brightness. Dividing by alpha restores the colour for normal blending.
    straight = np.clip(rgb / np.maximum(alpha, 1e-3)[..., None], 0, 1)
    return np.dstack([straight, alpha]) * 255.0


def bounds(frames: list[np.ndarray], threshold: int) -> tuple[int, int, int]:
    """Centre and half-size of a square that holds the shape in every frame."""
    union = np.zeros(frames[0].shape[:2], bool)
    for frame in frames[:: max(1, len(frames) // 60)]:
        union |= frame.max(axis=2) > threshold
    ys, xs = np.nonzero(union)
    if not len(xs):
        raise SystemExit("The video looks completely black.")
    cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
    half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.12)  # room for the glow
    return cx, cy, half


def read_frames(video: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        raise SystemExit(f"Could not read frames from {video}")
    return frames, fps


def write_frames(
    frames: list[np.ndarray], out: Path, box: tuple[int, int, int], fps: float,
    size: int = 240, body: float = 0.78, rim: int = 95, loop_start: int | None = None,
) -> float:
    """Key, crop to `box` (centre x, centre y, half-size) and save. Returns megabytes written."""
    cx, cy, half = box
    h, w = frames[0].shape[:2]
    pad = max(0, half - cx, half - cy, cx + half - w, cy + half - h)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, frame in enumerate(frames):
        if pad:
            frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = frame[cy + pad - half : cy + pad + half, cx + pad - half : cx + pad + half]
        small = cv2.resize(key_frame(crop, body, rim), (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{i:04d}.png"), small.astype(np.uint8))
    (out / "fps.txt").write_text(f"{fps:.3f}\n")
    if loop_start is not None:
        # The app plays frames before this index once, then repeats the rest.
        (out / "loop.txt").write_text(f"{loop_start}\n")
    return sum(p.stat().st_size for p in out.glob("*.png")) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=config.HOME / "orb")
    ap.add_argument("--size", type=int, default=240, help="frame width and height in pixels")
    ap.add_argument("--body", type=float, default=0.78, help="how solid the inside is, 0 (hollow) to 1")
    ap.add_argument("--rim", type=int, default=95, help="brightness (0-255) of the shape's bright edge")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()

    frames, fps = read_frames(args.video)
    kept = frames[:: args.step]
    megabytes = write_frames(kept, args.out, bounds(frames, 28), fps / args.step, args.size, args.body, args.rim)
    print(f"Wrote {len(kept)} frames ({megabytes:.0f} MB) at {fps / args.step:.0f} per second to {args.out}")
    print("Quit and reopen TacitVoice to use the new animation.")


if __name__ == "__main__":
    main()
