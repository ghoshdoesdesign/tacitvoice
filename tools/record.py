"""Record yourself whispering or mouthing a list of prompts (run on the Mac).

    uv run python -m tools.record data/prompts.txt --mode whisper
    uv run python -m tools.record data/prompts.txt --mode silent

A window shows the camera and the next prompt. SPACE starts and stops a take,
R redoes the last one, S skips, Q quits. Clips are saved under data/baseline/
on this Mac and nowhere else; rerunning continues where you stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from server.phrases import read_lines

RATE = 16000
WIDTH, HEIGHT = 960, 540


def draw(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    view = cv2.flip(frame, 1)  # mirror the preview only; saved frames are not mirrored
    cv2.rectangle(view, (0, 0), (view.shape[1], 34 * len(lines) + 16), (0, 0, 0), -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(view, text, (14, 34 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    return view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", type=Path, help="text file, one prompt per line")
    ap.add_argument("--mode", choices=["whisper", "silent", "aloud"], required=True)
    ap.add_argument("--out", type=Path, default=Path("data/baseline"))
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    prompts = read_lines(args.prompts)
    if not prompts:
        raise SystemExit(f"No prompts in {args.prompts}")
    out = args.out / args.mode
    out.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        raise SystemExit("Cannot open the camera. Allow camera access for your terminal in System Settings.")

    audio: list[np.ndarray] = []
    recording = False

    def on_audio(indata, _frames, _time, _status) -> None:
        if recording:
            audio.append(indata[:, 0].copy())

    hint = {
        "whisper": "WHISPER it: breath only, no voice",
        "silent": "MOUTH it: no sound at all",
        "aloud": "Say it ALOUD, normally",
    }[args.mode]

    i = 0
    while i < len(prompts) and (out / f"{i:03d}.json").exists():
        i += 1

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=on_audio):
        frames: list[np.ndarray] = []
        stamps: list[float] = []
        misses = 0
        while i < len(prompts):
            ok, frame = cam.read()
            if not ok:
                misses += 1
                if misses > 100:
                    raise SystemExit(
                        f"Camera {args.camera} is sending no pictures. Try the other one: --camera "
                        f"{1 - args.camera if args.camera in (0, 1) else 0}"
                    )
                time.sleep(0.05)
                continue
            misses = 0
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            if recording:
                frames.append(frame)
                stamps.append(time.perf_counter())

            if recording:
                status = ("REC  -  SPACE to stop", (60, 60, 255))
            else:
                status = (f"{i + 1}/{len(prompts)}  {hint}  -  SPACE to start", (120, 255, 120))
            wrapped = [(line, (255, 255, 255)) for line in textwrap.wrap(prompts[i], 62)]
            cv2.imshow("TacitVoice recorder", draw(frame, [status] + wrapped))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" ") and not recording:
                frames, stamps = [], []
                audio.clear()
                recording = True
            elif key == ord(" ") and recording:
                recording = False
                if len(frames) < 10:
                    continue  # an accidental double tap
                writer = cv2.VideoWriter(
                    str(out / f"{i:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT)
                )
                for f in frames:
                    writer.write(f)
                writer.release()
                sound = np.concatenate(audio) if audio else np.zeros(1, dtype="float32")
                sf.write(out / f"{i:03d}.wav", sound, RATE)
                meta = {
                    "text": prompts[i],
                    "mode": args.mode,
                    "timestamps": [round(t - stamps[0], 4) for t in stamps],
                }
                # Written last: its presence marks the take as complete.
                (out / f"{i:03d}.json").write_text(json.dumps(meta))
                i += 1
            elif key == ord("s") and not recording:
                i += 1
            elif key == ord("r") and not recording and i > 0:
                i -= 1
                (out / f"{i:03d}.json").unlink(missing_ok=True)

        # Every take is already on disk. Leave from inside the audio stream:
        # on macOS, shutting the microphone or camera down can hang for good.
        done = len(list(out.glob("*.json")))
        print(f"{done} takes saved in {out}. Next: uv run python -m tools.evaluate", flush=True)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # the window only closes once its event loop runs again
        os._exit(0)


if __name__ == "__main__":
    main()
