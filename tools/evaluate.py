"""Measure error rates and speed on your own recordings (run on the Mac).

    uv run python -m tools.evaluate                  # everything in data/baseline
    uv run python -m tools.evaluate --mode silent --limit 20

Prints a table and writes data/baseline/results.json. Only numbers and text
are produced; the recordings never leave this machine. With cleanup on, the
candidate transcripts (text) are sent to the cleanup provider's API.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from server import config
from server.phrases import Phrasebook
from server.pipeline import Pipeline, Utterance
from server.stt import resample
from server.text import corpus_wer, word_error_rate

STAGES = ("stt", "face", "encode", "beam", "cleanup", "total")


def load_clip(stem: Path) -> tuple[Utterance, str]:
    meta = json.loads(stem.with_suffix(".json").read_text())
    cap = cv2.VideoCapture(str(stem.with_suffix(".mp4")))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    audio, rate = sf.read(stem.with_suffix(".wav"), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    stamps = meta.get("timestamps")
    if not stamps or len(stamps) != len(frames):
        stamps = None
    video = np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)
    return Utterance(frames=video, audio=resample(audio, rate), timestamps=stamps, fps=30.0), meta["text"]


def summarise(rows: list[dict]) -> dict:
    def wer(pick) -> float:
        return corpus_wer([(r["truth"], pick(r)) for r in rows])

    def median(key: str) -> int:
        values = [r["timings_ms"][key] for r in rows if key in r["timings_ms"]]
        return int(statistics.median(values)) if values else 0

    def best_lip(r: dict) -> str:
        return min(r["lips"], key=lambda t: word_error_rate(r["truth"], t)) if r["lips"] else ""

    return {
        "clips": len(rows),
        "lips_ran": any(r["lips"] for r in rows),
        "wer_audio_only": wer(lambda r: r["audio"]),
        "wer_lips_top1": wer(lambda r: r["lips"][0] if r["lips"] else ""),
        # Best of the candidates: the ceiling a perfect chooser could reach.
        "wer_lips_best_of_n": wer(best_lip),
        "wer_final": wer(lambda r: r["final"]),
        "exact_final": sum(word_error_rate(r["truth"], r["final"]) == 0 for r in rows) / len(rows),
        "detected_as": {m: sum(r["mode_detected"] == m for r in rows) for m in ("whisper", "silent", "none")},
        "median_ms": {k: median(k) for k in STAGES},
        "median_clip_seconds": statistics.median(r["seconds"] for r in rows),
    }


def print_table(results: dict) -> None:
    print(f"\n{'mode':8} {'clips':>5} {'audio':>7} {'lips':>7} {'lips*':>7} {'final':>7} {'exact':>7} {'total ms':>9}")
    for mode, r in results.items():
        lips = f"{r['wer_lips_top1']:7.1%} {r['wer_lips_best_of_n']:7.1%}" if r["lips_ran"] else "      -       -"
        print(
            f"{mode:8} {r['clips']:5d} {r['wer_audio_only']:7.1%} {lips} "
            f"{r['wer_final']:7.1%} {r['exact_final']:7.1%} {r['median_ms']['total']:9d}"
        )
    print("Word error rates; lower is better. lips* = best of the lip candidates. exact = clips with no errors.")
    for mode, r in results.items():
        print(f"{mode}: detected as {r['detected_as']}; median ms by stage {r['median_ms']} "
              f"(clips ~{r['median_clip_seconds']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/baseline"))
    ap.add_argument("--mode", choices=["whisper", "silent", "aloud"], action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument(
        "--known-prompts",
        action="store_true",
        help="let the matcher use ~/.tacitvoice/prompts.txt. Off by default: if the test "
        "prompts are in that file, the score measures lookup, not recognition.",
    )
    args = ap.parse_args()

    modes = args.mode or ["whisper", "silent", "aloud"]
    clips = {m: sorted(p.with_suffix("") for p in (args.data / m).glob("*.json")) for m in modes}
    if args.limit:
        clips = {m: stems[: args.limit] for m, stems in clips.items()}
    if not any(clips.values()):
        raise SystemExit(f"No recordings under {args.data}. Run tools.record first.")

    pipe = Pipeline(cleanup=not args.no_cleanup)
    print(f"speech-to-text: {config.STT_BACKEND}; cleanup: {config.CLEANUP_PROVIDER} {config.CLEANUP_MODEL}")
    if pipe.lips is not None:
        print(f"lip model weights: {pipe.lips.weights}")
    for problem in pipe.errors:
        print("!", problem)
    if not args.known_prompts:
        pipe.phrasebook = Phrasebook(config.DICTIONARY_FILE, Path("/nonexistent/prompts.txt"))

    results: dict = {}
    for mode, stems in clips.items():
        rows = []
        for n, stem in enumerate(stems, 1):
            utterance, truth = load_clip(stem)
            o = pipe.run(utterance, log=False)
            rows.append({
                "clip": stem.name,
                "truth": truth,
                "final": o.text,
                "audio": o.stt.text,
                "lips": [t for t, _ in o.lips.candidates],
                "mode_detected": o.mode,
                "activity": round(o.stt.activity, 2),
                "seconds": round(len(utterance.audio) / 16000, 2),
                "timings_ms": o.timings_ms,
                "warnings": o.warnings,
            })
            print(f"[{mode} {n}/{len(stems)}] {word_error_rate(truth, o.text):4.0%}  {o.text!r}")
            for warning in o.warnings:
                print(f"      ! {warning}")
        if rows:
            results[mode] = {**summarise(rows), "rows": rows}

    print_table(results)
    out = args.data / "results.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"Details (includes your prompt text): {out}")


if __name__ == "__main__":
    main()
