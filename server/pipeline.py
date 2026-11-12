"""One utterance in, finished text out. Used by the server and the eval tool."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import config
from .cleanup import Cleaned, Cleaner, build_evidence, make_cleaner, raw_pick
from .lips import LipReader, LipResult
from .phrases import Phrasebook
from .stt import OpenAISTT, SttResult, WhisperSTT


@dataclass
class Utterance:
    frames: np.ndarray  # (T, H, W, 3) RGB uint8; may be empty
    audio: np.ndarray  # mono float32 at 16 kHz; may be empty
    timestamps: list[float] | None = None  # seconds, one per frame
    fps: float | None = None
    app: str = ""
    text_before_cursor: str = ""


@dataclass
class Outcome:
    id: str
    text: str
    alternatives: list[str]
    mode: str  # "whisper" | "silent" | "none"
    stt: SttResult
    lips: LipResult
    cleaned: Cleaned
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "alternatives": self.alternatives,
            "mode": self.mode,
            "warnings": self.warnings,
            "timingsMs": self.timings_ms,
            "raw": {
                "audio": self.stt.text,
                "audioConfidence": round(self.stt.confidence, 3),
                "peakDbfs": round(self.stt.peak_dbfs, 1),
                "activity": round(self.stt.activity, 2),
                "lips": [t for t, _ in self.lips.candidates],
                "faceFraction": round(self.lips.face_fraction, 2),
                "cleanup": self.cleaned.source,
            },
        }


class Pipeline:
    def __init__(self, lips: bool = True, cleanup: bool = True) -> None:
        self.errors: list[str] = []
        self.phrasebook = Phrasebook(config.DICTIONARY_FILE, config.PROMPTS_FILE)
        if config.STT_BACKEND == "openai":
            self.stt = OpenAISTT(config.STT_CLOUD_MODEL, config.SILENCE_DBFS, config.MIN_ACTIVITY)
        else:
            self.stt = WhisperSTT(
                config.STT_MODEL, config.STT_COMPUTE, config.SILENCE_DBFS, config.MIN_ACTIVITY
            )
        self.lips: LipReader | None = None
        if lips and config.LIPS_ENABLED:
            try:
                self.lips = LipReader(
                    config.AVSR_REPO,
                    config.AVSR_CONFIG,
                    config.LIPS_DEVICE,
                    config.LIPS_BEAM,
                    config.LIPS_NBEST,
                )
            except Exception as e:
                # Whisper mode still works without the lip model, so keep going
                # and report it on /health instead of refusing to start.
                self.errors.append(f"lip reading unavailable: {e}")
        self.cleaner: Cleaner | None = None
        if cleanup and config.CLEANUP_ENABLED:
            self.cleaner = make_cleaner(
                config.CLEANUP_PROVIDER, config.CLEANUP_MODEL, config.CLEANUP_TIMEOUT_S
            )
        self._pool = ThreadPoolExecutor(max_workers=2)

    def run(self, u: Utterance, log: bool = True) -> Outcome:
        t0 = time.perf_counter()
        self.phrasebook.reload()

        # Audio and lips are independent, so run them side by side.
        stt_job = self._pool.submit(self.stt.transcribe, u.audio, self.phrasebook.glossary)
        if self.lips is not None and len(u.frames):
            lips_job = self._pool.submit(self.lips.transcribe, u.frames, u.timestamps, u.fps)
        else:
            lips_job = None
        stt = stt_job.result()
        warnings: list[str] = []
        if stt.note:
            warnings.append(stt.note)
        try:
            lips = lips_job.result() if lips_job else LipResult()
        except Exception as e:
            lips = LipResult()
            warnings.append(f"lip reading failed: {e}")
        t1 = time.perf_counter()

        if lips_job and lips.frames >= 5:
            if lips.face_fraction == 0:
                warnings.append("No face found. Face the camera.")
            elif lips.face_fraction < 0.5:
                warnings.append("Face lost for much of the clip. Check lighting and camera angle.")

        candidates = ([stt.text] if stt.audible else []) + [t for t, _ in lips.candidates]
        similar = self.phrasebook.similar(candidates)
        mode = "whisper" if stt.audible else ("silent" if lips.candidates else "none")

        cleaned = raw_pick(stt, lips, similar)
        if cleaned.text and self.cleaner is not None:
            evidence = build_evidence(
                stt, lips, self.phrasebook.glossary, similar, u.app, u.text_before_cursor
            )
            cleaned = self.cleaner.clean(evidence, cleaned)
            if cleaned.note:
                warnings.append(cleaned.note)
        t2 = time.perf_counter()

        timings = {
            **stt.timings_ms,
            **lips.timings_ms,
            "recognise": int((t1 - t0) * 1000),
            "cleanup": cleaned.ms,
            "total": int((t2 - t0) * 1000),
        }
        outcome = Outcome(
            id=uuid.uuid4().hex[:12],
            text=cleaned.text,
            alternatives=cleaned.alternatives,
            mode=mode,
            stt=stt,
            lips=lips,
            cleaned=cleaned,
            warnings=warnings,
            timings_ms=timings,
        )
        if log:
            append_log({"event": "utterance", "app": u.app, **outcome.to_json()})
        return outcome


def append_log(record: dict) -> None:
    """Text-only usage log on this device: feeds the latency and habit metrics."""
    config.HOME.mkdir(parents=True, exist_ok=True)
    record = {"ts": round(time.time(), 3), **record}
    with config.LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
