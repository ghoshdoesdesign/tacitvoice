"""Speech-to-text for the whisper channel.

Two backends: Whisper run on this Mac (private, slow on a CPU) or OpenAI's
transcription API (fast, but the audio of each utterance leaves the device).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class SttResult:
    text: str = ""
    words: list[tuple[str, float]] = field(default_factory=list)  # (word, probability)
    peak_dbfs: float = -120.0
    activity: float = 0.0  # share of the clip above its own noise floor
    audible: bool = False
    note: str = ""  # why there is no text, when something went wrong
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return float(np.mean([p for _, p in self.words])) if self.words else 0.0


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    n = int(round(len(audio) * SAMPLE_RATE / rate))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def peak_dbfs(audio: np.ndarray, window_s: float = 0.1) -> float:
    """Level of the loudest 100 ms. A whisper is short bursts, so the mean misleads."""
    win = int(SAMPLE_RATE * window_s)
    if len(audio) < win:
        return -120.0
    n = len(audio) // win
    rms = np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))
    return float(20 * np.log10(max(float(rms.max()), 1e-6)))


def activity(audio: np.ndarray, frame_s: float = 0.025, rise_db: float = 8.0) -> float:
    """Share of the clip that stands clearly above its own background noise.

    Loudness alone cannot tell a whisper from silent mouthing: in a normal room
    the two overlap (a key click or the room's hum is as loud as a whisper).
    What differs is how much of the clip is busy. A whisper lifts the level
    above the room's floor for a large part of the clip; a silent clip does so
    only for the odd click. Measured on a pre-emphasised signal, which drops
    low hum and keeps the hiss that whispering is made of.
    """
    frame = int(SAMPLE_RATE * frame_s)
    if len(audio) < frame * 10:
        return 0.0
    x = np.diff(audio.astype(np.float32), prepend=audio[:1])
    n = len(x) // frame
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    floor = np.percentile(db, 10)
    return float(np.mean(db > floor + rise_db))


# What speech models tend to output when given no speech at all.
_PHANTOMS = {
    "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
    "i'll see you next time", "see you next time", "bye", "you", "so", "here we go", "let's go",
    "so let's go", "okay", "",
}


_PHANTOM_PARTS = ("for watching", "next video", "subscribe", "see you next time")


def is_phantom(text: str) -> bool:
    from .text import normalise

    text = normalise(text)
    return text in _PHANTOMS or any(part in text for part in _PHANTOM_PARTS)


def gate(result: "SttResult", audio: np.ndarray, silence_dbfs: float, min_activity: float) -> bool:
    """Fill in the clip's levels; return True if it is worth transcribing."""
    result.peak_dbfs = peak_dbfs(audio)
    result.activity = activity(audio)
    return result.peak_dbfs >= silence_dbfs and result.activity >= min_activity


class WhisperSTT:
    def __init__(
        self, model: str, compute_type: str = "int8", silence_dbfs: float = -50.0, min_activity: float = 0.12
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device="auto", compute_type=compute_type)
        self.silence_dbfs = silence_dbfs
        self.min_activity = min_activity

    def transcribe(self, audio: np.ndarray, glossary: list[str] | None = None) -> SttResult:
        """audio: mono float32 at 16 kHz."""
        result = SttResult()
        if not gate(result, audio, self.silence_dbfs, self.min_activity):
            return result  # silent mouthing: Whisper would hallucinate here

        t0 = time.perf_counter()
        # No voice-activity filter: those are trained on voiced speech and can
        # throw a whisper away as noise.
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=("Glossary: " + ", ".join(glossary[:80])) if glossary else None,
        )
        segments = list(segments)
        result.timings_ms["stt"] = int((time.perf_counter() - t0) * 1000)

        kept = [s for s in segments if not (s.no_speech_prob > 0.8 and s.avg_logprob < -1.0)]
        result.text = " ".join(s.text.strip() for s in kept).strip()
        result.words = [(w.word.strip(), float(w.probability)) for s in kept for w in (s.words or [])]
        if is_phantom(result.text):
            result.text, result.words = "", []
        result.audible = bool(result.text)
        return result


class OpenAISTT:
    """Cloud speech-to-text. Sends the clip's audio (never video) to OpenAI."""

    def __init__(
        self, model: str, silence_dbfs: float = -50.0, min_activity: float = 0.12, timeout_s: float = 20.0
    ) -> None:
        import openai

        self._sdk = openai
        self._client = openai.OpenAI(timeout=timeout_s, max_retries=0)  # reads OPENAI_API_KEY
        self.model = model
        self.silence_dbfs = silence_dbfs
        self.min_activity = min_activity

    def transcribe(self, audio: np.ndarray, glossary: list[str] | None = None) -> SttResult:
        import soundfile as sf

        result = SttResult()
        if not gate(result, audio, self.silence_dbfs, self.min_activity):
            return result

        wav = io.BytesIO()
        sf.write(wav, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        extra = {"prompt": "Glossary: " + ", ".join(glossary[:80])} if glossary else {}
        t0 = time.perf_counter()
        try:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", wav.getvalue(), "audio/wav"),
                language="en",
                response_format="json",
                **extra,
            )
        except self._sdk.APIError as e:
            status = getattr(e, "status_code", None)
            result.note = f"speech-to-text failed: OpenAI error {status or type(e).__name__}"
            return result
        result.timings_ms["stt"] = int((time.perf_counter() - t0) * 1000)
        result.text = (response.text or "").strip()
        if is_phantom(result.text):
            result.text = ""
        result.audible = bool(result.text)
        return result
