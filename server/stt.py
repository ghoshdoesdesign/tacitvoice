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


class WhisperSTT:
    def __init__(self, model: str, compute_type: str = "int8", silence_dbfs: float = -50.0) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device="auto", compute_type=compute_type)
        self.silence_dbfs = silence_dbfs

    def transcribe(self, audio: np.ndarray, glossary: list[str] | None = None) -> SttResult:
        """audio: mono float32 at 16 kHz."""
        result = SttResult(peak_dbfs=peak_dbfs(audio))
        if result.peak_dbfs < self.silence_dbfs:
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
        result.audible = bool(result.text)
        return result


class OpenAISTT:
    """Cloud speech-to-text. Sends the clip's audio (never video) to OpenAI."""

    def __init__(self, model: str, silence_dbfs: float = -50.0, timeout_s: float = 20.0) -> None:
        import openai

        self._sdk = openai
        self._client = openai.OpenAI(timeout=timeout_s, max_retries=0)  # reads OPENAI_API_KEY
        self.model = model
        self.silence_dbfs = silence_dbfs

    def transcribe(self, audio: np.ndarray, glossary: list[str] | None = None) -> SttResult:
        import soundfile as sf

        result = SttResult(peak_dbfs=peak_dbfs(audio))
        if result.peak_dbfs < self.silence_dbfs:
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
        result.audible = bool(result.text)
        return result
