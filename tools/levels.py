"""Print how loud each recorded clip is, to set the whisper/silent threshold.

    uv run python -m tools.levels

Takes a few seconds. Shows numbers only.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from server import config
from server.stt import peak_dbfs, resample


def main() -> None:
    data = Path("data/baseline")
    for mode in ("whisper", "silent", "aloud"):
        levels = []
        for wav in sorted((data / mode).glob("*.wav")):
            audio, rate = sf.read(wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            levels.append(round(peak_dbfs(resample(audio, rate)), 1))
        if levels:
            print(f"{mode:8} min {min(levels):6.1f}  max {max(levels):6.1f}  dBFS   {levels}")
    print(f"current threshold: {config.SILENCE_DBFS} dBFS (clips below it count as silent)")


if __name__ == "__main__":
    main()
