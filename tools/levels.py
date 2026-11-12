"""Print how loud each recorded clip is, to set the whisper/silent threshold.

    uv run python -m tools.levels

Takes a few seconds. Shows numbers only.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from server import config
from server.stt import activity, peak_dbfs, resample


def main() -> None:
    data = Path("data/baseline")
    for mode in ("whisper", "silent", "aloud"):
        levels, busy = [], []
        for wav in sorted((data / mode).glob("*.wav")):
            audio, rate = sf.read(wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = resample(audio, rate)
            levels.append(round(peak_dbfs(audio), 1))
            busy.append(round(activity(audio), 2))
        if levels:
            print(f"{mode:8} loudness min {min(levels):6.1f}  max {max(levels):6.1f} dBFS")
            print(f"{mode:8} activity min {min(busy):6.2f}  max {max(busy):6.2f}   {busy}")
    print(f"a clip counts as a whisper when loudness >= {config.SILENCE_DBFS} dBFS "
          f"and activity >= {config.MIN_ACTIVITY}")


if __name__ == "__main__":
    main()
