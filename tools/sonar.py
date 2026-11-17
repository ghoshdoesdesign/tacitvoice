"""Experiment: can the laptop's own speakers and microphone sense a silent mouth?

    uv run python -m tools.sonar

Plays three tones too high to hear (18.5, 19.5 and 20.5 kHz) through the
speakers and listens with the microphone while you follow prompts on screen.
A moving surface in front of the laptop changes the echo of a steady tone, so
the question is whether a mouth, at laptop distance, changes it enough.

It answers three things, in about five minutes:

1. Do the tones come through this laptop's speakers and microphone at all?
2. Does mouthing change the echo more than sitting still does, and how does
   that compare with simply moving the head?
3. Can three different phrases be told apart from the echo alone? This is the
   one that matters: each phrase is mouthed several times, and each take is
   matched to its nearest other take.

Nothing is saved. Sound is analysed in memory, and only the narrow bands
around the tones are kept, which hold no speech.

Before starting: unplug headphones, set the volume to about 70%, and sit as
you normally do. Dogs and some young people can hear these tones.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

RATE = 48_000
TONES = (18_500.0, 19_500.0, 20_500.0)
BASE_RATE = 200  # samples per second after shifting a tone down to 0 Hz
BLOCK = RATE // BASE_RATE
TAKE_S = 4.0
EDGE_S = 0.5  # dropped from each end: fade-in, and the delay between speaker and mic
MOTION_BAND = (2.0, 40.0)  # Hz around the tone; a mouth moving at 10 cm/s shifts it by about 11 Hz
PHRASES = ("hello how are you today", "please open the window", "what time is the meeting")


def tone_signal(seconds: float, tones=TONES, rate: int = RATE, level: float = 0.25) -> np.ndarray:
    """The tones added together, fading in and out so the speakers do not click."""
    t = np.arange(int(seconds * rate)) / rate
    signal = sum(np.sin(2 * np.pi * f * t) for f in tones) * (level / len(tones))
    fade = int(0.05 * rate)
    ramp = np.linspace(0.0, 1.0, fade)
    signal[:fade] *= ramp
    signal[-fade:] *= ramp[::-1]
    return signal.astype(np.float32)


def baseband(audio: np.ndarray, tone: float, rate: int = RATE) -> np.ndarray:
    """What happened to one tone over time: a complex value 200 times a second.

    The steady part is the tone travelling straight from speaker to microphone
    and bouncing off things that do not move. Anything that moves makes the
    value wobble.

    Averaging over blocks of 1/200 s removes the other tones exactly, because
    they sit whole multiples of 200 Hz away.
    """
    t = np.arange(len(audio)) / rate
    shifted = audio * np.exp(-2j * np.pi * tone * t)
    blocks = len(shifted) // BLOCK
    out = shifted[: blocks * BLOCK].reshape(blocks, BLOCK).mean(axis=1)
    edge = int(EDGE_S * BASE_RATE)
    return out[edge:-edge]


def carrier_snr_db(audio: np.ndarray, tone: float) -> float:
    """How far the tone stands above the room's noise at a nearby empty frequency."""
    signal = np.abs(baseband(audio, tone).mean()) ** 2
    noise = np.mean(np.abs(baseband(audio, tone + 400.0)) ** 2)
    return float(10 * np.log10(max(signal, 1e-20) / max(noise, 1e-20)))


def motion_db(bb: np.ndarray) -> float:
    """Strength of the wobble (2-40 Hz either side of the tone) relative to the steady tone."""
    steady = np.abs(bb.mean()) ** 2
    spectrum = np.abs(np.fft.fft((bb - bb.mean()) * np.hanning(len(bb)))) ** 2
    freqs = np.abs(np.fft.fftfreq(len(bb), 1 / BASE_RATE))
    band = (freqs >= MOTION_BAND[0]) & (freqs <= MOTION_BAND[1])
    power = spectrum[band].sum() / (np.hanning(len(bb)) ** 2).sum() / len(bb)
    return float(10 * np.log10(max(power, 1e-20) / max(steady, 1e-20)))


def signature(audio: np.ndarray, tones) -> np.ndarray:
    """A take as a short film of its wobble: (frames, features), one row per 80 ms."""
    rows = []
    window, hop = 64, 16
    freqs = np.fft.fftfreq(window, 1 / BASE_RATE)
    keep = (np.abs(freqs) >= MOTION_BAND[0]) & (np.abs(freqs) <= MOTION_BAND[1])
    for tone in tones:
        bb = baseband(audio, tone)
        bb = (bb - bb.mean()) / max(np.abs(bb.mean()), 1e-12)
        frames = [bb[i : i + window] * np.hanning(window) for i in range(0, len(bb) - window, hop)]
        rows.append(np.log10(np.abs(np.fft.fft(frames, axis=1))[:, keep] + 1e-6))
    features = np.concatenate(rows, axis=1)
    return features - features.mean(axis=0)  # remove the room's steady pattern


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """How different two takes are, allowing one to be said faster or later than the other."""
    na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cost = 1.0 - na @ nb.T
    total = np.full((len(a) + 1, len(b) + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    return float(total[-1, -1] / (len(a) + len(b)))


def nearest_take_score(signatures: list[np.ndarray], labels: list[int]) -> int:
    """Number of takes whose closest other take is the same phrase."""
    right = 0
    for i, sig in enumerate(signatures):
        others = [(distance(sig, other), labels[j]) for j, other in enumerate(signatures) if j != i]
        right += min(others)[1] == labels[i]
    return right


def record_take(sd, seconds: float = TAKE_S) -> np.ndarray:
    audio = sd.playrec(tone_signal(seconds), samplerate=RATE, channels=1, dtype="float32", blocking=True)
    return audio[:, 0].astype(np.float64)


def prompt(text: str) -> None:
    print(f"\n{text}")
    for n in (3, 2, 1):
        print(f"  {n}", end="", flush=True)
        time.sleep(0.7)
    print("  GO", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=4, help="times each phrase is mouthed (default 4)")
    args = ap.parse_args()

    import sounddevice as sd

    out_name = sd.query_devices(kind="output")["name"]
    in_name = sd.query_devices(kind="input")["name"]
    print(f"Speaker: {out_name}\nMicrophone: {in_name}")
    print("These should be the laptop's own. If not, change them in System Settings > Sound and start again.")
    input("\nUnplug headphones, set the volume to about 70%, then press Return. ")

    print("\nStep 1 of 3: checking that the tones come through. Sit still.")
    audio = record_take(sd, 3.0)
    loudest = float(np.abs(audio).max())
    snr = {tone: carrier_snr_db(audio, tone) for tone in TONES}
    for tone, value in snr.items():
        print(f"  {tone / 1000:.1f} kHz: {value:5.1f} dB above the room's noise")
    usable = [tone for tone in TONES if snr[tone] >= 25.0]
    if loudest >= 0.99:
        print("  The microphone is overloading. Lower the volume a little and start again.")
        sys.exit(1)
    if not usable:
        print("\nRESULT: none of the tones reach the microphone clearly on this laptop (25 dB is needed).\n"
              "Raise the volume and try once more. If it stays like this, the speakers or the\n"
              "microphone filter these tones out and sonar is not possible on this machine.")
        sys.exit(1)
    print(f"  Using {len(usable)} of {len(TONES)} tones.")

    print("\nStep 2 of 3: still, mouthing, and head movement. Each take is 4 seconds.")
    conditions = [
        ("still", "Sit STILL with your mouth closed."),
        ("mouth", "Keep your head still and silently MOUTH: one two three four five six seven eight."),
        ("head", "Mouth closed. Gently NOD and turn your head a little, as you would while talking."),
    ]
    motion: dict[str, list[float]] = {name: [] for name, _ in conditions}
    for _ in range(3):
        for name, text in conditions:
            prompt(text)
            audio = record_take(sd)
            motion[name].append(float(np.mean([motion_db(baseband(audio, tone)) for tone in usable])))
    level = {name: float(np.median(values)) for name, values in motion.items()}

    print(f"\nStep 3 of 3: three phrases, {args.repeats} times each, silently. Keep your head as still as you can.")
    print("Start mouthing at GO and take about two seconds.")
    signatures, labels = [], []
    for _ in range(args.repeats):
        for label, phrase in enumerate(PHRASES):
            prompt(f'Silently mouth: "{phrase.upper()}"')
            signatures.append(signature(record_take(sd), usable))
            labels.append(label)
    right = nearest_take_score(signatures, labels)
    total = len(labels)

    print("\n" + "=" * 62)
    print("RESULT (copy everything from here down)")
    print("tones above noise: " + ", ".join(f"{t / 1000:.1f} kHz {snr[t]:.0f} dB" for t in TONES))
    for name in ("still", "mouth", "head"):
        print(f"movement in the echo, {name:5}: {level[name]:6.1f} dB   takes: "
              + " ".join(f"{v:.1f}" for v in motion[name]))
    print(f"mouthing vs still: {level['mouth'] - level['still']:+.1f} dB    head vs mouthing: "
          f"{level['head'] - level['mouth']:+.1f} dB")
    print(f"phrases told apart: {right} of {total} takes matched their own phrase "
          f"(guessing would get about {total // len(PHRASES)})")
    print("=" * 62)
    gain = level["mouth"] - level["still"]
    if gain < 3:
        print("Reading: mouthing barely changes the echo. Sonar at this distance looks like a dead end.")
    elif right >= 0.7 * total:
        print("Reading: the echo carries the mouth, and different phrases look different. Worth building on.")
    else:
        print("Reading: the echo senses movement, but these phrases are not clearly told apart yet.")


if __name__ == "__main__":
    main()
