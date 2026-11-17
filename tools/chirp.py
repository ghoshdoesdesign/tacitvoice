"""Experiment: sonar with sweeps, keeping only the echo from where the face is.

    uv run python -m tools.chirp

tools.sonar played steady tones and found only a faint trace of the mouth:
with a steady tone, the echo from the face is mixed with the sound going
straight from speaker to microphone and with every other surface in the room.

This plays a rapid sweep from 18 to 22 kHz, fifty times a second. A sweep
tells echoes apart by how long they took to return, so the result is a
picture of the room by distance, redrawn fifty times a second, and the part
20-90 cm away (the face) can be looked at on its own.

It answers, in about eight minutes:

1. At which distances does the echo change when you mouth, compared with
   sitting still and with moving your head?
2. Can three phrases be told apart from the echo at face distance?
3. The control: can they still be told apart when you only THINK the phrase
   with your mouth closed? If so, the echo is following something other than
   the mouth (head, breathing, timing) and the answer to 2 means little.

Nothing is saved. Only the band above 18 kHz is analysed, which holds no speech.

Before starting: unplug headphones, set the volume to about 70%, sit as you
normally do. You may hear a faint buzz or hiss; dogs can hear the sweep.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from tools.sonar import PHRASES, nearest_take_score, prompt

RATE = 48_000
F_LOW, F_HIGH = 18_000.0, 22_000.0
FRAME = 960  # samples per sweep: 50 sweeps a second, echoes up to 3.4 m of path kept apart
FPS = RATE / FRAME
TAKE_S = 4.0
EDGE_S = 0.5  # dropped from each end of a take
SOUND_M_S = 343.0
SPEAKER_TO_MIC_M = 0.25  # rough path of the direct sound inside the laptop
FACE_M = (0.2, 0.9)  # distances treated as "where the face is"
PROFILE_STEP_M = 0.1


def sweep(level: float = 0.25) -> np.ndarray:
    """One 20 ms sweep from 18 to 22 kHz, faded at both ends so that repeating it does not click."""
    t = np.arange(FRAME) / RATE
    phase = 2 * np.pi * (F_LOW * t + (F_HIGH - F_LOW) / (2 * FRAME / RATE) * t**2)
    taper = np.ones(FRAME)
    edge = FRAME // 10
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    taper[:edge], taper[-edge:] = ramp, ramp[::-1]
    return (level * np.sin(phase) * taper).astype(np.float32)


def echo_profiles(audio: np.ndarray) -> np.ndarray:
    """(sweeps, FRAME) complex: for each sweep, the echo strength at each delay.

    Column 0 is the sound that went straight from speaker to microphone;
    column n is whatever arrived n samples later. Each row is divided by its
    own direct sound, which cancels volume changes and clock drift.
    """
    sent = np.fft.fft(sweep(1.0).astype(np.float64))
    freqs = np.fft.fftfreq(FRAME, 1 / RATE)
    band = (freqs >= F_LOW) & (freqs <= F_HIGH)  # positive frequencies only, so the result is complex
    window = np.zeros(FRAME)
    window[band] = np.hanning(band.sum())  # trades a little sharpness for much less smearing between delays
    inverse = np.zeros(FRAME, dtype=complex)
    inverse[band] = window[band] / sent[band]

    count = len(audio) // FRAME
    frames = audio[: count * FRAME].reshape(count, FRAME)
    edge = int(EDGE_S * FPS)
    frames = frames[edge:-edge]
    profiles = np.fft.ifft(np.fft.fft(frames, axis=1) * inverse, axis=1)
    direct = int(np.argmax(np.abs(profiles.mean(axis=0))))
    profiles = np.roll(profiles, -direct, axis=1)
    return profiles / profiles[:, :1]


def direct_snr_db(profiles: np.ndarray) -> float:
    """How far the direct sound stands above the emptiest part of the picture."""
    strength = np.abs(profiles.mean(axis=0))
    return float(20 * np.log10(strength[0] / max(np.median(strength[FRAME // 2 :]), 1e-12)))


def column(distance_m: float) -> int:
    """Delay, in samples after the direct sound, of an echo from this far in front of the laptop."""
    return max(1, int(round((2 * distance_m - SPEAKER_TO_MIC_M) / SOUND_M_S * RATE)))


def movement(profiles: np.ndarray) -> np.ndarray:
    """Per delay: how much the echo changed during the take, relative to the direct sound (power)."""
    return np.mean(np.abs(profiles - profiles.mean(axis=0)) ** 2, axis=0)


def band_db(power: np.ndarray, near_m: float, far_m: float) -> float:
    return float(10 * np.log10(max(power[column(near_m) : column(far_m)].mean(), 1e-20)))


def signature(profiles: np.ndarray) -> np.ndarray:
    """A take as a short film of the echo at face distance: (frames, 5 cm steps), 25 rows a second.

    Uses how much each distance changed over 60 ms, not the echo itself: the
    echo's exact phase shifts with a millimetre of head position and would
    never repeat between takes.
    """
    change = np.abs(profiles[3:] - profiles[:-3])
    edges = [column(d) for d in np.arange(FACE_M[0], FACE_M[1] + 0.001, 0.05)]
    steps = np.stack([change[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    steps = steps[: len(steps) // 2 * 2].reshape(-1, 2, steps.shape[1]).mean(axis=1)
    features = np.log10(steps + 1e-7)
    return features - features.mean(axis=0)


def record_take(sd, seconds: float = TAKE_S) -> np.ndarray:
    signal = np.tile(sweep(), int(seconds * FPS))
    audio = sd.playrec(signal, samplerate=RATE, channels=1, dtype="float32", blocking=True)
    return audio[:, 0].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5, help="times each phrase is done, mouthed and again mouth-closed (default 5)")
    args = ap.parse_args()

    import sounddevice as sd

    print(f"Speaker: {sd.query_devices(kind='output')['name']}\nMicrophone: {sd.query_devices(kind='input')['name']}")
    print("These should be the laptop's own. If not, change them in System Settings > Sound and start again.")
    input("\nUnplug headphones, set the volume to about 70%, then press Return. ")

    print("\nStep 1 of 4: checking that the sweep comes through. Sit still.")
    audio = record_take(sd, 3.0)
    if float(np.abs(audio).max()) >= 0.99:
        print("  The microphone is overloading. Lower the volume a little and start again.")
        sys.exit(1)
    snr = direct_snr_db(echo_profiles(audio))
    print(f"  Sweep is {snr:.0f} dB above the noise.")
    if snr < 25:
        print("\nRESULT: the sweep does not reach the microphone clearly (25 dB is needed). Raise the volume and try again.")
        sys.exit(1)

    print("\nStep 2 of 4: still, mouthing, and head movement. Each take is 4 seconds.")
    conditions = [
        ("still", "Sit STILL with your mouth closed."),
        ("mouth", "Keep your head still and silently MOUTH: one two three four five six seven eight."),
        ("head", "Mouth closed. Gently NOD and turn your head a little, as you would while talking."),
    ]
    takes: dict[str, list[np.ndarray]] = {name: [] for name, _ in conditions}
    for _ in range(3):
        for name, text in conditions:
            prompt(text)
            takes[name].append(movement(echo_profiles(record_take(sd))))
    power = {name: np.median(np.stack(rows), axis=0) for name, rows in takes.items()}

    def phrase_round(instruction: str) -> tuple[int, int]:
        signatures, labels = [], []
        for _ in range(args.repeats):
            for label, phrase in enumerate(PHRASES):
                prompt(f'{instruction}: "{phrase.upper()}"')
                signatures.append(signature(echo_profiles(record_take(sd))))
                labels.append(label)
        return nearest_take_score(signatures, labels), len(labels)

    print(f"\nStep 3 of 4: three phrases, {args.repeats} times each, silently. Head as still as you can.")
    print("Start mouthing at GO and take about two seconds.")
    mouthed, total = phrase_round("Silently MOUTH")

    print(f"\nStep 4 of 4: the same again with your MOUTH CLOSED. Say each phrase in your head,")
    print("at the same pace, starting at GO. Do not move your lips or jaw.")
    closed, _ = phrase_round("Mouth CLOSED, think")

    print("\n" + "=" * 66)
    print("RESULT (copy everything from here down)")
    print(f"sweep above noise: {snr:.0f} dB")
    print("movement in the echo by distance from the laptop (dB, higher = more):")
    print("  distance     still   mouth    head   mouth-still")
    best = (-99.0, 0.0)
    for near in np.arange(0.1, 1.2, PROFILE_STEP_M):
        far = near + PROFILE_STEP_M
        s, m, h = (band_db(power[name], near, far) for name in ("still", "mouth", "head"))
        best = max(best, (m - s, near))
        print(f"  {near * 100:3.0f}-{far * 100:3.0f} cm  {s:6.1f}  {m:6.1f}  {h:6.1f}   {m - s:+6.1f}")
    face = {name: band_db(power[name], *FACE_M) for name in power}
    print(f"at face distance ({FACE_M[0] * 100:.0f}-{FACE_M[1] * 100:.0f} cm): mouthing vs still "
          f"{face['mouth'] - face['still']:+.1f} dB, head vs mouthing {face['head'] - face['mouth']:+.1f} dB")
    print(f"largest mouthing vs still: {best[0]:+.1f} dB at {best[1] * 100:.0f}-{(best[1] + PROFILE_STEP_M) * 100:.0f} cm")
    chance = total // len(PHRASES)
    print(f"phrases told apart, mouthed:      {mouthed} of {total} (guessing would get about {chance})")
    print(f"phrases told apart, mouth closed: {closed} of {total} (the control: should be near guessing)")
    print("=" * 66)
    if mouthed >= 0.7 * total and closed <= 0.5 * total:
        print("Reading: phrases are told apart when mouthed and not when the mouth is closed. Sonar is worth building on.")
    elif mouthed >= 0.7 * total:
        print("Reading: phrases are told apart even with the mouth closed, so the echo is following something other than the mouth.")
    else:
        print("Reading: phrases are not reliably told apart at face distance. Sonar on this laptop is not enough.")


if __name__ == "__main__":
    main()
