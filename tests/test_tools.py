import json

import cv2
import numpy as np
import soundfile as sf

from tools import evaluate


def test_load_clip_round_trip(tmp_path):
    stem = tmp_path / "000"
    writer = cv2.VideoWriter(str(stem.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 64))
    for i in range(12):
        writer.write(np.full((64, 96, 3), i * 20, dtype=np.uint8))
    writer.release()
    sf.write(stem.with_suffix(".wav"), np.zeros(8000, dtype="float32"), 16000)
    stamps = [round(i / 30, 4) for i in range(12)]
    stem.with_suffix(".json").write_text(json.dumps({"text": "hello there", "timestamps": stamps}))

    utterance, truth = evaluate.load_clip(stem)
    assert truth == "hello there"
    assert utterance.frames.shape == (12, 64, 96, 3) and utterance.timestamps == stamps
    assert len(utterance.audio) == 8000


def test_summarise_and_table(capsys):
    rows = [
        {"truth": "summarise this thread", "final": "Summarise this thread.", "audio": "summarise this bread",
         "lips": ["some eyes this red", "summarise this thread"], "mode_detected": "whisper", "seconds": 2.0,
         "timings_ms": {"stt": 300, "total": 900}},
        {"truth": "send it now", "final": "send it", "audio": "", "lips": [], "mode_detected": "none",
         "seconds": 1.0, "timings_ms": {"total": 100}},
    ]
    s = evaluate.summarise(rows)
    assert s["clips"] == 2 and s["exact_final"] == 0.5
    assert s["wer_final"] == 1 / 6 and s["wer_audio_only"] == 4 / 6
    assert s["wer_lips_best_of_n"] == 3 / 6 < s["wer_lips_top1"]
    assert s["detected_as"] == {"whisper": 1, "silent": 0, "none": 1}
    assert s["median_ms"]["total"] == 500 and s["median_ms"]["beam"] == 0
    evaluate.print_table({"whisper": s})
    assert "16.7%" in capsys.readouterr().out


def test_orb_keying_makes_black_transparent_and_keeps_the_shape_solid():
    from tools.make_orb import bounds, key_frame

    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(frame, (100, 100), 60, (40, 0, 60), -1)      # dim inside
    cv2.circle(frame, (100, 100), 60, (255, 200, 80), 4)    # bright rim
    out = key_frame(frame, body=0.78, rim=95)
    assert out.shape == (200, 200, 4)
    assert out[5, 5, 3] == 0                                 # background fully transparent
    assert out[100, 160, 3] > 240                            # rim opaque
    assert 180 < out[100, 100, 3] < 215                      # inside mostly solid (0.78)
    assert out[100, 160, 0] > 240                            # rim keeps its colour (blue channel)
    cx, cy, half = bounds([frame], 28)
    assert abs(cx - 100) <= 1 and abs(cy - 100) <= 1 and 62 <= half <= 75


def test_loading_frames_are_written_with_a_loop_point(tmp_path):
    from tools.make_orb import bounds, write_frames

    frames = []
    for i in range(12):
        frame = np.zeros((120, 120, 3), np.uint8)
        cv2.circle(frame, (60, 60), 30 + i, (255, 220, 120), 4)
        frames.append(frame)
    megabytes = write_frames(frames, tmp_path / "orb-loading", bounds(frames, 28), 30.0, size=64, loop_start=5)
    out = tmp_path / "orb-loading"
    assert len(list(out.glob("*.png"))) == 12 and megabytes > 0
    assert (out / "loop.txt").read_text().strip() == "5" and (out / "fps.txt").read_text().strip() == "30.000"
    image = cv2.imread(str(out / "0000.png"), cv2.IMREAD_UNCHANGED)
    assert image.shape == (64, 64, 4) and image[0, 0, 3] == 0


def test_sonar_measures_a_moving_reflector_and_tells_patterns_apart():
    from tools import sonar

    rng = np.random.default_rng(0)
    t = np.arange(int(sonar.TAKE_S * sonar.RATE)) / sonar.RATE

    def take(wobble):
        """The tones straight from the speaker, plus a weak echo whose path length changes."""
        out = rng.normal(0, 1e-4, len(t))
        for f in sonar.TONES:
            out += 0.05 * np.sin(2 * np.pi * f * t) + 0.002 * np.sin(2 * np.pi * f * t + wobble)
        return out

    still = take(np.zeros(len(t)))
    slow = [take(3 * np.sin(2 * np.pi * 3 * t + p)) for p in (0.0, 0.4, 0.8)]
    fast = [take(3 * np.sin(2 * np.pi * 9 * t + p)) for p in (0.0, 0.4, 0.8)]

    assert all(sonar.carrier_snr_db(still, f) > 25 for f in sonar.TONES)
    assert sonar.carrier_snr_db(rng.normal(0, 1e-4, len(t)), sonar.TONES[0]) < 10  # no tone, no carrier
    quiet = sonar.motion_db(sonar.baseband(still, sonar.TONES[0]))
    moving = sonar.motion_db(sonar.baseband(slow[0], sonar.TONES[0]))
    assert moving > quiet + 15
    signatures = [sonar.signature(a, sonar.TONES) for a in slow + fast]
    assert sonar.nearest_take_score(signatures, [0, 0, 0, 1, 1, 1]) == 6
    assert abs(float(np.abs(sonar.tone_signal(1.0)).max())) <= 0.25
