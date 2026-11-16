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
