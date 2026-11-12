import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config, main, pipeline
from server.cleanup import Cleaned
from server.lips import LipResult
from server.phrases import Phrasebook
from server.stt import SttResult


class FakePipeline:
    lips = None
    cleaner = None
    errors: list = []

    def __init__(self, tmp_path):
        self.seen = None
        self.phrasebook = Phrasebook(tmp_path / "dictionary.txt", tmp_path / "prompts.txt")

    def run(self, u):
        self.seen = u
        return pipeline.Outcome(
            id="abc", text="Hello.", alternatives=["Yellow."], mode="whisper",
            stt=SttResult(text="hello", words=[("hello", 0.9)], audible=True),
            lips=LipResult(candidates=[("yellow", -2.0)], face_fraction=1.0),
            cleaned=Cleaned(text="Hello.", source="model"), timings_ms={"total": 5},
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    for name in ("HOME", "TOKEN_FILE", "LOG_FILE"):
        monkeypatch.setattr(config, name, tmp_path / {"HOME": "", "TOKEN_FILE": "token", "LOG_FILE": "log.jsonl"}[name])
    monkeypatch.setattr(main, "_load", lambda: None)
    fake = FakePipeline(tmp_path)
    main._state.update(pipeline=fake, error=None)
    with TestClient(main.app) as c:
        c.fake = fake
        c.token = {"Authorization": f"Bearer {config.local_token()}"}
        yield c
    main._state.update(pipeline=None, error=None)


def jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
    return buf.tobytes()


def test_health_needs_no_token(client):
    assert client.get("/health").json()["ready"] is True


def test_transcribe_requires_token(client):
    assert client.post("/transcribe", data={"meta": "{}"}).status_code == 401
    assert client.post("/transcribe", data={"meta": "{}"}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_transcribe_round_trip(client):
    pcm = (np.ones(48000) * 1000).astype("<i2").tobytes()
    meta = {"sampleRate": 48000, "timestamps": [0.0, 0.033, 0.066], "app": "Slack", "textBeforeCursor": "Hi, "}
    files = [("audio", ("a.pcm", pcm, "application/octet-stream"))]
    files += [("frames", (f"{i}.jpg", jpeg(i * 40), "image/jpeg")) for i in range(3)]
    r = client.post("/transcribe", data={"meta": json.dumps(meta)}, files=files, headers=client.token)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hello." and body["alternatives"] == ["Yellow."] and body["mode"] == "whisper"
    assert body["raw"]["lips"] == ["yellow"]
    u = client.fake.seen
    assert u.frames.shape == (3, 48, 64, 3) and len(u.audio) == 16000
    assert u.app == "Slack" and u.text_before_cursor == "Hi, " and u.timestamps == [0.0, 0.033, 0.066]
    # frames arrive in order
    assert [int(f.mean()) for f in u.frames] == pytest.approx([0, 40, 80], abs=2)


def test_transcribe_rejects_bad_input(client):
    assert client.post("/transcribe", data={"meta": "{nope"}, headers=client.token).status_code == 400
    files = [("frames", ("0.jpg", b"not an image", "image/jpeg"))]
    assert client.post("/transcribe", data={"meta": "{}"}, files=files, headers=client.token).status_code == 400


def test_correction_is_logged_and_remembered(client):
    r = client.post("/correction", json={"id": "abc", "text": "Summarise this thread please"}, headers=client.token)
    assert r.status_code == 200
    log = [json.loads(line) for line in config.LOG_FILE.read_text().splitlines()]
    assert log[-1]["event"] == "correction" and log[-1]["id"] == "abc"
    assert "Summarise this thread please" in client.fake.phrasebook.prompts


def test_loading_state(client):
    main._state.update(pipeline=None, error=None)
    assert client.get("/health").json()["ready"] is False
    assert client.post("/transcribe", data={"meta": "{}"}, headers=client.token).status_code == 503
