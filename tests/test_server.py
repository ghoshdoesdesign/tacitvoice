import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config, main, pipeline, worker
from server.cleanup import Cleaned
from server.lips import LipResult
from server.phrases import Phrasebook
from server.stt import SttResult


class FakePipeline:
    def __init__(self, tmp_path):
        self.seen = None
        self.phrasebook = Phrasebook(tmp_path / "dictionary.txt", tmp_path / "prompts.txt")

    def run(self, u, trace=None):
        self.seen = u
        return pipeline.Outcome(
            id="abc", text="Hello.", alternatives=["Yellow."], mode="whisper",
            stt=SttResult(text="hello", words=[("hello", 0.9)], audible=True),
            lips=LipResult(candidates=[("yellow", -2.0)], face_fraction=1.0),
            cleaned=Cleaned(text="Hello.", source="model"), timings_ms={"total": 5},
        )


class InlineWorker:
    """A worker that answers in this process, through the real request handler."""

    def __init__(self, pipe):
        self.pipe, self.error = pipe, None
        self.info = {"stt": "openai", "lips": True, "cleanup": True, "problems": []}

    @property
    def ready(self):
        return self.info is not None

    def start(self):
        pass

    def stop(self):
        pass

    def call(self, message):
        if not self.ready:
            raise worker.WorkerError("Models are still loading.", 503)
        answer = worker.handle(self.pipe, message)
        if answer[0] == "ok":
            return answer[1]
        raise worker.WorkerError(answer[1], answer[2])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "TOKEN_FILE", tmp_path / "token")
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "log.jsonl")
    fake = FakePipeline(tmp_path)
    inline = InlineWorker(fake)
    monkeypatch.setattr(main, "make_worker", lambda: inline)
    with TestClient(main.app) as c:
        c.fake, c.worker = fake, inline
        c.token = {"Authorization": f"Bearer {config.local_token()}"}
        yield c


def jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
    return buf.tobytes()


def test_health_needs_no_token(client):
    body = client.get("/health").json()
    assert body["ready"] is True and body["lips"] is True and body["stt"] == "openai"


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
    assert client.post("/transcribe", data={"meta": "[1]"}, headers=client.token).status_code == 400
    files = [("frames", ("0.jpg", b"not an image", "image/jpeg"))]
    assert client.post("/transcribe", data={"meta": "{}"}, files=files, headers=client.token).status_code == 400


def test_correction_is_logged_and_remembered(client):
    r = client.post("/correction", json={"id": "abc", "text": "Summarise this thread please"}, headers=client.token)
    assert r.status_code == 200
    log = [json.loads(line) for line in config.LOG_FILE.read_text().splitlines()]
    assert log[-1]["event"] == "correction" and log[-1]["id"] == "abc"
    assert "Summarise this thread please" in client.fake.phrasebook.prompts


def test_loading_state(client):
    client.worker.info = None
    assert client.get("/health").json()["ready"] is False
    assert client.post("/transcribe", data={"meta": "{}"}, headers=client.token).status_code == 503


def wait_ready(w, seconds=20):
    deadline = time.monotonic() + seconds
    while not w.ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert w.ready, w.error


def test_model_process_survives_its_own_crash():
    """A real child process: answers, dies, is reported and replaced."""
    w = worker.Worker(target=worker.echo)
    w.start()
    try:
        wait_ready(w)
        assert w.call(("hello", 1)) == ("hello", 1)
        with pytest.raises(worker.WorkerError) as failure:
            w.call(("crash",))
        assert "exit code 7" in str(failure.value) and failure.value.status == 500
        wait_ready(w)  # a fresh process took over
        assert w.call(("again",)) == ("again",)
    finally:
        w.stop()


def test_describe_exit():
    assert worker.describe_exit(0) == "exit code 0" and worker.describe_exit(-9) == "killed by SIGKILL"
