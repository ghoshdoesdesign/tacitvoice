"""The models run in a process of their own, on its main thread.

Inside the web server's process (worker threads beside an event loop) the
recognition step stalled and then the whole process vanished without an
error. As a plain process, the way tools/evaluate.py runs them, the same
models work. Keeping them apart also means a crash here cannot take the
server down: it reports the exit code, starts a fresh process, and carries on.
"""

from __future__ import annotations

import multiprocessing
import signal
import sys
import threading
import time

import numpy as np


class WorkerError(Exception):
    """The request could not be processed. `status` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def say(message: str) -> None:
    print(f"[tacitvoice] {message}", file=sys.stderr, flush=True)


def decode_frames(blobs: list[bytes]) -> np.ndarray:
    import cv2

    frames = []
    for blob in blobs:
        img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("A frame is not a readable image.")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    if len({f.shape for f in frames}) != 1:
        raise ValueError("Frames differ in size.")
    return np.stack(frames)


def handle(pipe, message: tuple) -> tuple:
    """Answer one request. Runs in the model process."""
    from .pipeline import Utterance
    from .stt import pcm16_to_float, resample

    kind = message[0]
    if kind == "remember":
        pipe.phrasebook.remember(message[1])
        return ("ok", None)
    if kind != "run":
        return ("error", f"unknown request {kind!r}", 500)

    _, jpegs, pcm, info = message
    try:
        frames = decode_frames(jpegs)
    except ValueError as e:
        return ("error", str(e), 400)
    audio = resample(pcm16_to_float(pcm), int(info.get("sampleRate") or 16000))
    say(f"clip decoded: {len(frames)} frames, {len(audio) / 16000:.1f} s of audio")
    utterance = Utterance(
        frames=frames,
        audio=audio,
        timestamps=info.get("timestamps"),
        fps=info.get("fps"),
        app=str(info.get("app") or ""),
        text_before_cursor=str(info.get("textBeforeCursor") or ""),
    )
    result = pipe.run(utterance).to_json()
    say(f"done ({result['mode']}, {result['timingsMs'].get('total')} ms): {result['text']!r} {result['warnings']}")
    return ("ok", result)


def serve(conn) -> None:
    """Entry point of the model process: load once, then answer requests."""
    from . import config
    from .pipeline import Pipeline

    try:
        pipe = Pipeline()
    except Exception as e:
        conn.send(("failed", f"{type(e).__name__}: {e}"))
        return
    conn.send(("ready", {
        "stt": config.STT_BACKEND,
        "lips": pipe.lips is not None,
        "lipsDevice": pipe.lips.device if pipe.lips else None,
        "cleanup": pipe.cleaner is not None,
        "cleanupModel": pipe.cleaner.model if pipe.cleaner else None,
        "problems": pipe.errors,
    }))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return  # the server went away
        try:
            conn.send(handle(pipe, message))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}", 500))


def echo(conn) -> None:
    """Stand-in model process for tests: no models, optional crash."""
    conn.send(("ready", {"lips": False, "cleanup": False, "problems": []}))
    while True:
        message = conn.recv()
        if message[0] == "crash":
            sys.exit(7)
        conn.send(("ok", message))


def describe_exit(code: int | None) -> str:
    if code is None:
        return "still running"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


class Worker:
    """Owns the model process: starts it, talks to it, restarts it if it dies."""

    def __init__(self, target=serve, timeout_s: float = 45.0) -> None:  # under the app's 60 s
        self._target = target
        self._timeout_s = timeout_s
        self._lock = threading.Lock()  # one utterance at a time
        self._proc = None
        self._conn = None
        self.info: dict | None = None  # set once the models are loaded
        self.error: str | None = None

    def start(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=self._target, args=(child,), daemon=True)
        self.info = None
        self._proc.start()
        child.close()
        threading.Thread(target=self._await_ready, args=(self._proc, self._conn), daemon=True).start()

    def _await_ready(self, proc, conn) -> None:
        while not conn.poll(0.2):
            if not proc.is_alive():
                self.error = f"The model process stopped while loading ({describe_exit(proc.exitcode)})."
                say(self.error)
                return
        try:
            kind, payload = conn.recv()
        except (EOFError, OSError):
            self.error = f"The model process stopped while loading ({describe_exit(proc.exitcode)})."
            say(self.error)
            return
        if kind == "ready":
            self.info, self.error = payload, None
            say("models loaded")
        else:
            self.error = payload
            say(f"models failed to load: {payload}")

    @property
    def ready(self) -> bool:
        return self.info is not None

    def call(self, message: tuple):
        """Send one request and wait for its answer. Raises WorkerError."""
        with self._lock:
            if not self.ready:
                raise WorkerError(self.error or "Models are still loading.", 503)
            proc, conn = self._proc, self._conn
            try:
                conn.send(message)
                deadline = time.monotonic() + self._timeout_s
                while not conn.poll(0.2):
                    if not proc.is_alive():
                        raise EOFError
                    if time.monotonic() > deadline:
                        say(f"recognition gave no answer in {self._timeout_s:.0f} s; restarting the model process")
                        proc.kill()
                        proc.join(5)
                        self.start()
                        raise WorkerError("Recognition took too long and was restarted. Try again.", 504)
                answer = conn.recv()
            except (EOFError, OSError):
                proc.join(5)
                reason = describe_exit(proc.exitcode)
                say(f"the model process stopped mid-request ({reason}); restarting it")
                self.start()
                raise WorkerError(f"Recognition crashed ({reason}) and was restarted. Try again.", 500)
        if answer[0] == "ok":
            return answer[1]
        raise WorkerError(answer[1], answer[2] if len(answer) > 2 else 500)

    def stop(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
