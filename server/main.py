"""Local HTTP server between the menu-bar app and the models.

Run with: uv run python -m server.main
Listens on 127.0.0.1 only and requires the token in ~/.tacitvoice/token.
The models themselves live in a separate process (server/worker.py).
"""

from __future__ import annotations

import hmac
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import agent, config
from .pipeline import append_log
from .worker import Worker, WorkerError, say

_state: dict = {"worker": None}
# The app name the menu-bar app reports while a task, not dictation, is being recognised.
COMMAND_APP = "TacitVoice command"


def make_worker() -> Worker:
    return Worker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.local_token()
    if _state["worker"] is None:
        _state["worker"] = make_worker()
        _state["worker"].start()  # loads the models in the background
    yield
    _state["worker"].stop()
    _state["worker"] = None


app = FastAPI(title="TacitVoice", lifespan=lifespan)


def authorised(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {config.local_token()}"
    if not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong token.")


async def ask(message: tuple):
    worker: Worker = _state["worker"]
    try:
        return await run_in_threadpool(worker.call, message)
    except WorkerError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@app.get("/health")
def health() -> dict:
    worker: Worker = _state["worker"]
    info = worker.info or {}
    return {
        "ready": worker.ready,
        "error": worker.error,
        "stt": info.get("stt"),
        "lips": bool(info.get("lips")),
        "lipsDevice": info.get("lipsDevice"),
        "cleanup": bool(info.get("cleanup")),
        "cleanupModel": info.get("cleanupModel"),
        "problems": info.get("problems", []),
    }


@app.post("/transcribe", dependencies=[Depends(authorised)])
async def transcribe(
    meta: str = Form("{}"),
    audio: UploadFile | None = File(default=None),
    frames: list[UploadFile] = File(default=[]),
) -> dict:
    """meta: JSON {sampleRate, timestamps?, fps?, app?, textBeforeCursor?}.
    audio: raw 16-bit little-endian mono PCM. frames: JPEG images in order.
    """
    try:
        info = json.loads(meta)
    except ValueError:
        raise HTTPException(status_code=400, detail="meta is not valid JSON.")
    if not isinstance(info, dict):
        raise HTTPException(status_code=400, detail="meta must be a JSON object.")

    pcm = await audio.read() if audio else b""
    jpegs = [await f.read() for f in frames]
    say(f"clip received: {len(jpegs)} frames, audio {'yes' if pcm else 'no'}")
    return await ask(("run", jpegs, pcm, info))


class Correction(BaseModel):
    id: str
    text: str  # what the user ended up with
    chosen: str = "edit"  # "alternative" | "edit"


@app.post("/correction", dependencies=[Depends(authorised)])
async def correction(body: Correction) -> dict:
    """Record a fix. Corrected text is ground truth, so it joins the known prompts."""
    append_log({"event": "correction", **body.model_dump()})
    await ask(("remember", body.text))
    return {"ok": True}


class Command(BaseModel):
    text: str


@app.post("/command", dependencies=[Depends(authorised)])
async def command(body: Command) -> dict:
    """Give a recognised sentence to the OpenClaw agent on this Mac and return its reply."""
    text = " ".join(body.text.split())[:1000]
    if not text:
        raise HTTPException(status_code=400, detail="No command.")
    say(f"command: {len(text.split())} words")
    started = time.monotonic()
    try:
        reply, ok = await run_in_threadpool(agent.run, text), True
    except agent.AgentError as e:
        reply, ok = str(e), False
    append_log({"event": "command", "heard": text, "reply": reply, "ok": ok,
                "seconds": round(time.monotonic() - started, 1)})
    if not ok:
        raise HTTPException(status_code=502, detail=reply)
    return {"reply": reply}


def history_entries(limit: int = 200) -> list[dict]:
    """The newest dictations and tasks from the text log, newest first."""
    try:
        lines = config.LOG_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "command":
            entries.append({"kind": "command", "ts": record.get("ts", 0), "heard": record.get("heard", ""),
                            "reply": record.get("reply", ""), "ok": record.get("ok", True),
                            "seconds": record.get("seconds", 0)})
        elif record.get("event") == "utterance" and record.get("app") != COMMAND_APP:
            # (A task's own recognition is logged too; it is listed once, as the task.)
            seconds = (record.get("timingsMs") or {}).get("total", 0) / 1000
            entries.append({"kind": "dictation", "ts": record.get("ts", 0), "heard": record.get("text", ""),
                            "reply": "", "ok": True, "seconds": seconds})
        if len(entries) >= limit:
            break
    return entries


@app.get("/history", dependencies=[Depends(authorised)])
def history() -> dict:
    return {"entries": history_entries()}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """The history page. It holds no data itself: its script asks /history with the token."""
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


def main() -> None:
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
