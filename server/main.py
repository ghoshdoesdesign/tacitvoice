"""Local HTTP server between the menu-bar app and the models.

Run with: uv run python -m server.main
Listens on 127.0.0.1 only and requires the token in ~/.tacitvoice/token.
"""

from __future__ import annotations

import asyncio
import atexit
import hmac
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from . import config
from .pipeline import Pipeline, Utterance, append_log
from .stt import pcm16_to_float, resample

_state: dict = {"pipeline": None, "error": None}
# Every model call, loading included, runs on this one long-lived thread. The
# face detector keeps graphics state tied to the thread that created it, and
# using it from another thread after that one has exited can kill the process.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="models")


def _load() -> None:
    try:
        _state["pipeline"] = Pipeline()
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.local_token()
    # Load models off the event loop so /health answers while they warm up.
    _worker.submit(_load)
    yield


app = FastAPI(title="TacitVoice", lifespan=lifespan)


def authorised(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {config.local_token()}"
    if not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong token.")


def pipeline() -> Pipeline:
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "Models are still loading.")
    return _state["pipeline"]


@app.get("/health")
def health() -> dict:
    p: Pipeline | None = _state["pipeline"]
    return {
        "ready": p is not None,
        "error": _state["error"],
        "lips": bool(p and p.lips),
        "lipsDevice": p.lips.device if p and p.lips else None,
        "stt": config.STT_BACKEND,
        "cleanup": bool(p and p.cleaner),
        "cleanupModel": p.cleaner.model if p and p.cleaner else None,
        "problems": p.errors if p else [],
    }


def decode_frames(blobs: list[bytes]) -> np.ndarray:
    frames = []
    for blob in blobs:
        img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="A frame is not a readable image.")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    if len({f.shape for f in frames}) != 1:
        raise HTTPException(status_code=400, detail="Frames differ in size.")
    return np.stack(frames)


@app.post("/transcribe", dependencies=[Depends(authorised)])
async def transcribe(
    meta: str = Form("{}"),
    audio: UploadFile | None = File(default=None),
    frames: list[UploadFile] = File(default=[]),
) -> dict:
    """meta: JSON {sampleRate, timestamps?, fps?, app?, textBeforeCursor?}.
    audio: raw 16-bit little-endian mono PCM. frames: JPEG images in order.
    Nothing received here is written to disk.
    """
    p = pipeline()
    try:
        info = json.loads(meta)
    except ValueError:
        raise HTTPException(status_code=400, detail="meta is not valid JSON.")

    say(f"clip received: {len(frames)} frames, audio {'yes' if audio else 'no'}")
    pcm = pcm16_to_float(await audio.read()) if audio else np.zeros(0, dtype=np.float32)
    pcm = resample(pcm, int(info.get("sampleRate") or 16000))
    images = decode_frames([await f.read() for f in frames])
    utterance = Utterance(
        frames=images,
        audio=pcm,
        timestamps=info.get("timestamps"),
        fps=info.get("fps"),
        app=str(info.get("app") or ""),
        text_before_cursor=str(info.get("textBeforeCursor") or ""),
    )

    say(f"clip decoded: {images.shape[0]} frames, {len(pcm) / 16000:.1f} s of audio")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_worker, lambda: p.run(utterance).to_json())
    say(f"done ({result['mode']}, {result['timingsMs'].get('total')} ms): {result['text']!r} {result['warnings']}")
    return result


class Correction(BaseModel):
    id: str
    text: str  # what the user ended up with
    chosen: str = "edit"  # "alternative" | "edit"


@app.post("/correction", dependencies=[Depends(authorised)])
def correction(body: Correction) -> dict:
    """Record a fix. Corrected text is ground truth, so it joins the known prompts."""
    append_log({"event": "correction", **body.model_dump()})
    pipeline().phrasebook.remember(body.text)
    return {"ok": True}


def say(message: str) -> None:
    """Progress line on the console, so a stall or exit shows how far a clip got."""
    print(f"[tacitvoice] {message}", file=sys.stderr, flush=True)


def main() -> None:
    atexit.register(lambda: say("server process is exiting"))
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
