# TacitVoice: project notes

Silent dictation prototype for macOS (PRD working name "Murmur"). One user, research use.

## Requirements (do not regress)
- Hold-to-talk only: camera and microphone run only while right Option is held.
- No video or audio is written to disk outside `tools/record.py`. Only text leaves the device, and only for the cleanup pass.
- The server listens on 127.0.0.1 only and requires the token in `~/.tacitvoice/token`.
- Take ideas from github.com/evanhu1/telepathy, but do not copy its code (it has no licence).
- Auto-AVSR code and weights stay out of the repo (`third_party/` is gitignored; research-only licence).

## Layout
- `server/`: local FastAPI server. `pipeline.py` runs `stt.py` (Whisper) and `lips.py` (Auto-AVSR, n-best) in parallel, then `phrases.py` (glossary and past prompts) and `cleanup.py` (OpenAI or Claude, chosen by `TACIT_CLEANUP_PROVIDER` or the key present; structured output; falls back to the raw transcript on any failure).
- `tools/record.py`, `tools/evaluate.py`: record labelled clips and measure word error rate and latency.
- `app/`: Swift menu-bar app (SwiftPM, bundled by `scripts/build_app.sh`). Hotkey, capture, pill, paste-insert.
- `tests/`: pytest; models are faked. `uv run pytest`.

## Conventions
- `/transcribe` is multipart: `meta` JSON, `audio` raw 16-bit mono PCM, `frames` JPEGs in order.
- Frames are resampled to 25 fps by timestamp before lip reading.
- A clip whose loudest 100 ms is under `TACIT_SILENCE_DBFS` skips Whisper (it invents text on silence).
- `tools.evaluate` ignores `~/.tacitvoice/prompts.txt` unless `--known-prompts`, so scores measure recognition, not lookup.
- The Swift app cannot be built on Linux; changes to it need a build on the Mac.
