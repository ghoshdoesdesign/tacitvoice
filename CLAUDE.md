# TacitVoice: project notes

Silent dictation prototype for macOS (PRD working name "Murmur"). One user, research use.

## Requirements (do not regress)
- Hold-to-talk only: camera and microphone run only while right Option is held.
- No video or audio is written to disk outside `tools/record.py`. Video never leaves the device. The owner allowed audio to go to the cloud for speed: with an OpenAI key, each utterance's audio goes to OpenAI for speech-to-text (`TACIT_STT=local` keeps it on device). Text goes out for the cleanup pass.
- The server listens on 127.0.0.1 only and requires the token in `~/.tacitvoice/token`.
- Take ideas from github.com/evanhu1/telepathy, but do not copy its code (it has no licence).
- Auto-AVSR code and weights stay out of the repo (`third_party/` is gitignored; research-only licence).

## Layout
- `server/`: local FastAPI server (`main.py`). The models run in a separate process on its main thread (`worker.py`): inside the web server's process recognition stalled and the process vanished without an error, while a plain process works. The worker is restarted if it dies or takes over 45 s. `pipeline.py` runs `stt.py` (OpenAI transcription API, or local Whisper) and `lips.py` (Auto-AVSR, n-best) in parallel, then `phrases.py` (glossary and past prompts) and `cleanup.py` (OpenAI or Claude, chosen by `TACIT_CLEANUP_PROVIDER` or the key present; structured output; falls back to the raw transcript on any failure).
- `tools/record.py`, `tools/evaluate.py`: record labelled clips and measure word error rate and latency.
- `app/`: Swift menu-bar app (SwiftPM, bundled by `scripts/build_app.sh`). Hotkey, capture, pill, paste-insert.
- `tests/`: pytest; models are faked. `uv run pytest`.

## Conventions
- `/transcribe` is multipart: `meta` JSON, `audio` raw 16-bit mono PCM, `frames` JPEGs in order.
- Frames are resampled to 25 fps by timestamp before lip reading.
- A clip counts as silent mouthing, and skips speech-to-text (it invents text on silence), when its loudest 100 ms is under `TACIT_SILENCE_DBFS` or less than `TACIT_MIN_ACTIVITY` of it rises above its own noise floor. Loudness alone does not separate the two in the owner's room. Stock phrases such as "Thank you." are also discarded.
- Cloud speech-to-text uses `whisper-1`; `gpt-4o-transcribe` truncates and paraphrases whispered audio.
- Lips are read only when no whisper is heard (`TACIT_LIPS_WITH_AUDIO=on` reads them always). Measured on 20 whispered clips: 17.9% word error from audio alone, 17.6% with lips, at 3-4 s extra.
- Lip reading runs on the CPU: Auto-AVSR's beam search fails on Apple's GPU (mps) with a device mismatch.
- `tools.evaluate` ignores `~/.tacitvoice/prompts.txt` unless `--known-prompts`, so scores measure recognition, not lookup.
- The Swift app cannot be built on Linux; changes to it need a build on the Mac.
