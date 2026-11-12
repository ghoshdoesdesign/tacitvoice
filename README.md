# TacitVoice

Silent dictation for the Mac. Hold right Option, whisper or silently mouth a
prompt, release, and the text appears at your cursor in any app. It uses only
the laptop's webcam and microphone.

Working name in the PRD: "Murmur". This is a one-user research prototype.

## How it works

| Step | Where | What |
|---|---|---|
| Capture | `app/` (Swift menu-bar app) | Camera and microphone, only while the key is held, kept in memory |
| Audio | `server/stt.py` | Speech-to-text of the whisper: OpenAI's API (fast) or Whisper on this Mac (private) |
| Lips | `server/lips.py` | Auto-AVSR lip reading of the mouth region, top five guesses |
| Your vocabulary | `server/phrases.py` | Glossary and past prompts from `~/.tacitvoice/` |
| Cleanup | `server/cleanup.py` | A hosted model (OpenAI or Claude) picks between the guesses and punctuates. Text only |
| Insert | `app/` | Pasted at the cursor; the clipboard is restored |

If the microphone hears nothing, the clip is treated as silent mouthing and
only the lips and your vocabulary are used. Silent mode is rough until the
model is fine-tuned on your own face.

## Setup (Apple Silicon Mac)

Needs Xcode and Homebrew.

```bash
brew install uv
```

```bash
./scripts/setup.sh
```

Store an API key for the cleanup pass in the Keychain (it prompts for the
key). Use either OpenAI:

```bash
security add-generic-password -a "$USER" -s tacitvoice-openai -w
```

or Anthropic:

```bash
security add-generic-password -a "$USER" -s tacitvoice-anthropic -w
```

With both stored, Anthropic is used unless `TACIT_CLEANUP_PROVIDER=openai`.

## Day 0: measure your baseline

Put 100 prompts you really send in `data/prompts.txt`, one per line. `data/`
is ignored by git. Then record them twice and measure:

```bash
uv run python -m tools.record data/prompts.txt --mode whisper
```

```bash
uv run python -m tools.record data/prompts.txt --mode silent
```

```bash
uv run python -m tools.evaluate
```

The table shows the word error rate for audio alone, lips alone, and the
final text, plus the time each stage takes.

## Daily use

Start the server and leave it running:

```bash
./scripts/server.sh
```

Build and open the app:

```bash
./scripts/build_app.sh
```

```bash
open app/build/TacitVoice.app
```

Allow Camera, Microphone and Accessibility when asked, then quit and reopen
the app once so the hold key starts working. Add names and terms you use to
`~/.tacitvoice/dictionary.txt` and prompts you send often to
`~/.tacitvoice/prompts.txt`; both are read on every use.

## Privacy

- The camera and microphone run only while the key is held.
- Video is processed on this Mac. Nothing is written to disk, except by
  `tools/record.py`, which saves clips to `data/` on purpose.
- With an OpenAI key, the audio of each utterance is sent to OpenAI for
  speech-to-text. Set `TACIT_STT=local` to keep audio on this Mac (much slower).
- Only text goes to the cleanup provider (OpenAI or Anthropic): the candidate transcripts, your
  glossary, similar past prompts, the app name and up to 600 characters before
  the cursor. The last one can be turned off in the menu. Password fields are
  never read.
- `~/.tacitvoice/log.jsonl` keeps a text-only log on this Mac for the latency
  and habit metrics.

## Settings

Environment variables for `scripts/server.sh`; see `server/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `TACIT_STT` | `openai` with an OpenAI key, else `local` | Where speech-to-text runs |
| `TACIT_STT_CLOUD_MODEL` | `whisper-1` | OpenAI transcription model |
| `TACIT_STT_MODEL` | `large-v3-turbo` | Local Whisper model. `small.en` is faster and less accurate |
| `TACIT_SILENCE_DBFS` | `-50` | Below this level a clip counts as silent |
| `TACIT_MIN_ACTIVITY` | `0.12` | Share of the clip that must rise above its noise floor to count as a whisper |
| `TACIT_LIPS` | `on` | `off` skips lip reading |
| `TACIT_LIPS_BEAM` | `10` | Search width. `0` uses the model's 40: slower, a little more accurate |
| `TACIT_CLEANUP_PROVIDER` | follows the key present | `openai` or `anthropic` |
| `TACIT_CLEANUP_MODEL` | `gpt-4.1-mini` or `claude-haiku-5-5` | Model for the cleanup pass |
| `TACIT_CLEANUP` | `on` | `off` inserts the raw transcript |

## Licence note

The lip-reading model (Auto-AVSR, trained on LRS3) is downloaded by
`scripts/setup.sh` and is not part of this repository. Its licence allows
individual research and private study only, not commercial use.

## Tests

```bash
uv run pytest
```
