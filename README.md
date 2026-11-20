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

If the microphone hears a whisper, only the audio is used. If it hears
nothing, the clip is treated as silent mouthing and your lips and vocabulary
are used instead. Silent mode is rough until the
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

## Command mode: give OpenClaw a task

Hold **right Command** instead of right Option, whisper or mouth a task, and release. The words are recognised the same way, but they go to an [OpenClaw](https://openclaw.ai) agent on this Mac instead of being typed. While it works, the orb keeps spinning and a dark card at the top centre of the screen shows the words it heard. Then the card shows the result: "Done" and what was done for an action, or the content itself (the email, the answer) for something you asked to see, without leaving the app you are in. **Open history…** in the menu shows every task and dictation in a browser page served from this Mac.

The words are shown for a second and a half before the agent gets them. **Control+X** cancels: in that moment if they were heard wrong, or later to stop a task that is running (it may already have done part of it). Control+X also drops a dictation that has not been typed yet.

While either key is held the Mac's sound is muted, so music from the speakers is not heard as part of your whisper; it comes back when you let go.

It needs OpenClaw installed and its gateway running on the same Mac. What the agent is able to do (shell commands, files, a browser, and screen control if you turn that on in OpenClaw) is set in OpenClaw, not here. Recognition makes mistakes. The agent sends messages without asking first and reports exactly what it sent; it is told to ask before deleting anything or spending money. Treat that as a request to the model, not a guarantee.

## Fine-tune the lip model on yourself

Silent mode is weak until the model has seen your own mouth. One day's work:

1. Record the calibration sentences three times (aloud, whispered, silent).
   Each run resumes where you stopped:

```bash
uv run python -m tools.record sentences/calibration.txt --mode aloud --out data/train
```

```bash
uv run python -m tools.record sentences/calibration.txt --mode whisper --out data/train
```

```bash
uv run python -m tools.record sentences/calibration.txt --mode silent --out data/train
```

2. Pack the recordings into one small file of mouth crops:

```bash
uv run python -m tools.pack
```

3. Upload `data/pack.npz` to the top level of your Google Drive. Open
   `train/finetune_colab.ipynb` in Google Colab (File > Upload notebook),
   choose a T4 GPU runtime, and run all cells. It prints the word error rate
   on your `data/baseline` clips before and after.

4. Download `lips_finetuned.pth` from Drive to `~/.tacitvoice/`. The server and
   the evaluator use it automatically; `TACIT_LIPS_WEIGHTS=stock` switches back.

The tuned model fits your face only. It will not help other people.

## Collect recordings from other people

A model tuned on one person does not help anyone else. To test a shared
model you need other faces. `collect/index.html` is a recording page people
open in a browser: it shows sentences and records video only (no sound)
while they hold a key.

1. Put the page online. GitHub Pages serves it from this repository at
   `https://<user>.github.io/tacitvoice/collect/` (Settings > Pages > deploy
   from this branch).
2. Choose how recordings come back:
   - **Sent to your Google Drive.** Follow the steps at the top of
     `collect/apps_script.gs` to deploy the receiver from your Google
     account, then rebuild the page with its address:
     `python collect/build.py --upload-url https://script.google.com/macros/s/.../exec`.
     The page then has a "Send your recordings" button, and each person gets
     their own folder under "Experiment recordings" in your Drive.
   - **Downloaded and sent by hand.** With no upload address the page
     offers a .zip download instead, and sends nothing anywhere.
3. Send people the link. Sixty sentences, three ways each, takes about half
   an hour; they can stop early and still send what they have.
4. Download a person's folder from Drive (or take the .zip they sent), then
   import it and repack:

```bash
uv run python -m tools.import_friend ~/Downloads/asha-x7k2p9
```

```bash
uv run python -m tools.pack
```

5. Train as before. To measure how well the model reads someone it never
   trained on, edit the last cell of the notebook to add
   `--holdout-speaker asha` (the folder name under `data/friends`).

Everyone reads the same 10 sentences first (`sentences/common.txt`), then 50 picked at random from about 4,450, so two people rarely read the same ones. Most of that pool is built from templates by `sentences/generate.py`.

To change how many sentences each person reads: `python collect/build.py --per-person 30`.

Face video is personal data. Keep the recordings private, tell people what
they are for, and delete them if someone asks. Anyone who has the page can
send files to the Drive receiver; archive its deployment when you are done.

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
| `TACIT_LIPS_WITH_AUDIO` | `off` | `on` also reads lips when a whisper is heard (slower, little gain so far) |
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
