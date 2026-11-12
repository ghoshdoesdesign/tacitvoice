"""Settings, all overridable through TACIT_* environment variables."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-user data: dictionary, known prompts, the text-only log, the local token.
HOME = Path(os.getenv("TACIT_HOME", "~/.tacitvoice")).expanduser()

AVSR_REPO = Path(
    os.getenv(
        "TACIT_AVSR_REPO",
        REPO_ROOT / "third_party" / "Visual_Speech_Recognition_for_Multiple_Languages",
    )
).expanduser()
AVSR_CONFIG = os.getenv("TACIT_AVSR_CONFIG", "configs/LRS3_V_WER19.1.ini")
LIPS_DEVICE = os.getenv("TACIT_LIPS_DEVICE", "auto")  # auto | mps | cpu | cuda
LIPS_BEAM = int(os.getenv("TACIT_LIPS_BEAM", "0"))  # 0 = the model config's value
LIPS_NBEST = int(os.getenv("TACIT_LIPS_NBEST", "5"))
LIPS_ENABLED = os.getenv("TACIT_LIPS", "on") != "off"

# Where speech-to-text runs. "openai" sends each utterance's audio to OpenAI
# (about a second); "local" keeps it on this Mac (10+ seconds on a CPU).
# By default it is "openai" when an OpenAI key is present.
STT_BACKEND = os.getenv("TACIT_STT") or ("openai" if os.getenv("OPENAI_API_KEY") else "local")
STT_CLOUD_MODEL = os.getenv("TACIT_STT_CLOUD_MODEL", "gpt-4o-transcribe")
STT_MODEL = os.getenv("TACIT_STT_MODEL", "large-v3-turbo")
STT_COMPUTE = os.getenv("TACIT_STT_COMPUTE", "int8")
# Loudest 100 ms of the clip, in dBFS. Below this the clip counts as silent
# mouthing and speech-to-text is skipped (Whisper invents text on silence).
SILENCE_DBFS = float(os.getenv("TACIT_SILENCE_DBFS", "-50"))

CLEANUP_ENABLED = os.getenv("TACIT_CLEANUP", "on") != "off"
# Which service runs the cleanup pass: "anthropic" or "openai". By default it
# follows whichever API key is present (Anthropic if both are).
CLEANUP_PROVIDER = os.getenv("TACIT_CLEANUP_PROVIDER") or (
    "openai" if os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY") else "anthropic"
)
# Fast models without a reasoning step: the user is waiting on a key release.
CLEANUP_MODEL = os.getenv("TACIT_CLEANUP_MODEL") or (
    "gpt-4.1-mini" if CLEANUP_PROVIDER == "openai" else "claude-haiku-5-5"
)
CLEANUP_TIMEOUT_S = float(os.getenv("TACIT_CLEANUP_TIMEOUT", "6"))

HOST = "127.0.0.1"  # never listen on other interfaces: frames of your face come here
PORT = int(os.getenv("TACIT_PORT", "8765"))

DICTIONARY_FILE = HOME / "dictionary.txt"
PROMPTS_FILE = HOME / "prompts.txt"
LOG_FILE = HOME / "log.jsonl"
TOKEN_FILE = HOME / "token"


def local_token() -> str:
    """The shared secret between the menu-bar app and this server.

    Without it any web page open in a browser could post to localhost and
    trigger transcriptions. The file is readable by the owner only.
    """
    HOME.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    TOKEN_FILE.touch(mode=0o600)
    TOKEN_FILE.chmod(0o600)
    TOKEN_FILE.write_text(token)
    return token
