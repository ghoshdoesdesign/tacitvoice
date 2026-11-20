"""Command mode: hand a recognised sentence to an OpenClaw agent on this Mac.

Dictation pastes text at the cursor. A command is the same recognition with a
different destination: the text goes to `openclaw agent`, which can run shell
commands, work with files and drive a browser, and its short reply is shown
to the user.

OpenClaw is installed and configured separately (its own installer, its own
API key). Nothing here gives it more access than it already has.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from . import config

# The reply is shown in a small panel above the orb, and the request was
# whispered or mouthed, so it may contain a misheard word.
STYLE = (
    "[This request was dictated to a silent-speech tool on the user's Mac and may contain a "
    "misheard word; go by the evident intent. Do the task yourself, on this Mac, without "
    "opening windows the user has to look at when a quieter way exists. "
    "If it was something to do (change a setting, open an app, send a message), do it and "
    "reply 'Done' followed by a few words saying exactly what you did, including the "
    "recipient and the text of anything you sent. "
    "If it was something to find or show (an email, a fact, a file), reply with the content "
    "itself in plain text, at most eight short lines, most important first. "
    "Before deleting anything or spending money, say what you would do and ask the user to "
    "confirm. If you could not do it, say why in one sentence. No markdown.] "
)


class AgentError(Exception):
    pass


def find_openclaw() -> str:
    """The openclaw command: TACIT_OPENCLAW, the PATH, or the usual install places.

    OpenClaw's installer puts it under whichever Node.js it finds. With nvm
    that folder is on the PATH only in shells where `nvm use` has been run, so
    it is looked for there too, newest Node first.
    """
    if config.OPENCLAW_PATH:
        return config.OPENCLAW_PATH
    path = shutil.which("openclaw")
    if path:
        return path
    home = Path.home()
    places = [home / ".npm-global/bin", home / ".volta/bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")]

    def version(folder: Path) -> tuple:
        return tuple(int(n) for n in re.findall(r"\d+", folder.parent.name))

    places += sorted((home / ".nvm/versions/node").glob("*/bin"), key=version, reverse=True)
    for place in places:
        if (place / "openclaw").exists():
            return str(place / "openclaw")
    raise AgentError("OpenClaw is not installed on this Mac.")


def extract_reply(result: dict) -> str:
    """The text of `openclaw agent --json`: top-level payloads, or under "result" in older versions."""
    payloads = result.get("payloads")
    if payloads is None:
        payloads = (result.get("result") or {}).get("payloads") or []
    texts = [p.get("text") or "" for p in payloads if isinstance(p, dict)]
    return "\n".join(t.strip() for t in texts if t.strip())


# The task in progress, so that it can be stopped. The app sends one at a time.
_lock = threading.Lock()
_running: dict = {"process": None, "cancelled": False}


def cancel() -> bool:
    """Stop the task in progress. Ending the openclaw command aborts the agent's run."""
    with _lock:
        process = _running["process"]
        if process is None or process.poll() is not None:
            return False
        _running["cancelled"] = True
        process.terminate()
        return True


def run(text: str, timeout_s: int | None = None) -> str:
    """One agent turn. The text is passed as an argument, never through a shell."""
    timeout_s = timeout_s or config.AGENT_TIMEOUT_S
    openclaw = find_openclaw()
    # openclaw is a Node.js script: put the Node it was installed with first on the PATH.
    env = {**os.environ, "PATH": str(Path(openclaw).parent) + os.pathsep + os.environ.get("PATH", "")}
    command = [
        openclaw, "agent",
        "--agent", config.AGENT_NAME,
        "--session-key", config.AGENT_SESSION,
        "--message", STYLE + text,
        "--timeout", str(timeout_s),
        "--json",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    except OSError as e:
        raise AgentError(f"OpenClaw could not be started: {e.strerror or e}")
    with _lock:
        _running["process"], _running["cancelled"] = process, False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s + 10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise AgentError("OpenClaw took too long and was stopped.")
    finally:
        with _lock:
            cancelled = _running["cancelled"]
            _running["process"] = None
    if cancelled:
        raise AgentError("Cancelled.")
    done = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    try:
        # With --json an error is also reported as JSON, on stdout or stderr.
        result = json.loads(done.stdout or done.stderr or "")
    except ValueError:
        raise AgentError("OpenClaw gave no readable answer. Is its gateway running?")
    if done.returncode != 0 or result.get("ok") is False:
        detail = result.get("error") or result.get("message") or f"exit {done.returncode}"
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail)
        raise AgentError(f"OpenClaw failed: {str(detail)[:200]}")
    reply = extract_reply(result)
    if not reply:
        raise AgentError("OpenClaw finished without saying anything.")
    return reply
