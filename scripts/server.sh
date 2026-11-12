#!/usr/bin/env bash
# Start the local recognition server. Leave it running while you use the app.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The cleanup pass needs an API key, from Anthropic or OpenAI. Keep it in the
# macOS Keychain (the command prompts for the key):
#   security add-generic-password -a "$USER" -s tacitvoice-openai -w
#   security add-generic-password -a "$USER" -s tacitvoice-anthropic -w
from_keychain() { security find-generic-password -a "$USER" -s "$1" -w 2>/dev/null || true; }
if command -v security >/dev/null; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    ANTHROPIC_API_KEY="$(from_keychain tacitvoice-anthropic)"
    [[ -n "$ANTHROPIC_API_KEY" ]] && export ANTHROPIC_API_KEY || unset ANTHROPIC_API_KEY
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$(from_keychain tacitvoice-openai)"
    [[ -n "$OPENAI_API_KEY" ]] && export OPENAI_API_KEY || unset OPENAI_API_KEY
  fi
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "No API key found: text will be inserted without the cleanup pass."
fi

# faulthandler prints where Python was if a native library crashes the process.
uv run python -X faulthandler -m server.main
code=$?
echo "The server stopped (exit code $code)."
exit $code
