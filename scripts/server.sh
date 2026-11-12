#!/usr/bin/env bash
# Start the local recognition server. Leave it running while you use the app.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The cleanup pass needs an Anthropic API key. Keep it in the macOS Keychain:
#   security add-generic-password -a "$USER" -s tacitvoice-anthropic -w
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && command -v security >/dev/null; then
  ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s tacitvoice-anthropic -w 2>/dev/null || true)"
  export ANTHROPIC_API_KEY
fi
[[ -n "${ANTHROPIC_API_KEY:-}" ]] || echo "No Anthropic API key found: text will be inserted without the cleanup pass."

exec uv run python -m server.main
