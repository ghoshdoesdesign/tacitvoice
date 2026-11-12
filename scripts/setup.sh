#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: Python environment and lip-reading model.
# Safe to rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

command -v uv >/dev/null || { echo "Install uv first:  brew install uv"; exit 1; }
uv sync

# The lip-reading model (Auto-AVSR). Its code and weights are licensed for
# private research only, so they are downloaded here and never committed.
REPO=third_party/Visual_Speech_Recognition_for_Multiple_Languages
mkdir -p third_party
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.git "$REPO"
fi

# Google Drive files from the "Model Zoo > LRS3" table in that repo's README:
# the visual-only model (891 MB) and the language model (191 MB).
fetch() {  # fetch <drive id> <unzip dir> <file that proves it worked>
  [[ -f "$REPO/$3" ]] && return
  uv run gdown "$1" -O third_party/download.zip
  unzip -o -q third_party/download.zip -d "$REPO/$2"
  rm -f third_party/download.zip
  [[ -f "$REPO/$3" ]] || {
    echo "Download did not produce $3."
    echo "Get it by hand from the Model Zoo table in $REPO/README.md and unzip into $REPO/$2"
    exit 1
  }
}
fetch 1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6 benchmarks/LRS3/models benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
fetch 1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv benchmarks/LRS3/language_models benchmarks/LRS3/language_models/lm_en_subword/model.pth

mkdir -p ~/.tacitvoice data
[[ -f ~/.tacitvoice/dictionary.txt ]] || printf '# Names and terms you use, one per line\n' > ~/.tacitvoice/dictionary.txt
[[ -f ~/.tacitvoice/prompts.txt ]] || printf '# Prompts you send often, one per line\n' > ~/.tacitvoice/prompts.txt

echo "Setup done. Start the server with:  ./scripts/server.sh"
