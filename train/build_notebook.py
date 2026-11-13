"""Regenerate finetune_colab.ipynb from finetune.py:  python train/build_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).parent

INTRO = """# TacitVoice: fine-tune the lip model on your own recordings

Before you start:

1. On your Mac, run `uv run python -m tools.pack`. It writes `data/pack.npz`.
2. Upload `pack.npz` to the top level of your Google Drive.
3. In this notebook choose **Runtime > Change runtime type > T4 GPU**.
4. Choose **Runtime > Run all** and allow access to Drive when asked.

When it finishes, `lips_finetuned.pth` is in your Google Drive. Download it and
move it to `~/.tacitvoice/lips_finetuned.pth` on your Mac.

The model and its training data are licensed for research use only."""

MOUNT = """from google.colab import drive
import os

drive.mount('/content/drive')
PACK = '/content/drive/MyDrive/pack.npz'
OUT = '/content/drive/MyDrive/lips_finetuned.pth'
assert os.path.exists(PACK), 'Upload pack.npz to the top level of your Google Drive first.'
print(f'Found the data pack: {os.path.getsize(PACK) / 1e6:.0f} MB')
!nvidia-smi -L || echo 'NO GPU: choose Runtime > Change runtime type > T4 GPU, then run all again.'"""

SETUP = """%cd /content
!test -d vsr || git clone -q --depth 1 https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages vsr
!test -d auto_avsr || git clone -q --depth 1 https://github.com/mpc001/auto_avsr auto_avsr
!pip -q install sentencepiece gdown
%cd /content/vsr
!test -f benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth || (gdown -q 1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6 -O vsr.zip && unzip -o -q vsr.zip -d benchmarks/LRS3/models)
!test -f benchmarks/LRS3/language_models/lm_en_subword/model.pth || (gdown -q 1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv -O lm.zip && unzip -o -q lm.zip -d benchmarks/LRS3/language_models)
!ls -la benchmarks/LRS3/models/LRS3_V_WER19.1/ benchmarks/LRS3/language_models/lm_en_subword/"""

RUN = """!python /content/finetune.py --data "$PACK" --repo /content/vsr \\
    --spm /content/auto_avsr/spm/unigram/unigram5000.model --out "$OUT" \\
    --epochs 15"""


def cell(kind: str, text: str) -> dict:
    body = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        body.update(execution_count=None, outputs=[])
    return body


def build() -> dict:
    script = (HERE / "finetune.py").read_text()
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "cells": [
            cell("markdown", INTRO),
            cell("code", MOUNT),
            cell("code", SETUP),
            cell("code", "%%writefile /content/finetune.py\n" + script),
            cell("code", RUN),
        ],
    }


if __name__ == "__main__":
    out = HERE / "finetune_colab.ipynb"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out}")
