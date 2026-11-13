import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "train" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = load("finetune")


def clips(lengths):
    return [{"x": np.zeros((n, 96, 96), np.uint8), "y": [5, 6, 7], "text": "a b c", "mode": "silent"} for n in lengths]


def test_batches_respect_the_frame_budget_and_keep_every_clip():
    random.seed(0)
    items = clips([40, 200, 90, 150, 60, 300, 120, 80])
    for shuffle in (False, True):
        groups = ft.batches(items, max_frames=400, shuffle=shuffle)
        assert sorted(len(c["x"]) for g in groups for c in g) == sorted(len(c["x"]) for c in items)
        for g in groups:
            assert len(g) == 1 or max(len(c["x"]) for c in g) * len(g) <= 400


def test_collate_pads_video_and_labels():
    batch = clips([30, 50])
    batch[1]["y"] = [9]
    x, lengths, y = ft.collate(batch, train=False, device="cpu")
    assert x.shape == (2, 1, 50, 88, 88) and lengths.tolist() == [30, 50]
    assert y.tolist() == [[5, 6, 7], [9, -1, -1]]
    assert float(x[0, 0, 30:].abs().sum()) == 0  # padding stays zero


def test_input_matches_inference_preprocessing():
    from server.lips import to_model_input

    patches = np.random.default_rng(0).integers(0, 255, (12, 96, 96), dtype=np.uint8)
    ours = ft.to_input(patches, train=False).numpy()  # (1, T, 88, 88)
    theirs = to_model_input(patches).transpose(1, 0, 2, 3)
    assert np.allclose(ours, theirs, atol=1e-5)
    random.seed(1)
    assert ft.to_input(patches, train=True).shape == (1, 12, 88, 88)


def test_word_helpers_and_distutils_shim():
    assert ft.words("Don’t STOP, now!") == ["don't", "stop", "now"]
    assert ft.edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    ft.shim_distutils()
    from distutils.util import strtobool
    from distutils.version import LooseVersion

    assert strtobool("true") == 1 and LooseVersion("2.5.1") >= LooseVersion("1.7.0")


def test_notebook_carries_the_current_script():
    notebook = json.loads((ROOT / "train" / "finetune_colab.ipynb").read_text())
    assert load("build_notebook").build()["cells"] == notebook["cells"], "run: python train/build_notebook.py"


def test_calibration_sentences_fit_the_model_vocabulary():
    lines = [l for l in (ROOT / "sentences" / "calibration.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) > 150 and len(set(lines)) == len(lines)
    assert not any(ch.isdigit() for line in lines for ch in line)
