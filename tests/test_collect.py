import importlib.util
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate import load_clip
from tools.import_friend import clean_text, import_zip, slug

ROOT = Path(__file__).resolve().parents[1]


def clip_bytes(tmp_path, frames=20) -> bytes:
    path = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (96, 64))
    for i in range(frames):
        writer.write(np.full((64, 96, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def make_zip(tmp_path, takes, extra=()) -> Path:
    archive = tmp_path / "friend.zip"
    video = clip_bytes(tmp_path)
    with zipfile.ZipFile(archive, "w") as z:
        listed = [{k: v for k, v in t.items() if not k.startswith("_")} for t in takes]
        z.writestr("manifest.json", json.dumps({"version": 1, "name": "Asha K.", "takes": listed}))
        for take in takes:
            if take.get("_write", True):
                z.writestr(take["file"], take.get("_bytes", video))
        for name in extra:
            z.writestr(name, b"x")
    return archive


def test_import_unpacks_a_friends_recordings(tmp_path):
    takes = [
        {"file": "takes/0000_aloud.mp4", "text": "Hello there friend", "mode": "aloud"},
        {"file": "takes/0001_whisper.mp4", "text": "Hello there friend", "mode": "whisper"},
        {"file": "takes/0002_silent.mp4", "text": "Hello there friend", "mode": "silent"},
        {"file": "takes/0003_silent.mp4", "text": "Second one", "mode": "silent"},
    ]
    folder, kept, rejected = import_zip(make_zip(tmp_path, takes), tmp_path / "friends")
    assert folder.name == "asha-k" and (kept, rejected) == (4, 0)
    assert sorted(p.name for p in (folder / "silent").iterdir()) == ["000.json", "000.mp4", "001.json", "001.mp4"]
    utterance, text = load_clip(folder / "silent" / "001")
    assert text == "Second one" and len(utterance.frames) == 20 and len(utterance.audio) == 0
    assert json.loads((folder / "aloud" / "000.json").read_text())["speaker"] == "asha-k"
    # a second import of the same person gets its own folder
    again, _, _ = import_zip(make_zip(tmp_path, takes), tmp_path / "friends")
    assert again.name == "asha-k-2"


def test_import_ignores_anything_the_page_would_not_write(tmp_path):
    takes = [
        {"file": "../../evil.mp4", "text": "escape", "mode": "aloud"},              # path traversal
        {"file": "takes/0001_aloud.sh", "text": "script", "mode": "aloud"},          # wrong type
        {"file": "takes/0002_aloud.mp4", "text": "missing file", "_write": False},   # not in the zip
        {"file": "takes/0003_aloud.mp4", "text": "not a video", "_bytes": b"junk"},  # unplayable
        {"file": "takes/0004_aloud.mp4", "text": "", "mode": "aloud"},               # no sentence
        {"file": "takes/0005_aloud.mp4", "text": "the good one", "mode": "aloud"},
    ]
    root = tmp_path / "friends"
    folder, kept, rejected = import_zip(make_zip(tmp_path, takes, extra=["../outside.txt"]), root)
    assert (kept, rejected) == (1, 5)
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file() and "friends" in p.parts)
    assert written == ["friends/asha-k/aloud/000.json", "friends/asha-k/aloud/000.mp4"]
    assert not (tmp_path / "evil.mp4").exists() and not (tmp_path.parent / "outside.txt").exists()


def test_names_and_text_are_made_safe():
    assert slug("../../Etc Passwd!") == "etc-passwd" and slug("") == "friend"
    assert clean_text("a\n\tb\x00  c") == "a b c" and len(clean_text("x" * 999)) == 300


def test_recording_page_is_built_from_the_current_template():
    spec = importlib.util.spec_from_file_location("build", ROOT / "collect" / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    page = (ROOT / "collect" / "index.html").read_text(encoding="utf-8")
    assert build.build() == page, "run: python collect/build.py"
    assert "__" not in page and len(build.sentences()) == 380
    assert "audio: false" in page  # the page promises that no sound is recorded
    assert "fetch(" not in page and "XMLHttpRequest" not in page  # and that it uploads nothing
