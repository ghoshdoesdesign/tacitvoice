"""Build collect/index.html, the recording page to send to other people.

    python collect/build.py              # 60 sentences per person
    python collect/build.py --per-person 30

    python collect/build.py --upload-url https://script.google.com/macros/s/.../exec

The page is one self-contained file. Host it anywhere that serves it over
https (GitHub Pages, Netlify Drop, ...) and send people the link.

Without an upload address it sends nothing: each person downloads a .zip and
sends it back themselves. With one (the web app URL of collect/apps_script.gs,
deployed from your Google account) the page has a "Send your recordings"
button that puts each person's clips in their own folder in your Google
Drive. Either way, tools.import_friend adds what comes back to the training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
SENTENCE_FILES = ("calibration.txt", "calibration2.txt")
SECONDS_PER_TAKE = 10  # read, hold, say, release


def sentences() -> list[str]:
    out = []
    for name in SENTENCE_FILES:
        for line in (HERE.parent / "sentences" / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def settings() -> dict:
    """What the published page was built with, kept beside it so it can be rebuilt the same."""
    path = HERE / "settings.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    return {"per_person": 60, "upload_url": "", **saved}


def build(per_person: int | None = None, upload_url: str | None = None) -> str:
    conf = settings()
    per_person = conf["per_person"] if per_person is None else per_person
    upload_url = conf["upload_url"] if upload_url is None else upload_url
    if upload_url and not upload_url.startswith("https://script.google.com/macros/s/"):
        raise SystemExit("The upload address must be a Google Apps Script web app URL (https://script.google.com/macros/s/...).")
    minutes = round(per_person * 3 * SECONDS_PER_TAKE / 60 / 5) * 5
    page = (HERE / "page.html").read_text(encoding="utf-8")
    return (
        page.replace("__SENTENCES__", json.dumps(sentences(), ensure_ascii=False))
        .replace("__PER_PERSON__", str(per_person))
        .replace("__UPLOAD_URL__", json.dumps(upload_url))
        .replace("__MINUTES__", str(minutes))
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-person", type=int, help="sentences each person reads, three ways each (default 60)")
    ap.add_argument("--upload-url", help="web app URL of collect/apps_script.gs, so recordings are sent to "
                    "your Google Drive; pass an empty string to go back to downloads")
    args = ap.parse_args()
    conf = settings()
    if args.per_person is not None:
        conf["per_person"] = args.per_person
    if args.upload_url is not None:
        conf["upload_url"] = args.upload_url.strip()
    page = build(conf["per_person"], conf["upload_url"])
    (HERE / "settings.json").write_text(json.dumps(conf, indent=1) + "\n")
    out = HERE / "index.html"
    out.write_text(page, encoding="utf-8")
    where = "sends to Google Drive" if conf["upload_url"] else "offers a download (no upload address set)"
    print(f"wrote {out}: {conf['per_person']} sentences per person, {where}")
