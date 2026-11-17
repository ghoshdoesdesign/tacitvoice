"""Turn an orb video into the app's "working" and "done" animations.

    uv run python -m tools.make_loading ~/Downloads/orb-loading.mp4

For a video of a glowing orb on black that settles into a loading state
(for example dots circling inside it) and then closes. It writes two sets of
transparent frames:

  ~/.tacitvoice/orb-loading/   shown after the key is released, while the
                               text is being worked out. Plays its opening
                               once, then repeats the looping part.
  ~/.tacitvoice/orb-done/      played once when the text has been inserted.

The frame numbers below fit the video this was written for (30 frames per
second: orb settles from frame 96, the loop is frames 169 to 239, the orb
closes by frame 277). For another video, pass your own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from server import config
from tools.make_orb import bounds, orb_diameter, orb_share, read_frames, write_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=int, default=96, help="first frame of the working state")
    ap.add_argument("--loop-start", type=int, default=169, help="first frame of the part that repeats")
    ap.add_argument("--loop-end", type=int, default=239, help="frame that looks like --loop-start again")
    ap.add_argument("--end", type=int, default=277, help="frame after the orb has closed")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--body", type=float, default=0.78)
    ap.add_argument("--rim", type=int, default=95)
    args = ap.parse_args()

    frames, fps = read_frames(args.video)
    if not 0 <= args.start < args.loop_start < args.loop_end < args.end <= len(frames):
        raise SystemExit(f"Frame numbers must rise and fit the video, which has {len(frames)} frames.")

    # One crop for both sets, so the orb does not jump between them.
    cx, cy, half = bounds(frames[args.start : args.end], 28)
    # Every animation is shown in the same square, so the orb must fill the
    # same share of its frame as it does in the listening animation, or it
    # appears to grow or shrink when the states change.
    share = orb_share(config.HOME / "orb")
    diameter = orb_diameter(frames[args.loop_start : args.loop_end], args.rim)
    if share and diameter:
        half = int(round(diameter / share / 2))
        print(f"Matched the listening orb, which fills {share:.0%} of its frame.")
    else:
        print("No listening animation found in ~/.tacitvoice/orb to match; using the orb's own size.")
    box = (cx, cy, half)
    options = dict(size=args.size, body=args.body, rim=args.rim)
    working = write_frames(frames[args.start : args.loop_end], config.HOME / "orb-loading", box, fps,
                           loop_start=args.loop_start - args.start, **options)
    done = write_frames(frames[args.loop_end : args.end], config.HOME / "orb-done", box, fps, **options)
    print(f"Wrote {args.loop_end - args.start} working frames ({working:.0f} MB) and "
          f"{args.end - args.loop_end} done frames ({done:.0f} MB) under {config.HOME}")
    print("Quit and reopen TacitVoice to use them.")


if __name__ == "__main__":
    main()
