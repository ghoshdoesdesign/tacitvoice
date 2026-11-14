"""Fine-tune the Auto-AVSR lip model on one person's recordings.

Self-contained on purpose: train/finetune_colab.ipynb writes this file onto a
Colab GPU machine and runs it there. It needs only the data pack made by
`tools.pack`, the two upstream repositories, and PyTorch.

    python finetune.py --data pack.npz --repo <inference repo> --spm <unigram5000.model> \
        --out lips_finetuned.pth

It prints the word error rate on the held-out test clips before and after, so
the gain is measured on prompts the model never trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import types
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch

CROP, MEAN, STD = 88, 0.421, 0.165


def shim_distutils() -> None:
    """The model code imports distutils, which Python 3.12 removed."""
    try:
        from distutils.util import strtobool  # noqa: F401
        from distutils.version import LooseVersion  # noqa: F401
        return
    except Exception:
        pass

    class LooseVersion:
        def __init__(self, text: str) -> None:
            self.parts = tuple(int(n) for n in re.findall(r"\d+", str(text))[:3])

        def __lt__(self, other: "LooseVersion") -> bool:
            return self.parts < other.parts

        def __ge__(self, other: "LooseVersion") -> bool:
            return self.parts >= other.parts

    def strtobool(value: str) -> int:
        return int(str(value).lower() in ("y", "yes", "t", "true", "on", "1"))

    root, util, version = (types.ModuleType(n) for n in ("distutils", "distutils.util", "distutils.version"))
    util.strtobool, version.LooseVersion = strtobool, LooseVersion
    root.util, root.version = util, version
    sys.modules.update({"distutils": root, "distutils.util": util, "distutils.version": version})


def load_model(repo: Path, config: str):
    """Build the model and beam-search decoder exactly as inference does."""
    sys.path.insert(0, str(repo))
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

    ini = ConfigParser()
    if not ini.read(repo / config):
        raise FileNotFoundError(repo / config)

    def path(key: str) -> str:
        value = ini.get("model", key, fallback="")
        return str(repo / value) if value else ""

    confs = json.loads(Path(path("model_conf")).read_text())
    train_args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))
    units = (repo / "pipelines" / "tokens" / "unigram5000_units.txt").read_text().splitlines()
    token_list = ["<blank>"] + [u.split()[0] for u in units] + ["<eos>"]

    model = E2E(len(token_list), train_args)
    model.load_state_dict(torch.load(path("model_path"), map_location="cpu"))
    return model, token_list, train_args, ini, path


def make_decoder(model, token_list, ini, path, beam: int, device: str):
    from pipelines.model import get_beam_search_decoder

    search = get_beam_search_decoder(
        model, token_list, path("rnnlm"), path("rnnlm_conf"),
        ini.getfloat("decode", "penalty"), ini.getfloat("decode", "ctc_weight"),
        ini.getfloat("decode", "lm_weight"), beam,
    )
    return search.to(device).eval()


class Tokenizer:
    def __init__(self, spm_path: Path, token_list: list[str]) -> None:
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(spm_path))
        self.index = {token: i for i, token in enumerate(token_list)}

    def __call__(self, text: str) -> list[int]:
        # The model's vocabulary is upper-case letters and apostrophes.
        clean = " ".join(re.sub(r"[^A-Z' ]", " ", text.upper().replace("’", "'")).split())
        unk = self.index["<unk>"]
        return [self.index.get(piece, unk) for piece in self.spm.EncodeAsPieces(clean)]


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("’", "'")).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def to_input(patches: np.ndarray, train: bool) -> torch.Tensor:
    """(T, 96, 96) uint8 -> (1, T, 88, 88) float, with augmentation when training."""
    t, h, w = patches.shape
    if train:
        top, left = random.randint(0, h - CROP), random.randint(0, w - CROP)
    else:
        top, left = (h - CROP) // 2, (w - CROP) // 2
    x = patches[:, top : top + CROP, left : left + CROP]
    if train and random.random() < 0.5:
        x = x[:, :, ::-1]  # mirror: lips read the same either way round
    x = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
    if train:
        # Blank out short stretches (up to 0.4 s per second of video) so the
        # model cannot lean on any single moment.
        for _ in range(max(1, t // 25)):
            length = random.randint(0, 10)
            if 0 < length < t:
                start = random.randint(0, t - length)
                x[start : start + length] = 0
    return ((x - MEAN) / STD).unsqueeze(0)


def batches(items: list[dict], max_frames: int, shuffle: bool) -> list[list[dict]]:
    """Group clips of similar length so each batch's padded size stays under max_frames."""
    order = list(items)
    if shuffle:
        random.shuffle(order)
        order = [c for i in range(0, len(order), 60) for c in sorted(order[i : i + 60], key=lambda c: len(c["x"]))]
    else:
        order.sort(key=lambda c: len(c["x"]))
    out, cur, longest = [], [], 0
    for clip in order:
        longest_if_added = max(longest, len(clip["x"]))
        if cur and longest_if_added * (len(cur) + 1) > max_frames:
            out.append(cur)
            cur, longest_if_added = [], len(clip["x"])
        cur.append(clip)
        longest = longest_if_added
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(batch: list[dict], train: bool, device: str):
    xs = [to_input(c["x"], train) for c in batch]
    lengths = torch.tensor([x.shape[1] for x in xs])
    x = torch.zeros(len(xs), 1, int(lengths.max()), CROP, CROP)
    for i, clip in enumerate(xs):
        x[i, :, : clip.shape[1]] = clip
    longest = max(len(c["y"]) for c in batch)
    y = torch.full((len(batch), longest), -1, dtype=torch.long)
    for i, c in enumerate(batch):
        y[i, : len(c["y"])] = torch.tensor(c["y"])
    return x.to(device), lengths.to(device), y.to(device)


def loss_for(model, x, lengths, labels, ctc_weight: float):
    """The model's own training objective: CTC plus attention-decoder loss."""
    from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
    hidden, _ = model.encoder(x, mask)
    ctc = model.ctc(hidden, lengths, labels)
    loss_ctc = ctc[0] if isinstance(ctc, tuple) else ctc
    ys_in, ys_out = add_sos_eos(labels, model.sos, model.eos, model.ignore_id)
    pred, _ = model.decoder(ys_in, target_mask(ys_in, model.ignore_id), hidden, mask)
    loss_att = model.criterion(pred, ys_out)
    return ctc_weight * loss_ctc + (1 - ctc_weight) * loss_att


@torch.no_grad()
def average_loss(model, items, max_frames, ctc_weight, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in batches(items, max_frames, shuffle=False):
        x, lengths, y = collate(batch, False, device)
        total += float(loss_for(model, x, lengths, y, ctc_weight)) * len(batch)
        count += len(batch)
    return total / max(count, 1)


@torch.no_grad()
def word_error_rates(model, search, token_list, items, device) -> dict[str, float]:
    """Word error rate per recording mode on the given clips, by beam search."""
    model.eval()
    errors: dict[str, list[int]] = {}
    for clip in items:
        x = to_input(clip["x"], False).to(device)
        hyps = search(model.encode(x))
        ids = [int(i) for i in hyps[0].asdict()["yseq"]][1:] if hyps else []
        text = "".join(token_list[i] for i in ids).replace("<eos>", "").replace("▁", " ")
        ref, hyp = words(clip["text"]), words(text)
        e = errors.setdefault(clip["mode"], [0, 0])
        e[0] += edit_distance(ref, hyp)
        e[1] += len(ref)
    return {mode: e / max(n, 1) for mode, (e, n) in sorted(errors.items())}


def show(label: str, rates: dict[str, float]) -> None:
    print(f"{label}: " + ", ".join(f"{mode} {rate:.1%}" for mode, rate in rates.items()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="pack.npz from tools.pack")
    ap.add_argument("--repo", type=Path, required=True, help="Visual_Speech_Recognition_for_Multiple_Languages")
    ap.add_argument("--spm", type=Path, required=True, help="unigram5000.model from the auto_avsr repo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="configs/LRS3_V_WER19.1.ini")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-frames", type=int, default=800, help="padded frames per batch; lower if out of memory")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    shim_distutils()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU. Training on a CPU would take days.")

    model, token_list, train_args, ini, path = load_model(args.repo, args.config)
    model.to(device)
    tokenize = Tokenizer(args.spm, token_list)
    ctc_weight = float(getattr(train_args, "mtlalpha", 0.1))

    pack = np.load(args.data, allow_pickle=False)
    manifest = json.loads(str(pack["manifest"]))
    clips = [{**m, "x": pack[m["key"]], "y": tokenize(m["text"])} for m in manifest]
    clips = [c for c in clips if len(c["y"]) > 0 and len(c["x"]) >= 2 * len(c["y"])]
    test = [c for c in clips if c["split"] == "test"]
    pool = [c for c in clips if c["split"] == "train"]
    # Hold out whole sentences, not clips. Each sentence is recorded in several
    # modes, so holding out single clips would leave the same words in training
    # and the validation loss would reward memorising them.
    sentences = sorted({c["text"] for c in pool})
    random.shuffle(sentences)
    held = set(sentences[: max(1, len(sentences) // 10)])
    val = [c for c in pool if c["text"] in held]
    train = [c for c in pool if c["text"] not in held]
    minutes = sum(len(c["x"]) for c in train) / 25 / 60
    print(f"{len(train)} training clips ({minutes:.0f} min), {len(val)} validation clips of "
          f"{len(held)} held-out sentences, {len(test)} test, on {device}")
    if not train:
        raise SystemExit("No training clips in the pack.")

    search = make_decoder(model, token_list, ini, path, args.beam, device)
    before = word_error_rates(model, search, token_list, test, device) if test else {}
    show("word error rate BEFORE fine-tuning (test clips)", before)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.03, betas=(0.9, 0.98))
    steps_per_epoch = len(batches(train, args.max_frames, shuffle=True))
    total_steps, warmup = steps_per_epoch * args.epochs, steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    best, started = float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in batches(train, args.max_frames, shuffle=True):
            x, lengths, y = collate(batch, True, device)
            loss = loss_for(model, x, lengths, y, ctc_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += float(loss) * len(batch)
            seen += len(batch)
        val_loss = average_loss(model, val, args.max_frames, ctc_weight, device)
        note = ""
        if val_loss < best:
            # Keep the weights that do best on clips held out of training.
            best = val_loss
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)
            note = "  saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {running / seen:.3f}  validation loss {val_loss:.3f}"
              f"  {int(time.time() - started) // 60} min{note}", flush=True)

    model.load_state_dict(torch.load(args.out, map_location="cpu"))
    model.to(device)
    after = word_error_rates(model, search, token_list, test, device) if test else {}
    show("word error rate BEFORE fine-tuning (test clips)", before)
    show("word error rate AFTER  fine-tuning (test clips)", after)
    print(f"Saved the fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
