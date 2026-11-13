"""Lip reading with the open Auto-AVSR model, entirely in memory.

The model code and weights are not part of this repository: scripts/setup.sh
puts them in third_party/. Their licence allows private research use only.

Unlike the upstream command-line pipeline, this wrapper takes frames straight
from memory (nothing is written to disk) and returns several candidate
transcripts rather than one, so the cleanup pass can choose between them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_FPS = 25.0
CROP = 88  # the model's input is the centre 88x88 of a 96x96 mouth patch
MEAN, STD = 0.421, 0.165


@dataclass
class LipResult:
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (text, score), best first
    face_fraction: float = 0.0  # share of frames in which a face was found
    frames: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)


def resample_indices(timestamps: list[float] | None, n_frames: int, fps: float | None) -> np.ndarray:
    """Pick, for each 1/25 s tick, the captured frame nearest in time.

    Webcams deliver 30 fps with jitter; the model expects a steady 25.
    """
    if n_frames == 0:
        return np.zeros(0, dtype=np.int64)
    if timestamps and len(timestamps) == n_frames and n_frames > 1:
        t = clean_times(np.asarray(timestamps, dtype=np.float64))
    else:
        t = np.arange(n_frames, dtype=np.float64) / float(fps or MODEL_FPS)
    duration = float(t[-1])
    ticks = np.arange(0.0, duration + 1e-9, 1.0 / MODEL_FPS)
    idx = np.searchsorted(t, ticks)
    idx = np.clip(idx, 1, n_frames - 1) if n_frames > 1 else np.zeros_like(idx)
    if n_frames > 1:
        left = idx - 1
        idx = np.where(ticks - t[left] <= t[idx] - ticks, left, idx)
    return idx.astype(np.int64)


def clean_times(t: np.ndarray, max_gap_s: float = 0.5) -> np.ndarray:
    """Frame times starting at 0, with impossible jumps squeezed out.

    A camera can hand over one stale frame stamped long before the recording
    began. Taken at face value that turns a 3-second clip into hours, and
    resampling it asks for more memory than the machine has. Any step that
    runs backwards, is not a number, or exceeds `max_gap_s` is replaced by
    the clip's typical step.
    """
    steps = np.diff(t)
    good = np.isfinite(steps) & (steps >= 0) & (steps <= max_gap_s)
    typical = float(np.median(steps[good])) if good.any() else 1.0 / 30.0
    steps = np.where(good, steps, typical)
    return np.concatenate([[0.0], np.cumsum(steps)])


def to_model_input(patches: np.ndarray) -> np.ndarray:
    """(T, 96, 96) uint8 mouth patches -> (T, 1, 88, 88) normalised float32."""
    t, h, w = patches.shape
    top, left = (h - CROP) // 2, (w - CROP) // 2
    x = patches[:, top : top + CROP, left : left + CROP].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x[:, None, :, :]


def decode_nbest(hyps, token_list: list[str], n: int) -> list[tuple[str, float]]:
    """Turn beam-search hypotheses into distinct lowercase transcripts."""
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hyp in hyps:
        d = hyp.asdict() if hasattr(hyp, "asdict") else hyp
        ids = [int(i) for i in d["yseq"]][1:]  # drop the start token
        text = "".join(token_list[i] for i in ids)
        text = text.replace("<eos>", "").replace("<space>", " ").replace("▁", " ")
        text = " ".join(text.split()).lower()
        if text and text not in seen:
            seen.add(text)
            out.append((text, float(d["score"])))
        if len(out) >= n:
            break
    return out


class FaceLandmarks:
    """Eye, nose and mouth points for the largest face in each frame."""

    def __init__(self) -> None:
        import mediapipe as mp

        fd = mp.solutions.face_detection
        # model_selection=0 is tuned for faces within 2 m, i.e. a laptop webcam.
        self._detector = fd.FaceDetection(min_detection_confidence=0.5, model_selection=0)

    def __call__(self, frames: np.ndarray) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for frame in frames:
            h, w = frame.shape[:2]
            result = self._detector.process(frame)
            best, best_area = None, 0.0
            for det in result.detections or []:
                box = det.location_data.relative_bounding_box
                area = box.width * box.height
                if area > best_area:
                    best, best_area = det, area
            if best is None:
                out.append(None)
                continue
            kp = best.location_data.relative_keypoints
            # 0 right eye, 1 left eye, 2 nose tip, 3 mouth centre
            out.append(np.array([[kp[i].x * w, kp[i].y * h] for i in range(4)], dtype=np.float64))
        return out


class MouthCropper:
    """Frames in, aligned (T, 96, 96) grey mouth patches out. No speech model.

    Used by the lip reader, and on its own to prepare training data.
    """

    def __init__(self, repo_dir: Path) -> None:
        repo_dir = Path(repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f"{repo_dir} is missing. Run scripts/setup.sh first.")
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        from pipelines.detectors.mediapipe.video_process import VideoProcess

        self._landmarks = FaceLandmarks()
        self._crop = VideoProcess(convert_gray=True)

    def __call__(
        self, frames: np.ndarray, timestamps: list[float] | None, fps: float | None
    ) -> tuple[np.ndarray | None, float]:
        """Resample to 25 fps, find the face, cut out the mouth.

        Returns the patches (None if no face was found) and the share of
        frames in which a face was found.
        """
        idx = resample_indices(timestamps, len(frames), fps)
        frames = frames[idx]
        landmarks = self._landmarks(frames)
        found = sum(lm is not None for lm in landmarks)
        fraction = found / len(landmarks) if landmarks else 0.0
        if not found:
            return None, 0.0
        return self._crop(frames, landmarks), fraction


class LipReader:
    def __init__(
        self,
        repo_dir: Path,
        config: str = "configs/LRS3_V_WER19.1.ini",
        device: str = "auto",
        beam_size: int = 0,
        nbest: int = 5,
        weights: Path | None = None,
    ) -> None:
        repo_dir = Path(repo_dir)
        self.mouth_patches = MouthCropper(repo_dir)  # also puts the repo on sys.path

        import torch
        from pipelines.model import get_beam_search_decoder

        self._torch = torch
        self.nbest = nbest
        self.device = self._pick_device(device)

        ini = ConfigParser()
        cfg_path = repo_dir / config
        if not ini.read(cfg_path):
            raise FileNotFoundError(f"Model config not found: {cfg_path}")
        if ini.get("input", "modality") != "video":
            raise ValueError("LipReader needs a video-only model config.")

        def path(key: str) -> str:
            value = ini.get("model", key, fallback="")
            return str(repo_dir / value) if value else ""

        with open(path("model_conf"), "rb") as f:
            confs = json.load(f)
        train_args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))
        if getattr(train_args, "labels_type", "char") == "unigram5000":
            units = (repo_dir / "pipelines" / "tokens" / "unigram5000_units.txt").read_text().splitlines()
            self.token_list = ["<blank>"] + [u.split()[0] for u in units] + ["<eos>"]
        else:
            self.token_list = list(train_args.char_list)

        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        self.model = E2E(len(self.token_list), train_args)
        # A personal fine-tune (train/finetune_colab.ipynb) replaces the stock weights.
        self.weights = str(weights) if weights and Path(weights).exists() else path("model_path")
        self.model.load_state_dict(torch.load(self.weights, map_location="cpu"))
        self.model.to(self.device).eval()

        self.beam_search = get_beam_search_decoder(
            self.model,
            self.token_list,
            path("rnnlm"),
            path("rnnlm_conf"),
            ini.getfloat("decode", "penalty"),
            ini.getfloat("decode", "ctc_weight"),
            ini.getfloat("decode", "lm_weight"),
            beam_size or ini.getint("decode", "beam_size"),
        )
        self.beam_search.to(self.device).eval()

    def _pick_device(self, wanted: str) -> str:
        torch = self._torch
        wanted = (wanted or "auto").lower()
        if wanted in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
        # Apple's GPU is opt-in: the model's beam search mixes devices on it.
        if wanted == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def transcribe(
        self,
        frames: np.ndarray,
        timestamps: list[float] | None = None,
        fps: float | None = None,
    ) -> LipResult:
        """frames: (T, H, W, 3) RGB uint8."""
        torch = self._torch
        result = LipResult(frames=len(frames))
        if len(frames) < 5:
            return result

        t0 = time.perf_counter()
        patches, result.face_fraction = self.mouth_patches(frames, timestamps, fps)
        t1 = time.perf_counter()
        result.timings_ms["face"] = int((t1 - t0) * 1000)
        if patches is None or result.face_fraction < 0.5:
            return result

        x = torch.from_numpy(to_model_input(patches)).permute(1, 0, 2, 3)  # (1, T, 88, 88)
        with torch.no_grad():
            enc = self.model.encode(x.to(self.device))
            t2 = time.perf_counter()
            hyps = self.beam_search(enc)
        t3 = time.perf_counter()
        result.timings_ms["encode"] = int((t2 - t1) * 1000)
        result.timings_ms["beam"] = int((t3 - t2) * 1000)
        result.candidates = decode_nbest(hyps, self.token_list, self.nbest)
        return result
