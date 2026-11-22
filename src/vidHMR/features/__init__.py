"""Frozen video-DiT features over 3DPW: frame loading, windows and storage for every backbone.

Each backbone module (`wan22`, `cosmos3`) runs in its own env and gives `run` a function
pixels (F, 3, H, W) in [-1, 1] -> {block: (F_lat, H/32, W/32, dim)}; `run` does the rest.
"""

import json
import time

import cv2
import numpy as np
import torch

from vidHMR import GENERATED, threedpw

WINDOW = 81  # frames per forward pass = 21 latent frames (the causal VAE needs 4k + 1)
SHORT, LONG = 480, 832  # sides frames are resized to: 480p, both multiples of 32
ROOT = GENERATED / "features"


def target_size(path) -> tuple[int, int]:
    """(W, H) to resize to, keeping the frame's orientation (3DPW has portrait and landscape)."""
    h, w = cv2.imread(str(path)).shape[:2]
    return (SHORT, LONG) if h > w else (LONG, SHORT)


def load_frames(paths, size) -> torch.Tensor:
    """Frames -> (F, 3, H, W) float32 in [-1, 1], RGB, resized to size (W, H)."""
    frames = []
    for p in paths:
        img = cv2.resize(cv2.imread(str(p)), size, interpolation=cv2.INTER_AREA)
        frames.append(img[:, :, ::-1])

    a = np.stack(frames).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def to_fp16(block: int, x: torch.Tensor) -> torch.Tensor:
    """Stored format: fp16 on the CPU. Refuses values fp16 can't hold (and inf / NaN)."""
    amax = x.abs().max().item()
    if not amax < torch.finfo(torch.float16).max:
        raise ValueError(f"block {block}: max |x| = {amax} does not fit fp16")
    return x.half().contiguous().cpu()


def run(extract_window, name: str, seqs: list[str], frames: int, meta: dict) -> None:
    """Extract every whole WINDOW of each sequence into ROOT/name/<seq>/w<start>.pt."""
    for seq in seqs:
        paths = threedpw.frames(seq)[: frames or None]
        out = ROOT / name / seq
        out.mkdir(parents=True, exist_ok=True)
        starts = list(range(0, len(paths) - WINDOW + 1, WINDOW))
        size = target_size(paths[0])
        timings = []

        for start in starts:
            torch.cuda.synchronize()
            t0 = time.time()
            feats = extract_window(load_frames(paths[start : start + WINDOW], size))
            feats = {k: to_fp16(k, v) for k, v in feats.items()}
            torch.cuda.synchronize()
            timings.append(time.time() - t0)
            torch.save(feats, out / f"w{start:05d}.pt")

        info = {
            **meta,
            "seq": seq,
            "window": WINDOW,
            "starts": starts,
            "size_wh": size,
            "sec_per_window": timings,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        (out / "meta.json").write_text(json.dumps(info, indent=1))
        print(
            f"{name}/{seq}: {len(timings)} windows, {np.mean(timings):.1f} s each, "
            f"peak {info['peak_gpu_gb']:.1f} GB"
        )
