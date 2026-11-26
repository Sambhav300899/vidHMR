"""Pre-render 3DPW sequence videos, one process per video.

Videos go to generated/videos/pw3d/{name}_{kind}_{height}p.mp4. Kinds:
  raw       - the plain frames
  skeleton  - projected 3D joints + 2D OpenPose, actor labels
  mesh      - ground-truth SMPL meshes (GPU, nvdiffrast)
  sam3d     - SAM 3D Body predicted meshes (needs generated/sam3d/pw3d/{name}.npz from
  vidHMR.sam3d.pw3d;
              sequences without predictions are skipped)
  mask      - GT SMPL person masks (needs generated/masks/pw3d_smpl from vidHMR.masks)

CLI:
  python -m vidHMR_viewer.prerender [--kinds raw skeleton] [--workers 14] [--height 960]
    [--only NAME ...] [--force]
"""

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from vidHMR import paths, video
from vidHMR.datasets import pw3d
from vidHMR_viewer import render

KINDS = ("raw", "skeleton", "mesh", "sam3d", "mask")
GPU_KINDS = ("mesh", "sam3d")
DEFAULT_HEIGHT = 960
# A keyframe every 0.5 s: the player seeks without decoding long GOPs
GOP = 15


def video_path(name: str, kind: str, height: int = DEFAULT_HEIGHT) -> Path:
    return paths.GENERATED / "videos" / "pw3d" / f"{name}_{kind}_{height}p.mp4"


_MESH_RENDERER = None  # one GPU renderer per worker process


def _frames(seq, kind: str, height: int):
    if kind in GPU_KINDS:
        global _MESH_RENDERER
        if _MESH_RENDERER is None:
            from vidHMR_viewer.render_mesh import MeshOverlayRenderer

            _MESH_RENDERER = MeshOverlayRenderer()
        if kind == "mesh":
            yield from _MESH_RENDERER.frames(seq, height)
        else:
            from vidHMR.sam3d.pw3d import load_predictions

            yield from _MESH_RENDERER.prediction_frames(seq, load_predictions(seq.name), height)
        return
    for f in range(seq.n_frames):
        if kind == "raw":
            img = cv2.cvtColor(cv2.imread(str(seq.image_path(f))), cv2.COLOR_BGR2RGB)
            yield render.resize_to_height(img, height)[0]
        elif kind == "skeleton":
            yield render.draw_pw3d_frame(seq, f, height=height)
        elif kind == "mask":
            yield render.draw_mask_frame(seq, f, height=height)
        else:
            raise ValueError(f"unknown kind {kind!r}")


def render_one(
    name: str, kind: str, height: int = DEFAULT_HEIGHT, force: bool = False
) -> tuple[Path, float]:
    """Render one video. Returns (path, seconds). Skips existing files unless force."""
    path = video_path(name, kind, height)
    if path.exists() and not force:
        return path, 0.0
    t = time.time()
    video.write_video(
        _frames(pw3d.load_sequence(name), kind, height), path, fps=pw3d.FPS, threads=2, gop=GOP
    )
    return path, time.time() - t


def _has_sam3d(name: str) -> bool:
    from vidHMR.sam3d.pw3d import pred_path

    return pred_path(name).exists()


def available_kinds(name: str) -> list[str]:
    """The video kinds that exist on disk for a sequence (the player offers only these)."""
    return [k for k in KINDS if video_path(name, k).exists()]


def render_all(names=None, kinds=KINDS, height=DEFAULT_HEIGHT, workers=None, force=False):
    names = names or [r["name"] for r in pw3d.index()]
    jobs = [(n, k) for n in names for k in kinds if k != "sam3d" or _has_sam3d(n)]
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    print(f"{len(jobs)} videos, {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn")
    ) as ex:  # spawn: CUDA-safe
        futs = {ex.submit(render_one, n, k, height, force): (n, k) for n, k in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            n, k = futs[fut]
            try:
                p, dt = fut.result()
                print(
                    f"[{i}/{len(jobs)}] {n} {k}: {p.stat().st_size / 1e6:.1f} MB, {dt:.0f}s",
                    flush=True,
                )
            except Exception as e:  # keep the other videos going
                print(f"[{i}/{len(jobs)}] {n} {k}: FAILED {e!r}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    render_all(a.only, a.kinds, a.height, a.workers, a.force)


if __name__ == "__main__":
    main()
