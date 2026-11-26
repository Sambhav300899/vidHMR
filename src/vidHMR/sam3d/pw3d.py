"""SAM 3D Body predictions on 3DPW sequences (single-frame inference, GT intrinsics).

Boxes: "gt" (default, the usual 3DPW protocol) = one box per annotated actor from its projected GT
joints,
so every row has an actor id; "vitdet" = the repo's detector (all people, actor = -1).
Inference type "body" (default) skips the hand decoder; "full" adds it (about 2x slower).

One file per sequence: generated/sam3d/pw3d/{name}.npz with one row per person:
  frame (N,) int32, actor (N,) int32, bbox (N,4) xyxy px,
  verts (N,18439,3) float16 camera-frame metres (OpenCV axes, already + cam_t),
  cam_t (N,3), kp3d (N,70,3) camera frame (+ cam_t), kp2d (N,70,2) px,
  shape (N,45), scale (N,28), body_pose (N,133), hand_pose (N,108), global_rot (N,3), focal (N,)
  faces (F,3) int32 (MHR mesh, same for every row).

Run with the sam-3d-body environment, from the project root (one GPU worker per process):
  sam-3d-body/.venv/bin/python -m vidHMR.sam3d.pw3d --subset occ --workers 3
  sam-3d-body/.venv/bin/python -m vidHMR.sam3d.pw3d --only downtown_bus_00 --max-frames 50
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from vidHMR import geometry, paths
from vidHMR.datasets import pw3d

BBOX_THR = 0.5
OUT_DIR = paths.GENERATED / "sam3d" / "pw3d"


def pred_path(name: str) -> Path:
    return OUT_DIR / f"{name}.npz"


def load_predictions(name: str) -> dict:
    with np.load(pred_path(name)) as z:
        return {k: z[k] for k in z.files}


def actor_boxes(seq, frame: int, margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """xyxy boxes (M,4) around the projected GT joints of each actor, and their ids.
    The box grows by `margin` x its height on each side, plus extra room above the head joint.
    When the GT projection is behind the camera, the box comes from the actor's 2D detections
    (poses2d).
    This happens in outdoors_crosscountry_00, whose camera translation in the 3DPW file is
    wrong."""
    boxes, ids = [], []
    W, H = seq.K[0, 2] * 2, seq.K[1, 2] * 2
    for a in range(seq.n_actors):
        uv, z = geometry.project(seq.joints3d[a, frame], seq.cam_poses[frame], seq.K)
        if not (z > 0).all():
            x, y, conf = seq.poses2d[a, frame]
            ok = conf > 0.1
            if ok.sum() < 4:
                continue
            uv = np.stack([x[ok], y[ok]], 1)
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        h = y1 - y0
        box = np.array(
            [x0 - margin * h, y0 - (margin + 0.1) * h, x1 + margin * h, y1 + margin * h]
        )
        box = np.clip(box, 0, [W, H, W, H])
        if box[2] - box[0] > 8 and box[3] - box[1] > 8:
            boxes.append(box)
            ids.append(a)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(ids, np.int32)


def predict_sequence(
    est, name: str, max_frames: int | None = None, boxes: str = "gt", inference_type: str = "body"
) -> dict:
    """Run the estimator on every frame of a 3DPW sequence with the GT intrinsics."""
    import torch

    seq = pw3d.load_sequence(name)
    K = torch.tensor(seq.K[None], dtype=torch.float32)
    rows = {
        k: []
        for k in (
            "frame",
            "actor",
            "bbox",
            "verts",
            "cam_t",
            "kp3d",
            "kp2d",
            "shape",
            "scale",
            "body_pose",
            "hand_pose",
            "global_rot",
            "focal",
        )
    }
    n = seq.n_frames if max_frames is None else min(max_frames, seq.n_frames)
    for f in range(n):
        if boxes == "gt":
            bb, ids = actor_boxes(seq, f)
            if not len(bb):
                continue
            outs = est.process_one_image(
                str(seq.image_path(f)), bboxes=bb, cam_int=K, inference_type=inference_type
            )
        else:
            outs = est.process_one_image(
                str(seq.image_path(f)), cam_int=K, bbox_thr=BBOX_THR, inference_type=inference_type
            )
            ids = np.full(len(outs), -1, np.int32)
        for o, a in zip(outs, ids, strict=False):
            t = o["pred_cam_t"]
            rows["frame"].append(f)
            rows["actor"].append(a)
            rows["bbox"].append(o["bbox"])
            rows["verts"].append((o["pred_vertices"] + t).astype(np.float16))
            rows["cam_t"].append(t)
            rows["kp3d"].append(o["pred_keypoints_3d"] + t)
            rows["kp2d"].append(o["pred_keypoints_2d"])
            rows["shape"].append(o["shape_params"])
            rows["scale"].append(o["scale_params"])
            rows["body_pose"].append(o["body_pose_params"])
            rows["hand_pose"].append(o["hand_pose_params"])
            rows["global_rot"].append(o["global_rot"])
            rows["focal"].append(o["focal_length"])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["frame"] = out["frame"].astype(np.int32)
    out["actor"] = out["actor"].astype(np.int32)
    out["faces"] = np.asarray(est.faces, np.int32)
    out["n_frames"] = np.int32(n)
    return out


def _worker(names, max_frames, force, log_q):
    import contextlib
    import io

    from vidHMR.sam3d.run import build_estimator

    est = build_estimator(fov=False)
    for name in names:
        if pred_path(name).exists() and not force:
            log_q.put(f"{name}: exists, skipped")
            continue
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):  # the estimator prints on every frame
            out = predict_sequence(est, name, max_frames)
        path = pred_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part.npz")
        np.savez(tmp, **out)
        tmp.replace(path)
        dt = time.time() - t
        log_q.put(
            f"{name}: {int(out['n_frames'])} frames, {len(out['frame'])} detections, "
            f"{dt:.0f}s ({out['n_frames'] / dt:.1f} fps)"
        )


def run_many(names, workers=3, max_frames=None, force=False):
    """Split the sequences over `workers` GPU processes (longest first, greedy balance)."""
    frames = {r["name"]: r["frames"] for r in pw3d.index()}
    names = sorted(names, key=lambda n: -frames[n])
    buckets, load = [[] for _ in range(workers)], [0] * workers
    for n in names:
        i = int(np.argmin(load))
        buckets[i].append(n)
        load[i] += frames[n]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(b, max_frames, force, q)) for b in buckets if b]
    t0 = time.time()
    for p in procs:
        p.start()
    for i in range(len(names)):
        print(f"[{i + 1}/{len(names)}] {q.get()}  (t={time.time() - t0:.0f}s)", flush=True)
    for p in procs:
        p.join()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--subset", choices=["occ", "pc", "test", "all"])
    sel.add_argument("--only", nargs="+")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if a.only:
        names = a.only
    else:
        names = [
            r["name"] for r in pw3d.index() if a.subset == "all" or a.subset in pw3d.subsets_of(r)
        ]
    run_many(names, a.workers, a.max_frames, a.force)


if __name__ == "__main__":
    main()
