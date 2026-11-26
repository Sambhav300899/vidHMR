"""Evaluate SAM 3D Body predictions (vidHMR.sam3d.pw3d) against the 3DPW ground truth.

Protocol (camera frame, one row per predicted actor-frame):
- Joints: 12 body joints that the GT SMPL skeleton and the MHR70 keypoints both have (shoulders,
  elbows,
  wrists, hips, knees, ankles). This is NOT the official 14-joint protocol (H36M regressor on the
  SMPL mesh),
  because SAM 3D Body outputs MHR, not SMPL. The hips are defined differently in the two skeletons
  (SMPL hip
  joint centers vs. surface-like MHR/COCO hips), which adds a constant offset; see the per-joint
  errors.
- MPJPE: after subtracting the pelvis (mean of the two hips) from both.  PA-MPJPE: after Procrustes
  alignment.
- Root error: distance between the predicted and GT pelvis in the camera frame (translation/depth
  error).
- Only frames with campose_valid = True are scored.

Outputs: generated/eval/sam3d_pw3d/{name}.npz (per-row errors) and
generated/eval/sam3d_pw3d/summary.json.
CLI:  .venv/bin/python -m vidHMR.eval.sam3d_pw3d [--only NAME ...]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from vidHMR import geometry, paths
from vidHMR.datasets import pw3d
from vidHMR.eval.metrics import mpjpe, pa_mpjpe
from vidHMR.sam3d.pw3d import OUT_DIR as PRED_DIR
from vidHMR.sam3d.pw3d import load_predictions

JOINT_NAMES = [
    "L shoulder",
    "R shoulder",
    "L elbow",
    "R elbow",
    "L wrist",
    "R wrist",
    "L hip",
    "R hip",
    "L knee",
    "R knee",
    "L ankle",
    "R ankle",
]
SMPL_IDX = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
MHR70_IDX = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
ROOT = [6, 7]  # hips, in the 12-joint order
OUT = paths.GENERATED / "eval" / "sam3d_pw3d"


def gt_camera_joints(seq, frames: np.ndarray, actors: np.ndarray) -> np.ndarray:
    """GT 12 joints in the camera frame for each (frame, actor) row: (N,12,3)."""
    return geometry.world_to_cam(seq.joints3d[actors, frames][:, SMPL_IDX], seq.cam_poses[frames])


def evaluate_sequence(name: str) -> dict:
    seq, p = pw3d.load_sequence(name), load_predictions(name)
    keep = (
        (p["actor"] >= 0) & seq.valid[np.maximum(p["actor"], 0), p["frame"]]
        if len(p["frame"])
        else np.zeros(0, bool)
    )
    f, a = p["frame"][keep], p["actor"][keep]
    pred = p["kp3d"][keep][:, MHR70_IDX].astype(np.float64)
    gt = gt_camera_joints(seq, f, a)
    e = mpjpe(pred, gt, ROOT)
    pa = pa_mpjpe(pred, gt)
    root = np.linalg.norm(pred[:, ROOT].mean(1) - gt[:, ROOT].mean(1), axis=-1) * 1000
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", frame=f, actor=a, mpjpe=e, pa_mpjpe=pa, root=root)
    n_gt = int(seq.valid.sum())
    return dict(
        name=name,
        split=seq.split,
        actors=seq.n_actors,
        scored=int(keep.sum()),
        valid_gt=n_gt,
        mpjpe=float(e.mean()) if len(e) else None,
        pa_mpjpe=float(pa.mean()) if len(pa) else None,
        root=float(np.median(root)) if len(root) else None,
        per_joint_mpjpe=e.mean(0).tolist() if len(e) else None,
        excluded=pw3d.BAD_GT_SEQUENCES.get(name),  # reported, not averaged
    )


def per_frame(name: str) -> dict | None:
    """Per-frame errors for the player: {actor: {"frame": [...], "mpjpe": [...], "pa_mpjpe": "
    "[...]}}."""
    path = OUT / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    out = {}
    for a in np.unique(z["actor"]):
        m = z["actor"] == a
        out[int(a)] = dict(
            frame=z["frame"][m].tolist(),
            mpjpe=np.round(z["mpjpe"][m].mean(1), 1).tolist(),
            pa_mpjpe=np.round(z["pa_mpjpe"][m].mean(1), 1).tolist(),
            root=np.round(z["root"][m]).tolist(),
        )
    return out


def _pooled(rows: list[dict], key: str) -> float | None:
    """Mean over all scored actor-frames (not over sequences)."""
    rows = [r for r in rows if r[key] is not None and not r["excluded"]]
    n = sum(r["scored"] for r in rows)
    return sum(r[key] * r["scored"] for r in rows) / n if n else None


def evaluate(names=None) -> dict:
    names = names or sorted(p.stem for p in PRED_DIR.glob("*.npz") if not p.stem.endswith(".part"))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(evaluate_sequence, names))
    occ = set(pw3d.oc_sequences())
    groups = {
        "all": rows,
        "occ": [r for r in rows if r["name"] in occ],
        **{s: [r for r in rows if r["split"] == s] for s in pw3d.SPLITS},
    }
    totals = {}
    for k, g in groups.items():
        ok = [r for r in g if not r["excluded"]]
        if not ok:
            continue
        n = sum(r["scored"] for r in ok)
        totals[k] = dict(
            sequences=len(ok),
            scored=n,
            mpjpe=_pooled(g, "mpjpe"),
            pa_mpjpe=_pooled(g, "pa_mpjpe"),
            per_joint_mpjpe=(
                np.sum([np.array(r["per_joint_mpjpe"]) * r["scored"] for r in ok], 0) / n
            ).tolist(),
        )
    summary = dict(
        joints=JOINT_NAMES,
        protocol=__doc__.split("Outputs:")[0].strip(),
        sequences=rows,
        totals=totals,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_summary() -> dict | None:
    path = OUT / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+")
    s = evaluate(ap.parse_args().only)
    for k, t in s["totals"].items():
        print(
            f"{k:11s} seqs={t['sequences']:2d} n={t['scored']:6d}  MPJPE {t['mpjpe']:.1f}  "
            f"PA-MPJPE {t['pa_mpjpe']:.1f} mm"
        )
    print(
        "per joint (all):",
        ", ".join(
            f"{j} {e:.0f}"
            for j, e in zip(s["joints"], s["totals"]["all"]["per_joint_mpjpe"], strict=False)
        ),
    )
    for r in s["sequences"]:
        print(
            f"  {r['name']:28s} {r['split']:10s} n={r['scored']:5d}/{r['valid_gt']:5d}  MPJPE "
            f"{r['mpjpe'] or 0:6.1f}  "
            f"PA {r['pa_mpjpe'] or 0:6.1f}  root {r['root'] or 0:6.0f}"
            + ("  EXCLUDED" if r["excluded"] else "")
        )


if __name__ == "__main__":
    main()
