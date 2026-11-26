"""One Rerun recording (.rrd) per 3DPW sequence, for the web viewer of the dashboard.

A recording holds, on the timeline "frame":
- the camera (pose and intrinsics) with the video in its image plane,
- each actor's GT SMPL mesh (reduced to GT_LOD_VERTS vertices) and skeleton (gray on invalid
  frames),
- each actor's SAM 3D Body mesh (reduced to SAM3D_LOD_VERTS vertices),
- the per-frame SAM 3D Body errors as plots.

The 3D scene is anchored to the camera: world points minus the camera center of each frame (axes
kept, so gravity stays up). This removes the 3DPW world drift that camera and actors share.

    python -m vidHMR_viewer.recording [--only NAME ...] [--force]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vidHMR import geometry, paths
from vidHMR.body.smpl import load_model
from vidHMR.datasets import pw3d
from vidHMR_viewer import prerender
from vidHMR_viewer.skeletons import ACTOR_COLORS, SMPL24_BONES

RECORDING_FORMAT = 3  # increase on each change of the recording content
SAM3D_LOD_VERTS = 4500
GT_LOD_VERTS = 3000  # of 6890 SMPL vertices: smaller files for slow connections
SAM3D_ALPHA = 150  # see-through, so that the GT body and the video show through
FOOT_JOINTS = [7, 8, 10, 11]
INVALID_COLOR = (150, 150, 150)
CAMERA_COLOR = (230, 230, 230)
APP_ID = "vidHMR-3dpw"


def recording_path(name: str) -> Path:
    return paths.GENERATED / "rerun" / "pw3d" / f"{name}.rrd"


# -------------------------------------------------------------------- SAM 3D Body mesh reduction
def mesh_lod(
    key: str, verts: np.ndarray, faces: np.ndarray, n_verts: int
) -> tuple[np.ndarray, np.ndarray]:
    """A reduced mesh of a fixed topology that uses original vertices only: (keep, faces), where
    keep indexes the full vertices and faces index keep. Computed once from one mesh `verts` of
    that topology and cached under `key`."""
    cache = paths.generated("cache", "mesh_lod", f"{key}_{n_verts}_v2.npz")
    if cache.exists():
        z = np.load(cache)
        return z["keep"], z["faces"]

    import fast_simplification

    # Edge collapses join only connected vertices. The replay gives, for each original vertex,
    # the output vertex that it went into; each output vertex then uses the original vertex of
    # its group that is nearest to the group mean, so that every frame can use true positions.
    v = verts.astype(np.float32)
    _, _, collapses = fast_simplification.simplify(
        v, faces, 1 - n_verts / len(v), return_collapses=True
    )
    _, tri, group = fast_simplification.replay_simplification(v, faces, collapses)
    n_out = group.max() + 1
    mean = np.zeros((n_out, 3))
    np.add.at(mean, group, v)
    mean /= np.bincount(group, minlength=n_out)[:, None]
    dist = ((v - mean[group]) ** 2).sum(1)
    order = np.lexsort((dist, group))  # by group, nearest first
    first = np.flatnonzero(np.r_[True, np.diff(group[order]) != 0])
    keep = order[first]  # keep[j]: the original vertex for output vertex j
    tri = tri.astype(np.int32)
    np.savez(cache, keep=keep, faces=tri)
    return keep, tri


# -------------------------------------------------------------------- scene helpers
def _ground(feet_y: np.ndarray, valid: np.ndarray) -> float:
    """Ground height: a low percentile of the foot joint heights (A x F x 4) on valid frames."""
    sel = feet_y[valid]
    return float(np.percentile(sel, 2)) if sel.size else float(feet_y.min())


def _rgb(a: int) -> tuple[int, int, int]:
    return ACTOR_COLORS[a % len(ACTOR_COLORS)]


def _tint(rgb, t: float = 0.55) -> tuple[int, int, int]:
    """The color moved t of the way to white (SAM 3D Body: a lighter tint of the actor color)."""
    return tuple(int(c + (255 - c) * t) for c in rgb)


def _rgba(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    """(N, 3) uint8-range colors -> (N,) packed 0xRRGGBBAA."""
    c = np.asarray(rgb, np.uint32)
    return (c[:, 0] << 24) | (c[:, 1] << 16) | (c[:, 2] << 8) | alpha


class Tracks:
    """Per-frame data of several entities, written in blocks of BLOCK frames: the viewer can show
    the first frames while the rest of the file still loads."""

    BLOCK = 60

    def __init__(self):
        self.items = []  # (entity path, frames (N,), columns(sel) -> list of ComponentColumn)

    def add(self, path: str, frames: np.ndarray, columns) -> None:
        self.items.append((path, np.asarray(frames), columns))

    def mesh(self, path: str, frames, verts: np.ndarray, rgba: np.ndarray) -> None:
        """Per-frame vertices (N, V, 3) and colors (N,) of a mesh with fixed faces."""
        import rerun as rr

        def columns(sel):
            v = verts[sel]
            return [
                *rr.Mesh3D.columns(vertex_positions=v.reshape(-1, 3)).partition(
                    [v.shape[1]] * len(v)
                ),
                *rr.Mesh3D.columns(albedo_factor=rgba[sel]),
            ]

        self.add(path, frames, columns)

    def write(self, rec, n_frames: int) -> None:
        import rerun as rr

        for b in range(0, n_frames, self.BLOCK):
            for path, frames, columns in self.items:
                sel = np.flatnonzero((frames >= b) & (frames < b + self.BLOCK))
                if len(sel):
                    index = [rr.TimeColumn("frame", sequence=frames[sel])]
                    rec.send_columns(path, index, columns(sel))


def _gt_meshes(seq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT SMPL vertices (A, F, V, 3) and joints (A, F, 24, 3) in world space, and faces."""
    import torch

    from vidHMR.body.smpl import SMPL

    verts, joints, faces = [], [], None
    with torch.no_grad():
        for a in range(seq.n_actors):
            model = SMPL(seq.genders[a])
            v, j = model(seq.poses[a], seq.betas[a], seq.trans[a])
            verts.append(v.cpu().numpy())
            joints.append(j.cpu().numpy())
            faces = model.faces.cpu().numpy()
    return np.stack(verts), np.stack(joints), faces


# -------------------------------------------------------------------- the recording
def build(name: str, force: bool = False) -> Path:
    import rerun as rr

    out = recording_path(name)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".rrd.part")

    seq = pw3d.load_sequence(name)
    F = seq.n_frames
    frames = np.arange(F)
    c2w = np.stack([geometry.camera_to_world(e) for e in seq.cam_poses])  # F x 4 x 4
    center = c2w[:, :3, 3]

    rec = rr.RecordingStream(APP_ID, recording_id=name)
    rec.save(tmp)
    rec.send_blueprint(blueprint(seq))
    tracks = Tracks()

    # The anchored scene: everything under world/ moves with minus the camera center.
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    tracks.add("world", frames, lambda sel: rr.Transform3D.columns(translation=-center[sel]))

    # Camera: pose per frame, intrinsics, and the video in the image plane. The pinhole is in the
    # pixels of the pre-rendered video, which is smaller than the 3DPW images.
    w, h = video_size(name)
    K = np.diag([h / (2 * seq.K[1, 2]), h / (2 * seq.K[1, 2]), 1.0]) @ seq.K
    rec.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=K,
            resolution=[w, h],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.6,
            color=CAMERA_COLOR,
        ),
        static=True,
    )
    tracks.add(
        "world/camera",
        frames,
        lambda sel: rr.Transform3D.columns(translation=center[sel], mat3x3=c2w[sel, :3, :3]),
    )
    video = prerender.video_path(name, "raw")
    asset = rr.AssetVideo(path=video)
    rec.log("video", asset, static=True)
    rec.log(
        "world/camera/video",
        rr.VideoFrameReference.from_fields(video_reference="video"),
        static=True,
    )
    ts = asset.read_frame_timestamps_nanos()
    if len(ts) < F:
        raise RuntimeError(f"{video}: {len(ts)} video frames, {F} expected")
    tracks.add(
        "world/camera/video", frames, lambda sel: rr.VideoFrameReference.columns_nanos(ts[sel])
    )

    # GT: SMPL mesh and skeleton per actor; gray on invalid frames.
    gt_v, gt_j, gt_faces = _gt_meshes(seq)
    keep, gt_faces = mesh_lod("smpl", load_model("m")["v_template"], gt_faces, GT_LOD_VERTS)
    gt_v = gt_v[:, :, keep]
    for a in range(seq.n_actors):
        rgba = _rgba(np.where(seq.valid[a][:, None], _rgb(a), INVALID_COLOR))
        path = f"world/gt/actor{a}"
        rec.log(f"{path}/mesh", rr.Mesh3D.from_fields(triangle_indices=gt_faces), static=True)
        tracks.mesh(f"{path}/mesh", frames, gt_v[a], rgba)

        bones = gt_j[a][:, SMPL24_BONES]  # F x B x 2 x 3
        rec.log(f"{path}/skeleton", rr.LineStrips3D.from_fields(radii=0.012), static=True)

        def skeleton(sel, bones=bones, rgba=rgba):
            return [
                *rr.LineStrips3D.columns(strips=bones[sel].reshape(-1, 2, 3)).partition(
                    [len(SMPL24_BONES)] * len(sel)
                ),
                *rr.LineStrips3D.columns(colors=rgba[sel]),
            ]

        tracks.add(f"{path}/skeleton", frames, skeleton)

    _add_sam3d(rec, tracks, name, c2w)
    tracks.write(rec, F)
    rec.flush()
    del rec
    os.replace(tmp, out)
    return out


def _add_sam3d(rec, tracks: Tracks, name: str, c2w: np.ndarray) -> None:
    """SAM 3D Body meshes (reduced) in world space, and the per-frame errors against the GT."""
    import rerun as rr

    from vidHMR.eval.sam3d_pw3d import per_frame
    from vidHMR.sam3d.pw3d import load_predictions, pred_path

    if not pred_path(name).exists():
        return
    p = load_predictions(name)
    keep, faces = mesh_lod("mhr", p["verts"][0], p["faces"], SAM3D_LOD_VERTS)

    for a in np.unique(p["actor"]):
        rows = np.flatnonzero(p["actor"] == a)
        rows = rows[np.argsort(p["frame"][rows], kind="stable")]
        f = p["frame"][rows]
        cam = p["verts"][rows][:, keep].astype(np.float32)  # camera frame of each frame
        world = np.einsum("fij,fvj->fvi", c2w[f, :3, :3], cam) + c2w[f, None, :3, 3]
        path = f"world/sam3d/actor{a}"
        rec.log(path, rr.Mesh3D.from_fields(triangle_indices=faces), static=True)
        rgba = _rgba(np.tile(_tint(_rgb(int(a))), (len(f), 1)), SAM3D_ALPHA)
        tracks.mesh(path, f, world, rgba)

    for a, e in (per_frame(name) or {}).items():
        for key, label in (("mpjpe", "MPJPE"), ("pa_mpjpe", "PA-MPJPE")):
            path = f"errors/{key}/actor{a}"
            width = 2.0 if key == "mpjpe" else 1.0
            rec.log(
                path,
                rr.SeriesLines(colors=[_rgb(a)], names=[f"actor {a} {label}"], widths=[width]),
                static=True,
            )
            values = np.asarray(e[key])
            tracks.add(path, e["frame"], lambda sel, v=values: rr.Scalars.columns(scalars=v[sel]))

        path = f"root_error/actor{a}"
        rec.log(path, rr.SeriesLines(colors=[_rgb(a)], names=[f"actor {a} root"]), static=True)
        values = np.asarray(e["root"])
        tracks.add(path, e["frame"], lambda sel, v=values: rr.Scalars.columns(scalars=v[sel]))


def blueprint(seq):
    """Layout: the camera view (video with the 3D content projected on it) and the 3D view side
    by side, the error plot under them, and the timeline."""
    import rerun as rr
    import rerun.blueprint as rrb

    c2w = np.stack([geometry.camera_to_world(e) for e in seq.cam_poses])
    roots = seq.joints3d[:, :, 0] - c2w[None, :, :3, 3]  # A x F x 3, in the anchored scene
    A = seq.n_actors
    feet_y = seq.joints3d[:, :, FOOT_JOINTS, 1] - c2w[None, :, 1, 3, None]
    ground = _ground(feet_y, seq.valid)

    # The first 3D view: from behind and above the camera, toward the actors.
    target = np.median(roots[seq.valid], 0) if seq.valid.any() else np.zeros(3)
    forward = np.median(c2w[:, :3, 2], 0) * [1, 0, 1]  # the camera z axis, horizontal
    forward /= max(np.linalg.norm(forward), 1e-6)
    eye = -2.5 * forward + [0.0, 1.2, 0.0]
    pc = pw3d.PC_RANGES.get(seq.name)
    time_panel = dict(timeline="frame", fps=pw3d.FPS, playback_speed=1.0, play_state="paused")
    if pc:
        time_panel["time_selection"] = rr.encodings.AbsoluteTimeRange(min=pc[0], max=pc[1])

    # Camera view: the video, the GT skeletons and the see-through SAM 3D Body meshes. The GT
    # meshes would hide the person: off by default (the eye icon in the left panel shows them).
    gt_meshes = {f"world/gt/actor{a}/mesh": rrb.EntityBehavior(visible=False) for a in range(A)}
    views = rrb.Horizontal(
        rrb.Spatial2DView(
            name="Camera", origin="world/camera", contents=["/**"], overrides=gt_meshes
        ),
        rrb.Spatial3DView(
            name="3D",
            origin="world",
            line_grid=rrb.LineGrid3D(visible=True, spacing=1.0, plane=rr_plane_y(ground)),
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=eye.tolist(),
                look_target=target.tolist(),
                eye_up=(0.0, 1.0, 0.0),
            ),
        ),
    )
    return rrb.Blueprint(
        rrb.Vertical(
            views,
            rrb.Tabs(
                rrb.TimeSeriesView(name="SAM 3D Body error (mm)", origin="errors"),
                rrb.TimeSeriesView(name="SAM 3D Body root error (mm)", origin="root_error"),
            ),
            row_shares=[3, 1],
        ),
        rrb.TimePanel(**time_panel),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="expanded"),
        collapse_panels=False,
    )


def video_size(name: str) -> tuple[int, int]:
    """Width and height of the pre-rendered raw video."""
    import cv2

    cap = cv2.VideoCapture(str(prerender.video_path(name, "raw")))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def rr_plane_y(height: float):
    """The horizontal plane y = height (the ground)."""
    import rerun as rr

    return rr.components.Plane3D(normal=(0.0, 1.0, 0.0), distance=height)


def _build_one(name: str, force: bool) -> str:
    try:
        path = build(name, force)
        return f"{name}: {path.stat().st_size / 1e6:.0f} MB"
    except Exception as e:  # report and go on with the other sequences
        return f"{name}: FAILED {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--only", nargs="*", help="sequence names (default: all)")
    ap.add_argument("--force", action="store_true", help="build again when the file exists")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = args.only or pw3d.sequences()
    with ProcessPoolExecutor(args.workers) as ex:
        for line in ex.map(_build_one, names, [args.force] * len(names)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
