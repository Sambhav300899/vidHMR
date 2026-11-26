"""BEDLAM (CVPR 2023): synthetic 30 fps videos with neutral SMPL ground truth.

Data in generated/datasets/bedlam/ (Hugging Face Intelligent-Systems/BEDLAM and the BEDLAM site):
  <scene>/be_seq.csv                      the sequences of a scene and their bodies
  <scene>/ground_truth/camera/<seq>_camera.csv   the camera of each frame (Unreal units: cm, deg)
  <scene>/mp4/<seq>.mp4                   the video (1280 x 720; 720 x 1280 in closeup scenes)
  <scene>/masks/<seq>/<seq>_<frame>_<person>_{body,clothing}.png   visible masks of each person
  labels/smpl_ground_truth_corr/<subject>/<motion>/motion_seq.npz   neutral SMPL, 11 betas, 30 fps

We use one 81-frame window per sequence. A window is named "bedlam/<scene>/<sequence>" and is
stored like a 3DPW sequence of 81 frames (frame 0 = video frame `start`). The camera conversion is
a port of the official data_processing/df_full_body_smpl.py (github.com/pixelite1201/BEDLAM).
"""

import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
import torch

from vidHMR import geometry, paths
from vidHMR.body.smpl import SMPL, axis_angle_to_matrix, fit_neutral_betas, rest_pelvis
from vidHMR.datasets.pw3d_targets import axis_angles, targets_path
from vidHMR.features import N_LATENT, WINDOW, frames_to_tensor, save_int8, target_size_of
from vidHMR.features import ROOT as FEATURES

ROOT = paths.GENERATED / "datasets" / "bedlam"
SMPL_GT = ROOT / "labels" / "smpl_ground_truth_corr"
SEED = 0


@dataclass
class Body:
    subject: str  # folder in SMPL_GT
    motion: str  # subfolder in SMPL_GT/<subject>
    position: np.ndarray  # (3,) cm, Unreal world frame
    yaw: float  # degrees
    start_frame: int  # SMPL frame of video frame 0


@dataclass
class Sequence:
    scene: str
    name: str  # seq_000123
    n_frames: int
    bodies: list[Body]  # in the order of the mask files (person 00, 01, ...)

    @property
    def closeup(self) -> bool:
        """Closeup scenes are rendered with the camera rolled by 90 degrees (720 x 1280)."""
        return "closeup" in self.scene


@cache
def sequences() -> list[Sequence]:
    """All sequences of all scenes, from be_seq.csv."""
    out = []
    for path in sorted(ROOT.glob("*/be_seq.csv")):
        scene = path.parent.name
        for row in csv.DictReader(path.open()):
            if row["Type"] == "Group":
                name = re.search(r"sequence_name=(\w+)", row["Comment"]).group(1)
                frames = int(re.search(r"frames=(\d+)", row["Comment"]).group(1))
                out.append(Sequence(scene, name, frames, []))
            elif row["Type"] == "Body":
                subject, motion = row["Body"].rsplit("_", 1)
                start = int(re.search(r"start_frame=(\d+)", row["Comment"]).group(1))
                position = np.array([float(row[k]) for k in "XYZ"])
                out[-1].bodies.append(Body(subject, motion, position, float(row["Yaw"]), start))
    return out


def select(n: int) -> list[tuple[Sequence, int]]:
    """n (sequence, window start) pairs: sequences in a seeded random order, without the zoom
    scenes (their focal length changes during the sequence; the model takes one K per item),
    each with a seeded random start (a multiple of 4). The first n of a larger selection are the
    same n."""
    candidates = [s for s in sequences() if "zoom" not in s.scene]
    random = np.random.default_rng(SEED)
    order = random.permutation(len(candidates))
    starts = 4 * random.integers(0, [(s.n_frames - WINDOW) // 4 + 1 for s in candidates])
    return [(candidates[i], int(starts[i])) for i in order[:n]]


def window_name(sequence: Sequence) -> str:
    return f"bedlam/{sequence.scene}/{sequence.name}"


def read_frames(sequence: Sequence, start: int) -> list[np.ndarray]:
    """The BGR video frames start .. start + 80, decoded in order from frame 0 (no seek)."""
    video = cv2.VideoCapture(str(ROOT / sequence.scene / "mp4" / f"{sequence.name}.mp4"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == sequence.n_frames, sequence.name
    frames = []
    for index in range(start + WINDOW):
        ok, frame = video.read()
        assert ok, f"{sequence.name}: frame {index}"
        if index >= start:
            frames.append(frame)
    return frames


def camera(sequence: Sequence) -> dict[str, np.ndarray]:
    """The camera CSV of a sequence as arrays (one value per frame)."""
    path = ROOT / sequence.scene / "ground_truth" / "camera" / f"{sequence.name}_camera.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == sequence.n_frames, sequence.name
    keys = ("x", "y", "z", "yaw", "pitch", "roll", "focal_length", "sensor_width", "sensor_height")
    return {key: np.array([float(row[key]) for row in rows]) for key in keys}


def image_size(sequence: Sequence) -> tuple[int, int]:
    """(width, height) of the video."""
    return (720, 1280) if sequence.closeup else (1280, 720)


def intrinsics(sequence: Sequence, focal_mm: float) -> np.ndarray:
    """K in pixels (official get_cam_int). Closeup scenes swap the sensor sides."""
    width, height = image_size(sequence)
    sensor_w, sensor_h = (20.25, 36.0) if sequence.closeup else (36.0, 20.25)
    fx, fy = focal_mm / sensor_w * width, focal_mm / sensor_h * height
    return np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])


def _rotation(axis: int, degrees) -> np.ndarray:
    """(N, 3, 3) rotations about one axis (0 = x, 1 = y, 2 = z)."""
    vectors = np.zeros((len(degrees), 3))
    vectors[:, axis] = np.deg2rad(degrees)
    return np.stack([cv2.Rodrigues(v)[0] for v in vectors])


def _unreal_to_opencv(points: np.ndarray) -> np.ndarray:
    """Unreal (cm; x forward, y right, z up) -> meters with the axes of the official code."""
    return np.roll(points / 100, 2, axis=-1) * np.array([1.0, -1.0, 1.0])


def camera_frame(sequence: Sequence, body: Body, frames: np.ndarray, pose, trans, j0):
    """The official conversion of one body to the camera frame for the given video frames.
    pose (F, 72) axis-angle and trans (F, 3) from motion_seq.npz, j0 the rest pelvis of the body
    (3,). Returns the camera-frame global orientation (F, 3, 3) and pelvis point (F, 3), meters.
    (Official: SMPL(pose_cam, betas, c_trans) + cam_trans, with c_trans = roll_pitch @ (body
    rotation @ (j0 + trans)) - j0, so the pelvis J0 lands at the value returned here.)"""
    cam = {key: value[frames] for key, value in camera(sequence).items()}
    pitch, yaw = -cam["pitch"], -cam["yaw"]
    roll = -cam["roll"] + (90 if sequence.closeup else 0)
    n = len(frames)

    # Camera rotation (roll @ pitch @ yaw) and the body position relative to the camera.
    rotation = _rotation(2, roll) @ _rotation(0, pitch) @ _rotation(1, yaw)
    camera_position = _unreal_to_opencv(np.stack([cam["x"], cam["y"], cam["z"]], -1))
    offset = _unreal_to_opencv(body.position)[None] - camera_position
    cam_trans = np.einsum("nij,nj->ni", rotation, offset)

    # The body: turned by yaw - 90 (+ camera yaw) about y, then flipped to the camera axes.
    body_rotation = _rotation(1, np.full(n, body.yaw - 90) + yaw) @ np.diag([1.0, -1.0, -1.0])
    roll_pitch = _rotation(2, roll) @ _rotation(0, pitch)
    root = np.stack([cv2.Rodrigues(p[:3])[0] for p in pose])
    global_orient = roll_pitch @ body_rotation @ root
    pelvis = np.einsum("nij,nj->ni", roll_pitch @ body_rotation, j0[None] + trans) + cam_trans
    return global_orient, pelvis


def masks(sequence: Sequence, frame: int, n_people: int) -> np.ndarray:
    """(A, H, W) bool: the visible body + clothing of each person in one video frame. A missing
    file is an empty mask."""
    width, height = image_size(sequence)
    folder = ROOT / sequence.scene / "masks" / sequence.name
    out = np.zeros((n_people, height, width), bool)
    for person in range(n_people):
        for part in ("body", "clothing"):
            path = folder / f"{sequence.name}_{frame:04d}_{person:02d}_{part}.png"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if sequence.closeup:  # stored unrotated (1280 x 720); the video is upright
                    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                out[person] |= mask > 127
    return out


def extract_masks(selection: list[tuple[Sequence, int]]) -> None:
    """Unpack the mask folders of the selected sequences from the scene mask tars (all masks
    would be about 4 M files)."""
    by_scene = {}
    for sequence, _ in selection:
        if not (ROOT / sequence.scene / "masks" / sequence.name).exists():
            by_scene.setdefault(sequence.scene, []).append(sequence.name)
    for scene, names in by_scene.items():
        tar = ROOT / scene / "masks" / f"{scene}_masks.tar"
        members = [f"{scene}/masks/{name}" for name in names]
        subprocess.run(["tar", "-xf", str(tar), "-C", str(ROOT), *members], check=True)


def build(sequence: Sequence, start: int) -> dict:
    """Write the targets of one window (the arrays of vidHMR.datasets.pw3d_targets, 81 frames)
    and return a short report. valid: all joints in front of the camera (> 0.1 m) and at least
    one visible pixel of the person."""
    frames = np.arange(start, start + WINDOW)
    n_people = len(sequence.bodies)
    width, height = image_size(sequence)
    focal = camera(sequence)["focal_length"][frames]
    assert np.ptp(focal) < 1e-3, f"{sequence.name}: focal length changes"
    K = intrinsics(sequence, focal[0])
    size = target_size_of(width, height)
    smpl = SMPL("n", device="cpu", n_betas=11)

    out = {
        "global_orient": np.zeros((n_people, WINDOW, 3), np.float32),
        "body_pose": np.zeros((n_people, WINDOW, 69), np.float32),
        "betas_neutral": np.zeros((n_people, 10), np.float32),
        "pelvis": np.zeros((n_people, WINDOW, 3), np.float32),
        "joints_3d": np.zeros((n_people, WINDOW, 24, 3), np.float32),
        "joints_2d": np.zeros((n_people, WINDOW, 24, 2), np.float32),
        "valid": np.zeros((n_people, WINDOW), bool),
        "prompt_masks": np.zeros((n_people, N_LATENT, size[1], size[0]), bool),
        "K": K.astype(np.float32),
        "image_size": np.array([width, height], np.int32),
    }
    report = {"name": window_name(sequence), "start": start, "neutral_fit_mm": []}

    visible = np.stack([masks(sequence, frame, n_people) for frame in frames], 1)  # (A, F, H, W)
    for person, body in enumerate(sequence.bodies):
        with np.load(SMPL_GT / body.subject / body.motion / "motion_seq.npz") as gt:
            index = frames + body.start_frame
            assert index[-1] < len(gt["poses"]), f"{sequence.name} {person}: motion too short"
            pose, trans, betas = gt["poses"][index], gt["trans"][index], gt["betas"]
        j0 = rest_pelvis("n", betas)
        orient, pelvis = camera_frame(sequence, body, frames, pose, trans, j0)

        joint_rotations = axis_angle_to_matrix(
            torch.as_tensor(pose[:, 3:]).float().view(-1, 23, 3)
        )
        rotations = torch.cat([torch.as_tensor(orient).float()[:, None], joint_rotations], 1)
        joints = smpl.joints(rotations, betas).numpy() - j0 + pelvis[:, None]

        out["global_orient"][person] = axis_angles(orient)
        out["body_pose"][person] = pose[:, 3:]
        out["pelvis"][person] = pelvis
        out["joints_3d"][person] = joints
        out["joints_2d"][person] = geometry.pixels(joints, K)
        out["valid"][person] = (joints[..., 2] > 0.1).all(-1) & visible[person].any((-1, -2))
        out["betas_neutral"][person], fit_error = fit_neutral_betas("n", betas)
        report["neutral_fit_mm"].append(round(1000 * fit_error, 1))

        for latent, frame in enumerate(range(0, WINDOW, 4)):
            mask = visible[person, frame].astype(np.float32)
            out["prompt_masks"][person, latent] = (
                cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0.5
            )

    out["gender"] = np.array(["n"] * n_people)  # the GT mesh is the neutral body
    out["betas_gendered"] = out["betas_neutral"]
    path = targets_path(window_name(sequence))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    report["valid_frames"] = out["valid"].sum(1).tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("n", type=int, help="number of sequences (one window each)")
    selection = select(parser.parse_args().n)

    extract_masks(selection)
    todo = [(s, start) for s, start in selection if not targets_path(window_name(s)).exists()]
    with ProcessPoolExecutor(16) as pool:
        for report in pool.map(build, *zip(*todo, strict=True)) if todo else []:
            print(report, flush=True)
    print(f"{len(selection)} windows, {len(todo)} built", flush=True)


if __name__ == "__main__":
    main()


def extract_features(extract_window, backbone: str, n: int, layer: int, meta: dict) -> None:
    """Features of the selected windows: features/<backbone>/bedlam/<scene>/<seq>/ with
    layer_XX_int8.npy, layer_XX_scale.npy and meta.json (one chunk: the window's 21 latents).
    extract_window(pixels) -> {layer: (21, H, W, C)}. Done windows are skipped; the next two
    windows are decoded while the GPU works."""
    todo = [
        (sequence, start)
        for sequence, start in select(n)
        if not (FEATURES / backbone / window_name(sequence) / "meta.json").exists()
    ]

    def load(item):
        sequence, start = item
        return frames_to_tensor(
            read_frames(sequence, start), target_size_of(*image_size(sequence))
        )

    pool = ThreadPoolExecutor(max_workers=2)
    pending = [pool.submit(load, item) for item in todo[:2]]
    begin = time.time()
    for i, (sequence, start) in enumerate(todo):
        pixels = pending[i].result()
        if i + 2 < len(todo):
            pending.append(pool.submit(load, todo[i + 2]))
        features = extract_window(pixels)[layer].float().cpu()
        pending[i] = None  # free the frames

        out = FEATURES / backbone / window_name(sequence)
        out.mkdir(parents=True, exist_ok=True)
        save_int8(out, layer, features)
        info = {
            **meta,
            "seq": window_name(sequence),
            "video_start_frame": start,
            "n_frames": WINDOW,
            "n_latents": N_LATENT,
            "chunks": [[0, 0, N_LATENT]],
            "size_wh": list(pixels.shape[:-3:-1]),
            "dtype": "int8",
        }
        (out / "meta.json").write_text(json.dumps(info, indent=1))
        if i % 100 == 0 or i == len(todo) - 1:
            rate = (time.time() - begin) / (i + 1)
            print(f"{i + 1}/{len(todo)} windows, {rate:.2f} s each", flush=True)
