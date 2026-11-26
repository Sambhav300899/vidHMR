"""Camera-frame training targets for 3DPW (task 1 of docs/training_plan.md).

One file per sequence, generated/targets/pw3d/<sequence>.npz, with these arrays for each person
(A) and frame (F):

  global_orient  (A, F, 3)        body rotation in the camera frame, axis-angle
  body_pose      (A, F, 69)       the 23 joint rotations, axis-angle (relative to the parent joint,
                                  so the same in every frame of reference)
  betas_neutral  (A, 10)          neutral-SMPL betas fitted to the person's gendered body shape
  betas_gendered (A, 10)          the 3DPW betas, for the gendered GT mesh (with `gender`)
  gender         (A,)             "m" or "f"
  pelvis         (A, F, 3)        SMPL joint 0 in the camera frame, meters (the translation target)
  joints_3d      (A, F, 24, 3)    the 24 SMPL joints in the camera frame, meters (gendered GT)
  joints_2d      (A, F, 24, 2)    joints_3d projected with K, pixels of the original image
  valid          (A, F)           3DPW valid flag, and False on GT camera jumps
  prompt_masks   (A, ceil(F/4), H, W)  where the person's body is visible, on every 4th frame
                                  (the frames that latents are aligned with), at the DiT input
                                  size (H, W = 832, 480 portrait; 480, 832 landscape)
  K              (3, 3)           intrinsics, pixels of the original image
  image_size     (2,)             width, height of the original image

A "person" is a 3DPW "actor". The 3DPW world frame is used only here. The camera frame has x
right, y down, z forward. The global_orient of an upright body has an angle near 180° in this
frame, so its axis-angle can flip sign between frames: compare rotations as matrices or 6D.

    python -m vidHMR.datasets.pw3d_targets            # all sequences
    python -m vidHMR.datasets.pw3d_targets NAME ...
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from vidHMR import geometry, paths
from vidHMR.body.smpl import fit_neutral_betas, rest_pelvis
from vidHMR.datasets import pw3d
from vidHMR.features import target_size
from vidHMR.masks import load_masks

OUT = paths.GENERATED / "targets" / "pw3d"


def targets_path(name: str) -> Path:
    """The targets file of a 3DPW sequence, or of a window named "<dataset>/..." (for example
    bedlam/<scene>/<sequence>, vidHMR.datasets.bedlam), which has the same arrays."""
    return OUT.parent / f"{name}.npz" if "/" in name else OUT / f"{name}.npz"


def rotation_matrices(axis_angle: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices."""
    return np.stack([cv2.Rodrigues(a.reshape(3, 1))[0] for a in axis_angle])


def axis_angles(rotations: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 3) axis-angle."""
    return np.stack([cv2.Rodrigues(r)[0].reshape(3) for r in rotations])


def prompt_masks(name: str, n_frames: int, n_people: int) -> np.ndarray:
    """(A, L, H, W) bool: the visible-body mask of each person on frames 0, 4, 8, ..., resized
    from the full-size mask to the DiT input size the same way as the frames
    (vidHMR.features.load_frames). A pixel is on when more than half of its area is visible."""
    size = target_size(pw3d.frames(name)[0])  # (W, H)
    masks = []
    for frame in range(0, n_frames, 4):
        _, visible = load_masks(name, frame, n_people)  # (A, H, W) at the image size
        resized = [
            cv2.resize(v.astype(np.float32), size, interpolation=cv2.INTER_AREA) for v in visible
        ]
        masks.append(np.stack(resized) > 0.5)
    return np.stack(masks, axis=1)


def build(name: str) -> dict:
    """Write the targets of one sequence and return a short report."""
    sequence = pw3d.load_sequence(name)
    n_people, n_frames = sequence.n_actors, sequence.n_frames
    rotation = sequence.cam_poses[:, :3, :3]  # world-to-camera R of each frame
    height, width = cv2.imread(str(pw3d.frames(name)[0])).shape[:2]

    out = {
        "global_orient": np.zeros((n_people, n_frames, 3), np.float32),
        "body_pose": sequence.poses[:, :, 3:].astype(np.float32),
        "betas_neutral": np.zeros((n_people, 10), np.float32),
        "betas_gendered": sequence.betas.astype(np.float32),
        "gender": np.array(sequence.genders),
        "pelvis": np.zeros((n_people, n_frames, 3), np.float32),
        "joints_3d": np.zeros((n_people, n_frames, 24, 3), np.float32),
        "joints_2d": np.zeros((n_people, n_frames, 24, 2), np.float32),
        "valid": sequence.valid & ~pw3d.camera_jump_frames(sequence.cam_poses)[None],
        "prompt_masks": prompt_masks(name, n_frames, n_people),
        "K": sequence.K.astype(np.float32),
        "image_size": np.array([width, height], np.int32),
    }
    report = {"name": name, "neutral_fit_mm": []}

    for person in range(n_people):
        gender, betas = sequence.genders[person], sequence.betas[person]

        # World -> camera: camera point = R @ world point + T, for every frame.
        joints_world = sequence.joints3d[person]  # (F, 24, 3)
        joints_camera = geometry.world_to_cam(joints_world, sequence.cam_poses)
        out["joints_3d"][person] = joints_camera
        out["pelvis"][person] = joints_camera[:, 0]

        # The body rotation changes frame too; the joint rotations do not (they are relative).
        orient_world = rotation_matrices(sequence.poses[person, :, :3])
        out["global_orient"][person] = axis_angles(rotation @ orient_world)

        out["joints_2d"][person] = geometry.pixels(joints_camera, sequence.K)

        out["betas_neutral"][person], fit_error = fit_neutral_betas(gender, betas)
        report["neutral_fit_mm"].append(round(1000 * fit_error, 1))

        # Check: 3DPW joint 0 is J0 + trans (SMPL turns around J0, then adds trans).
        pelvis_world = rest_pelvis(gender, betas) + sequence.trans[person]
        assert np.abs(pelvis_world - joints_world[:, 0]).max() < 1e-3, f"{name} {person}: J0 + t"

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(targets_path(name), **out)
    report["valid_person_frames"] = int(out["valid"].sum())
    report["masked_by_camera_jump"] = int((sequence.valid & ~out["valid"]).sum())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="sequence names (default: all)")
    names = parser.parse_args().names or pw3d.sequences()

    with ProcessPoolExecutor(12) as pool:  # 12 processes: about 3 GB of RAM each at most
        for report in pool.map(build, names):
            print(report, flush=True)


if __name__ == "__main__":
    main()
