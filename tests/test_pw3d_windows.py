"""Checks of the 3DPW training targets and items (vidHMR.datasets.pw3d_targets, pw3d_windows).

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_pw3d_windows.py -q

Needs the targets of all sequences: python -m vidHMR.datasets.pw3d_targets
"""

import json

import cv2
import numpy as np
import pytest
import torch

from vidHMR.body.smpl import SMPL, rest_pelvis
from vidHMR.datasets import pw3d
from vidHMR.datasets.pw3d_targets import targets_path
from vidHMR.datasets.pw3d_windows import (
    Pw3dWindows,
    SameGridBatches,
    make_loader,
    split_sequences,
    window_first_frames,
)
from vidHMR.features import N_LATENT, ROOT, WINDOW, load_frames, target_size
from vidHMR.masks import load_masks

# Two people (portrait), one person (landscape), and the train sequence with a camera jump.
SAMPLE = ["courtyard_arguing_00", "outdoors_freestyle_00", "outdoors_slalom_00"]


def load_targets(name: str) -> dict:
    with np.load(targets_path(name)) as targets:
        return dict(targets)


def rebuild_joints(targets: dict, person: int, gender: str, betas: np.ndarray) -> np.ndarray:
    """SMPL joints (F, 24, 3) in the camera frame from the stored camera-frame parameters:
    the body turns around its rest pelvis J0, so the SMPL translation is pelvis - J0."""
    model = SMPL(gender, device="cpu")
    poses = np.concatenate([targets["global_orient"][person], targets["body_pose"][person]], 1)
    translation = targets["pelvis"][person] - rest_pelvis(gender, betas)
    with torch.no_grad():
        _, joints = model(poses, betas, translation)
    return joints.numpy()


def test_window_layout():
    """Every window of every sequence uses latents first_frame // 4 .. + 20 (asserted inside)."""
    for name in pw3d.sequences():
        meta = json.loads((ROOT / "wan22_fun5b_t0" / name / "meta.json").read_text())
        assert meta["n_frames"] == pw3d.load_sequence(name).n_frames
        first_frames = window_first_frames(meta)
        assert first_frames[0] == 0 and first_frames[-1] + WINDOW <= meta["n_frames"]


@pytest.mark.parametrize("name", SAMPLE)
def test_gendered_rebuild_matches_joints(name):
    """Camera-frame global_orient, body_pose and pelvis give back the GT joints (< 1 mm)."""
    sequence, targets = pw3d.load_sequence(name), load_targets(name)
    for person in range(sequence.n_actors):
        gender, betas = sequence.genders[person], sequence.betas[person]
        joints = rebuild_joints(targets, person, gender, betas)
        assert np.abs(joints - targets["joints_3d"][person]).max() < 1e-3


@pytest.mark.parametrize("name", SAMPLE)
def test_neutral_rebuild_is_close(name):
    """Neutral SMPL with the fitted betas and the GT pose, against the gendered GT joints: about
    1 mm for women (the neutral and female files share the joint regressor) and about 12 mm for
    men (the male file has its own joint regressor)."""
    sequence, targets = pw3d.load_sequence(name), load_targets(name)
    for person in range(sequence.n_actors):
        betas = targets["betas_neutral"][person]
        joints = rebuild_joints(targets, person, "n", betas)
        error = np.linalg.norm(joints - targets["joints_3d"][person], axis=-1).mean()
        limit = 0.002 if sequence.genders[person] == "f" else 0.015
        assert error < limit, f"{name} {person}: {1000 * error:.1f} mm"


def test_camera_jump_mask():
    """Valid person-frames on a GT camera jump: 8 train, 17 validation, 38 test."""
    on_jump = dict.fromkeys(pw3d.SPLITS, 0)
    for name in pw3d.sequences():
        sequence = pw3d.load_sequence(name)
        on_jump[sequence.split] += int((sequence.valid & ~load_targets(name)["valid"]).sum())
    assert on_jump == {"train": 8, "validation": 17, "test": 38}

    # outdoors_slalom_00: the camera jumps on frames 139-141; 3DPW already marks 141 as invalid.
    sequence = pw3d.load_sequence("outdoors_slalom_00")
    assert np.flatnonzero(pw3d.camera_jump_frames(sequence.cam_poses)).tolist() == [139, 140, 141]
    newly_masked = sequence.valid[0] & ~load_targets("outdoors_slalom_00")["valid"][0]
    assert np.flatnonzero(newly_masked).tolist() == [139, 140]


@pytest.mark.parametrize("name", SAMPLE)
def test_joints_2d_fall_on_the_person(name):
    """Projected GT joints are near the 3DPW 2D keypoints (median 11 px over the dataset)."""
    smpl_index = [17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]  # shoulders .. ankles
    openpose_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    sequence, targets = pw3d.load_sequence(name), load_targets(name)
    for person in range(sequence.n_actors):
        keypoints = sequence.poses2d[person][:, :2, openpose_index].transpose(0, 2, 1)
        confident = sequence.poses2d[person][:, 2, openpose_index] > 0.3
        good = confident & targets["valid"][person][:, None]
        distance = np.linalg.norm(targets["joints_2d"][person][:, smpl_index] - keypoints, axis=-1)
        assert np.median(distance[good]) < 20


def test_prompt_masks_cover_the_joints():
    """The visible masks have the size of the DiT input frames and cover the projected joints of
    valid frames (every 4th frame). One latent (4 frames) of misalignment gives 0.94."""
    name = "courtyard_arguing_00"
    targets = load_targets(name)
    first_image = pw3d.frames(name)[0]
    frame_size = load_frames([first_image], target_size(first_image)).shape[2:]
    assert targets["prompt_masks"].shape[-2:] == frame_size

    mask, joints = targets["prompt_masks"], targets["joints_2d"][:, ::4]
    scale = np.array([mask.shape[-1], mask.shape[-2]]) / targets["image_size"]  # (W, H)
    column, row = (joints * scale).astype(int).transpose(3, 0, 1, 2)
    column, row = column.clip(0, mask.shape[-1] - 1), row.clip(0, mask.shape[-2] - 1)
    people, frames = np.indices(joints.shape[:2])
    hit = mask[people[..., None], frames[..., None], row, column]
    assert hit[targets["valid"][:, ::4]].mean() > 0.98


def test_items_and_batches():
    """Every item once per epoch, no batch mixes portrait and landscape grids, correct shapes."""
    dataset = Pw3dWindows(SAMPLE)
    batches = SameGridBatches(dataset, batch_size=4, seed=0)
    for _ in range(2):  # two epochs
        seen = []
        for indices in batches:
            assert len({dataset.grid_shape(i) for i in indices}) == 1
            seen += indices
        assert sorted(seen) == list(range(len(dataset)))
    assert len(batches) == sum(1 for _ in batches)

    batch = next(iter(make_loader(dataset, batch_size=4, seed=0)))
    size = len(batch["sequence"])
    assert len(batch["features"]) == size  # a list: stacked on the GPU by vidHMR.train.to_cuda
    assert batch["features"][0].shape[0] == N_LATENT and batch["features"][0].dtype == torch.int8
    assert batch["joints_3d"].shape == (size, WINDOW, 24, 3)
    assert batch["valid"].shape == (size, WINDOW)


def test_item_content():
    """The last (overlapping) window of a sequence: the right latents, person and frames."""
    name, person = "courtyard_arguing_00", 1
    dataset = Pw3dWindows([name])
    first_frame = max(f for n, f, p in dataset.items if p == person)
    item = dataset[dataset.items.index((name, first_frame, person))]

    stored = np.load(ROOT / "wan22_fun5b_t0" / name / "layer_15_int8.npy", mmap_mode="r")
    assert np.array_equal(
        item["features"].numpy(), stored[first_frame // 4 : first_frame // 4 + N_LATENT]
    )
    targets = load_targets(name)
    frames = slice(first_frame, first_frame + WINDOW)
    assert np.array_equal(item["joints_3d"].numpy(), targets["joints_3d"][person, frames])
    assert np.array_equal(item["valid"].numpy(), targets["valid"][person, frames])
    assert item["gender"] == pw3d.load_sequence(name).genders[person]


def test_item_without_valid_latent_is_left_out():
    """courtyard_warmWelcome_00, window 80, person 1 has no valid frame."""
    dataset = Pw3dWindows(["courtyard_warmWelcome_00"])
    assert ("courtyard_warmWelcome_00", 80, 1) not in dataset.items
    assert dataset.items_without_valid_latent == 1


def test_split_sequences():
    assert "outdoors_crosscountry_00" not in split_sequences("validation")
    assert len(split_sequences("train")) == 24


@pytest.mark.parametrize(
    ("name", "first_frame", "person", "latent"),
    [
        ("courtyard_arguing_00", 240, 1, 0),
        ("courtyard_relaxOnBench_00", 320, 0, 1),
        ("outdoors_freestyle_00", 160, 0, 0),  # landscape
    ],
)
def test_prompt_mask(name, first_frame, person, latent):
    """The prompt is the visible mask of the first valid latent's frame, resized to the DiT input
    size (32 x 32 mask pixels per feature token); checked against a mask made here from the
    full-size PNG."""
    dataset = Pw3dWindows([name])
    item = dataset[dataset.items.index((name, first_frame, person))]
    valid_latents = item["valid"][::4].numpy()
    assert item["prompt_latent"] == latent and valid_latents[latent]
    assert not valid_latents[:latent].any()

    mask = item["prompt_mask"].numpy()
    assert mask.shape == tuple(32 * np.array(dataset.grid_shape(0)))
    frame = first_frame + 4 * latent
    _, visible = load_masks(name, frame, pw3d.load_sequence(name).n_actors)
    size = target_size(pw3d.frames(name)[frame])
    expected = cv2.resize(visible[person].astype(np.float32), size, interpolation=cv2.INTER_AREA)
    assert np.array_equal(mask, expected > 0.5) and mask.any()
