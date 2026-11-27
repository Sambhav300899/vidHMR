"""3DPW training items: one 81-frame feature window × one person (task 1 of
docs/training_plan.md).

An item holds the frozen DiT features of one window (one backbone layer, loaded into RAM once)
and the camera-frame targets of one person for the 81 frames of that window
(vidHMR.datasets.pw3d_targets).

Window layout (from vidHMR.features): a window that starts at frame `first_frame` (a multiple of
4) covers frames first_frame .. first_frame + 80 and uses the stored latents
first_frame // 4 .. first_frame // 4 + 20. Latent k of the window belongs to frame
first_frame + 4k. Each extraction pass stores only its new latents, so latent 0 of every window
after the first comes from the pass before it, and the last window of a sequence (which overlaps
the window before it) takes some of its first latents (up to 19) from the earlier pass. The frame
alignment is the same either way.

Prompt (task 3): the person's visible mask on the first latent whose frame is valid, at the DiT
input size (one feature token = 32 x 32 mask pixels). Items with no valid latent frame are left
out.

    dataset = Pw3dWindows(split_sequences("train"))
    loader = make_loader(dataset, batch_size=8, seed=0)
"""

import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from vidHMR.datasets import pw3d
from vidHMR.datasets.pw3d_targets import targets_path
from vidHMR.features import LONG, N_LATENT, ROOT, SHORT, WINDOW


def split_sequences(split: str) -> list[str]:
    """The sequences of a split, without the ones with broken GT (pw3d.BAD_GT_SEQUENCES)."""
    return [
        name
        for name in pw3d.sequences()
        if pw3d.find_split(name) == split and name not in pw3d.BAD_GT_SEQUENCES
    ]


def first_valid_latent(valid_window: np.ndarray, masks_window: np.ndarray) -> int | None:
    """The latent that holds the prompt: the first latent whose frame (every 4th frame of the
    window's valid flags) is valid and whose visible mask (masks_window: the window's 21 prompt
    masks, bit-packed or not) is not empty; None if there is none."""
    usable = np.flatnonzero(valid_window[::4] & masks_window.any(axis=(-1, -2)))
    return int(usable[0]) if len(usable) else None


def window_first_frames(meta: dict) -> list[int]:
    """The first frame of each extraction window, from a features meta.json."""
    for first_frame, _, latent_end in meta["chunks"]:
        assert first_frame % 4 == 0 and latent_end == first_frame // 4 + N_LATENT, meta["seq"]
    return [first_frame for first_frame, _, _ in meta["chunks"]]


class Pw3dWindows(Dataset):
    """Items (sequence, first_frame, person). Items without a valid latent frame (first_frame +
    4k) with a visible mask are left out: the prompt needs one. A sequence is a 3DPW sequence or
    a window of another dataset (vidHMR.datasets.bedlam), stored the same way.

    live: the items have no features (vidHMR.train runs the DiT on the window's stored VAE
    latents, vidHMR.features.wan22.window_latents), and with all_starts a window starts at every
    4th frame, not only at the extraction windows."""

    def __init__(
        self,
        sequences: list[str],
        backbone: str = "wan22_fun5b_t0",
        layer: int = 15,
        live: bool = False,
        all_starts: bool = False,
    ):
        self.live = live
        self.features = {}  # sequence -> (latents, grid height, grid width, channels) int8
        self.scales = {}  # sequence -> (channels,) float32: features = int8 * scale
        self.targets = {}  # sequence -> the arrays of vidHMR.datasets.pw3d_targets
        self.items = []  # (sequence, first_frame, person)
        self.items_without_valid_latent = 0

        for name in sequences:
            folder = ROOT / backbone / name
            meta = json.loads((folder / "meta.json").read_text())
            if not live:
                self.features[name] = np.load(folder / f"layer_{layer:02d}_int8.npy")
                self.scales[name] = np.load(folder / f"layer_{layer:02d}_scale.npy")
            with np.load(targets_path(name)) as targets:
                self.targets[name] = dict(targets)
            masks = self.targets[name]["prompt_masks"]  # 8 times smaller in RAM: one bit each
            self.targets[name]["prompt_masks"] = np.packbits(masks, axis=-1)

            valid = self.targets[name]["valid"]  # (people, frames)
            masks = self.targets[name]["prompt_masks"]  # (people, latents, H, W / 8)
            assert valid.shape[1] == meta["n_frames"], name
            starts = window_first_frames(meta)
            if all_starts:
                starts = range(0, meta["n_frames"] - WINDOW + 1, 4)
            for first_frame in starts:
                window = slice(first_frame, first_frame + WINDOW)
                latents = slice(first_frame // 4, first_frame // 4 + N_LATENT)
                for person in range(valid.shape[0]):
                    if (
                        first_valid_latent(valid[person, window], masks[person, latents])
                        is not None
                    ):
                        self.items.append((name, first_frame, person))
                    else:
                        self.items_without_valid_latent += 1

    def __len__(self) -> int:
        return len(self.items)

    def pin_memory(self) -> None:
        """Move the features to page-locked RAM (one sequence at a time), so that their copies to
        the GPU can run while the GPU computes (vidHMR.train.on_gpu)."""
        for name in self.features:
            self.features[name] = torch.from_numpy(self.features[name]).pin_memory()

    def grid_shape(self, index: int) -> tuple[int, int]:
        """(height, width) of the feature grid: 26 x 15 for portrait, 15 x 26 for landscape."""
        width, height = self.targets[self.items[index][0]]["image_size"]
        return (LONG // 32, SHORT // 32) if height > width else (SHORT // 32, LONG // 32)

    def __getitem__(self, index: int) -> dict:
        name, first_frame, person = self.items[index]
        frames = slice(first_frame, first_frame + WINDOW)
        latents = slice(first_frame // 4, first_frame // 4 + N_LATENT)
        targets = self.targets[name]

        # Prompt: the visible mask of the first latent whose frame is valid (and mask not empty).
        masks = targets["prompt_masks"][person, latents]
        prompt_latent = first_valid_latent(targets["valid"][person, frames], masks)
        prompt_mask = np.unpackbits(masks[prompt_latent], axis=-1).astype(bool)

        def person_frames(key):
            return torch.from_numpy(targets[key][person, frames])

        stored = {}
        if not self.live:
            stored = {
                "features": torch.as_tensor(self.features[name][latents]),  # (21, H, W, C) int8
                "feature_scale": torch.as_tensor(self.scales[name]),  # (C,): int8 * scale
            }
        return stored | {
            "global_orient": person_frames("global_orient"),  # (81, 3) camera frame
            "body_pose": person_frames("body_pose"),  # (81, 69)
            "betas_neutral": torch.from_numpy(targets["betas_neutral"][person]),  # (10,)
            "betas_gendered": torch.from_numpy(targets["betas_gendered"][person]),  # (10,) GT mesh
            "gender": str(targets["gender"][person]),  # "m" or "f", for the GT mesh
            "pelvis": person_frames("pelvis"),  # (81, 3) meters, camera frame
            "joints_3d": person_frames("joints_3d"),  # (81, 24, 3) meters, camera frame
            "joints_2d": person_frames("joints_2d"),  # (81, 24, 2) pixels
            "valid": person_frames("valid"),  # (81,) False: no loss on this frame
            "prompt_mask": torch.from_numpy(prompt_mask),  # (32H, 32W) bool, DiT input size
            "prompt_latent": prompt_latent,  # the latent (0..20) that the prompt mask belongs to
            "K": torch.from_numpy(targets["K"]),  # (3, 3) pixels
            "image_size": torch.from_numpy(targets["image_size"]),  # (2,) width, height
            "sequence": name,
            "first_frame": first_frame,
            "person": person,
        }


class SameGridBatches:
    """Batches of item indices in a new random order each epoch. All items in one batch have the
    same feature grid, because portrait and landscape videos give different grids."""

    def __init__(self, dataset: Pw3dWindows, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.random = np.random.default_rng(seed)
        self.groups = {}  # grid shape -> item indices
        for index in range(len(dataset)):
            self.groups.setdefault(dataset.grid_shape(index), []).append(index)

    def __iter__(self):
        batches = []
        for indices in self.groups.values():
            indices = self.random.permutation(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size].tolist())
        self.random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def collate(items: list[dict]) -> dict:
    """default_collate, but the features stay a list of per-item tensors: stacking 50 MB per item
    on the CPU was the slowest step of training. vidHMR.train.to_cuda stacks them on the GPU."""
    features = [item.pop("features") for item in items]
    return default_collate(items) | {"features": features}


def make_loader(dataset: Pw3dWindows, batch_size: int, seed: int) -> DataLoader:
    """The data is in RAM already, so no worker processes (they were slower: each batch then goes
    through shared memory)."""
    batches = SameGridBatches(dataset, batch_size, seed)
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, collate_fn=collate)
