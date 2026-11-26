"""3DPW (3D Poses in the Wild) loading, subsets and statistics.

CLI:  python -m vidHMR.datasets.pw3d stats
"""

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from vidHMR import paths

SPLITS = ("train", "validation", "test")
FPS = 30

# 3DPW-OCC: the occlusion subset (sequence-name prefixes), as used by OOH / ROMP / PARE.
OCC_PREFIXES = (
    "courtyard_backpack",
    "courtyard_basketball",
    "courtyard_bodyScannerMotions",
    "courtyard_box",
    "courtyard_golf",
    "courtyard_jacket",
    "courtyard_laceShoe",
    "downtown_stairs",
    "flat_guitar",
    "flat_packBags",
    "outdoors_climbing",
    "outdoors_crosscountry",
    "outdoors_fencing",
    "outdoors_freestyle",
    "outdoors_golf",
    "outdoors_parcours",
    "outdoors_slalom",
)
# 3DPW-PC: person-crowd frame ranges [start, end), from ROMP (romp/lib/dataset/pw3d.py).
PC_RANGES = {
    "courtyard_basketball_00": (200, 280),
    "courtyard_captureSelfies_00": (500, 600),
    "courtyard_dancing_00": (60, 370),
    "courtyard_dancing_01": (60, 270),
    "courtyard_hug_00": (100, 500),
    "downtown_bus_00": (1620, 1900),
}
# Image folders without an annotation file (do not use for evaluation).
UNANNOTATED = ("downtown_cafe_01",)
# Sequences whose GT camera pose is wrong in the 3DPW file: not for training or averaged metrics.
BAD_GT_SEQUENCES = {
    "outdoors_crosscountry_00": (
        "camera translation in the 3DPW file is wrong (GT behind the camera)"
    ),
}
# A GT camera turn larger than this between two frames (1/30 s) is an error in the 3DPW file.
MAX_CAMERA_TURN_DEGREES = 10.0


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load 3DPW pickles without allowing arbitrary classes."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("numpy", "builtins"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked class {module}.{name}")


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f, encoding="latin1").load()


@dataclass
class Sequence:
    name: str
    split: str
    K: np.ndarray  # 3x3 intrinsics
    cam_poses: np.ndarray  # F x 4 x 4 world-to-camera
    joints3d: np.ndarray  # A x F x 24 x 3 world SMPL joints
    poses2d: np.ndarray  # A x F x 3 x 18 (x, y, conf), OpenPose COCO-18
    valid: np.ndarray  # A x F bool (campose_valid)
    trans: np.ndarray  # A x F x 3
    poses: np.ndarray  # A x F x 72 SMPL axis-angle
    betas: np.ndarray  # A x 10
    genders: list

    @property
    def n_frames(self) -> int:
        return self.cam_poses.shape[0]

    @property
    def n_actors(self) -> int:
        return self.joints3d.shape[0]

    def image_path(self, frame: int) -> Path:
        return paths.PW3D / "imageFiles" / self.name / f"image_{frame:05d}.jpg"


def find_split(name: str) -> str:
    for split in SPLITS:
        if (paths.PW3D / "sequenceFiles" / split / f"{name}.pkl").exists():
            return split
    raise FileNotFoundError(f"no 3DPW sequence named {name}")


@lru_cache(maxsize=8)
def load_sequence(name: str) -> Sequence:
    split = find_split(name)
    d = _load_pickle(paths.PW3D / "sequenceFiles" / split / f"{name}.pkl")
    n_frames = len(d["cam_poses"])
    return Sequence(
        name=name,
        split=split,
        K=np.asarray(d["cam_intrinsics"], dtype=np.float64),
        cam_poses=np.asarray(d["cam_poses"], dtype=np.float64),
        joints3d=np.stack([np.asarray(j).reshape(n_frames, 24, 3) for j in d["jointPositions"]]),
        poses2d=np.stack([np.asarray(p) for p in d["poses2d"]]),
        valid=np.stack([np.asarray(v).astype(bool) for v in d["campose_valid"]]),
        trans=np.stack([np.asarray(t) for t in d["trans"]]),
        poses=np.stack([np.asarray(p) for p in d["poses"]]),
        betas=np.stack([np.asarray(b).reshape(-1)[:10] for b in d["betas"]]),
        genders=list(d["genders"]),
    )


def camera_jump_frames(cam_poses: np.ndarray) -> np.ndarray:
    """Bool (F,): True on both frames of each GT camera turn > MAX_CAMERA_TURN_DEGREES between two
    frames. The valid flags do not mark these frames, but the GT body is off the person there.
    cam_poses: (F, 4, 4) world-to-camera."""
    rotation = cam_poses[:, :3, :3]
    relative = rotation[1:] @ rotation[:-1].transpose(0, 2, 1)
    cos_turn = (np.trace(relative, axis1=1, axis2=2) - 1) / 2
    turn_degrees = np.degrees(np.arccos(np.clip(cos_turn, -1, 1)))

    jump = np.zeros(len(cam_poses), bool)
    before = np.flatnonzero(turn_degrees > MAX_CAMERA_TURN_DEGREES)
    jump[before] = jump[before + 1] = True
    return jump


def sequences() -> list[str]:
    """All annotated sequences (train, validation, test), by name."""
    return sorted(p.stem for p in (paths.PW3D / "sequenceFiles").glob("*/*.pkl"))


def oc_sequences() -> list[str]:
    """The 3DPW-OC (object occlusion) sequences. Same set as the "occ" subset."""
    return [s for s in sequences() if is_occ(s)]


def frames(seq: str) -> list[Path]:
    """The sequence's RGB frames in order (portrait or landscape, 1080x1920)."""
    return sorted((paths.PW3D / "imageFiles" / seq).glob("image_*.jpg"))


def is_occ(name: str) -> bool:
    return any(name.startswith(p + "_") for p in OCC_PREFIXES)


def subsets_of(row: dict) -> list[str]:
    """Names of the subsets that a sequence belongs to."""
    out = [row["split"]]
    if row["occ"]:
        out.append("occ")
    if row["pc_range"]:
        out.append("pc")
    if row["actors"] == 2:
        out.append("two_actors")
    return out


def build_index() -> list[dict]:
    """One metadata row per annotated sequence."""
    rows = []
    for split in SPLITS:
        for pkl in sorted((paths.PW3D / "sequenceFiles" / split).glob("*.pkl")):
            s = load_sequence(pkl.stem)
            n_img = len(list((paths.PW3D / "imageFiles" / s.name).glob("*.jpg")))
            rows.append(
                dict(
                    name=s.name,
                    split=split,
                    scene=s.name.split("_")[0],
                    frames=s.n_frames,
                    images=n_img,
                    actors=s.n_actors,
                    genders="".join(s.genders),
                    valid_pf=int(s.valid.sum()),
                    all_pf=int(s.valid.size),
                    occ=is_occ(s.name),
                    pc_range=list(PC_RANGES.get(s.name, [])),
                )
            )
            load_sequence.cache_clear()  # keep memory low during the full scan
    return rows


def index(refresh: bool = False) -> list[dict]:
    """Cached sequence index (generated/cache/pw3d_index.json)."""
    cache = paths.generated("cache", "pw3d_index.json")
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    rows = build_index()
    cache.write_text(json.dumps(rows, indent=1))
    return rows


def summary(rows: list[dict]) -> dict[str, dict]:
    """Aggregate counts per split and in total."""
    out = {}
    for key in (*SPLITS, "total"):
        sel = [r for r in rows if key == "total" or r["split"] == key]
        out[key] = dict(
            sequences=len(sel),
            frames=sum(r["frames"] for r in sel),
            actor_tracks=sum(r["actors"] for r in sel),
            two_actor_seqs=sum(r["actors"] == 2 for r in sel),
            valid_person_frames=sum(r["valid_pf"] for r in sel),
            occ_seqs=sum(r["occ"] for r in sel),
            occ_frames=sum(r["frames"] for r in sel if r["occ"]),
        )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    rows = index(refresh=args.refresh)
    print(f"{'split':<11}{'sequence':<33}{'frames':>7}{'actors':>7}{'valid':>14}  subsets")
    for r in rows:
        print(
            f"{r['split']:<11}{r['name']:<33}{r['frames']:>7}{r['actors']:>7}{r['valid_pf']:>7}"
            f"/{r['all_pf']:<6}  {','.join(subsets_of(r)[1:])}"
        )
    for k, v in summary(rows).items():
        print(k, v)
