"""3DPW layout: frames per sequence (imageFiles/) and SMPL ground truth (sequenceFiles/)."""

from pathlib import Path

from vidHMR import GENERATED

ROOT = GENERATED / "datasets" / "3DPW"
FPS = 30

# 3DPW-OC (object occlusion) from OOH (Zhang et al., CVPR 2020), as ROMP defines it
# (romp/lib/dataset/pw3d.py): base names; every recording of each is in the subset, from all
# three splits. On our copy: 23 sequences, 20,529 frames.
OC_BASES = (
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


def sequences() -> list[str]:
    """All 60 sequences (train, validation, test), by name."""
    return sorted(p.stem for p in (ROOT / "sequenceFiles").glob("*/*.pkl"))


def oc_sequences() -> list[str]:
    """The 3DPW-OC sequences."""
    return [s for s in sequences() if s[:-3] in OC_BASES]


def frames(seq: str) -> list[Path]:
    """The sequence's RGB frames in order (portrait or landscape, 1080x1920)."""
    return sorted((ROOT / "imageFiles" / seq).glob("image_*.jpg"))
