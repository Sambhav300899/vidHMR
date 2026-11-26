"""Project paths. Data and artifacts live in generated/ (gitignored); see vidHMR.GENERATED."""

from pathlib import Path

from vidHMR import GENERATED, REPO

PW3D = GENERATED / "datasets" / "3DPW"
SMPL_DIR = GENERATED / "datasets" / "smpl"
CHECKPOINTS = GENERATED / "checkpoints"
SAM3D_REPO = REPO / "external" / "sam-3d-body"


def generated(*parts: str) -> Path:
    """Return a path under generated/, creating its parent folder."""
    p = GENERATED.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
