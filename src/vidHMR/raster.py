"""nvdiffrast helpers shared by the viewer overlays and the mask renderer."""

import numpy as np
import torch

from vidHMR.body.smpl import SMPL

NEAR, FAR = 0.05, 100.0


def to_clip(verts_cam: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """OpenCV camera coordinates (B,V,3) -> nvdiffrast clip coordinates (B,V,4).
    Output row 0 is the top image row (nvdiffrast puts y=-1 at row 0)."""
    X, Y, Z = verts_cam.unbind(-1)
    Zs = Z.clamp_min(1e-6)
    x = 2 * (K[0, 0] * X / Zs + K[0, 2]) / width - 1
    y = 2 * (K[1, 1] * Y / Zs + K[1, 2]) / height - 1
    z = (Z * (FAR + NEAR) - 2 * FAR * NEAR) / (FAR - NEAR)
    return torch.stack([x * Z, y * Z, z, Z], -1)


def gt_verts_cam(seq, actor: int, frames: np.ndarray, model: SMPL) -> torch.Tensor:
    """GT SMPL vertices of one 3DPW actor in OpenCV camera coordinates, (B,V,3)."""
    E = torch.as_tensor(seq.cam_poses[frames], dtype=torch.float32, device=model.device)
    v, _ = model(seq.poses[actor, frames], seq.betas[actor], seq.trans[actor, frames])
    return (E[:, None, :3, :3] @ v[..., None])[..., 0] + E[:, None, :3, 3]
