"""3D pose error metrics (numpy, batched). Inputs are (N, J, 3) in metres; outputs are per-sample "
"(N, J) in mm."""

import numpy as np


def root_align(x: np.ndarray, root: list[int]) -> np.ndarray:
    """Subtract the mean of the `root` joints (e.g. the two hips -> pelvis)."""
    return x - x[:, root].mean(1, keepdims=True)


def joint_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint Euclidean error in mm."""
    return np.linalg.norm(pred - gt, axis=-1) * 1000


def procrustes(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align each pred (N,J,3) to its gt: scale, rotation and translation (Umeyama)."""
    mp, mg = pred.mean(1, keepdims=True), gt.mean(1, keepdims=True)
    p, g = pred - mp, gt - mg
    U, S, Vt = np.linalg.svd(np.einsum("nji,njk->nik", p, g))  # p^T g
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    D = np.ones_like(S)
    D[:, -1] = d
    R = np.einsum("nij,nj,njk->nik", U, D, Vt)  # rotation applied as p @ R
    scale = (S * D).sum(1) / (p**2).sum((1, 2))
    return scale[:, None, None] * np.einsum("nji,nik->njk", p, R) + mg


def mpjpe(pred, gt, root):
    """Root-aligned per-joint error (N,J) in mm."""
    return joint_errors(root_align(pred, root), root_align(gt, root))


def pa_mpjpe(pred, gt):
    """Per-joint error after Procrustes alignment (N,J) in mm."""
    return joint_errors(procrustes(pred, gt), gt)
