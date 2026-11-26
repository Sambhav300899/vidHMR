"""Camera geometry helpers. Extrinsics are 4x4 world-to-camera matrices (OpenCV convention)."""

import numpy as np


def world_to_cam(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """World points (..., N, 3) -> camera points (..., N, 3): R @ point + T. `extrinsic` is one
    (4, 4) matrix for all points, or (..., 4, 4) with one matrix per leading index (e.g. frame)."""
    rotation, translation = extrinsic[..., :3, :3], extrinsic[..., None, :3, 3]
    return np.einsum("...ij,...kj->...ki", rotation, points) + translation


def pixels(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection of camera points (..., 3) to pixels (..., 2): K @ (point / depth)."""
    depth = np.maximum(points_camera[..., 2:3], 1e-6)
    focal = np.array([K[0, 0], K[1, 1]])
    return points_camera[..., :2] / depth * focal + K[:2, 2]


def project(
    points_world: np.ndarray, extrinsic: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels. Returns (..., 2) pixels and (...) depth."""
    points_camera = world_to_cam(points_world, extrinsic)
    return pixels(points_camera, K), points_camera[..., 2]


def camera_to_world(extrinsic: np.ndarray) -> np.ndarray:
    """Invert a world-to-camera 4x4 matrix."""
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out
