"""The losses of task 4 (docs/training_plan.md) and the training metrics. Float32, outside
autocast. Every per-frame term counts valid frames only.

  rotation_geodesic  angle between predicted and GT rotation (24 per frame), radians
  rotation_mse       mean squared difference of the 9 matrix entries (24 per frame)
  betas              l1 on the 10 neutral betas, once per item
  joints_relative    l1 on the 24 joints minus the pelvis, meters
  joints_3d          l1 on the 24 joints in the camera frame, meters * REFERENCE_DEPTH / GT depth
  pelvis             l1 on the pelvis point in the camera frame, the same depth scale
  joints_2d          l1 on the projected joints, image coordinates 0..1 (pixel / image size)
  direct_joints_relative, direct_joints_2d: as joints_relative and joints_2d, on the decoder's
                     direct joints (only when the prediction has them)

The depth scale makes the camera-frame error relative to the person's distance: without it, the
depth error of far BEDLAM people (median 6.6 m, 3DPW 3.4 m) took about 90 % of the gradient.
"""

import torch

from vidHMR.body.smpl import axis_angle_to_matrix
from vidHMR.model.decoder import START_DEPTH as REFERENCE_DEPTH  # the median 3DPW train depth

ANGLE_EPS = 1e-6  # keeps arccos away from +-1, where its gradient is infinite


def gt_rotations(batch):
    """(B, F, 24, 3, 3) from the camera-frame global_orient and the 23 body_pose rotations."""
    B, n_frames = batch["global_orient"].shape[:2]
    axis_angles = torch.cat([batch["global_orient"], batch["body_pose"]], -1)
    return axis_angle_to_matrix(axis_angles.reshape(B, n_frames, 24, 3))


def rotation_angles(rotations, gt):
    """Angle between two sets of rotation matrices (..., 3, 3) -> (...), radians. The arccos
    clamp leaves a floor of about 0.08 degrees."""
    cos_angle = ((rotations.transpose(-1, -2) @ gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(cos_angle.clamp(-1 + ANGLE_EPS, 1 - ANGLE_EPS))


def valid_mean(per_frame, valid):
    """Mean of per_frame (B, F) over the valid frames."""
    return (per_frame * valid).sum() / valid.sum().clamp_min(1)


def l1(prediction, target):
    """Mean absolute difference over everything after the (B, F) axes -> (B, F)."""
    return (prediction - target).abs().flatten(2).mean(-1)


def losses(prediction: dict, batch: dict) -> dict:
    """Unweighted loss terms (scalars). prediction from vidHMR.model.body.decode."""
    valid = batch["valid"].float()
    image_size = batch["image_size"][:, None, None].float()
    rotations = gt_rotations(batch)
    joints_3d = batch["joints_3d"]

    angle = rotation_angles(prediction["rotations"], rotations)
    squared = (prediction["rotations"] - rotations).pow(2)
    depth_scale = REFERENCE_DEPTH / batch["pelvis"][..., 2].clamp_min(0.1)  # (B, F)

    def joints_relative(key):
        return valid_mean(l1(prediction[key], joints_3d - joints_3d[:, :, :1]), valid)

    def joints_2d(key):
        return valid_mean(l1(prediction[key] / image_size, batch["joints_2d"] / image_size), valid)

    terms = {
        "rotation_geodesic": valid_mean(angle.mean(-1), valid),
        "rotation_mse": valid_mean(squared.flatten(2).mean(-1), valid),
        "betas": (prediction["betas"] - batch["betas_neutral"]).abs().mean(),
        "joints_relative": joints_relative("joints_relative"),
        "joints_3d": valid_mean(l1(prediction["joints_3d"], joints_3d) * depth_scale, valid),
        "pelvis": valid_mean(l1(prediction["pelvis"], batch["pelvis"]) * depth_scale, valid),
        "joints_2d": joints_2d("joints_2d"),
    }
    if "direct_joints_2d" in prediction:
        terms["direct_joints_relative"] = joints_relative("direct_joints_relative")
        terms["direct_joints_2d"] = joints_2d("direct_joints_2d")
    return terms


@torch.no_grad()
def metrics(prediction: dict, batch: dict) -> dict:
    """Errors in readable units, on valid frames:
    mpjpe_pelvis_mm         the 24 joints, after the pelvis is subtracted (its error is 0 there)
                            (not the 12-joint evaluation MPJPE of vidHMR.eval.sam3d_pw3d)
    pelvis_mm               the pelvis point in the camera frame
    joints_2d_px            the 24 projected joints
    rotation_deg            the 24 rotations; global_rotation_deg: the global rotation only
    rotations_over_90_pct   % of the 24 rotations that are more than 90 degrees off
    direct_joints_2d_px, pnp_accepted_pct: the direct 2D joints, and the % of frames whose pelvis
                            comes from the mixed PnP (when the prediction has them)"""
    valid = batch["valid"].float()
    joints_3d = batch["joints_3d"]
    per_joint = (prediction["joints_relative"] - (joints_3d - joints_3d[:, :, :1])).norm(dim=-1)
    mpjpe = per_joint.mean(-1)  # (B, F)
    pelvis = (prediction["pelvis"] - batch["pelvis"]).norm(dim=-1)
    pixels = (prediction["joints_2d"] - batch["joints_2d"]).norm(dim=-1).mean(-1)
    angle = torch.rad2deg(rotation_angles(prediction["rotations"], gt_rotations(batch)))

    out = {
        "mpjpe_pelvis_mm": 1000 * valid_mean(mpjpe, valid).item(),
        "pelvis_mm": 1000 * valid_mean(pelvis, valid).item(),
        "joints_2d_px": valid_mean(pixels, valid).item(),
        "rotation_deg": valid_mean(angle.mean(-1), valid).item(),
        "global_rotation_deg": valid_mean(angle[..., 0], valid).item(),
        "rotations_over_90_pct": 100 * valid_mean((angle > 90).float().mean(-1), valid).item(),
    }
    if "direct_joints_2d" in prediction:
        direct = (prediction["direct_joints_2d"] - batch["joints_2d"]).norm(dim=-1).mean(-1)
        out["direct_joints_2d_px"] = valid_mean(direct, valid).item()
        out["pnp_accepted_pct"] = 100 * valid_mean(prediction["pnp_accepted"], valid).item()
    return out
