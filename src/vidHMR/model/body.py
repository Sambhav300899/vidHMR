"""From the decoder outputs to the body (task 2 of docs/training_plan.md): rotation matrices, the
pelvis point (inverse projection with K, or ACE's mixed PnP with the direct 2D joints), and the
joints of the neutral SMPL body in the camera frame and in the image. Runs in float32, outside
autocast.
"""

import torch
import torch.nn.functional as F  # noqa: N812 (the usual PyTorch name)

from vidHMR.features import LONG, SHORT

TOKEN_PX = 32  # pixels of the DiT input for each feature token

MIN_DEPTH = 0.1  # meters; projected points closer than this are clamped to it

# ACE's mixed PnP gates (memory_projector.py, _decode_cam_trans_mixed_pnp).
PNP_Z_NEAR = 0.05  # meters: a joint votes only in front of the camera
PNP_MARGIN = 0.02  # ... and this fraction of the image inside the outer token centres
PNP_MIN_JOINTS = 6  # a frame needs this many votes, and an RMS residual of at most
PNP_RESIDUAL_FRACTION = 0.25  # max(0.25 * the diagonal of the voting 2D joints' box,
PNP_RESIDUAL_MIN_PX = 15.0  # 15 px); else it keeps the pelvis of the head


def rotation_6d_to_matrix(x):
    """ACE's Gram-Schmidt: (..., 6) -> (..., 3, 3). The 6 numbers are the first two columns."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2, dim=-1)], dim=-1)


def focal_and_centre(K):
    """K (B, 3, 3) -> focal lengths (B, 2) and principal point (B, 2), pixels."""
    return K[:, [0, 1], [0, 1]], K[:, :2, 2]


def pelvis_point(pelvis_uv_log_depth, K, image_size):
    """pelvis_uv_log_depth (B, F, 3): pelvis u, v (-1 left/top .. 1 right/bottom) and log depth ->
    the pelvis in the camera frame (B, F, 3), meters: the point on the pixel's ray at that
    depth."""
    pixel = (pelvis_uv_log_depth[..., :2] + 1) / 2 * image_size[:, None]
    depth = pelvis_uv_log_depth[..., 2:].exp()
    focal, centre = focal_and_centre(K)
    return torch.cat([(pixel - centre[:, None]) / focal[:, None] * depth, depth], -1)


def project(points, K):
    """points (B, ..., 3) in the camera frame -> pixels (B, ..., 2). The depth is clamped to
    MIN_DEPTH."""
    shape = (K.shape[0],) + (1,) * (points.dim() - 2) + (2,)  # broadcast over the middle axes
    focal, centre = (x.view(shape) for x in focal_and_centre(K))
    return points[..., :2] / points[..., 2:].clamp_min(MIN_DEPTH) * focal + centre


def mesh(rotations, betas, pelvis, smpl):
    """Neutral mesh vertices (N, V, 3) for rotations (N, 24, 3, 3), betas (N, 10) or (10,), with
    the pelvis joint at `pelvis` (N, 3): SMPL turns the body about its rest pelvis, so move that
    point to the pelvis."""
    vertices, joints = smpl(rotations, betas)
    return vertices - joints[:, :1] + torch.as_tensor(pelvis, device=smpl.device)[:, None]


def mixed_pnp(joints_relative, joints_2d, pelvis, K, image_size):
    """ACE's mixed PnP: keep the depth of the head's pelvis (B, F, 3), and solve the pelvis x, y by
    least squares so that the SMPL joints joints_relative (B, F, 24, 3) project onto the direct 2D
    joints joints_2d (B, F, 24, 2; pixels). Only joints that pass the gates vote, and a frame
    that fails the gates keeps the head's pelvis. Returns the pelvis (B, F, 3) and the accepted
    frames (B, F) bool. The gradients go through the selected pelvis only."""
    focal, centre = (x[:, None, None] for x in focal_and_centre(K))  # (B, 1, 1, 2)
    depth = pelvis[..., 2:]
    z = joints_relative[..., 2] + depth  # (B, F, 24)
    scale = focal / z.clamp_min(PNP_Z_NEAR)[..., None]  # d pixel / d pelvis x, y
    offset = joints_2d - centre - joints_relative[..., :2] * scale  # = scale * pelvis x, y
    with torch.no_grad():
        # Our soft-argmax runs over the token centres, so a joint outside the image stops at
        # half a token from the edge (ACE's linspace grid puts it at the edge, ACE's margin 0).
        portrait = (image_size[:, 1] > image_size[:, 0])[:, None]
        sides = torch.tensor([[LONG, SHORT]], device=image_size.device)
        tokens = torch.where(portrait, sides.flip(-1), sides) / TOKEN_PX  # (B, 2) along x, y
        margin = (PNP_MARGIN + 0.5 / tokens)[:, None, None]
        inside = joints_2d / image_size[:, None, None]
        inside = ((inside > margin) & (inside < 1 - margin)).all(-1)
        votes = ((z > PNP_Z_NEAR) & inside).float()[..., None]  # (B, F, 24, 1)
    xy = (votes * scale * offset).sum(2) / (votes * scale**2).sum(2).clamp_min(1e-8)
    solved = torch.cat([xy, depth], -1)

    with torch.no_grad():
        residual = ((scale * xy[:, :, None] - offset) ** 2).sum(-1, keepdim=True)
        n_votes = votes.sum((2, 3))
        rms = ((votes * residual).sum((2, 3)) / n_votes.clamp_min(1)).sqrt()
        far = torch.tensor(1e9, device=joints_2d.device)
        box = torch.where(votes > 0, joints_2d, -far).amax(2)
        box = box - torch.where(votes > 0, joints_2d, far).amin(2)
        limit = (PNP_RESIDUAL_FRACTION * box.norm(dim=-1)).clamp_min(PNP_RESIDUAL_MIN_PX)
        accepted = (n_votes >= PNP_MIN_JOINTS) & (rms <= limit)
    return torch.where(accepted[..., None], solved, pelvis), accepted


def decode(rotations_6d, pelvis_uv_log_depth, betas, K, image_size, smpl, direct=None):
    """Decoder outputs -> dict of rotations (B, F, 24, 3, 3), pelvis (B, F, 3), joints_relative
    (B, F, 24, 3; minus the pelvis), joints_3d (B, F, 24, 3; camera frame), joints_2d (B, F, 24, 2;
    pixels), betas (B, 10). smpl is the neutral vidHMR.body.smpl.SMPL. With direct = the decoder's
    (joints_2d 0..1, joints_relative), the pelvis comes from mixed_pnp, and the dict also has
    direct_joints_2d (pixels), direct_joints_relative and pnp_accepted (B, F; 1.0 or 0.0)."""
    B, n_frames = pelvis_uv_log_depth.shape[:2]
    rotations = rotation_6d_to_matrix(rotations_6d)
    pelvis = pelvis_point(pelvis_uv_log_depth, K, image_size)

    frame_betas = betas[:, None].expand(B, n_frames, 10).reshape(-1, 10)
    joints = smpl.joints(rotations.reshape(-1, 24, 3, 3), frame_betas).reshape(B, n_frames, 24, 3)
    joints_relative = joints - joints[:, :, :1]
    out = {}
    if direct is not None:
        direct_2d = direct[0] * image_size[:, None, None]
        pelvis, accepted = mixed_pnp(joints_relative, direct_2d, pelvis, K, image_size)
        out = {
            "direct_joints_2d": direct_2d,
            "direct_joints_relative": direct[1],
            "pnp_accepted": accepted.float(),
        }
    joints_3d = joints_relative + pelvis[:, :, None]

    return out | {
        "rotations": rotations,
        "pelvis": pelvis,
        "joints_relative": joints_relative,
        "joints_3d": joints_3d,
        "joints_2d": project(joints_3d, K),
        "betas": betas,
    }
