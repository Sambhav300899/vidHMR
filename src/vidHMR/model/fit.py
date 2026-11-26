"""Fit SMPL parameters directly to the targets of one window with our losses and Adam, without
the network: a check that the losses lead to the GT (tests/test_model.py, scripts/fit_video.py).
"""

import torch

from vidHMR.body.smpl import axis_angle_to_matrix
from vidHMR.model.body import decode
from vidHMR.model.losses import gt_rotations, losses


def start(batch, max_degrees=None):
    """Start values (rotations_6d, pelvis_uv_log_depth, betas) for the frames of `batch`: each GT
    rotation turned by a random rotation of up to max_degrees (None: fully random 6D values),
    random betas, a random pelvis pixel and a random depth of 1.6 to 4.5 m."""
    rotations = gt_rotations(batch)
    if max_degrees is None:
        rotations_6d = torch.randn(*rotations.shape[:3], 6, device=rotations.device)
    else:
        shape = rotations.shape[:3]
        axis = torch.nn.functional.normalize(
            torch.randn(*shape, 3, device=rotations.device), dim=-1
        )
        angle = torch.deg2rad(torch.rand(*shape, 1, device=rotations.device) * max_degrees)
        turned = rotations @ axis_angle_to_matrix(axis * angle)
        rotations_6d = torch.cat([turned[..., :, 0], turned[..., :, 1]], -1)
    B, n_frames = rotations.shape[:2]
    uv = torch.rand(B, n_frames, 2, device=rotations.device) * 2 - 1
    log_depth = torch.rand(B, n_frames, 1, device=rotations.device) + 0.5
    betas = torch.randn(B, 10, device=rotations.device)
    return rotations_6d, torch.cat([uv, log_depth], -1), betas


def fit(batch, rotations_6d, pelvis_uv_log_depth, betas, weights, smpl, steps, on_step=None):
    """Optimize the three start tensors (changed in place) for `steps` Adam steps, learning rate
    0.02 with cosine decay. on_step(step, prediction, terms) is called before each step and once
    after the last one. Returns the final prediction and loss terms."""
    parameters = [x.requires_grad_() for x in (rotations_6d, pelvis_uv_log_depth, betas)]
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    for step in range(steps + 1):
        prediction = decode(*parameters, batch["K"], batch["image_size"], smpl)
        terms = losses(prediction, batch)
        if on_step is not None:
            on_step(step, prediction, terms)
        if step == steps:
            return prediction, terms

        optimizer.zero_grad()
        sum(weights[key] * value for key, value in terms.items()).backward()
        optimizer.step()
        schedule.step()
