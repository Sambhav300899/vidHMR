"""Checks of the person decoder, the body decode and the losses (vidHMR.model).

PYTHONPATH=src .venv/bin/python -m pytest tests/test_model.py -q
"""

import math

import numpy as np
import pytest
import tomllib
import torch

from vidHMR import REPO
from vidHMR.body.smpl import SMPL, axis_angle_to_matrix
from vidHMR.datasets.pw3d_targets import targets_path
from vidHMR.features import WINDOW
from vidHMR.model.body import decode, project
from vidHMR.model.decoder import PersonDecoder, ray_angles
from vidHMR.model.fit import fit, start
from vidHMR.model.losses import gt_rotations, losses, metrics
from vidHMR.train import learning_rate_factor, predict, to_cuda

PER_FRAME_KEYS = ("global_orient", "body_pose", "pelvis", "joints_3d", "joints_2d", "valid")


def gt_batch(name="outdoors_climbing_00", person=0, frames=slice(0, WINDOW)):
    """The targets of one window as a batch of one item (the keys that the losses use)."""
    with np.load(targets_path(name)) as targets:
        batch = {key: targets[key][person, frames] for key in PER_FRAME_KEYS}
        batch["betas_neutral"] = targets["betas_neutral"][person]
        batch["K"], batch["image_size"] = targets["K"], targets["image_size"]
    return {key: torch.from_numpy(np.asarray(value))[None] for key, value in batch.items()}


def gt_outputs(batch):
    """The GT as decoder outputs: 6D = the first two columns of each rotation matrix; pelvis u, v
    from the projected GT pelvis and its log depth; the neutral betas."""
    rotations = gt_rotations(batch)
    rotations_6d = torch.cat([rotations[..., :, 0], rotations[..., :, 1]], -1)
    uv = 2 * project(batch["pelvis"], batch["K"]) / batch["image_size"][:, None] - 1
    pelvis_uv_log_depth = torch.cat([uv, batch["pelvis"][..., 2:].log()], -1)
    return rotations_6d, pelvis_uv_log_depth, batch["betas_neutral"].clone()


def test_gt_through_decode_gives_zero_loss():
    """The GT as decoder outputs gives ~0 loss for a woman (the neutral body is ~1 mm from her
    joints), and finite gradients."""
    batch = gt_batch()
    outputs = gt_outputs(batch)
    for x in outputs:
        x.requires_grad_()

    prediction = decode(*outputs, batch["K"], batch["image_size"], SMPL("n", device="cpu"))
    terms = losses(prediction, batch)
    assert terms["rotation_geodesic"] < 2e-3  # the arccos clamp leaves 0.08 degrees
    assert terms["rotation_mse"] < 1e-10 and terms["betas"] == 0 and terms["pelvis"] < 1e-6
    assert terms["joints_relative"] < 1e-3 and terms["joints_3d"] < 1e-3  # meters
    assert terms["joints_2d"] < 1e-3  # image fractions
    assert metrics(prediction, batch)["mpjpe_pelvis_mm"] < 2

    sum(terms.values()).backward()
    assert all(x.grad.isfinite().all() for x in outputs)


def test_mixed_pnp_finds_the_pelvis():
    """With the GT 2D joints as the direct joints, the mixed PnP moves a pelvis that is 0.5 m off
    in x and y (correct depth) back to the GT pelvis; with 2D joints far from the body, a frame
    keeps the head's pelvis."""
    batch = gt_batch()
    rotations_6d, pelvis_uv_log_depth, betas = gt_outputs(batch)
    shifted = batch["pelvis"] + torch.tensor([0.5, -0.5, 0.0])
    uv = 2 * project(shifted, batch["K"]) / batch["image_size"][:, None] - 1
    pelvis_uv_log_depth = torch.cat([uv, pelvis_uv_log_depth[..., 2:]], -1)
    gt_2d = batch["joints_2d"] / batch["image_size"][:, None, None]
    gt_relative = batch["joints_3d"] - batch["joints_3d"][:, :, :1]
    smpl = SMPL("n", device="cpu")

    def pelvis(joints_2d):
        prediction = decode(
            rotations_6d, pelvis_uv_log_depth, betas, batch["K"], batch["image_size"], smpl,
            direct=(joints_2d, gt_relative),
        )  # fmt: skip
        return prediction["pelvis"][batch["valid"]], prediction["pnp_accepted"][batch["valid"]]

    solved, accepted = pelvis(gt_2d)
    assert accepted.all() and (solved - batch["pelvis"][batch["valid"]]).norm(dim=-1).max() < 0.02
    scattered = gt_2d.clone()
    scattered[..., ::2, 0] = 0.1  # half the joints on the left edge, half on the right
    scattered[..., 1::2, 0] = 0.9
    kept, accepted = pelvis(scattered)
    assert not accepted.any() and torch.allclose(kept, shifted[batch["valid"]], atol=1e-4)


def test_ray_angles_known_values():
    """3DPW portrait camera (fx = 1961.9, cx = 540, cy = 960), 26 x 15 tokens of 72 x 73.8 px:
    the column whose centre is at cx has azimuth 0; the last column (centre u = 1044) has azimuth
    atan(504 / fx); the rows have elevation atan((v - cy) / fy)."""
    batch = gt_batch()
    K, image_size = batch["K"], batch["image_size"].float()
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
    angles = ray_angles(K, image_size, 26, 15)[0].reshape(26, 15, 2)

    assert abs(angles[0, 7, 0].item()) < 1e-6  # (7 + 0.5) * 72 = 540
    assert math.isclose(angles[0, 14, 0].item(), math.atan(504 / fx), rel_tol=1e-5)
    v = (25 + 0.5) * 1920 / 26
    assert math.isclose(angles[25, 0, 1].item(), math.atan((v - 960) / fy), rel_tol=1e-5)


@pytest.mark.parametrize("grid", [(26, 15), (15, 26)])  # portrait, landscape
def test_decoder_shapes_and_interpolation(grid):
    """Output shapes; frame 4k gets exactly the person slot token of latent k; the mask encoder
    reads the prompt latent's encoder conditioning; the prompt mask changes the output."""
    torch.manual_seed(0)
    height, width = grid
    model = PersonDecoder(channels=64)
    features = torch.randn(2, 21, height, width, 64).half()
    mask = torch.zeros(2, 32 * height, 32 * width, dtype=torch.bool)
    mask[:, 100:300, 100:300] = True
    image_size = [1080, 1920] if height > width else [1920, 1080]
    inputs = (
        torch.tensor([0, 3]),
        gt_batch()["K"].expand(2, 3, 3),
        torch.tensor([image_size] * 2),
    )

    seen = {}
    model.final_norm.register_forward_hook(lambda m, i, out: seen.update(person=out[:, :, 0]))
    model.frame_fuse.register_forward_hook(lambda m, i, out: seen.update(frames=i[0]))
    model.mask_encoder.register_forward_hook(lambda m, i, out: seen.update(prompt=i[1]))
    with torch.no_grad():
        rotations_6d, pelvis_uv_log_depth, betas, joints_2d, joints_relative = model(
            features, mask, *inputs
        )
        prompt = seen["prompt"]
        other = model(features, ~mask, *inputs)
        for latent, changes in [(3, True), (5, False)]:  # item 1 has its prompt on latent 3
            changed = features.clone()
            changed[1, latent] += 1
            model(changed, mask, *inputs)
            assert torch.equal(seen["prompt"][0], prompt[0])
            assert changes != torch.equal(seen["prompt"][1], prompt[1])

    assert rotations_6d.shape == (2, 81, 24, 6) and pelvis_uv_log_depth.shape == (2, 81, 3)
    assert betas.shape == (2, 10)
    assert joints_2d.shape == (2, 81, 24, 2) and joints_relative.shape == (2, 81, 24, 3)
    assert ((joints_2d > 0) & (joints_2d < 1)).all()  # a soft-argmax inside the image
    assert (joints_relative[:, :, 0] == 0).all()
    assert torch.allclose(seen["frames"][:, ::4], seen["person"], atol=1e-6)
    assert not torch.allclose(rotations_6d, other[0])


def test_losses_with_known_errors():
    """Known errors give the hand-computed values: one joint turned by 0.1 rad replaces that
    joint's floor angle with 0.1 in the mean over 24; +10 px in x gives a 2D metric of 10 px and a
    loss of 10 / width / 2."""
    batch = gt_batch()
    smpl = SMPL("n", device="cpu")
    exact = decode(*gt_outputs(batch), batch["K"], batch["image_size"], smpl)
    floor = losses(exact, batch)["rotation_geodesic"]

    turned = dict(exact)
    turn = axis_angle_to_matrix(torch.tensor([0.1, 0.0, 0.0]))
    turned["rotations"] = exact["rotations"].clone()
    turned["rotations"][..., 5, :, :] = exact["rotations"][..., 5, :, :] @ turn
    expected = floor * 23 / 24 + 0.1 / 24
    assert torch.isclose(losses(turned, batch)["rotation_geodesic"], expected, atol=1e-5)

    shifted = dict(exact)
    shifted["joints_2d"] = batch["joints_2d"] + torch.tensor([10.0, 0.0])  # from the exact GT
    width = batch["image_size"][0, 0].item()
    assert abs(metrics(shifted, batch)["joints_2d_px"] - 10) < 1e-3
    assert abs(losses(shifted, batch)["joints_2d"].item() - 10 / width / 2) < 1e-6


def test_losses_fit_a_body_to_the_gt():
    """Without the network: SMPL parameters optimized with Adam on our weighted losses reach the
    GT. Start: each GT rotation turned by a random rotation of up to 90 degrees, random betas,
    random pelvis pixel and depth. (From a fully random start, some trunk rotations stay about
    180 degrees off: both rotation terms are flat there, and a wrong parent and child rotation
    can cancel in the joint terms.) Every 4th frame of the window, for speed."""
    torch.manual_seed(0)
    batch = gt_batch(frames=slice(0, WINDOW, 4))
    with open(REPO / "configs" / "overfit_one_person.toml", "rb") as file:
        weights = tomllib.load(file)["loss_weights"]

    smpl = SMPL("n", device="cpu")
    prediction, terms = fit(batch, *start(batch, max_degrees=90), weights, smpl, steps=1500)

    errors = metrics(prediction, batch)
    assert errors["mpjpe_pelvis_mm"] < 3 and errors["pelvis_mm"] < 1
    assert errors["joints_2d_px"] < 1
    assert torch.rad2deg(terms["rotation_geodesic"]) < 0.5 and terms["betas"] < 0.05


def test_learning_rate_schedule():
    """Linear warmup to 1, cosine down to 0 at the last step, and 0 after it."""
    assert learning_rate_factor(0, 50, 2000) == 1 / 50
    assert learning_rate_factor(49, 50, 2000) == 1.0 and learning_rate_factor(50, 50, 2000) == 1.0
    assert math.isclose(learning_rate_factor(1025, 50, 2000), 0.5, abs_tol=1e-9)
    assert (
        learning_rate_factor(2000, 50, 2000) < 1e-12
        and learning_rate_factor(3000, 50, 2000) < 1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prediction_on_the_gpu_is_float32():
    """Under the bf16 autocast of train.predict, the heads still give float32 outputs, and the
    decoded body is finite."""
    torch.manual_seed(0)
    batch = to_cuda(gt_batch())
    batch["features"] = torch.randint(-127, 128, (1, 21, 26, 15, 64), device="cuda").to(torch.int8)
    batch["feature_scale"] = torch.full((1, 64), 0.01, device="cuda")
    batch["prompt_mask"] = torch.ones(1, 832, 480, dtype=torch.bool, device="cuda")
    batch["prompt_latent"] = torch.tensor([0], device="cuda")
    model = PersonDecoder(channels=64).cuda()
    prediction = predict(model, SMPL("n", device="cuda"), batch)
    assert all(value.dtype == torch.float32 for value in prediction.values())
    assert all(value.isfinite().all() for value in prediction.values())
