"""Videos of the loss check (vidHMR.model.fit): SMPL parameters fitted to the GT of one window
with our losses and Adam, without the network. Two starts: each GT rotation turned by up to 90
degrees, and fully random. Each video frame is one optimization step:

  top row     the fitted neutral mesh on 4 frames of the window, the GT skeleton (green) and the
              fitted skeleton (red)
  bottom row  the same skeletons seen from the side (turned 90 degrees about the GT pelvis), so
              that depth errors show

    PYTHONPATH=src .venv/bin/python scripts/fit_video.py   # -> generated/videos/fit_check/
"""

import cv2
import numpy as np
import tomllib
import torch
from torch.utils.data import default_collate

from vidHMR import REPO, geometry, paths
from vidHMR.body.smpl import SMPL, axis_angle_to_matrix
from vidHMR.datasets import pw3d
from vidHMR.datasets.pw3d_windows import Pw3dWindows
from vidHMR.model.body import mesh
from vidHMR.model.fit import fit, start
from vidHMR.model.losses import metrics
from vidHMR.train import to_cuda
from vidHMR.video import write_video
from vidHMR_viewer.render_mesh import MeshOverlayRenderer
from vidHMR_viewer.skeletons import SMPL24_BONES

ITEM = ("outdoors_climbing_00", 0, 0)  # sequence, first frame of the window, person
PANELS = [0, 28, 56, 80]  # frames of the window that the video shows
HEIGHT = 480  # pixels of one panel
STEPS = 1500
GREEN, RED = (60, 220, 60), (240, 70, 50)
SIDE_TURN = axis_angle_to_matrix(torch.tensor([0.0, np.pi / 2, 0.0])).numpy()  # about camera y


def shown_steps() -> set[int]:
    """Dense at the start, where most of the change happens."""
    return set(range(0, 50)) | set(range(50, 300, 5)) | set(range(300, STEPS + 1, 20))


def read_frame(sequence, frame, height):
    """RGB image of a 3DPW frame, resized to `height`."""
    image = cv2.imread(str(pw3d.frames(sequence)[frame]))[:, :, ::-1]
    width = round(image.shape[1] * height / image.shape[0])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def color(rgb, vertices):
    """One RGB color (0..255) for every vertex, 0..1 on the GPU."""
    return torch.tensor(rgb, device="cuda").float().div(255).expand_as(vertices)


def put_lines(image, lines):
    for row, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip


def draw_skeleton(image, joints_2d, color):
    for parent, child in SMPL24_BONES:
        a, b = joints_2d[parent].round().astype(int), joints_2d[child].round().astype(int)
        cv2.line(image, tuple(a), tuple(b), color, 2, cv2.LINE_AA)


def side_view(joints_3d, gt_pelvis, K, scale):
    """Joints turned by 90 degrees about the camera y axis through the GT pelvis, projected."""
    turned = (joints_3d - gt_pelvis) @ SIDE_TURN.T + gt_pelvis
    return geometry.pixels(turned, K) * scale


class Frames:
    """Collects one video frame per shown optimization step."""

    def __init__(self, batch, images, renderer, title):
        self.batch, self.images, self.renderer, self.title = batch, images, renderer, title
        self.smpl = SMPL("n", device="cuda")
        self.K = batch["K"][0].cpu().numpy()
        self.scale = HEIGHT / batch["image_size"][0, 1].item()
        self.frames = []

    def __call__(self, step, prediction, terms):
        if step not in shown_steps():
            return
        with torch.no_grad():
            vertices = mesh(
                prediction["rotations"][0, PANELS],
                prediction["betas"][0],
                prediction["pelvis"][0, PANELS],
                self.smpl,
            )
            top = self.renderer.blend(
                self.images, vertices, self.smpl.faces, color(RED, vertices), self.K
            )

        gt_3d = self.batch["joints_3d"][0, PANELS].cpu().numpy()
        fitted_3d = prediction["joints_3d"][0, PANELS].detach().cpu().numpy()
        fitted_2d = prediction["joints_2d"][0, PANELS].detach().cpu().numpy() * self.scale
        gt_2d = self.batch["joints_2d"][0, PANELS].cpu().numpy() * self.scale
        bottom = np.full_like(top, 40)
        for panel in range(len(PANELS)):
            draw_skeleton(top[panel], gt_2d[panel], GREEN)
            draw_skeleton(top[panel], fitted_2d[panel], RED)
            pelvis = gt_3d[panel, 0]
            gt_side = side_view(gt_3d[panel], pelvis, self.K, self.scale)
            draw_skeleton(bottom[panel], gt_side, GREEN)
            draw_skeleton(
                bottom[panel], side_view(fitted_3d[panel], pelvis, self.K, self.scale), RED
            )

        errors = metrics(prediction, self.batch)
        frame = np.concatenate([np.concatenate(list(top), 1), np.concatenate(list(bottom), 1)], 0)
        lines = [
            f"{self.title}   step {step} / {STEPS}",
            f"MPJPE {errors['mpjpe_pelvis_mm']:.1f} mm   pelvis {errors['pelvis_mm']:.1f} mm"
            f"   2D {errors['joints_2d_px']:.1f} px",
            f"rotation error: mean {errors['rotation_deg']:.1f} deg,"
            f" {errors['rotations_over_90_pct']:.1f} % of rotations > 90 deg",
        ]
        put_lines(frame, lines)
        self.frames.append(frame)


def main():
    sequence, first_frame, _ = ITEM
    dataset = Pw3dWindows([sequence])
    item = default_collate([dataset[dataset.items.index(ITEM)]])
    batch = to_cuda(item)
    with open(REPO / "configs" / "overfit_one_person.toml", "rb") as file:
        weights = tomllib.load(file)["loss_weights"]

    images = np.stack([read_frame(sequence, first_frame + frame, HEIGHT) for frame in PANELS])
    renderer = MeshOverlayRenderer(alpha=0.6)
    smpl = SMPL("n", device="cuda")
    out = paths.GENERATED / "videos" / "fit_check"

    for name, max_degrees in [("start_within_90_degrees", 90), ("start_random", None)]:
        torch.manual_seed(0)
        frames = Frames(batch, images, renderer, name.replace("_", " "))
        fit(batch, *start(batch, max_degrees), weights, smpl, STEPS, on_step=frames)
        frames.frames += [frames.frames[-1]] * 40  # hold the result for 2 s
        path = write_video(frames.frames, out / f"{name}.mp4", fps=20)
        cv2.imwrite(str(out / f"{name}_last.png"), frames.frames[-1][:, :, ::-1])
        print(path, len(frames.frames), "frames", flush=True)


if __name__ == "__main__":
    main()
