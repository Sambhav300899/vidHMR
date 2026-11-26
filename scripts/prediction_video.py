"""Videos of a run's predictions on a whole sequence (generated/runs/<run>/<sequence>_<person>.npz
from vidHMR.infer), one video frame per frame. Side views turn by 90 degrees about the GT pelvis.

  <run>_<file>.mp4         the predicted neutral mesh (red) on the image with the GT skeleton
                           (green) and the predicted skeleton (red) | the skeletons from the side
  <run>_<file>_meshes.mp4  the GT gendered mesh (green) on the image | the predicted mesh (red) on
                           the image | both meshes from the side

    PYTHONPATH=src .venv/bin/python scripts/prediction_video.py <run> <file without .npz>
    # -> generated/videos/predictions/
"""

import sys

import numpy as np
import torch
from fit_video import (
    GREEN,
    RED,
    SIDE_TURN,
    color,
    draw_skeleton,
    put_lines,
    read_frame,
    side_view,
)

from vidHMR import paths
from vidHMR.body.smpl import SMPL
from vidHMR.datasets import pw3d
from vidHMR.datasets.pw3d_targets import targets_path
from vidHMR.model.body import mesh
from vidHMR.raster import gt_verts_cam
from vidHMR.video import write_video
from vidHMR_viewer.render_mesh import MeshOverlayRenderer

HEIGHT = 960  # pixels of the video
CHUNK = 64  # frames rendered at once


class Video:
    def __init__(self, prediction):
        self.prediction = prediction
        self.sequence, self.person = str(prediction["sequence"]), int(prediction["person"])
        self.n = len(prediction["pelvis"])
        self.frames = np.arange(self.n)
        with np.load(targets_path(self.sequence)) as targets:
            gt = {
                key: targets[key][self.person, self.frames]
                for key in ("joints_3d", "joints_2d", "valid")
            }
            self.K, image_size = targets["K"], targets["image_size"]
        self.gt_3d, self.gt_2d, self.valid = gt["joints_3d"], gt["joints_2d"], gt["valid"]
        self.scale = HEIGHT / image_size[1]

        self.seq = pw3d.load_sequence(self.sequence)
        self.smpl = SMPL("n", device="cuda")
        self.gt_smpl = SMPL(self.seq.genders[self.person], device="cuda")
        self.renderer = MeshOverlayRenderer(alpha=0.6)
        self.turn = torch.as_tensor(SIDE_TURN, device="cuda").float()

        gt_relative = self.gt_3d - self.gt_3d[:, :1]
        error = np.linalg.norm(prediction["joints_relative"] - gt_relative, axis=-1)
        self.mpjpe = 1000 * error.mean(-1)
        self.pelvis = 1000 * np.linalg.norm(prediction["pelvis"] - self.gt_3d[:, 0], axis=-1)

    def lines(self, frame):
        return [
            f"{self.sequence}  frame {frame}  (latent offset {frame % 4},"
            f" window {self.prediction['window'][frame]})",
            f"MPJPE {self.mpjpe[frame]:.1f} mm   pelvis {self.pelvis[frame]:.1f} mm"
            if self.valid[frame]
            else "invalid frame (no GT)",
        ]

    @torch.no_grad()
    def render(self, meshes):
        """Yield the video frames, CHUNK frames rendered at a time."""
        for start in range(0, self.n, CHUNK):
            chunk = slice(start, min(start + CHUNK, self.n))
            images = np.stack([read_frame(self.sequence, f, HEIGHT) for f in self.frames[chunk]])
            p, smpl, K = self.prediction, self.smpl, self.K
            vertices = mesh(p["rotations"][chunk], p["betas"][chunk], p["pelvis"][chunk], smpl)
            overlays = self.renderer.blend(images, vertices, smpl.faces, color(RED, vertices), K)

            if meshes:
                gt = gt_verts_cam(self.seq, self.person, self.frames[chunk], self.gt_smpl)
                gt_overlays = self.renderer.blend(images, gt, smpl.faces, color(GREEN, gt), K)
                pelvis = torch.as_tensor(self.gt_3d[chunk, :1], device="cuda")
                both = torch.cat([gt, vertices], 1)  # the gendered and neutral meshes share faces
                both = (both - pelvis) @ self.turn.T + pelvis
                faces = torch.cat([smpl.faces, smpl.faces + vertices.shape[1]])
                colors = torch.cat([color(GREEN, gt), color(RED, vertices)], 1)
                sides = self.renderer.blend(np.full_like(images, 40), both, faces, colors, K)

            for i, top in enumerate(overlays):
                frame = start + i
                if meshes:
                    image = np.concatenate([gt_overlays[i], top, sides[i]], 1)
                    lines = self.lines(frame) + ["GT mesh (green) | predicted (red) | side view"]
                else:
                    gt_3d, pelvis = self.gt_3d[frame], self.gt_3d[frame, 0]
                    draw_skeleton(top, self.gt_2d[frame] * self.scale, GREEN)
                    draw_skeleton(top, self.prediction["joints_2d"][frame] * self.scale, RED)
                    side = np.full_like(top, 40)
                    draw_skeleton(side, side_view(gt_3d, pelvis, self.K, self.scale), GREEN)
                    predicted = self.prediction["joints_3d"][frame]
                    draw_skeleton(side, side_view(predicted, pelvis, self.K, self.scale), RED)
                    image = np.concatenate([top, side], 1)
                    lines = self.lines(frame)
                put_lines(image, lines)
                yield image


def main(run, file):
    video = Video(dict(np.load(paths.generated("runs", run, f"{file}.npz"))))
    for suffix, meshes in [("", False), ("_meshes", True)]:
        out = paths.generated("videos", "predictions", f"{run}_{file}{suffix}.mp4")
        print(write_video(video.render(meshes), out, fps=30), video.n, "frames", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
