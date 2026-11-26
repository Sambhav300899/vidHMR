"""GPU SMPL mesh overlays with nvdiffrast.

r = MeshOverlayRenderer()
for rgb in r.frames(seq, height=960): ...   # RGB uint8 frames with the meshes blended in
"""

from collections.abc import Iterator

import cv2
import numpy as np
import torch

from vidHMR.body.smpl import SMPL
from vidHMR.raster import gt_verts_cam, to_clip
from vidHMR_viewer import render
from vidHMR_viewer.render import INVALID_COLOR
from vidHMR_viewer.skeletons import ACTOR_COLORS


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """verts (B,V,3), faces (F,3) long -> unit normals (B,V,3)."""
    v0, v1, v2 = (verts[:, faces[:, i]] for i in range(3))
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    n = torch.zeros_like(verts)
    for i in range(3):
        n.index_add_(1, faces[:, i], fn)
    return torch.nn.functional.normalize(n, dim=-1)


def prediction_tint(rgb) -> np.ndarray:
    """The 3D view's predColor(): the actor color 60 % toward white, mixed in linear light as
    three.js Color.lerp does. sRGB 0..255 in, sRGB 0..1 out."""
    c = np.asarray(rgb) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lin = 0.4 * lin + 0.6
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


class MeshOverlayRenderer:
    def __init__(self, device="cuda", alpha=0.8, batch=8):
        import nvdiffrast.torch as dr

        self.dr, self.ctx = dr, dr.RasterizeCudaContext(device=device)
        self.device, self.alpha, self.batch = device, alpha, batch
        self._models = {}

    def model(self, gender: str) -> SMPL:
        if gender not in self._models:
            self._models[gender] = SMPL(gender, self.device)
        return self._models[gender]

    @torch.no_grad()
    def blend(
        self,
        images: np.ndarray,
        verts: torch.Tensor,
        faces: torch.Tensor,
        cols: torch.Tensor,
        K: np.ndarray,
    ) -> np.ndarray:
        """Blend camera-frame meshes into images.
        images (B,H,W,3) uint8 at any size; verts (B,N,3) OpenCV camera coordinates (rows with z <
        0 are
        clipped away, so absent people can be parked behind the camera); faces (F,3); cols (B,N,3)
        in 0..1;
        K: intrinsics of the ORIGINAL image (principal point at the image center)."""
        dev = self.device
        H, W = images.shape[1:3]
        Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
        faces = faces.int().contiguous()
        normals = vertex_normals(verts, faces.long())
        clip = to_clip(verts, Kt, K[0, 2] * 2, K[1, 2] * 2).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, faces, (H, W))
        out, _ = self.dr.interpolate(torch.cat([cols, normals], -1).contiguous(), rast, faces)
        col, n = out[..., :3], torch.nn.functional.normalize(out[..., 3:], dim=-1)
        shade = 0.3 + 0.7 * (-n[..., 2]).clamp(0, 1)  # light at the camera; OpenCV z looks forward
        rgba = torch.cat([col * shade[..., None], (rast[..., 3:] > 0).float()], -1)
        rgba = self.dr.antialias(rgba.contiguous(), rast, clip, faces)
        img = torch.as_tensor(images, device=dev).float() / 255
        a = rgba[..., 3:] * self.alpha
        img = img * (1 - a) + rgba[..., :3] * a
        return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()

    @torch.no_grad()
    def render(self, seq, frames: np.ndarray, images: np.ndarray) -> np.ndarray:
        """Blend the GT SMPL meshes of `frames` (B,) into `images` (B,H,W,3 uint8, any size)."""
        dev, B = self.device, len(frames)
        verts, cols, faces = [], [], []
        for a in range(seq.n_actors):
            m = self.model(seq.genders[a])
            v = gt_verts_cam(seq, a, frames, m)
            base = (
                torch.tensor(
                    [ACTOR_COLORS[a % len(ACTOR_COLORS)], INVALID_COLOR],
                    dtype=torch.float32,
                    device=dev,
                )
                / 255
            )
            valid = torch.as_tensor(seq.valid[a, frames], device=dev)
            cols.append(
                torch.where(valid[:, None, None], base[0], base[1]).expand(B, v.shape[1], 3)
            )
            faces.append(m.faces + a * v.shape[1])
            verts.append(v)
        return self.blend(images, torch.cat(verts, 1), torch.cat(faces), torch.cat(cols, 1), seq.K)

    @staticmethod
    def read_frames(seq, idx, height):
        return np.stack(
            [
                render.resize_to_height(
                    cv2.cvtColor(cv2.imread(str(seq.image_path(f))), cv2.COLOR_BGR2RGB), height
                )[0]
                for f in idx
            ]
        )

    def frames(self, seq, height: int = 960) -> Iterator[np.ndarray]:
        """Yield every frame of the sequence, resized to `height`, with the GT meshes."""
        for s in range(0, seq.n_frames, self.batch):
            idx = np.arange(s, min(s + self.batch, seq.n_frames))
            yield from self.render(seq, idx, self.read_frames(seq, idx, height))

    @torch.no_grad()
    def prediction_frames(self, seq, pred: dict, height: int = 960) -> Iterator[np.ndarray]:
        """Yield every frame with predicted camera-frame meshes, e.g.
        vidHMR.sam3d.pw3d.load_predictions().
        pred needs frame (N,), actor (N,) (-1 = unmatched), verts (N,V,3) and faces (F,3). Actor
        colors match
        the GT overlay; unmatched people are white."""
        dev = self.device
        n_slots = max(1, int(np.bincount(pred["frame"]).max()) if len(pred["frame"]) else 1)
        V = int(pred["faces"].max()) + 1  # also right when there are no rows
        faces1 = torch.as_tensor(pred["faces"], device=dev).long()
        faces = torch.cat([faces1 + i * V for i in range(n_slots)])
        palette = [prediction_tint(ACTOR_COLORS[a % len(ACTOR_COLORS)]) for a in range(8)]
        rows_of = {}
        for i, f in enumerate(pred["frame"]):
            rows_of.setdefault(int(f), []).append(i)
        for s in range(0, seq.n_frames, self.batch):
            idx = np.arange(s, min(s + self.batch, seq.n_frames))
            verts = np.full(
                (len(idx), n_slots * V, 3), [0, 0, -1], np.float32
            )  # empty slot: behind the camera
            cols = np.ones((len(idx), n_slots * V, 3), np.float32)
            for b, f in enumerate(idx):
                for k, i in enumerate(rows_of.get(int(f), [])):
                    verts[b, k * V : (k + 1) * V] = pred["verts"][i]
                    a = int(pred["actor"][i])
                    cols[b, k * V : (k + 1) * V] = palette[a % 8] if a >= 0 else 1.0
            yield from self.blend(
                self.read_frames(seq, idx, height),
                torch.as_tensor(verts, device=dev),
                faces,
                torch.as_tensor(cols, device=dev),
                seq.K,
            )
