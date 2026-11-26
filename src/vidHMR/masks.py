"""GT SMPL person masks for 3DPW, rendered with nvdiffrast at the native image size.

One PNG per frame in generated/masks/pw3d_smpl/<seq>/image_XXXXX.png, uint8, bits per actor a:
  bit a      amodal: the actor's full mesh silhouette, also where another actor covers it
  bit 4 + a  visible: the pixels where the actor's mesh is the nearest surface
The masks cover the body surface only (no clothes or hair) and ignore occlusion by objects.
counts.npy (F, A, 2) holds the amodal and visible pixel counts of each actor and frame.

    python -m vidHMR.masks all          # or a list of sequence names
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from vidHMR import GENERATED
from vidHMR.body.smpl import SMPL
from vidHMR.datasets import pw3d
from vidHMR.raster import gt_verts_cam, to_clip

ROOT = GENERATED / "masks" / "pw3d_smpl"
MAX_ACTORS = 4
BATCH = 32


def load_masks(seq: str, frame: int, n_actors: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode one frame -> amodal (A,H,W) bool, visible (A,H,W) bool."""
    m = cv2.imread(str(ROOT / seq / f"image_{frame:05d}.png"), cv2.IMREAD_UNCHANGED)
    bits = np.arange(n_actors)
    amodal = (m[None] >> bits[:, None, None]) & 1
    visible = (m[None] >> (bits[:, None, None] + 4)) & 1
    return amodal.astype(bool), visible.astype(bool)


@torch.no_grad()
def render_batch(dr, ctx, verts: list[torch.Tensor], faces: torch.Tensor, K, size) -> torch.Tensor:
    """verts: A tensors (B,V,3) in camera coordinates -> packed uint8 masks (B,H,W)."""
    W, H = size
    A, (B, V, _) = len(verts), verts[0].shape
    Kt = torch.as_tensor(K, dtype=torch.float32, device=verts[0].device)
    faces = faces.int().contiguous()

    # Amodal: each actor alone, all actors stacked in the batch.
    clip = to_clip(torch.cat(verts), Kt, W, H).contiguous()
    rast, _ = dr.rasterize(ctx, clip, faces, (H, W))
    amodal = (rast[..., 3] > 0).view(A, B, H, W)

    # Visible: all actors in one scene; the depth test keeps the nearest triangle.
    all_faces = torch.cat([faces + a * V for a in range(A)]).contiguous()
    clip = to_clip(torch.cat(verts, 1), Kt, W, H).contiguous()
    rast, _ = dr.rasterize(ctx, clip, all_faces, (H, W))
    tri = rast[..., 3].long()  # triangle id + 1, 0 = empty
    owner = torch.where(tri > 0, (tri - 1) // faces.shape[0], -1)

    packed = torch.zeros(B, H, W, dtype=torch.int32, device=clip.device)
    for a in range(A):
        packed |= amodal[a].int() << a
        packed |= (owner == a).int() << (4 + a)
    return packed.to(torch.uint8)


def render_sequence(name: str, dr, ctx, models: dict, pool: ThreadPoolExecutor) -> None:
    out = ROOT / name
    meta_file = out / "meta.json"
    seq = pw3d.load_sequence(name)
    if meta_file.exists() and json.loads(meta_file.read_text())["n_frames"] == seq.n_frames:
        print(f"{name}: done; skipped", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    assert seq.n_actors <= MAX_ACTORS
    h, w = cv2.imread(str(seq.image_path(0))).shape[:2]
    counts = np.zeros((seq.n_frames, seq.n_actors, 2), np.int32)
    t0, writes = time.time(), []

    for s in range(0, seq.n_frames, BATCH):
        idx = np.arange(s, min(s + BATCH, seq.n_frames))
        verts = [gt_verts_cam(seq, a, idx, models[seq.genders[a]]) for a in range(seq.n_actors)]
        faces = models[seq.genders[0]].faces
        masks = render_batch(dr, ctx, verts, faces, seq.K, (w, h))

        for a in range(seq.n_actors):
            counts[idx, a, 0] = ((masks >> a) & 1).sum((1, 2)).cpu().numpy()
            counts[idx, a, 1] = ((masks >> (4 + a)) & 1).sum((1, 2)).cpu().numpy()

        for f, m in zip(idx, masks.cpu().numpy(), strict=True):
            path = str(out / f"image_{f:05d}.png")
            writes.append(pool.submit(cv2.imwrite, path, m))

    for job in writes:
        assert job.result()

    np.save(out / "counts.npy", counts)
    meta = {
        "seq": name,
        "n_frames": seq.n_frames,
        "n_actors": seq.n_actors,
        "size_wh": [w, h],
        "encoding": "uint8; bit a = amodal actor a, bit 4+a = visible actor a",
        "source": "GT SMPL (gendered), 3DPW cam_poses + K, nvdiffrast",
    }
    meta_file.write_text(json.dumps(meta, indent=1))
    print(
        f"{name}: {seq.n_frames} frames, {seq.n_actors} actors, {time.time() - t0:.1f}s",
        flush=True,
    )


def main() -> None:
    import nvdiffrast.torch as dr

    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="+", help="sequence names, or `all`")
    args = ap.parse_args()
    seqs = pw3d.sequences() if args.seqs == ["all"] else args.seqs

    ctx = dr.RasterizeCudaContext(device="cuda")
    models = {g: SMPL(g, "cuda") for g in ("m", "f")}
    with ThreadPoolExecutor(max_workers=24) as pool:
        for name in seqs:
            render_sequence(name, dr, ctx, models, pool)


if __name__ == "__main__":
    main()
