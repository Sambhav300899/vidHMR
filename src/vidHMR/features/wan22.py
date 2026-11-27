"""Wan2.2-Fun-5B-Control features: one forward pass over the clean video latents at t = 0.

The ACE-Ego-Hand recipe (feature_mode noised_video, sigma 0): x = VAE(video), t = 0, control
channels zero, one fixed caption, hidden states read after the chosen blocks; no denoising
loop. Loaded through ACE's code. Runs in the ACE env:

    PYTHONPATH=src python -m vidHMR.features.wan22 --seqs downtown_walking_00 --frames 162
    PYTHONPATH=src python -m vidHMR.features.wan22 --bedlam 3200 --layer 15  # int8, 1 window each
    PYTHONPATH=src python -m vidHMR.features.wan22 --latents train validation  # live training
"""

import argparse
import sys
import time
from functools import partial

import cv2
import numpy as np
import torch

from vidHMR import REPO
from vidHMR.datasets import pw3d
from vidHMR.features import ROOT, WINDOW, cached_text, resized_rgb, rgb_to_tensor, run, target_size

ACE = REPO / "external" / "ACE-Ego-Hand"
sys.path.insert(0, str(ACE))
from ace_ego_hand.archs.wan_backbone import WanBackbone, fold_tokens  # noqa: E402
from ace_ego_hand.video_vae import MODEL_ROOT, VIDEOX_CFG, encode, load_vae  # noqa: E402

NAME = "wan22_fun5b_t0"
LAYERS = (5, 10, 15, 20, 25, 29)  # blocks (of 30) whose output is stored; ACE taps 15
CAPTION = (
    "Renders of a 3D human body mesh: color-coded depth, body joint skeleton, surface normals."
)


def caption_embed(text: str, device) -> torch.Tensor:
    """umT5 embedding (L, 4096) of the fixed caption, cached in generated/text_embeds/. The 11 GB
    text encoder loads only on the first call."""
    return cached_text(
        "wan22_fun5b", text, {"model": MODEL_ROOT.name}, lambda: _encode(text, device)
    )


def _encode(text: str, device) -> torch.Tensor:
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from videox_fun.models import WanT5EncoderModel

    cfg = OmegaConf.load(str(VIDEOX_CFG))["text_encoder_kwargs"]
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT / cfg.get("tokenizer_subpath")))
    te = WanT5EncoderModel.from_pretrained(
        str(MODEL_ROOT / cfg.get("text_encoder_subpath")),
        additional_kwargs=OmegaConf.to_container(cfg),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    te = te.eval().to(device)

    ids = tok(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    with torch.no_grad():
        emb = te(ids.input_ids.to(device), attention_mask=ids.attention_mask.to(device))[0]

    del te
    torch.cuda.empty_cache()
    return emb[0, : int(ids.attention_mask.sum())].float().cpu()


@torch.no_grad()
def dit_features(dit, ctx, device, latents, layers=LAYERS) -> dict[int, torch.Tensor]:
    """The VAE latents of one window (48, F_lat, H/16, W/16), float16 as ACE's `encode` gives
    them -> {block: (F_lat, H/32, W/32, dim)}. The DiT stops after the last block in
    `layers`."""
    lat = latents.unsqueeze(0).to(device, torch.bfloat16)
    ctrl = lat.new_zeros(1, dit.control_channels, *lat.shape[2:])
    t = torch.zeros(1, device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        taps, _, grid = dit.forward_features(lat, ctrl, [ctx], t, layers, run_head=False)
    return {k: fold_tokens(v, grid)[0].permute(1, 2, 3, 0) for k, v in taps.items()}


def latents_path(name: str):
    """The VAE latents of every window of a 3DPW sequence that starts at a multiple of 4 frames:
    (starts, 48, 21, H/16, W/16) float16, row i = the window at frame 4i (live training)."""
    return ROOT / f"{NAME}_latents" / f"{name}.npy"


def encode_latents(vae, device, name: str) -> None:
    """Write latents_path(name); the frames are loaded once, as load_frames gives them."""
    path = latents_path(name)
    if path.exists():
        return
    paths = pw3d.frames(name)
    rgb = resized_rgb([cv2.imread(str(p)) for p in paths], target_size(paths[0]))
    starts = range(0, len(paths) - WINDOW + 1, 4)

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial.npy")
    out = None
    start_time = time.time()
    for i, first_frame in enumerate(starts):
        latents = encode(vae, rgb_to_tensor(rgb[first_frame : first_frame + WINDOW]), device)
        if out is None:
            shape = (len(starts), *latents.shape)
            out = np.lib.format.open_memmap(partial_path, "w+", np.float16, shape)
        out[i] = latents.numpy()
    out.flush()
    partial_path.rename(path)
    print(f"{name}: {len(starts)} windows, {(time.time() - start_time) / len(starts):.2f} s each")


def window_latents(name: str, first_frame: int) -> torch.Tensor:
    """The stored VAE latents (latents_path) of the window that starts at first_frame."""
    return torch.from_numpy(np.load(latents_path(name), mmap_mode="r")[first_frame // 4].copy())


def load_dit(device, layers=LAYERS):
    """The DiT on `device`, as a function: VAE latents -> {block: features} (dit_features)."""
    ctx = caption_embed(CAPTION, device).to(device, torch.bfloat16)
    dit = WanBackbone(str(MODEL_ROOT), mh_expand=False).to(device)
    return partial(dit_features, dit, ctx, device, layers=layers)


def load_extractor(device, layers=LAYERS):
    """The VAE and the DiT on `device`, as a function: pixels (81, 3, H, W) -> {block: features}
    for one window."""
    vae, dit = load_vae(device), load_dit(device, layers)

    def extract(pixels):
        return dit(encode(vae, pixels, device))

    return extract


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="*", default=[], help="3DPW sequence names, or all")
    ap.add_argument("--text-only", action="store_true", help="only cache the text embedding")
    ap.add_argument("--frames", type=int, default=0, help="first N frames only (0: all)")
    ap.add_argument("--bedlam", type=int, default=0, help="N BEDLAM windows (int8, one layer)")
    ap.add_argument("--layer", type=int, default=15, help="the block stored for --bedlam")
    ap.add_argument("--latents", nargs="*", default=[], help="3DPW splits: VAE latents only")
    args = ap.parse_args()
    if not (args.seqs or args.text_only or args.bedlam or args.latents):
        ap.error("give --seqs, --bedlam, --latents or --text-only")

    device = torch.device("cuda")
    if args.latents:
        from vidHMR.datasets.pw3d_windows import split_sequences

        vae = load_vae(device)
        for split in args.latents:
            for name in split_sequences(split):
                encode_latents(vae, device, name)
        return
    if args.text_only:
        caption_embed(CAPTION, device)
        return

    meta = {"model": "Wan2.2-Fun-5B-Control", "layers": LAYERS, "caption": CAPTION, "t": 0}
    if args.bedlam:
        from vidHMR.datasets import bedlam

        extract = load_extractor(device, layers=(args.layer,))
        bedlam.extract_features(
            extract, NAME, args.bedlam, args.layer, meta | {"layers": [args.layer]}
        )
    else:
        run(load_extractor(device), NAME, args.seqs, args.frames, meta)


if __name__ == "__main__":
    main()
