"""Wan2.2-Fun-5B-Control features: one forward pass over the clean video latents at t = 0.

The ACE-Ego-Hand recipe (feature_mode noised_video, sigma 0): x = VAE(video), t = 0, control
channels zero, one fixed caption, hidden states read after the chosen blocks; no denoising
loop. Loaded through ACE's code. Runs in the ACE env:

    PYTHONPATH=src python -m vidHMR.features.wan22 --seqs downtown_walking_00 --frames 162
    PYTHONPATH=src python -m vidHMR.features.wan22 --bedlam 3200 --layer 15  # int8, 1 window each
"""

import argparse
import sys
from functools import partial

import torch

from vidHMR import REPO
from vidHMR.features import cached_text, run

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
def extract_window(vae, dit, ctx, device, pixels, layers=LAYERS) -> dict[int, torch.Tensor]:
    """One window of frames -> {block: (F_lat, H/32, W/32, dim)}. The DiT stops after the last
    block in `layers`."""
    lat = encode(vae, pixels, device).unsqueeze(0).to(device, torch.bfloat16)
    ctrl = lat.new_zeros(1, dit.control_channels, *lat.shape[2:])
    t = torch.zeros(1, device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        taps, _, grid = dit.forward_features(lat, ctrl, [ctx], t, layers, run_head=False)
    return {k: fold_tokens(v, grid)[0].permute(1, 2, 3, 0) for k, v in taps.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="*", default=[], help="3DPW sequence names, or all")
    ap.add_argument("--text-only", action="store_true", help="only cache the text embedding")
    ap.add_argument("--frames", type=int, default=0, help="first N frames only (0: all)")
    ap.add_argument("--bedlam", type=int, default=0, help="N BEDLAM windows (int8, one layer)")
    ap.add_argument("--layer", type=int, default=15, help="the block stored for --bedlam")
    args = ap.parse_args()
    if not (args.seqs or args.text_only or args.bedlam):
        ap.error("give --seqs, --bedlam or --text-only")

    device = torch.device("cuda")
    ctx = caption_embed(CAPTION, device).to(device, torch.bfloat16)
    if args.text_only:
        return

    vae = load_vae(device)
    dit = WanBackbone(str(MODEL_ROOT), mh_expand=False).to(device)

    meta = {"model": "Wan2.2-Fun-5B-Control", "layers": LAYERS, "caption": CAPTION, "t": 0}
    if args.bedlam:
        from vidHMR.datasets import bedlam

        extract = partial(extract_window, vae, dit, ctx, device, layers=(args.layer,))
        bedlam.extract_features(
            extract, NAME, args.bedlam, args.layer, meta | {"layers": [args.layer]}
        )
    else:
        run(partial(extract_window, vae, dit, ctx, device), NAME, args.seqs, args.frames, meta)


if __name__ == "__main__":
    main()
