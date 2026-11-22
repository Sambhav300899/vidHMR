"""Wan2.2-Fun-5B-Control features: one forward pass over the clean video latents at t = 0.

The ACE-Ego-Hand recipe (feature_mode noised_video, sigma 0): x = VAE(video), t = 0, control
channels zero, one fixed caption, hidden states read after the chosen blocks; no denoising
loop. Loaded through ACE's code. Runs in the ACE env:

    PYTHONPATH=src python -m vidHMR.features.wan22 --seqs downtown_walking_00 --frames 162
"""

import argparse
import sys
from functools import partial

import torch

from vidHMR import REPO
from vidHMR.features import ROOT, run

ACE = REPO / "external" / "ACE-Ego-Hand"
sys.path.insert(0, str(ACE))
from ace_ego_hand.archs.wan_backbone import WanBackbone, fold_tokens  # noqa: E402
from ace_ego_hand.video_vae import MODEL_ROOT, VIDEOX_CFG, encode, load_vae  # noqa: E402

NAME = "wan22_fun5b_t0"
LAYERS = (5, 10, 15, 20, 25, 29)  # blocks (of 30) whose output is stored; ACE taps 15
CAPTION = "A video of a person."


def caption_embed(text: str, device) -> torch.Tensor:
    """umT5 embedding (L, 4096) of the fixed caption, cached after the first run."""
    cache = ROOT / NAME / "caption.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["caption"] == text:
            return saved["embed"]

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
    emb = emb[0, : int(ids.attention_mask.sum())].float().cpu()

    del te
    torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embed": emb, "caption": text}, cache)
    return emb


@torch.no_grad()
def extract_window(vae, dit, ctx, device, pixels) -> dict[int, torch.Tensor]:
    """One window of frames -> {block: (F_lat, H/32, W/32, dim)}."""
    lat = encode(vae, pixels, device).unsqueeze(0).to(device, torch.bfloat16)
    ctrl = lat.new_zeros(1, dit.control_channels, *lat.shape[2:])
    t = torch.zeros(1, device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        taps, _, grid = dit.forward_features(lat, ctrl, [ctx], t, LAYERS, run_head=False)
    return {k: fold_tokens(v, grid)[0].permute(1, 2, 3, 0) for k, v in taps.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="+", required=True, help="3DPW sequence names")
    ap.add_argument("--frames", type=int, default=0, help="first N frames only (0: all)")
    args = ap.parse_args()

    device = torch.device("cuda")
    ctx = caption_embed(CAPTION, device).to(device, torch.bfloat16)
    vae = load_vae(device)
    dit = WanBackbone(str(MODEL_ROOT), mh_expand=False).to(device)

    meta = {"model": "Wan2.2-Fun-5B-Control", "layers": LAYERS, "caption": CAPTION, "t": 0}
    run(partial(extract_window, vae, dit, ctx, device), NAME, args.seqs, args.frames, meta)


if __name__ == "__main__":
    main()
