"""Cosmos3-Edge features: one forward pass of the generator tower over the clean video latents.

Same recipe as Wan2.2: every frame labelled noisy at t = 0 (Cosmos3 adds the timestep
embedding of 0 to every vision token, once, before the first layer). Inputs are built with the
pipeline's own helpers (prompt template, packing, 3D mRoPE with the real 3DPW fps). Runs in the
cosmos3 env (envs/cosmos3/install.sh):

    python -m vidHMR.features.cosmos3 --seqs downtown_walking_00 --frames 162
"""

import argparse
from functools import partial

import torch
from diffusers import Cosmos3OmniPipeline
from huggingface_hub import snapshot_download

from vidHMR.datasets import pw3d
from vidHMR.features import LONG, SHORT, WINDOW, cached_text, run

MODEL_ID = "nvidia/Cosmos3-Edge"
REVISION = "6afc1256321c3a8e623d89be3bc5117a1e90f96b"  # main on 2026-11-20
NAME = "cosmos3_edge_t0"
LAYERS = (5, 9, 14, 19, 23, 27)  # layers (of 28) at the depths of the Wan2.2 blocks (of 30)
PROMPT = (
    "Renders of a 3D human body mesh: color-coded depth, body joint skeleton, surface normals."
)


_SEGMENTS: dict[str, dict] = {}  # text_segment results already read from disk


class _StopError(Exception):
    """Raised after the last tapped layer: nothing later is needed."""


def text_segment(pipe, n_frames: int, h: int, w: int, device) -> dict:
    """The packed text of the joint sequence for one window size, cached on disk and in memory.

    Cosmos3 has no text encoder: the transformer embeds token IDs itself. The IDs depend on the
    window (the prompt gets a duration and a resolution sentence), so each size has its own entry.
    """
    key = {
        "revision": REVISION,
        "n_frames": n_frames,
        "height": h,
        "width": w,
        "fps": pw3d.FPS,
        "system_prompt": pipe.config.default_use_system_prompt,
    }

    def build():
        ids, _ = pipe.tokenize_prompt(PROMPT, num_frames=n_frames, height=h, width=w, fps=pw3d.FPS)
        return pipe._prepare_text_segment(ids, device="cpu")

    name = f"cosmos3_edge_{n_frames}f_{h}x{w}"
    if name not in _SEGMENTS:
        _SEGMENTS[name] = cached_text(name, PROMPT, key, build)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in _SEGMENTS[name].items()}


@torch.no_grad()
def extract_window(pipe, pixels) -> dict[int, torch.Tensor]:
    """One window of frames -> {layer: (F_lat, H/32, W/32, 2048)}, vision tokens only."""
    device = pipe.transformer.device
    n_frames, _, h, w = pixels.shape
    lat = pipe._encode_video(pixels.permute(1, 0, 2, 3).unsqueeze(0).to(device, torch.bfloat16))

    text = text_segment(pipe, n_frames, h, w, device)
    vis = pipe._prepare_vision_segment(
        input_vision_tokens=lat,
        has_image_condition=False,
        mrope_offset=text["vision_start_temporal_offset"],
        vision_fps=pw3d.FPS,
        curr=text["und_len"],
        device=device,
        condition_frame_indexes=[],
    )

    taps = {}
    n_vis = vis["num_vision_tokens"]

    def tap(layer, module, args, out):
        taps[layer] = out[1][:n_vis]  # gen_seq starts with the vision tokens
        if layer == max(LAYERS):
            raise _StopError

    layers = pipe.transformer.layers
    hooks = [layers[k].register_forward_hook(partial(tap, k)) for k in LAYERS]
    try:
        pipe.transformer(
            input_ids=text["input_ids"],
            text_indexes=text["text_indexes"],
            position_ids=torch.cat([text["text_mrope_ids"], vis["vision_mrope_ids"]], dim=1),
            und_len=text["und_len"],
            sequence_length=text["und_len"] + n_vis,
            vision_tokens=[lat.to(torch.bfloat16)],
            vision_token_shapes=vis["vision_token_shapes"],
            vision_sequence_indexes=vis["vision_sequence_indexes"],
            vision_mse_loss_indexes=vis["vision_mse_loss_indexes"],
            vision_timesteps=torch.zeros(n_vis, device=device),
            vision_noisy_frame_indexes=vis["vision_noisy_frame_indexes"],
            return_dict=False,
        )
    except _StopError:
        pass
    finally:
        for hk in hooks:
            hk.remove()

    t, ph, pw = vis["vision_token_shapes"][0]
    return {k: v.view(t, ph, pw, -1) for k, v in taps.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="*", default=[], help="3DPW sequence names, or all")
    ap.add_argument("--text-only", action="store_true", help="only cache the text tokens")
    ap.add_argument("--frames", type=int, default=0, help="first N frames only (0: all)")
    args = ap.parse_args()
    if not (args.seqs or args.text_only):
        ap.error("give --seqs or --text-only")

    pipe = Cosmos3OmniPipeline.from_pretrained(
        snapshot_download(MODEL_ID, revision=REVISION),
        torch_dtype=torch.bfloat16,
        enable_safety_checker=False,
    )
    if args.text_only:  # landscape and portrait windows; the weights stay on the CPU
        for h, w in ((SHORT, LONG), (LONG, SHORT)):
            text_segment(pipe, WINDOW, h, w, "cpu")
        return
    pipe = pipe.to("cuda")

    meta = {
        "model": f"{MODEL_ID}@{REVISION}",
        "layers": LAYERS,
        "prompt": PROMPT,
        "t": 0,
        "fps": pw3d.FPS,
    }
    run(partial(extract_window, pipe), NAME, args.seqs, args.frames, meta)


if __name__ == "__main__":
    main()
