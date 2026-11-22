"""Smoke test for Cosmos3-Edge: the official Diffusers image-to-video example from the model card.

Same inputs and settings as the card (480p, 121 frames, 20 steps, guidance 6, flow shift 12),
except that the safety checker is off: it needs the gated nvidia/Cosmos-1.0-Guardrail repo.
The inputs come from the `assets/` folder of the HF snapshot.

Run in the cosmos3 env (see envs/cosmos3/install.sh), e.g.
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4 HF_HOME=/edrive4/srohatgi/hf_home \
        /edrive4/srohatgi/envs/cosmos3/bin/python scripts/cosmos3_edge_i2v.py
"""

import json
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from huggingface_hub import snapshot_download

from vidHMR import GENERATED
from vidHMR.features.cosmos3 import MODEL_ID, REVISION

OUT_DIR = GENERATED / "cosmos3_edge_test"


def main():
    model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION))
    json_prompt = json.loads((model_dir / "assets/example_i2v_prompt.json").read_text())
    negative_prompt = json.loads((model_dir / "assets/negative_prompt.json").read_text())

    t0 = time.time()
    # The card uses enable_safety_checker=True. Our HF account has no access to the gated
    # nvidia/Cosmos-1.0-Guardrail repo that it needs, so use the model_index default (False).
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, enable_safety_checker=False
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=12.0, use_karras_sigmas=False
    )
    t_load = time.time() - t0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = pipe(
        prompt=json.dumps(json_prompt),
        negative_prompt=json.dumps(negative_prompt),
        image=load_image(str(model_dir / "assets/example_i2v_input.jpg")),
        num_frames=121,
        height=480,
        width=832,
        fps=24.0,
        num_inference_steps=20,
        guidance_scale=6.0,
        generator=torch.Generator(device="cuda").manual_seed(0),
        add_resolution_template=False,
        add_duration_template=False,
    )
    t_gen = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "edge_i2v_diffusers.mp4"
    # macro_block_size=1 preserves the requested 832x480 resolution.
    export_to_video(result.video, str(out_path), fps=24, macro_block_size=1)
    print(f"Saved {out_path} ({len(result.video)} frames)")
    print(f"load {t_load:.1f} s, generate {t_gen:.1f} s")
    print(f"peak GPU memory: allocated {peak_gb:.1f} GB, reserved {reserved_gb:.1f} GB")


if __name__ == "__main__":
    main()
