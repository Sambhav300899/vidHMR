#!/usr/bin/env bash
# Rebuild the Cosmos3-Edge env + weights (nvidia/Cosmos3-Edge, Diffusers path from the model card).
# These are the commands actually run on genos (Ubuntu 20.04, A100, driver 535 / CUDA 12.2), 2026-11-20.
# The model card's Diffusers recipe needs no code repo: Cosmos3OmniPipeline lives in diffusers main.
# Test: scripts/cosmos3_edge_i2v.py (official image-to-video example, 480p, 121 frames).
set -euo pipefail

REPO=/edrive4/srohatgi/vidHMR
ENV=/edrive4/srohatgi/envs/cosmos3
UV=/edrive4/srohatgi/egocube_deps/bin/uv
export UV_CACHE_DIR=/edrive4/srohatgi/egocube_deps/uv_cache
# Keep uv's managed Pythons on /edrive4 (the default ~/.local/share/uv is on the full / disk).
export UV_PYTHON_INSTALL_DIR=/edrive4/srohatgi/egocube_deps/uv_python
export HF_HOME=/edrive4/srohatgi/hf_home  # keep the HF cache (and saved token) off /
DIFFUSERS_COMMIT=c215605a189169236544c1c9828789bfca0891d1  # diffusers main on 2026-11-20
MODEL_REV=6afc1256321c3a8e623d89be3bc5117a1e90f96b         # Cosmos3-Edge main on 2026-11-20

# The login profile puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which breaks torch
# (undefined symbol __nvJitLinkComplete_12_4). Also run the model with `env -u LD_LIBRARY_PATH`.
unset LD_LIBRARY_PATH
export VIRTUAL_ENV=$ENV  # target of the `uv pip install` calls below

# 1. Env. Model card: `uv venv --python 3.12 --seed --managed-python` (resolved to 3.12.13).
# huggingface_hub goes in first so that the weight download (step 5) can run in parallel.
$UV venv --python 3.12 --seed --managed-python "$ENV"
$UV pip install "huggingface_hub[hf_xet]==2.4.0"

# 2. PyTorch. The card installs plain `torch`, but the default PyPI wheel is built for a newer
# CUDA than driver 535 supports. cu128 wheels run on this driver (minor-version compatibility;
# torch 2.11.0+cu128 is already in use in other envs on genos).
$UV pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Diffusers from git (Cosmos3OmniPipeline is not in a release yet) + the card's other deps.
# diffusers is pinned to a commit; the others are the versions uv resolved on 2026-11-20.
$UV pip install \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@$DIFFUSERS_COMMIT" \
    accelerate==1.16.0 av==19.0.2 cosmos-guardrail==0.3.2 \
    imageio==2.38.2 imageio-ffmpeg==0.8.1 transformers==5.23.0

# 4. vidHMR package itself (scripts import vidHMR.GENERATED); no deps.
$UV pip install --no-deps -e "$REPO"

# 5. Weights (~8.6 GB) into the HF cache. Not gated.
# The card example uses enable_safety_checker=True (cosmos_guardrail). That needs the GATED repo
# nvidia/Cosmos-1.0-Guardrail (+ Qwen/Qwen3Guard-Gen-0.6B); our HF account got 403, so
# scripts/cosmos3_edge_i2v.py runs with enable_safety_checker=False (the model_index default).
"$ENV/bin/hf" download nvidia/Cosmos3-Edge --revision "$MODEL_REV"
