#!/usr/bin/env bash
# ACE-Ego-Hand env (conda, Python 3.10) for vidHMR.features.wan22: ACE's Wan2.2 backbone and VAE
# code (submodule external/ACE-Ego-Hand) on the Wan2.2-Fun-5B-Control weights.
# Copied from egoCUBE's envs/ace-ego-hand/install.sh, where it ran on PSC (H100) and genos
# (A100, driver 535). Removed: ACE's hand-head checkpoints, MANO and ACE's caption precompute
# (wan22 reads no heads and caches its own caption embedding in generated/features/).
# torch 2.5.1+cu124 (PyPI wheel, bundles its CUDA runtime); no Blackwell (sm_100/sm_120) kernels.
# Disk: env ~8 GB, Wan2.2-Fun-5B-Control 24 GB. Time: ~15 min of pip plus the download.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV=${ENV:-/edrive4/srohatgi/envs/ace-ego-hand}
CONDA=${CONDA:-conda}  # e.g. CONDA="$REPO/.toolchains/bin/micromamba" on a machine without conda
[ ! -e "$ENV" ] || { echo "$ENV exists; not touching it (set ENV=<new prefix>)"; exit 1; }
ACE=$REPO/external/ACE-Ego-Hand
export HF_HOME=${HF_HOME:-/edrive4/srohatgi/hf_home}  # keep the HF cache off the full / disk

# The login profile puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which breaks torch
# (undefined symbol __nvJitLinkComplete_12_4). Also run the model with `env -u LD_LIBRARY_PATH`.
unset LD_LIBRARY_PATH
# The global pip config adds https://pypi.ngc.nvidia.com, which does not resolve: without
# --isolated pip retries it for every package and hangs (and a killed install leaves a broken
# torch: "No module named 'torch.fft'").
PIP="$ENV/bin/pip install --isolated -q"

# 1. Code: ACE and VideoX-Fun (the Wan backbone code) are submodules. ACE imports VideoX-Fun from
#    its gitignored third_party/ folder, so link it there.
git -C "$REPO" submodule update --init external/ACE-Ego-Hand external/VideoX-Fun
[ -e "$ACE/third_party" ] || ln -s ../VideoX-Fun "$ACE/third_party"

# 2. Env (the conda base on genos is read-only -> --prefix)
$CONDA create -y -q -p "$ENV" python=3.10

# 3. torch first, pinned (ACE's requirements.txt says CUDA 12.1, but PyPI's 2.5.1 is the cu124
#    build, which is what ran); torchvision 0.20.1 is needed by VideoX-Fun and must match
$PIP torch==2.5.1 torchvision==0.20.1

# 4. ACE's own pins (numpy 1.24.4, diffusers 0.36.0, transformers 4.57.3, peft 0.18.1, ...) plus
#    the hf CLI. Do NOT install chumpy (only ACE's MANO code needs it).
$PIP -r "$ACE/requirements.txt" "huggingface_hub[cli]==0.36.2"

# 5. VideoX-Fun imports these but ACE's requirements miss them (found by import-and-retry);
#    do NOT install VideoX-Fun's own requirements.txt over ACE's pins
$PIP librosa==0.11.0 imageio==2.38.0

# 5b. The live training mode (vidHMR.train, live = true) runs in this env: MLflow (skinny, the
#     same version as the project venv; the full mlflow would move numpy off ACE's pin) and
#     tomli (Python 3.10 has no tomllib)
$PIP "mlflow-skinny==3.18.0" sqlalchemy alembic tomli "numpy==1.24.4"

# 6. Weights (public, no login) into ACE's gitignored ckpt/, where ace_ego_hand.video_vae looks:
#    config, diffusion safetensors, Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth,
#    google/umt5-xxl/. On genos a copy already exists: link
#    /edrive4/srohatgi/egoCUBE/external/ACE-Ego-Hand/ckpt/Wan2.2-Fun-5B-Control instead.
WAN=$ACE/ckpt/Wan2.2-Fun-5B-Control
[ -e "$WAN" ] || "$ENV/bin/hf" download alibaba-pai/Wan2.2-Fun-5B-Control \
    --revision b8bc1a65ab71d054ba4636dc0dac104aa4df2686 --local-dir "$WAN"

# Verification. "Your transformers version is too old to load Mistral3Model" from VideoX-Fun is
# harmless. vidHMR is not installed in this env: run it with PYTHONPATH=src.
"$ENV/bin/pip" check
"$ENV/bin/python" -c "import torch, torchvision, diffusers, transformers, peft, cv2, omegaconf, \
einops; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
(cd "$REPO" && PYTHONPATH=src "$ENV/bin/python" -c "import vidHMR.features.wan22; print('wan22 ok')")
