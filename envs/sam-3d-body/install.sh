#!/usr/bin/env bash
# Rebuild the SAM 3D Body env + weights (facebookresearch/sam-3d-body, submodule external/sam-3d-body).
# These are the commands actually run on genos (Ubuntu 20.04, A100, driver 535 / CUDA 12.2), 2026-11-20.
# Follows external/sam-3d-body/INSTALL.md; deviations are commented with the reason.
# Optional SAM2 segmentor / SAM3 detector are not installed (the demo default is ViTDet + MoGe-2;
# mask prompts are only used with --use_mask).
set -euo pipefail

REPO=/edrive4/srohatgi/vidHMR
ENV=/edrive4/srohatgi/envs/sam-3d-body
CKPT=$REPO/generated/checkpoints
CUDA_HOME=/usr/local/cuda-12.1  # matches the torch cu121 wheels; needed to build detectron2
export HF_HOME=/edrive4/srohatgi/hf_home       # keep HF cache (and saved token) off /
export TORCH_HOME=/edrive4/srohatgi/torch_home # torch.hub clones facebookresearch/dinov3 code here

# The login profile puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH, which breaks torch
# (undefined symbol __nvJitLinkComplete_12_4). Also run the model itself with `env -u LD_LIBRARY_PATH`.
unset LD_LIBRARY_PATH
# --isolated: the global pip config adds an NGC extra index that can hang.
PIP="$ENV/bin/python -m pip install --isolated --no-cache-dir"

# 1. Env (conda base is read-only -> --prefix)
conda create --prefix "$ENV" python=3.11 -y

# 2. PyTorch. cu121 so that the local CUDA 12.1 toolkit can build detectron2 against it.
$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Python deps from INSTALL.md. numpy<2: xtcocotools/chumpy-era deps and torch 2.5 are built
# against numpy 1.x. ("chump" is spelled as in INSTALL.md.)
$PIP "numpy<2" pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas \
    rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump \
    networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest \
    xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub
# setuptools>=81 dropped pkg_resources, which detectron2.model_zoo imports (ViTDet config load).
$PIP setuptools==80.9.0

# 4. Detectron2 (ViTDet person detector), built with gcc 10 + CUDA 12.1 for A100 (sm_80).
# A dir with gcc/g++ symlinks goes first on PATH (never put /usr/bin itself first).
mkdir -p "$ENV/gcc10bin"
ln -sf /usr/bin/gcc-10 "$ENV/gcc10bin/gcc"
ln -sf /usr/bin/g++-10 "$ENV/gcc10bin/g++"
PATH=$ENV/gcc10bin:$CUDA_HOME/bin:$PATH CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=8.0 \
    FORCE_CUDA=1 MAX_JOBS=16 \
    $PIP 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# 5. MoGe (FOV estimator). Pinned to the last pre-V3 commit (2025-11-02): MoGe HEAD (V3, 2026-08)
# requires numpy>=2 and a CUDA build of flex-gemm; sam-3d-body only needs moge.model.v2.
$PIP "git+https://github.com/microsoft/MoGe.git@07444410f1e3" torch==2.5.1 torchvision==0.20.1 "numpy<2"

# 6. vidHMR itself (for vidHMR.GENERATED in scripts/sam3d_body_video.py)
$PIP --no-deps -e "$REPO"

# 7. Weights (gated HF repo: the account must have accepted the terms of facebook/sam-3d-body-dinov3)
mkdir -p "$CKPT/vitdet"
"$ENV/bin/hf" download facebook/sam-3d-body-dinov3 --local-dir "$CKPT/sam-3d-body-dinov3"  # model.ckpt + assets/mhr_model.pt
"$ENV/bin/hf" download Ruicheng/moge-2-vitl-normal --local-dir "$CKPT/moge-2-vitl-normal"
wget -c -O "$CKPT/vitdet/model_final_f05665.pkl" \
    https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
# The DINOv3 backbone *code* is fetched by torch.hub (github facebookresearch/dinov3, pretrained=False)
# into $TORCH_HOME/hub on first model load; the weights come from model.ckpt.
# The MHR body model runs from the TorchScript assets/mhr_model.pt; the optional `mhr`/pymomentum
# package is not installed (mhr_head.py falls back to torch.jit.load and warns "Momentum is not enabled").

# Smoke check
$ENV/bin/python -c "import torch, detectron2, moge.model.v2, pyrender; print(torch.__version__, torch.cuda.is_available())"
