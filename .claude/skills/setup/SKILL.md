---
name: setup
description: Set up vidHMR on a new machine (submodules, envs, weights, 3DPW), or rebuild or repair one env. Use when installing vidHMR, when an env, weight or dataset is missing, when an env is broken (import errors, CUDA symbols), or when someone asks what vidHMR depends on.
---

# Setting up vidHMR

Each model runs in its own env, because their torch and CUDA builds conflict. The `vidHMR` package
itself has no dependencies (`pyproject.toml`); each env installs it with `--no-deps -e`.
All data and weights go in `generated/` (gitignored), except the HF cache and the Wan weights.

## What exists

| env | install | type | used by | weights |
|---|---|---|---|---|
| ace-ego-hand | `envs/ace-ego-hand/install.sh` | conda, py3.10, torch 2.5.1+cu124 | `vidHMR.features.wan22` | Wan2.2-Fun-5B-Control, 24 GB (in ACE's `ckpt/`) |
| cosmos3 | `envs/cosmos3/install.sh` | uv venv, py3.12, torch 2.11.0+cu128, diffusers at a commit | `vidHMR.features.cosmos3`, `scripts/cosmos3_edge_i2v.py` | nvidia/Cosmos3-Edge, ~8.6 GB (HF cache) |
| sam-3d-body | `envs/sam-3d-body/install.sh` | conda, py3.11, torch 2.5.1+cu121, detectron2 + MoGe built from git | `scripts/sam3d_body_video.py`, `scripts/sam3d_body_3dpw_oc.sh` | SAM 3D Body, ViTDet, MoGe-2 in `generated/checkpoints/`, ~6.5 GB |

The cosmos3 and sam-3d-body scripts record the commands that ran on genos on 2026-11-20. The
ace-ego-hand script is egoCUBE's script, reduced to what wan22 needs. All pin their versions.
Their paths are variables at the top (genos layout); change them on another machine.

## Order

1. **Gated access first.** These need a person, not an agent; ask the user.
   - `hf auth login` with an account that accepted the terms of `facebook/sam-3d-body-dinov3`.
   - The 3DPW licence (https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html).
2. **Submodules:** `git submodule update --init` (ACE-Ego-Hand, VideoX-Fun, sam-3d-body, pinned
   commits). Do not copy external code into the repo.
3. **Envs:** run each install script (the weights download is the last step of each).
   - `bash envs/ace-ego-hand/install.sh`. It also links `external/VideoX-Fun` as ACE's gitignored
     `third_party/` and downloads the Wan weights into ACE's gitignored `ckpt/`, where ACE's code
     looks for them. vidHMR is not installed in this env: run it with `PYTHONPATH=src`.
   - `bash envs/cosmos3/install.sh`
   - `bash envs/sam-3d-body/install.sh` (builds detectron2 with gcc 10 and CUDA 12.1; ~20 min)

   On genos the ace-ego-hand env already exists at
   `/edrive4/srohatgi/egocube_deps/conda/envs/ace-ego-hand` (built for egoCUBE), and
   `external/ACE-Ego-Hand/ckpt/Wan2.2-Fun-5B-Control` links egoCUBE's copy of the weights.
4. **3DPW** (5.2 GB unpacked) into `generated/datasets/3DPW`:
   ```bash
   mkdir -p generated/datasets/3DPW && cd generated/datasets/3DPW
   for f in readme_and_demo sequenceFiles imageFiles; do
       wget -c https://virtualhumans.mpi-inf.mpg.de/3DPW/$f.zip
   done
   for f in *.zip; do unzip -q -n "$f"; done && rm *.zip
   ```
   The ground truth uses gendered SMPL. On genos only SMPL_NEUTRAL is available, in
   `/edrive4/srohatgi/HOSIG/body_models/smpl/`.
5. **Check:** run each user on a few frames.
   ```bash
   PYTHONPATH=src env -u LD_LIBRARY_PATH \
       /edrive4/srohatgi/egocube_deps/conda/envs/ace-ego-hand/bin/python \
       -m vidHMR.features.wan22 --seqs downtown_walking_00 --frames 81
   env -u LD_LIBRARY_PATH /edrive4/srohatgi/envs/cosmos3/bin/python \
       -m vidHMR.features.cosmos3 --seqs downtown_walking_00 --frames 81
   env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl /edrive4/srohatgi/envs/sam-3d-body/bin/python \
       scripts/sam3d_body_video.py --image_folder generated/datasets/3DPW/imageFiles/downtown_walking_00 \
       --num_frames 10
   ```

## genos

- 8× A100 80GB, shared with other users. No Slurm: pick a GPU with `CUDA_VISIBLE_DEVICES`. Free
  GPU memory changes fast; jobs of other users can take it and cause out-of-memory errors.
- Ubuntu 20.04, driver 535 (CUDA 12.2). The cu121, cu124 and cu128 torch wheels all run through
  CUDA minor-version compatibility.
- `/` is almost full: keep caches on /edrive4 (`HF_HOME=/edrive4/srohatgi/hf_home`,
  `TORCH_HOME=/edrive4/srohatgi/torch_home`, `UV_CACHE_DIR=/edrive4/srohatgi/egocube_deps/uv_cache`).
- uv is `/edrive4/srohatgi/egocube_deps/bin/uv`. The conda base (`/usr/local/anaconda3`) is
  read-only, so conda envs are prefixes.
- Ruff: `env UV_CACHE_DIR=/edrive4/srohatgi/egocube_deps/uv_cache /edrive4/srohatgi/egocube_deps/bin/uvx ruff format .`
  (then `ruff check .`).
- Long runs that must survive a closed session: `setsid nohup <cmd> > log 2>&1 < /dev/null &`.

## Common issues

- `undefined symbol: __nvJitLinkComplete_12_4` on `import torch`: the login profile puts
  `/usr/local/cuda/lib64` on `LD_LIBRARY_PATH`. Run every env with `env -u LD_LIBRARY_PATH`.
- pip hangs: the global pip config adds an NGC index that does not resolve. Use
  `pip install --isolated` (uv ignores pip.conf).
- `No module named pkg_resources` in detectron2: setuptools ≥ 81. Keep `setuptools==80.9.0`.
- MoGe HEAD (V3) needs numpy ≥ 2 and flex-gemm. sam-3d-body pins the pre-V3 commit `07444410f1e3`.
- Cosmos3 `enable_safety_checker=True` needs the gated `nvidia/Cosmos-1.0-Guardrail` (403 for our
  account). Our scripts pass `enable_safety_checker=False`.
- `nvidia-smi` and every new CUDA program hang in D state: the NVIDIA driver is stuck (2026-11-22,
  Xid 13, soft lockups in `nvidia_close`). `kill -9` has no effect. Only a reboot by an
  administrator repairs it.
