#!/bin/bash
# SAM 3D Body (body decoder only) on every frame of 3DPW-OC: the sequences split over the given
# GPUs by frame count, one process per GPU (models loaded once). Finished sequences (a `done`
# file) are skipped, so rerunning resumes. Outputs: generated/sam3d_body/3dpw_oc/<seq>/.
# Usage: bash scripts/sam3d_body_3dpw_oc.sh GPU [GPU ...]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/edrive4/srohatgi/envs/sam-3d-body/bin/python
OUT=generated/sam3d_body/3dpw_oc
mkdir -p "$OUT/logs"

# One list per GPU with about equal frame counts (largest sequences first, to the lightest list).
mapfile -t LISTS < <(PYTHONPATH=src $PY - "$#" <<'PYEOF'
import sys

from vidHMR import threedpw

n = int(sys.argv[1])
lists, counts = [[] for _ in range(n)], [0] * n
for seq in sorted(threedpw.oc_sequences(), key=lambda s: -len(threedpw.frames(s))):
    i = counts.index(min(counts))
    lists[i].append(f"{threedpw.ROOT}/imageFiles/{seq}")
    counts[i] += len(threedpw.frames(seq))
for folders in lists:
    print(*folders)
PYEOF
)

i=0
for gpu in "$@"; do
    env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src \
        $PY scripts/sam3d_body_video.py --image_folder ${LISTS[$i]} --num_frames 0 \
        --inference_type body --out_root "$OUT" > "$OUT/logs/gpu$gpu.log" 2>&1 &
    i=$((i + 1))
done
wait
