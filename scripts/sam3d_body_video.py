"""Run SAM 3D Body per frame on an image sequence; save raw predictions and an overlay video.

SAM 3D Body is a single-image model, so the model, detector (ViTDet) and FOV estimator (MoGe-2)
are loaded once (for all given folders) and each frame is processed independently (no
temporal smoothing).

Run in the sam-3d-body env (see envs/sam-3d-body/install.sh), e.g.
    env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=7 \
        /edrive4/srohatgi/envs/sam-3d-body/bin/python scripts/sam3d_body_video.py \
        --image_folder generated/datasets/3DPW/imageFiles/downtown_walking_00 --num_frames 150
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# vidHMR.sam3d.run puts the sam-3d-body repo on sys.path, so this import must come after it.
from tools.vis_utils import visualize_sample_together  # noqa: E402

from vidHMR import GENERATED
from vidHMR.sam3d.run import build_estimator


def save_preds(path, outputs):
    """Save one frame's per-person outputs as an npz with arrays stacked on a person axis."""
    keys = [k for k in (outputs[0] if outputs else {}) if outputs[0][k] is not None]
    np.savez(path, num_people=len(outputs), **{k: np.stack([o[k] for o in outputs]) for k in keys})


def run_sequence(estimator, image_folder, out_dir, args):
    """Process the frames of one folder: one npz per frame in out_dir/preds and an overlay.mp4."""
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    frames = sorted(image_folder.glob("*.jpg"))[: args.num_frames or None]
    torch.cuda.reset_peak_memory_stats()

    ffmpeg = None
    times = []
    for frame in frames:
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = estimator.process_one_image(
            str(frame), bbox_thr=args.bbox_thresh, inference_type=args.inference_type
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        save_preds(out_dir / "preds" / f"{frame.stem}.npz", outputs)

        img = cv2.imread(str(frame))
        if outputs:
            vis = visualize_sample_together(img, outputs, estimator.faces).astype(np.uint8)
        else:
            vis = np.concatenate([img] * 4, axis=1)
        scale = args.video_height / vis.shape[0]
        vis = cv2.resize(vis, (round(vis.shape[1] * scale / 2) * 2, args.video_height))

        if ffmpeg is None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
                + ["-s", f"{vis.shape[1]}x{vis.shape[0]}", "-r", str(args.fps), "-i", "-"]
                + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
                + [str(out_dir / "overlay.mp4")],
                stdin=subprocess.PIPE,
            )
        ffmpeg.stdin.write(vis.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()

    steady = times[1:] or times
    print(
        f"{len(frames)} frames -> {out_dir}\n"
        f"model+detector+FOV per frame: {np.mean(steady):.3f}s mean (first {times[0]:.2f}s)\n"
        f"peak torch GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image_folder", required=True, type=Path, nargs="+")
    parser.add_argument("--num_frames", default=150, type=int, help="0: all frames")
    parser.add_argument("--out_root", default=GENERATED / "sam3d_body_test", type=Path)
    parser.add_argument("--inference_type", default="full", choices=["full", "body"])
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--bbox_thresh", default=0.8, type=float)
    parser.add_argument("--video_height", default=540, type=int)
    args = parser.parse_args()

    estimator = build_estimator(torch.device("cuda"))

    # One model load for all folders; a `done` file marks a finished folder, so reruns resume.
    for folder in args.image_folder:
        out_dir = args.out_root / folder.name
        if (out_dir / "done").exists():
            continue
        run_sequence(estimator, folder, out_dir, args)
        (out_dir / "done").touch()


if __name__ == "__main__":
    main()
