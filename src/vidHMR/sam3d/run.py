"""Run SAM 3D Body (single-frame) on a folder of frames or a 3DPW sequence.

Run with the sam-3d-body environment, from the project root:
  PYOPENGL_PLATFORM=egl sam-3d-body/.venv/bin/python -m vidHMR.sam3d.run --pw3d
      downtown_warmWelcome_00 --num-frames 150
  PYOPENGL_PLATFORM=egl sam-3d-body/.venv/bin/python -m vidHMR.sam3d.run --frames path/to/jpgs
      --name my_clip

Outputs go to generated/sam3d_runs/<name>/: result.mp4, frames/, predictions.npy.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from vidHMR import paths
from vidHMR.video import write_video

# Importing this module puts the sam-3d-body repo on sys.path (for `sam_3d_body` and `tools`).
sys.path.insert(0, str(paths.SAM3D_REPO))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

KEEP_KEYS = (
    "bbox",
    "focal_length",
    "pred_cam_t",
    "pred_keypoints_2d",
    "pred_keypoints_3d",
    "shape_params",
    "scale_params",
    "expr_params",
    "body_pose_params",
    "hand_pose_params",
    "global_rot",
)


def build_estimator(device=None, fov=True):
    """Load SAM 3D Body (DINOv3 checkpoint), the ViTDet person detector and, if fov, MoGe-2.

    Checkpoints come from generated/checkpoints. With fov=False, pass the intrinsics to
    process_one_image(cam_int=...) instead.
    """
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector
    from tools.build_fov_estimator import FOVEstimator

    ckpt = paths.CHECKPOINTS / "sam-3d-body-dinov3"
    device = device or torch.device("cuda")
    model, model_cfg = load_sam_3d_body(
        str(ckpt / "model.ckpt"), device=device, mhr_path=str(ckpt / "assets" / "mhr_model.pt")
    )
    detector = HumanDetector(name="vitdet", device=device, path=str(paths.CHECKPOINTS / "vitdet"))
    fov_estimator = None
    if fov:
        moge = paths.CHECKPOINTS / "moge-2-vitl-normal" / "model.pt"
        fov_estimator = FOVEstimator(
            name="moge2", device=device, path=str(moge) if moge.exists() else None
        )
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )


def run(frame_paths, out_dir, fps=30):
    est = build_estimator()
    from tools.vis_utils import visualize_sample_together
    from tqdm import tqdm

    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    results = {}

    def frames():
        for f in tqdm(frame_paths):
            outputs = est.process_one_image(str(f), bbox_thr=0.8)
            img = cv2.imread(str(f))
            # The repo's visualizer fails on an empty list, so keep the raw frame when no person is
            # found.
            vis = (
                visualize_sample_together(img, outputs, est.faces).astype(np.uint8)
                if outputs
                else img
            )
            cv2.imwrite(str(out_dir / "frames" / f"{f.stem}.jpg"), vis)
            results[f.stem] = [{k: o[k] for k in KEEP_KEYS if k in o} for o in outputs]
            yield cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    write_video(frames(), out_dir / "result.mp4", fps=fps)
    np.save(out_dir / "predictions.npy", results, allow_pickle=True)
    print(
        f"DONE {len(frame_paths)} frames, {sum(map(len, results.values()))} detections -> "
        f"{out_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of .jpg frames")
    src.add_argument("--pw3d", help="3DPW sequence name")
    ap.add_argument("--name", help="output name (default: sequence or folder name)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=None)
    args = ap.parse_args()

    folder = paths.PW3D / "imageFiles" / args.pw3d if args.pw3d else paths.Path(args.frames)
    frame_paths = sorted(folder.glob("*.jpg"))[args.start_frame :]
    if args.num_frames:
        frame_paths = frame_paths[: args.num_frames]
    name = args.name or folder.name
    run(frame_paths, paths.GENERATED / "sam3d_runs" / name)


if __name__ == "__main__":
    main()
