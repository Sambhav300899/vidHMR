"""Run a trained person decoder (generated/runs/<run>/{last,best}.pt) on every window of one
sequence and join the windows into one prediction per frame.

    python -m vidHMR.infer overfit_one_person outdoors_climbing_00 [person] [--checkpoint best]

Writes generated/runs/<run>/<sequence>_<person>.npz: the decode outputs with one row per frame
(betas too), plus `window`, the first frame of the window that each frame comes from. A frame in
two windows takes the later window. The output ends with the last window. Prints the metrics
of each window.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import default_collate

from vidHMR import paths
from vidHMR.body.smpl import SMPL
from vidHMR.datasets.pw3d_windows import Pw3dWindows
from vidHMR.features import WINDOW
from vidHMR.model.decoder import PersonDecoder
from vidHMR.model.losses import metrics
from vidHMR.train import predict, to_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("sequence")
    parser.add_argument("person", type=int, nargs="?", default=0)
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last")
    args = parser.parse_args()

    checkpoint = torch.load(paths.generated("runs", args.run, f"{args.checkpoint}.pt"))
    config = checkpoint["config"]
    dataset = Pw3dWindows([args.sequence], config["backbone"], config["layer"])
    indices = [i for i, (_, _, person) in enumerate(dataset.items) if person == args.person]
    n_frames = dataset.targets[args.sequence]["valid"].shape[1]
    assert len(indices) > 0, f"{args.sequence} has no items for person {args.person}"

    model = PersonDecoder(channels=dataset[indices[0]]["features"].shape[-1]).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    smpl = SMPL("n", device="cuda")
    joined = {"window": np.full(n_frames, -1)}
    for index in indices:
        batch = to_cuda(default_collate([dataset[index]]))
        with torch.no_grad():
            prediction = predict(model, smpl, batch)

        first_frame = batch["first_frame"][0].item()
        errors = metrics(prediction, batch)
        print(
            f"window {first_frame:5d}  MPJPE {errors['mpjpe_pelvis_mm']:6.1f} mm"
            f"  pelvis {errors['pelvis_mm']:6.1f} mm  2D {errors['joints_2d_px']:5.1f} px",
            flush=True,
        )
        frames = slice(first_frame, first_frame + WINDOW)
        prediction["betas"] = prediction["betas"][:, None].expand(-1, WINDOW, -1)
        for key, value in prediction.items():
            value = value[0].cpu().numpy()
            joined.setdefault(key, np.full((n_frames, *value.shape[1:]), np.nan, np.float32))
            joined[key][frames] = value
        joined["window"][frames] = first_frame

    end = joined["window"].max() + WINDOW  # the frames after the last window have no features
    joined = {key: value[:end] for key, value in joined.items()}
    # shortcut: a person who is absent for a whole window (no valid latent, no item) leaves a gap
    # here; the output would need to mark it (needed for the evaluation of task 6).
    assert (joined["window"] >= 0).all(), "frames without a window"
    path = paths.generated("runs", args.run, f"{args.sequence}_{args.person}.npz")
    np.savez(path, sequence=args.sequence, person=args.person, **joined)
    print(path)


if __name__ == "__main__":
    main()
