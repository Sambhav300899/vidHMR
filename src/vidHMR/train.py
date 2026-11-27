"""Train the person decoder (task 5 of docs/training_plan.md).

    python -m vidHMR.train configs/train.toml               # the train split
    python -m vidHMR.train configs/overfit_one_person.toml  # milestones: a list of items

Logs to MLflow, local only (generated/mlflow/; view with
`mlflow ui --backend-store-uri sqlite:///generated/mlflow/mlflow.db`). Writes best.pt (lowest
validation MPJPE) and last.pt to generated/runs/<name>/.
"""

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: the ACE env, which runs the DiT in the live mode
    import tomli as tomllib

from vidHMR import paths
from vidHMR.body.smpl import SMPL
from vidHMR.datasets.bedlam import select, window_name
from vidHMR.datasets.pw3d_windows import (
    Pw3dWindows,
    make_loader,
    split_sequences,
)
from vidHMR.eval.metrics import mpjpe, pa_mpjpe
from vidHMR.eval.sam3d_pw3d import ROOT, SMPL_IDX
from vidHMR.model.body import decode
from vidHMR.model.decoder import PersonDecoder
from vidHMR.model.losses import losses, metrics


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over `warmup` steps, then cosine decay to 0 at `total` steps (and 0 after)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(step - warmup, total - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def start_mlflow() -> None:
    database = paths.generated("mlflow", "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{database}")
    if mlflow.get_experiment_by_name("vidHMR") is None:
        artifacts = (database.parent / "artifacts").as_uri()
        mlflow.create_experiment("vidHMR", artifact_location=artifacts)
    mlflow.set_experiment("vidHMR")


def flatten(config: dict, prefix: str = "") -> dict:
    """Nested config -> {"loss_weights.betas": "0.1", ...} for the MLflow parameters."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = str(value)
    return out


def predict(model, smpl, batch):
    """Decoder forward pass in bfloat16 (its heads run in float32), then the decode in float32.
    The stored int8 features become float16 here (int8 * scale per channel); live features
    (live_features) have no scale."""
    features = batch["features"].half()
    if "feature_scale" in batch:
        features.mul_(batch["feature_scale"].half()[:, None, None, None])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        *body, joints_2d, joints_relative = model(
            features, batch["prompt_mask"], batch["prompt_latent"], batch["K"],
            batch["image_size"],
        )  # fmt: skip
    return decode(
        *body, batch["K"], batch["image_size"], smpl, direct=(joints_2d, joints_relative)
    )


def to_cuda(batch: dict) -> dict:
    """The tensors on the GPU. A list of tensors (the features from make_loader) is stacked there,
    one copy per item."""

    def move(value):
        if torch.is_tensor(value):
            return value.cuda(non_blocking=True)
        if isinstance(value, list) and torch.is_tensor(value[0]):
            return torch.stack([item.cuda(non_blocking=True) for item in value])
        return value

    return {key: move(value) for key, value in batch.items()}


def on_gpu(loader):
    """The batches of `loader` on the GPU. The copy of the next batch runs on a side stream while
    the GPU computes the current one (this needs pinned memory: Pw3dWindows.pin_memory)."""
    stream = torch.cuda.Stream()

    def load(batch):
        with torch.cuda.stream(stream):
            return to_cuda(batch), stream.record_event()

    def ready(batch, copied):
        torch.cuda.current_stream().wait_event(copied)
        for value in batch.values():
            if torch.is_tensor(value):  # the memory stays reserved until the compute is done
                value.record_stream(torch.cuda.current_stream())
        return batch

    upcoming = None
    for batch in loader:
        batch = load(batch)
        if upcoming is not None:
            yield ready(*upcoming)
        upcoming = batch
    yield ready(*upcoming)


def live_features(loader, load_window, extract):
    """Live mode: the batches of `loader` on the GPU, with the features of each item's window
    from the DiT, run here: extract(load_window(sequence, first_frame)) -> (21, H, W, C). The
    windows of the next batch load in threads meanwhile."""
    pool = ThreadPoolExecutor(8)

    def windows(batch):  # (sequence, first frame) of each item; two people can share one
        return list(zip(batch["sequence"], batch["first_frame"].tolist(), strict=True))

    def load(batch):
        return batch, {window: pool.submit(load_window, *window) for window in windows(batch)}

    def ready(batch, inputs):
        with torch.no_grad():
            features = {window: extract(future.result()) for window, future in inputs.items()}
        stacked = torch.stack([features[window] for window in windows(batch)])
        return to_cuda(batch) | {"features": stacked}

    upcoming = None
    for batch in loader:
        batch = load(batch)
        if upcoming is not None:
            yield ready(*upcoming)
        upcoming = batch
    yield ready(*upcoming)


def train_items(config: dict) -> Pw3dWindows:
    """The config's `items` (the overfit milestones) or, without them, the 3DPW train split plus
    the first `bedlam_sequences` BEDLAM windows (vidHMR.datasets.bedlam.select). With `live`,
    a window starts at every 4th frame (Pw3dWindows all_starts)."""
    live = config.get("live", False)
    if "items" not in config:
        assert not (live and config["bedlam_sequences"]), "live: 3DPW only (wan22.latents_path)"
        bedlam_windows = [window_name(s) for s, _ in select(config["bedlam_sequences"])]
        sequences = split_sequences("train") + bedlam_windows
        return Pw3dWindows(sequences, config["backbone"], config["layer"], live, live)
    items = [tuple(item) for item in config["items"]]
    sequences = sorted({sequence for sequence, _, _ in items})
    dataset = Pw3dWindows(sequences, config["backbone"], config["layer"])
    assert set(items) <= set(dataset.items), "an item without a valid latent"
    dataset.items = items
    return dataset


def mean(rows: list[dict]) -> dict:
    """Mean over the batches of an epoch."""
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def validate(model, smpl, loader) -> dict:
    """The joint errors of vidHMR.eval.sam3d_pw3d (12 joints, mm) over the valid frames of the
    items: MPJPE after the midpoint of the hips is subtracted, and PA-MPJPE. shortcut: not the
    same frames as that evaluation: the frames of a sequence's last (overlapping) window count
    twice, camera-jump frames are out (our `valid`), and the 1-4 frames after the last window
    are not scored; fine for model selection, not for the final numbers."""
    errors = {"mpjpe": [], "pa_mpjpe": []}
    for batch in loader:
        batch = to_cuda(batch)
        valid = batch["valid"].cpu().numpy()
        predicted = predict(model, smpl, batch)["joints_3d"].cpu().numpy()[valid][:, SMPL_IDX]
        gt = batch["joints_3d"].cpu().numpy()[valid][:, SMPL_IDX]
        errors["mpjpe"].append(mpjpe(predicted, gt, ROOT).mean(-1))
        errors["pa_mpjpe"].append(pa_mpjpe(predicted, gt).mean(-1))
    return {key: float(np.concatenate(value).mean()) for key, value in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    with open(parser.parse_args().config, "rb") as file:
        config = tomllib.load(file)
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("high")  # TF32 for the float32 heads and decode
    run = paths.generated("runs", config["name"], "last.pt").parent  # creates the run folder

    live = config.get("live", False)  # run the DiT on the stored VAE latents of each batch
    train = train_items(config)
    if not live:
        train.pin_memory()
    train = make_loader(train, config["batch_size"], config["seed"])
    validation = Pw3dWindows(
        split_sequences("validation"), config["backbone"], config["layer"], live
    )
    validation = make_loader(validation, config["batch_size"], config["seed"])
    if live:
        from vidHMR.features import wan22  # needs the ACE env

        dit = wan22.load_dit(torch.device("cuda"), layers=(config["layer"],))

        def extract(latents):
            return dit(latents)[config["layer"]]

        def train_batches():
            return live_features(train, wan22.window_latents, extract)

        def validation_batches():
            return live_features(validation, wan22.window_latents, extract)

        channels = extract(wan22.window_latents(*train.dataset.items[0][:2])).shape[-1]
    else:

        def train_batches():
            return on_gpu(train)

        def validation_batches():
            return validation

        channels = train.dataset[0]["features"].shape[-1]

    model = PersonDecoder(channels).cuda()
    smpl = SMPL("n", device="cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    steps = len(train)  # per epoch
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(
            step, config["warmup_epochs"] * steps, config["epochs"] * steps
        ),
    )
    weights = config["loss_weights"]
    best = math.inf

    start_mlflow()
    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params(flatten(config) | {"steps_per_epoch": steps})
        start = time.time()
        for epoch in range(1, config["epochs"] + 1):
            log = epoch == 1 or epoch % config["log_every"] == 0  # epoch 1: the start
            rows = []
            for batch in train_batches():
                prediction = predict(model, smpl, batch)
                terms = losses(prediction, batch)
                total = sum(weights[key] * value for key, value in terms.items())
                if not torch.isfinite(total):
                    raise RuntimeError(f"loss {total.item()} at epoch {epoch}: {terms}")

                learning_rate = optimizer.param_groups[0]["lr"]  # the rate of this step
                optimizer.zero_grad()
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                schedule.step()

                if log:
                    row = {f"loss/{key}": value.item() for key, value in terms.items()}
                    row |= {"loss/total": total.item(), "learning_rate": learning_rate}
                    row |= {"gradient_norm": norm.item()}  # before the clip
                    rows.append(row | metrics(prediction, batch))

            if log:
                logged = mean(rows)
                mlflow.log_metrics(logged, step=epoch)
                print(
                    f"epoch {epoch:4d}  loss {logged['loss/total']:7.4f}"
                    f"  MPJPE {logged['mpjpe_pelvis_mm']:6.1f} mm"
                    f"  pelvis {logged['pelvis_mm']:6.1f} mm"
                    f"  2D {logged['joints_2d_px']:5.1f} px"
                    f" (direct {logged['direct_joints_2d_px']:5.1f} px,"
                    f" PnP {logged['pnp_accepted_pct']:3.0f} %)"
                    f"  rotation {logged['rotation_deg']:5.1f} deg"
                    f" ({logged['rotations_over_90_pct']:4.1f} % > 90)"
                    f"  {time.time() - start:5.0f} s",
                    flush=True,
                )

            if epoch % config["validate_every"] == 0 or epoch == config["epochs"]:
                errors = validate(model, smpl, validation_batches())
                mlflow.log_metrics({f"validation/{k}": v for k, v in errors.items()}, step=epoch)
                print(
                    f"validation  MPJPE {errors['mpjpe']:6.1f} mm"
                    f"  PA-MPJPE {errors['pa_mpjpe']:6.1f} mm",
                    flush=True,
                )
                state = {
                    "model": model.state_dict(), "config": config, "epoch": epoch,
                    "validation": errors,
                }  # fmt: skip
                torch.save(state, run / "last.pt")  # also at each validation, for a crash
                if errors["mpjpe"] < best:
                    best = errors["mpjpe"]
                    torch.save(state, run / "best.pt")

        mlflow.log_metric("validation/best_mpjpe", best)
        print(f"best validation MPJPE {best:.1f} mm")


if __name__ == "__main__":
    main()
