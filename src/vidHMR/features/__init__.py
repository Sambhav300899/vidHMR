"""Frozen video-DiT features over 3DPW: frame loading, windows and storage for every backbone.

Each backbone module (`wan22`, `cosmos3`) runs in its own env and gives `run` a function
pixels (F, 3, H, W) in [-1, 1] -> {block: (F_lat, H/32, W/32, dim)}; `run` does the rest.

Layout: ROOT/<name>/<seq>/layer_<block>.npy (fp16 memmap, (G, H/32, W/32, dim): one row per
latent frame of the whole sequence) and meta.json. Latent g covers frame 0 (g = 0) or frames
4g-3 .. 4g (g >= 1). The model sees WINDOW frames at once, so windows chain: each starts STRIDE
frames after the last and shares one frame with it. That frame is the first latent of the new
window, which is not stored again (the previous window already gave its latent).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from vidHMR import GENERATED
from vidHMR.datasets import pw3d

WINDOW = 81  # frames per forward pass = 21 latent frames (the causal VAE needs 4k + 1)
STRIDE = WINDOW - 1  # frames between window starts: windows share one frame (a multiple of 4)
N_LATENT = (WINDOW - 1) // 4 + 1  # latent frames of one window
SHORT, LONG = 480, 832  # sides frames are resized to: 480p, both multiples of 32
ROOT = GENERATED / "features"
TEXT = (
    GENERATED / "text_embeds"
)  # cached text conditioning, kept apart from the (rebuildable) features


def cached_text(name: str, text: str, key: dict, build):
    """Text conditioning stored in TEXT/<name>.pt; `build()` runs only when the file is missing or
    was made for another text or `key` (everything else the value depends on)."""
    file = TEXT / f"{name}.pt"
    if file.exists():
        saved = torch.load(file)
        if saved["text"] == text and saved["key"] == key:
            return saved["value"]

    value = build()
    TEXT.mkdir(parents=True, exist_ok=True)
    torch.save({"text": text, "key": key, "value": value}, file)
    return value


def target_size(path) -> tuple[int, int]:
    """(W, H) to resize to, keeping the frame's orientation (3DPW has portrait and landscape)."""
    h, w = cv2.imread(str(path)).shape[:2]
    return target_size_of(w, h)


def target_size_of(width: int, height: int) -> tuple[int, int]:
    """(W, H) to resize a width x height frame to, keeping its orientation."""
    return (SHORT, LONG) if height > width else (LONG, SHORT)


def load_frames(paths, size) -> torch.Tensor:
    """Frame files -> (F, 3, H, W) float32 in [-1, 1], RGB, resized to size (W, H)."""
    return frames_to_tensor([cv2.imread(str(p)) for p in paths], size)


def frames_to_tensor(images, size) -> torch.Tensor:
    """BGR images -> (F, 3, H, W) float32 in [-1, 1], RGB, resized to size (W, H)."""
    frames = [cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, ::-1] for img in images]
    a = np.stack(frames).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def to_fp16(block: int, x: torch.Tensor) -> torch.Tensor:
    """Stored format: fp16 on the CPU. Refuses values fp16 can't hold (and inf / NaN)."""
    amax = x.abs().max().item()
    if not amax < torch.finfo(torch.float16).max:
        raise ValueError(f"block {block}: max |x| = {amax} does not fit fp16")
    return x.half().contiguous().cpu()


def to_int8(x) -> tuple[np.ndarray, np.ndarray]:
    """Stored format of the training features: int8 with one float32 scale per channel (the
    channel's largest |value| over all tokens of the sequence, / 127). Dequantize with
    int8 * scale. On 3DPW validation it changes MPJPE by <= 0.02 mm (docs/training_plan.md)."""
    x = torch.as_tensor(x).float()
    scale = x.abs().flatten(0, -2).amax(0).clamp_min(1e-8) / 127
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8).numpy(), scale.numpy()


def save_int8(folder, layer: int, x) -> None:
    """Write layer_XX_int8.npy and layer_XX_scale.npy (see to_int8)."""
    values, scale = to_int8(x)
    np.save(folder / f"layer_{layer:02d}_int8.npy", values)
    np.save(folder / f"layer_{layer:02d}_scale.npy", scale)


def plan_chunks(n_frames: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Windows for a sequence: (G, [(start_frame, first_latent, end_latent)]).

    Latents [first, end) (global indices) are taken from the window; together the chunks cover
    all G latents once. A window gives all its latents except the first, which holds the frame
    shared with the previous window. The last window is moved back to end at the last full latent
    frame, so it overlaps the previous one by more than a frame; at most 3 trailing frames have no
    latent.
    """
    if n_frames < WINDOW:
        return 0, []

    starts = list(range(0, n_frames - WINDOW + 1, STRIDE))
    tail = 4 * ((n_frames - WINDOW) // 4)
    if tail > starts[-1]:
        starts.append(tail)

    total = (n_frames - 1) // 4 + 1
    chunks, prev = [], 0
    for i, start in enumerate(starts):
        end = total if i == len(starts) - 1 else start // 4 + N_LATENT
        chunks.append((start, prev, end))
        prev = end
    return total, chunks


def run(extract_window, name: str, seqs: list[str], frames: int, meta: dict) -> None:
    """Extract features of each sequence (`all`: every 3DPW sequence) into ROOT/name/<seq>/, as
    layer_XX_int8.npy + layer_XX_scale.npy (to_int8, one scale set per sequence).

    Finished sequences are skipped and an interrupted one resumes at its next window.
    """
    if seqs == ["all"]:
        seqs = pw3d.sequences()

    pool = ThreadPoolExecutor(max_workers=2)  # frame decoding overlaps the GPU work
    for seq in seqs:
        paths = pw3d.frames(seq)[: frames or None]
        out = ROOT / name / seq
        info_file, progress_file = out / "meta.json", out / "progress.json"
        total, chunks = plan_chunks(len(paths))
        if not chunks:
            print(f"{name}/{seq}: {len(paths)} frames, fewer than a window; skipped")
            continue
        if info_file.exists() and json.loads(info_file.read_text())["n_frames"] == len(paths):
            print(f"{name}/{seq}: done; skipped")
            continue

        out.mkdir(parents=True, exist_ok=True)
        size = target_size(paths[0])
        done = 0
        if progress_file.exists():
            progress = json.loads(progress_file.read_text())
            done = progress["done"] if progress["n_frames"] == len(paths) else 0

        def submit(i, paths=paths, chunks=chunks, size=size):
            return pool.submit(load_frames, paths[chunks[i][0] : chunks[i][0] + WINDOW], size)

        pending = {i: submit(i) for i in range(done, min(done + 2, len(chunks)))}
        arrays, timings = {}, []

        for i in range(done, len(chunks)):
            start, first, end = chunks[i]
            pixels = pending.pop(i).result()
            if i + 2 < len(chunks):
                pending[i + 2] = submit(i + 2)

            torch.cuda.synchronize()
            t0 = time.time()
            feats = extract_window(pixels)
            feats = {k: to_fp16(k, v) for k, v in feats.items()}
            torch.cuda.synchronize()
            timings.append(time.time() - t0)

            for k, v in feats.items():
                if k not in arrays:
                    file = out / f"layer_{k:02d}.npy"
                    if done:
                        arrays[k] = np.load(file, mmap_mode="r+")
                    else:
                        shape = (total, *v.shape[1:])
                        arrays[k] = np.lib.format.open_memmap(file, "w+", np.float16, shape)
                lo = first - start // 4
                arrays[k][first:end] = v[lo : lo + end - first].numpy()
                arrays[k].flush()
            progress_file.write_text(json.dumps({"n_frames": len(paths), "done": i + 1}))

        info = {
            **meta,
            "seq": seq,
            "n_frames": len(paths),
            "n_latents": total,
            "latent_frames": "latent 0: frame 0; latent g >= 1: frames 4g-3 .. 4g",
            "frames_without_latent": len(paths) - (4 * (total - 1) + 1),
            "window": WINDOW,
            "stride": STRIDE,
            "chunks": chunks,
            "size_wh": size,
            "shape": {f"layer_{k:02d}": list(a.shape) for k, a in arrays.items()},
            "dtype": "int8",
            "sec_per_window": timings,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        for k, array in arrays.items():  # the float16 file holds the windows until here
            save_int8(out, k, array)
            (out / f"layer_{k:02d}.npy").unlink()
        info_file.write_text(json.dumps(info, indent=1))
        progress_file.unlink()
        print(
            f"{name}/{seq}: {len(chunks)} windows, {np.mean(timings):.1f} s each, "
            f"peak {info['peak_gpu_gb']:.1f} GB",
            flush=True,
        )
