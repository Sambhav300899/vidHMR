"""Video writing."""

from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_video(
    frames: Iterable[np.ndarray],
    path: Path,
    fps: int = 30,
    crf: int = 23,
    threads: int = 0,
    gop: int | None = None,
) -> Path:
    """Write RGB frames to an H.264 mp4 (sizes are cropped to even numbers). Higher crf = smaller
    file.
    threads=0 lets x264 pick; use a small number when many encoders run in parallel.
    gop: maximum frames between keyframes (x264 default 250). A short GOP makes browser seeks
    fast."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    params = ["-crf", str(crf), "-movflags", "+faststart", "-threads", str(threads)] + (
        ["-g", str(gop)] if gop else []
    )
    with imageio.get_writer(
        tmp,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=params,
    ) as w:
        for f in frames:
            w.append_data(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
    tmp.replace(path)
    return path
