"""The vidHMR dashboard: 3DPW sequence tables and an embedded Rerun web viewer.

    python -m vidHMR_viewer.app [--port 7860]

The page (web/index.html) gets its tables from /api/* as JSON. A click on a sequence opens its
recording (vidHMR_viewer.recording, pre-built) in the Rerun web viewer, which the server hosts
from generated/cache (fetched once from the npm registry).
"""

import argparse
import hashlib
import io
import tarfile
import time
import urllib.request
from functools import cache
from pathlib import Path

import rerun
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from vidHMR import paths
from vidHMR.datasets import pw3d
from vidHMR.eval import sam3d_pw3d
from vidHMR_viewer.recording import RECORDING_FORMAT, recording_path

WEB = Path(__file__).parent / "web"
RERUN_VERSION = rerun.__version__
RERUN_FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm")
IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}
NO_CACHE = {"Cache-Control": "no-cache"}

SUBSETS = [
    ("all", "All", "All annotated sequences."),
    ("train", "Train", "Official training split."),
    ("validation", "Validation", "Official validation split."),
    ("test", "Test", "Official test split: the one that papers report 3DPW results on."),
    (
        "occ",
        "3DPW-OCC",
        "Occlusion subset (OOH / ROMP / PARE). Includes train, validation and test sequences.",
    ),
    (
        "pc",
        "3DPW-PC",
        "Person-crowd subset from ROMP: fixed frame ranges where people overlap. The timeline "
        "selects the range.",
    ),
]
SAM3D_CAUTION = (
    "The SAM 3D Body training data includes 3DPW train (its dataset has a 3dpw_train split). "
    "Only the validation and test rows measure generalization."
)
SAM3D_PROTOCOL = (
    "Protocol: 12 joints shared by SMPL and MHR70, camera frame, valid frames, GT intrinsics, one "
    "GT box per actor. MPJPE is pelvis-aligned; PA-MPJPE is after Procrustes. Root error is a "
    "median over frames. Not the official 14-joint protocol: compare sequences with each other, "
    "not with papers."
)


# -------------------------------------------------------------------- the Rerun web viewer
def rerun_viewer_dir() -> Path:
    """The web viewer files of the installed rerun-sdk version (fetched once)."""
    d = paths.GENERATED / "cache" / "rerun_web_viewer" / RERUN_VERSION
    if all((d / f).exists() for f in RERUN_FILES):
        return d
    url = f"https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-{RERUN_VERSION}.tgz"
    with urllib.request.urlopen(url) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tar:
        d.mkdir(parents=True, exist_ok=True)
        for f in RERUN_FILES:
            (d / f).write_bytes(tar.extractfile(f"package/{f}").read())
    return d


def web_version() -> str:
    """A hash of the web/ files: in the asset URLs, so that an edit gives new URLs."""
    h = hashlib.sha1()
    for f in sorted(WEB.rglob("*")):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.relative_to(WEB)}:{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()[:12]


# -------------------------------------------------------------------- table data
def sequence_rows() -> list[dict]:
    rows = []
    for r in pw3d.index():
        rec = recording_path(r["name"])
        rows.append(
            dict(
                name=r["name"],
                split=r["split"],
                scene=r["scene"],
                frames=r["frames"],
                seconds=round(r["frames"] / pw3d.FPS, 1),
                genders=r["genders"],
                valid=round(100 * r["valid_pf"] / r["all_pf"], 1),
                subsets=pw3d.subsets_of(r),
                pc_range=r["pc_range"],
                recording=(
                    f"/recordings/{r['name']}.rrd?v={RECORDING_FORMAT}-{int(rec.stat().st_mtime)}"
                    if rec.exists()
                    else None
                ),
                recording_mb=round(rec.stat().st_size / 1e6) if rec.exists() else None,
            )
        )
    return rows


@cache
def worst_frame(name: str) -> int | None:
    """The frame with the largest SAM 3D Body MPJPE over the actors."""
    e = sam3d_pw3d.per_frame(name)
    if not e:
        return None
    frame, _ = max(
        ((f, m) for a in e.values() for f, m in zip(a["frame"], a["mpjpe"], strict=True)),
        key=lambda fm: fm[1],
    )
    return int(frame)


def sam3d_results() -> dict | None:
    s = sam3d_pw3d.load_summary()
    if not s:
        return None
    seqs = [
        dict(
            name=r["name"],
            split=r["split"],
            actors=r["actors"],
            scored=r["scored"],
            valid_gt=r["valid_gt"],
            mpjpe=r["mpjpe"],
            pa_mpjpe=r["pa_mpjpe"],
            root=r["root"],
            excluded=r["excluded"] or "",
            worst_frame=worst_frame(r["name"]),
        )
        for r in s["sequences"]
    ]
    return dict(
        totals=s["totals"],
        joints=s["joints"],
        sequences=seqs,
        caution=SAM3D_CAUTION,
        protocol=SAM3D_PROTOCOL,
    )


def dataset_summary() -> dict:
    return pw3d.summary(pw3d.index())


# -------------------------------------------------------------------- the app
def build_app() -> FastAPI:
    app = FastAPI(title="vidHMR dashboard")
    viewer_dir = rerun_viewer_dir()
    web_root = WEB.resolve()

    @app.get("/", include_in_schema=False)
    def index():
        html = (WEB / "index.html").read_text()
        html = html.replace("{{WEB}}", f"/web/{web_version()}").replace(
            "{{RERUN}}", f"/rerun/{RERUN_VERSION}"
        )
        return HTMLResponse(html, headers=NO_CACHE)

    @app.get("/web/{version}/{file:path}", include_in_schema=False)
    def web_file(version: str, file: str):
        path = (web_root / file).resolve()
        if not path.is_relative_to(web_root) or not path.is_file():
            raise HTTPException(404, f"no file {file}")
        return FileResponse(path, headers=IMMUTABLE)

    @app.get("/rerun/{version}/{file}", include_in_schema=False)
    def rerun_file(version: str, file: str):
        if version != RERUN_VERSION or file not in RERUN_FILES:
            raise HTTPException(404, f"no file {file}")
        media = "application/wasm" if file.endswith(".wasm") else "text/javascript"
        return FileResponse(viewer_dir / file, media_type=media, headers=IMMUTABLE)

    @app.get("/recordings/{name}.rrd", include_in_schema=False)
    def recording(name: str):
        if name not in {r["name"] for r in pw3d.index()}:
            raise HTTPException(404, f"unknown sequence {name}")
        path = recording_path(name)
        if not path.exists():
            raise HTTPException(
                404, f"not built yet: run python -m vidHMR_viewer.recording --only {name}"
            )
        # The URL carries ?v=<format>-<mtime>: a new file gets a new URL.
        return FileResponse(path, media_type="application/octet-stream", headers=IMMUTABLE)

    client_log = paths.generated("logs", "client.log")

    @app.post("/api/client-log", include_in_schema=False)
    async def client_report(request: Request):
        """Diagnostics from the page (web/app.js): one JSON line per report."""
        body = (await request.body())[:4000].decode(errors="replace")
        with client_log.open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {body}\n")
        return Response(status_code=204)

    @app.get("/api/sequences")
    def api_sequences():
        return dict(rows=sequence_rows(), subsets=SUBSETS)

    @app.get("/api/sam3d")
    def api_sam3d():
        return sam3d_results()

    @app.get("/api/summary")
    def api_summary():
        return dataset_summary()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
