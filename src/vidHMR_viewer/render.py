"""2D drawing of annotations on frames."""

import cv2
import numpy as np

from vidHMR import geometry
from vidHMR_viewer.skeletons import ACTOR_COLORS, COCO18_BONES, SMPL24_BONES

INVALID_COLOR = (150, 150, 150)


def resize_to_height(img: np.ndarray, height: int) -> tuple[np.ndarray, float]:
    s = height / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), height), interpolation=cv2.INTER_AREA), s


def draw_skeleton(img, uv, bones, color, conf=None, thick=2, min_conf=0.1):
    """Draw joints `uv` (J x 2, pixels) with `bones`. Joints with conf < min_conf are skipped."""
    ok = np.ones(len(uv), bool) if conf is None else conf >= min_conf
    ok &= np.isfinite(uv).all(1)
    pts = np.round(uv).astype(int)
    for a, b in bones:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thick, cv2.LINE_AA)
    for j in np.flatnonzero(ok):
        cv2.circle(img, tuple(pts[j]), thick + 1, color, -1, cv2.LINE_AA)


def draw_label(img, text, org, color):
    x, y = org
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(
        img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )


def draw_mask_frame(seq, frame: int, height: int = 960) -> np.ndarray:
    """RGB frame with the GT SMPL person masks: the visible part filled, the occluded part
    (amodal minus visible) tinted, and the amodal silhouette outlined. One color per actor."""
    from vidHMR.masks import load_masks

    img = cv2.cvtColor(cv2.imread(str(seq.image_path(frame))), cv2.COLOR_BGR2RGB)
    img, _ = resize_to_height(img, height)
    size = (img.shape[1], img.shape[0])
    amodal, visible = load_masks(seq.name, frame, seq.n_actors)
    thick = max(1, round(height / 400))

    out, outlines = img.astype(np.float32), []
    for a in range(seq.n_actors):
        color = ACTOR_COLORS[a % len(ACTOR_COLORS)] if seq.valid[a, frame] else INVALID_COLOR
        am = cv2.resize(amodal[a].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        vi = cv2.resize(visible[a].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        hidden = (am > 0) & ~vi
        out[vi] = 0.4 * out[vi] + 0.6 * np.array(color)
        out[hidden] = 0.75 * out[hidden] + 0.25 * np.array(color)
        outlines.append((am, color))

    out = out.astype(np.uint8)
    for am, color in outlines:
        cnt, _ = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, color, thick, cv2.LINE_AA)
    return out


def draw_pw3d_frame(
    seq,
    frame: int,
    height: int = 960,
    show_2d=True,
    show_3d=True,
    show_bbox=True,
    show_labels=True,
) -> np.ndarray:
    """Return an RGB frame of a 3DPW sequence with its annotations drawn on it."""
    img = cv2.cvtColor(cv2.imread(str(seq.image_path(frame))), cv2.COLOR_BGR2RGB)
    img, s = resize_to_height(img, height)
    thick = max(1, round(height / 400))
    for a in range(seq.n_actors):
        valid = seq.valid[a, frame]
        color = ACTOR_COLORS[a % len(ACTOR_COLORS)] if valid else INVALID_COLOR
        uv3, z = geometry.project(seq.joints3d[a, frame], seq.cam_poses[frame], seq.K)
        uv3 = uv3 * s
        in_front = bool((z > 0).all()) and np.isfinite(uv3).all()
        if show_3d and in_front:
            draw_skeleton(img, uv3, SMPL24_BONES, color, thick=thick)
        p2 = seq.poses2d[a, frame]
        conf = p2[2]
        uv2 = p2[:2].T * s
        if show_2d and conf.max() > 0:
            draw_skeleton(
                img, uv2, COCO18_BONES, (255, 235, 59), conf=conf, thick=max(1, thick - 1)
            )
        if (show_bbox or show_labels) and in_front:  # no box for an actor behind the camera
            x0, y0 = np.clip(uv3.min(0) - 20 * s, -1e4, 1e4)
            x1, y1 = np.clip(uv3.max(0) + 20 * s, -1e4, 1e4)
            if show_bbox:
                cv2.rectangle(
                    img, (int(x0), int(y0)), (int(x1), int(y1)), color, thick, cv2.LINE_AA
                )
            if show_labels:
                tag = f"actor {a}" + ("" if valid else " (invalid)")
                draw_label(img, tag, (int(max(x0, 0)), int(max(y0, 18))), color)
    return img
