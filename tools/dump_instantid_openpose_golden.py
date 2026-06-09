#!/usr/bin/env python
"""Golden dump for the native OpenPose body-skeleton rasterizer (sc-3379).

`draw_bodypose` (worker `apps/worker/scene_worker/openpose_skeleton.py`) renders the COCO-18 body
skeleton control image the xinsir OpenPose-SDXL ControlNet was trained on (the controlnet_aux
`draw_bodypose` format). The Rust port (`mlx-gen-instantid/src/openpose.rs`) must bit-match OpenCV's
integer rasterization, so this dumps several (canvas, keypoints) cases → the exact RGB8 output.

The `square_fit` + `draw_bodypose` bodies below are copied **verbatim** from the worker source (only
cv2/numpy/math), so the golden is exactly what production renders. Pinned OpenCV: 4.13.0.

Cases cover: a real gallery pose on a square 1024² canvas (the production pose size); the same pose on
a non-square canvas (exercises the centered-square letterbox); a pose with occluded joints (None →
skipped limbs); and a tiny 128² canvas for easy pixel debugging.

Run from a cv2 venv (cv2 4.13 + numpy + safetensors):
    ~/repos/mflux/.venv-0312/bin/python ~/Repos/mlx-gen/tools/dump_instantid_openpose_golden.py
"""
import math
from pathlib import Path

import cv2
import numpy as np
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parent / "golden" / "instantid_openpose_golden.safetensors"

# ---- verbatim from apps/worker/scene_worker/openpose_skeleton.py ----
LIMB_SEQ = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)


def square_fit(canvas_w, canvas_h):
    side = min(canvas_w, canvas_h)
    return side, (canvas_w - side) // 2, (canvas_h - side) // 2


def draw_bodypose(canvas_w, canvas_h, keypoints, stickwidth=4):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    side, ox, oy = square_fit(canvas_w, canvas_h)
    pts = [None if p is None else (ox + float(p[0]) * side, oy + float(p[1]) * side) for p in keypoints]

    for i, (a, b) in enumerate(LIMB_SEQ):
        if a >= len(pts) or b >= len(pts) or pts[a] is None or pts[b] is None:
            continue
        xa, ya = pts[a]
        xb, yb = pts[b]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLORS[i])

    for i in range(min(18, len(pts))):
        if pts[i] is None:
            continue
        x, y = pts[i]
        cv2.circle(canvas, (int(x), int(y)), stickwidth, COLORS[i], thickness=-1)
    return canvas


# A real gallery pose (apps/web/public/poses/index.json :: dance_01), COCO-18 normalized.
DANCE_01 = [
    (0.5429, 0.1454), (0.515, 0.2608), (0.4469, 0.263), (0.3465, 0.3332), (0.2275, 0.4021),
    (0.5831, 0.2587), (0.6376, 0.3433), (0.6275, 0.4365), (0.4841, 0.4852), (0.553, 0.6616),
    (0.5859, 0.8867), (0.553, 0.4895), (0.4454, 0.6917), (0.3623, 0.8465), (0.5243, 0.1354),
    (0.5386, 0.134), (0.4784, 0.1569), (0.52, 0.1512),
]

# The same pose with the head occluded (nose/eyes/ears None) — exercises the limb-skip path.
OCCLUDED = [None if i in (0, 14, 15, 16, 17) else p for i, p in enumerate(DANCE_01)]


def flat_kps(keypoints):
    """Encode a keypoint list as a float32 [18, 3] array: (x, y, present) with present=0 for None."""
    rows = []
    for p in keypoints:
        if p is None:
            rows.append((0.0, 0.0, 0.0))
        else:
            rows.append((float(p[0]), float(p[1]), 1.0))
    return np.asarray(rows, dtype=np.float32)


CASES = {
    "square_1024": (1024, 1024, DANCE_01),
    "nonsquare_768x1024": (768, 1024, DANCE_01),
    "occluded_head_1024": (1024, 1024, OCCLUDED),
    "tiny_128": (128, 128, DANCE_01),
}


def main():
    tensors = {}
    for name, (w, h, kps) in CASES.items():
        img = draw_bodypose(w, h, kps)
        assert img.shape == (h, w, 3), img.shape
        tensors[f"{name}_wh"] = np.asarray([w, h], dtype=np.int32)
        tensors[f"{name}_kps"] = flat_kps(kps)
        tensors[f"{name}_img"] = img.astype(np.uint8)
        print(f"{name}: {w}x{h}, nonzero px = {int((img.any(axis=2)).sum())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
