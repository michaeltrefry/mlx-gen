#!/usr/bin/env python3
"""Golden dump for the native 5-point alignment / norm_crop (sc-3083).

Runs the authoritative insightface antelopev2 `FaceAnalysis` on `t1.jpg` and records, per face:
  - kps (5×2, original-image coords) and bbox,
  - `estimate_norm` M (2×3, the skimage SimilarityTransform / Umeyama fit),
  - the `norm_crop` 112² RGB crop (cv2.warpAffine, borderValue=0) — the exact ArcFace input,
  - the authoritative 512-d `embedding` (insightface `app.get`),
  - a 512² facexlib-template crop (SimilarityTransform → cv2.warpAffine, gray border) for the
    EVA-CLIP/parsing path (the "swappable" crop — uses the SAME SCRFD kps, per the sc-3080 spike).

Also dumps the original RGB image and the SCRFD 640 blob + det_scale so the Rust test can close the
full **detect → align → embed** loop with the native SCRFD too.

The Rust port warps the RGB image directly (swapRB folded in), so all crops here are dumped RGB.

Run with the dwpose-spike venv (insightface + onnxruntime + cv2 + skimage + numpy + safetensors):
  ~/.dwpose-spike/venv/bin/python tools/dump_face_align_golden.py
Outputs tools/golden/face_align_goldens.safetensors (gitignored).
"""
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "golden")
DET_SIZE = 640

# facexlib FaceRestoreHelper FFHQ template (face_size=512, crop_ratio=(1,1)); RGB gray border.
FACEXLIB_DST_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float64)
FACEXLIB_BORDER_RGB = (132.0, 133.0, 135.0)  # cv2 (135,133,132) BGR


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    import cv2
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from skimage.transform import SimilarityTransform

    img_path = os.path.join(os.path.dirname(insightface.__file__), "data", "images", "t1.jpg")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    def i32(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.int32))

    def f32(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

    # --- SCRFD 640 blob (insightface-identical, matches convert_scrfd.py) for the e2e leg.
    im_ratio = float(h) / w
    if im_ratio > 1.0:
        new_h, new_w = DET_SIZE, int(DET_SIZE / im_ratio)
    else:
        new_w, new_h = DET_SIZE, int(DET_SIZE * im_ratio)
    det_scale = float(new_h) / h
    det_img = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = cv2.resize(img_bgr, (new_w, new_h))
    blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True)

    # --- authoritative antelopev2 FaceAnalysis (det + 5kps + glintr100 embedding)
    antelope_root = os.environ.get("PULID_FLUX_INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
    app = FaceAnalysis(name="antelopev2", root=antelope_root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
    rec_sess = app.models["recognition"].session
    rec_in = rec_sess.get_inputs()[0].name

    def onnx_emb_rgb(crop_rgb_u8):
        # exactly the Rust path: (rgb-127.5)/127.5, NHWC->NCHW (R,G,B), raw glintr100 session.
        x = (crop_rgb_u8.astype(np.float32) - 127.5) / 127.5
        nchw = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])
        return rec_sess.run(None, {rec_in: nchw})[0].flatten()

    faces = app.get(img_bgr)
    # deterministic order: largest face first (PuLID uses the max face; we keep all for coverage)
    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    print(f"FaceAnalysis detected {len(faces)} faces")

    out = {
        "image": i32(img_rgb),  # [H,W,3] RGB
        "blob": f32(np.transpose(blob, (0, 2, 3, 1))),  # [1,640,640,3] NHWC for the Rust SCRFD
        "det_scale": f32(np.asarray(det_scale).reshape(())),
        "n_faces": i32(np.asarray(len(faces)).reshape(())),
    }
    for idx, face in enumerate(faces):
        kps = face.kps.astype(np.float64)  # (5,2) original coords
        out[f"kps.{idx}"] = f32(kps)
        out[f"bbox.{idx}"] = f32(face.bbox)
        out[f"embedding.{idx}"] = f32(face.embedding)

        # insightface norm_crop (112²) — warp the RGB image so the crop is RGB.
        m112 = face_align.estimate_norm(face.kps, image_size=112)  # 2×3
        out[f"M.{idx}"] = f32(m112)
        crop112 = cv2.warpAffine(img_rgb, m112, (112, 112), borderValue=0.0)
        out[f"norm_crop.{idx}"] = i32(crop112)
        # onnx embedding on THIS exact RGB crop via the Rust-equivalent normalization (isolates the
        # forward from insightface's app path; == app embedding up to f32).
        out[f"emb_onnx.{idx}"] = f32(onnx_emb_rgb(crop112))

        # facexlib-template 512² crop using the SAME kps (deterministic SimilarityTransform == the
        # LMEDS partial-affine fit for clean 5-pt) — the tolerant EVA-CLIP/parsing crop.
        t = SimilarityTransform()
        t.estimate(kps, FACEXLIB_DST_512)
        m512 = t.params[0:2, :]
        out[f"M512.{idx}"] = f32(m512)
        crop512 = cv2.warpAffine(img_rgb, m512, (512, 512), borderValue=FACEXLIB_BORDER_RGB)
        out[f"align512.{idx}"] = i32(crop512)

    from safetensors.numpy import save_file
    gpath = os.path.join(OUT_DIR, "face_align_goldens.safetensors")
    save_file(out, gpath)
    print("wrote", gpath, f"({len(out)} tensors, image {h}x{w})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
