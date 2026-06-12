#!/usr/bin/env python
"""SAM3 **tracker** single-frame (box-prompt PVS) parity oracle — epic 4910, sc-4924 (Phase F1).

Drives the public `transformers` SAM3 tracker (`facebook/sam3` → `Sam3VideoModel.tracker_model`,
a `Sam3TrackerVideoModel`) on one image + one box, replicating the reference's init-conditioning
single-frame path (`_run_single_frame_inference` for `is_init_cond_frame=True`, no memory):

    feats = tracker.get_image_features(pixel_values)        # [s0(288²,32), s1(144²,64), pix(72²,256)]
    high_res = feats[:-1] -> BCHW                            # the conv_s0/conv_s1-projected high-res maps
    pix = (feats[-1] + no_memory_embedding) -> BCHW          # 72² image embedding + no-memory bias
    image_pe = tracker.get_image_wide_positional_embeddings()
    sparse, dense = tracker.prompt_encoder(input_boxes=box)  # _embed_boxes (corners labels 2/3 + pad)
    masks, iou, _, obj = tracker.mask_decoder(pix, image_pe, sparse, dense, multimask=True, high_res)

Dumps the final best (argmax-IoU) low-res mask + iou + object-score, plus the staged inputs, as the
parity fixtures the Rust `Sam3Tracker` validates against. No MLX here.

Run:  /tmp/sam3ref/.venv/bin/python dump_tracker_fixture.py
"""

import hashlib
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3Processor, Sam3VideoModel

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

# zidane (two people) — same image the SAM2 spike + detector oracle use. Box is in the 1008-input
# space (what the prompt encoder consumes); fixed so Rust + oracle use an identical prompt.
URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
BOX_1008 = [430.0, 90.0, 700.0, 980.0]  # a tall person region in 1008² space


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    tracker = model.tracker_model
    processor = Sam3Processor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    W, H = image.size
    pixel_values = processor(images=image, text="person", return_tensors="pt")["pixel_values"]
    print(f"  image {W}x{H} -> pixel_values {list(pixel_values.shape)}")

    with torch.no_grad():
        # The tracker has no vision encoder of its own (`remove_vision_encoder=True`): the PE backbone
        # is shared from the detector. Mirror `get_vision_features_for_tracker` → conv_s0/s1-projected
        # high-res maps + the 72² image embedding, all flattened HWxBxC.
        vision_embeds = model.detector_model.vision_encoder(pixel_values)
        feats, _pos = model.get_vision_features_for_tracker(vision_embeds)  # [s0, s1, pix]
        sizes = tracker.backbone_feature_sizes  # [[288,288],[144,144],[72,72]]
        high_res = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(feats[:-1], sizes[:-1])
        ]
        B, C = feats[-1].size(1), feats[-1].size(2)
        h, w = sizes[-1]
        pix = (feats[-1] + tracker.no_memory_embedding).permute(1, 2, 0).view(B, C, h, w)
        image_pe = tracker.get_image_wide_positional_embeddings()

        box = torch.tensor(BOX_1008, dtype=torch.float32).view(1, 1, 4)
        sparse, dense = tracker.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=box, input_masks=None
        )
        masks, iou, _sam_tokens, obj = tracker.mask_decoder(
            image_embeddings=pix,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_resolution_features=high_res,
        )

    # masks: [B, point_batch, 3, mg, mg]; iou: [B, point_batch, 3]; obj: [B, point_batch, 1]
    iou_flat = iou.reshape(-1)
    best = int(torch.argmax(iou_flat).item())
    best_mask = masks.reshape(masks.shape[-3], masks.shape[-2], masks.shape[-1])[best]
    print(f"  masks {list(masks.shape)}  iou={[round(x,4) for x in iou_flat.tolist()]}  "
          f"best={best}  obj_score={float(obj.reshape(-1)[0]):.4f}")

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "image_size_wh": [W, H],
        "box_1008": BOX_1008,
        "multimask_output": True,
        "best_index": best,
        "object_score": float(obj.reshape(-1)[0]),
        "iou_scores": iou_flat.tolist(),
        "stages": {
            "pix_feat": stats(pix),
            "image_pe": stats(image_pe),
            "sparse": stats(sparse),
            "dense": stats(dense),
            "high_res_s0": stats(high_res[0]),
            "high_res_s1": stats(high_res[1]),
            "masks": stats(masks),
            "best_low_res": stats(best_mask),
        },
    }
    with open(os.path.join(OUT, "tracker_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    npy = lambda t: t.detach().cpu().float().numpy()
    np.savez_compressed(
        os.path.join(OUT, "tracker_fixture_zidane.npz"),
        pixel_values=npy(pixel_values),
        box_1008=np.array(BOX_1008, dtype=np.float32),
        pix_feat=npy(pix),
        image_pe=npy(image_pe),
        sparse=npy(sparse),
        dense=npy(dense),
        high_res_s0=npy(high_res[0]),
        high_res_s1=npy(high_res[1]),
        masks=npy(masks),
        iou_scores=npy(iou_flat),
        object_score=npy(obj.reshape(-1)),
        best_low_res=npy(best_mask),
        best_index=np.array([best], dtype=np.int64),
    )
    # safetensors fixture for the Rust parity harness (Weights::from_file).
    det = lambda t: t.detach().cpu().float().contiguous().clone()
    save_file(
        {
            "pixel_values": det(pixel_values),
            "box_1008": torch.tensor(BOX_1008, dtype=torch.float32),
            "pix_feat": det(pix),
            "image_pe": det(image_pe),
            "sparse": det(sparse),
            "dense": det(dense),
            "high_res_s0": det(high_res[0]),
            "high_res_s1": det(high_res[1]),
            "masks": det(masks),
            "iou_scores": det(iou_flat),
            "object_score": det(obj.reshape(-1)),
            "best_low_res": det(best_mask),
        },
        os.path.join(OUT, "tracker_fixture.safetensors"),
    )
    print("wrote tracker_fixture_manifest.json + .npz + .safetensors to", OUT)


if __name__ == "__main__":
    main()
