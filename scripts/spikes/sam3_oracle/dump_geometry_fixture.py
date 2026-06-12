#!/usr/bin/env python
"""SAM3 geometry/exemplar-encoder (PVS box-prompt path) fixtures for the mlx-gen-sam3 SAM3-E parity
test (sc-4923).

Two fixtures from one zidane + "person" + box-prompt run:
  * geometry_fixture.safetensors — the exact inputs the `Sam3GeometryEncoder` receives (captured via
    a forward hook: normalized cxcywh boxes, labels, the 72² FPN feature + its sine pos embed) and
    its output prompt tokens. Isolates geometry.rs from vision/text parity.
  * geometry_e2e_fixture.safetensors — pixel_values + input_ids + attention_mask + the model-space
    input_boxes/labels, and the post-processed box-prompted instance masks/scores/boxes, for the
    end-to-end PVS path.

    /tmp/sam3ref/.venv/bin/python dump_geometry_fixture.py
"""
import os
import urllib.request
from io import BytesIO

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3Model, Sam3Processor

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"

model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
img = Image.open(BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert("RGB")

# Two box prompts (xyxy pixels) around the two people in zidane.jpg (≈1280×720).
input_boxes = [[[744.0, 42.0, 1144.0, 712.0], [120.0, 200.0, 720.0, 712.0]]]
input_boxes_labels = [[1, 1]]
inputs = processor(
    images=img,
    text="person",
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    return_tensors="pt",
)

# Capture the geometry encoder's exact inputs + output.
cap = {}


def hook(_module, _args, kwargs, output):
    cap["box_embeddings"] = kwargs["box_embeddings"].detach().clone()
    cap["box_labels"] = kwargs["box_labels"].detach().clone()
    cap["img_feats_72"] = kwargs["img_feats"][-1].detach().clone()  # [1,256,72,72] NCHW
    cap["img_pos_72"] = kwargs["img_pos_embeds"][-1].detach().clone()  # [1,256,72,72] NCHW
    cap["geo_output"] = output.last_hidden_state.detach().clone()  # [1,N+1,256]


h = model.geometry_encoder.register_forward_hook(hook, with_kwargs=True)
with torch.no_grad():
    out = model(**inputs)
h.remove()

n = cap["box_embeddings"].shape[1]
print("boxes", tuple(cap["box_embeddings"].shape), "geo_output", tuple(cap["geo_output"].shape))
print("box_embeddings (cxcywh):", cap["box_embeddings"][0].tolist())

save_file(
    {
        "box_embeddings": cap["box_embeddings"].contiguous(),  # [1,N,4] cxcywh∈[0,1]
        "box_labels": cap["box_labels"].to(torch.int32).contiguous(),  # [1,N]
        "fpn_72": cap["img_feats_72"].contiguous(),  # [1,256,72,72] NCHW
        "vision_pos_72": cap["img_pos_72"].contiguous(),  # [1,256,72,72] NCHW
        "geo_output": cap["geo_output"].contiguous(),  # [1,N+1,256]
    },
    os.path.join(OUT, "geometry_fixture.safetensors"),
)
print("wrote geometry_fixture.safetensors")

# End-to-end box-prompted post-process (native 288² masks).
res = processor.image_processor.post_process_instance_segmentation(
    out, threshold=0.5, mask_threshold=0.5, target_sizes=None
)[0]
m = int(len(res["scores"]))
print("PVS instances", m, "scores", [round(s, 3) for s in res["scores"].tolist()])

save_file(
    {
        "pixel_values": inputs["pixel_values"].contiguous(),
        "input_ids": inputs["input_ids"].to(torch.int32).contiguous(),
        "attention_mask": inputs["attention_mask"].to(torch.int32).contiguous(),
        "input_boxes": inputs["input_boxes"].contiguous(),  # [1,N,4] cxcywh∈[0,1]
        "input_boxes_labels": inputs["input_boxes_labels"].to(torch.int32).contiguous(),  # [1,N]
        "instance_masks": res["masks"].to(torch.uint8).contiguous(),  # [m,288,288]
        "instance_scores": res["scores"].contiguous(),  # [m]
        "instance_boxes": res["boxes"].contiguous(),  # [m,4] xyxy∈[0,1]
    },
    os.path.join(OUT, "geometry_e2e_fixture.safetensors"),
)
print("wrote geometry_e2e_fixture.safetensors")
