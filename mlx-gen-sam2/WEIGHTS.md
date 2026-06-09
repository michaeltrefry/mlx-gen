# `mlx-gen-sam2` weights — conversion, hosting, provisioning (sc-3707)

Zero Python in the runtime or weight path: the Rust model loads a converted MLX `.safetensors`
directly. This doc records how that artifact is produced, hosted, and provisioned.

## Conversion (we own it, from the canonical Meta weights)

`tools/convert_sam2_to_mlx.py` converts an official Meta SAM2.1 `.pt` → the MLX layout this crate
loads. It is a self-contained vendor of `avbiswas/sam2-mlx`'s converter (Apache-2.0), so the output
is **bit-identical** to `avbiswas/sam2.1-hiera-large-mlx` (verified: max |Δ| = 0 over all 900
tensors) — but produced from the canonical `facebook/sam2.1-hiera-large` source, so we don't depend
on a third party's converted upload.

```
~/mlx-flux-venv/bin/python tools/convert_sam2_to_mlx.py \
    --hf-id facebook/sam2.1-hiera-large \
    --output sam2.1_hiera_large.safetensors
```

The mapping: Torch `Conv2d` OIHW → MLX OHWI; `ConvTranspose2d` IOHW → OHWI; the learned `pos_embed`
is bicubic-interpolated to 256² and fused with the window pos-embed into `trunk.pos_embed_full`.
Output is a full segmenter (`trunk.` / `neck.` / `sam_prompt_encoder.` / `sam_mask_decoder.` /
`memory_*` / obj-ptr), f32.

| size | Meta source | output sha256 |
|------|-------------|---------------|
| large (default) | `facebook/sam2.1-hiera-large` (`sam2.1_hiera_large.pt`, sha `2647878d…dd318`) | `bbbd94abd316a0867d906c6cdf2d51c780c3fd3e804ab47bdcdc9b29763628e1` |

Default = **large** (matches the Python `sam2_hiera_large.pt` baseline); base-plus is the speed
option (spike sc-3635: large ≈ baseline quality, base-plus ≈ 2× faster at IoU 0.977 vs large) —
convert with `--hf-id facebook/sam2.1-hiera-base-plus`.

## Hosting

Mirror under **`SceneWorks/sam2-mlx`** (the `SceneWorks/real-esrgan-onnx` convention from sc-3489),
with a README recording the provenance table above. Upload (run by a maintainer with the SceneWorks
HF token — this publishes a public artifact):

```
hf upload SceneWorks/sam2-mlx sam2.1_hiera_large.safetensors sam2.1_hiera_large.safetensors --repo-type model
```

## Engine load contract

The crate is provisioning-agnostic — it loads a path:

```rust
let w = mlx_gen::weights::Weights::from_file(path)?;          // converted .safetensors
let seg = mlx_gen_sam2::Sam2Segmenter::from_weights(&w, &Sam2ImageEncoderConfig::large())?;
let mask = seg.segment(rgb, h, w, [x1, y1, x2, y2])?;        // binary L mask
```

`tests/weights_load.rs` (`#[ignore]`) exercises exactly this against a real converted checkpoint
via `SCENEWORKS_SAM2_WEIGHTS=<path>`.

## SceneWorks-side provisioning (implemented with the worker wiring, sc-3709)

The download-on-first-use provisioning lives in the SceneWorks worker (where `pose_jobs.rs` /
`upscale_jobs.rs` `ensure_weights` live), not in this engine crate:

- env pin `SCENEWORKS_SAM2_MODEL` (default `SceneWorks/sam2-mlx` / `sam2.1_hiera_large.safetensors`),
- cache at `<data_dir>/cache/sam2/`,
- download-on-first-use + **sha256 verification** against the provenance table,
- then `Weights::from_file(cached_path)` → `Sam2Segmenter::from_weights`.
