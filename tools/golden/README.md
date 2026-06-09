# `tools/golden/` — real-weights parity goldens

This directory holds the reference tensors (`*.safetensors`) and inspection images (`*.png`) that
the **`#[ignore]`d real-weights parity tests** compare against. **Everything here except this
`README.md` and `CHECKSUMS.txt` is gitignored** — the goldens are large (the 1024² Z-Image txt2img
golden alone is ~15 MB), regenerable from the scripts below, and sensitive to the MLX version
(e.g. the 0.31.1 bump shifted VAE-decode precision), so committing them would bloat history for no
gain. They also can't be produced — or consumed — without the licensed multi-GB HuggingFace
weights and a Mac with Metal, which is exactly why the tests that read them are `#[ignore]`d.

## Fixtures vs. goldens — the convention

| | committed? | runs in default `cargo test`? | needs model weights? |
|---|---|---|---|
| **`tests/fixtures/*.safetensors`** (per crate) | yes | yes | no — synthetic / small dumped intermediates |
| **`tools/golden/*` (here)** | no (gitignored) | no (`#[ignore]`) | yes — real HF weights + Metal |

Default-running tests only depend on committed inputs; anything needing un-committable inputs is
`#[ignore]`d. So a fresh clone's `cargo test` is green without this directory.

## Regenerating

Goldens are produced by running the dump scripts **from the frozen Python `mflux` fork** (which has
the reference implementation + its `.venv`), pointed at this repo's `tools/`:

```sh
cd ~/repos/mflux
uv run python /path/to/mlx-gen/tools/dump_<name>.py        # writes into this dir
```

Each script writes into `tools/golden/` next to itself (paths are `__file__`-relative, so this
works from any checkout/worktree). Run the matching `#[ignore]`d test with, e.g.:

```sh
cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
```

Prerequisites: macOS + Metal; the frozen `mflux` fork at `~/repos/mflux`; the model weights in
`~/.cache/huggingface/hub/` (auto-downloaded by the fork on first run).

## Manifest

### Z-Image (`mlx-gen-z-image`)

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `z_image_golden.safetensors` (+ `.png`) | `dump_z_image_golden.py` | `tests/e2e_real_weights.rs` | txt2img stage + full pipeline. Env: `ZIMAGE_PROMPT/SEED/STEPS/W/H` (use `W=H=1024` for the e2e size; default 256²). Emits the **static shift=3.0** schedule (sc-2536). |
| `z_image_q8_golden.safetensors`, `z_image_q4_golden.safetensors` (+ `.png`) | `dump_z_image_golden.py` with `QUANTIZE=8`/`4` **and `ZIMAGE_W=1024 ZIMAGE_H=1024`** | `tests/e2e_real_weights.rs` (`transformer_q8/q4_pipeline_matches_fork` + `q8/q4_full_generate_renders`) | sc-2532 Q4/Q8 parity, **regenerated at 1024²** (production res; at 256² the per-pixel metric is a pessimistic artifact). `ZImage(quantize=N)` runs the fork's real whole-model quantized path; the full-`generate()` tests match it end-to-end (transformer + text encoder + VAE), the cap_feats-fed `transformer_q*` tests isolate the transformer. |
| `zq8_pack_probe.safetensors` | `dump_z_image_q8_pack_probe.py` | `tests/e2e_real_weights.rs` (`q8_packing_byte_identical_to_fork`) | sc-2532 byte-level Q8 packing proof on a **real bf16 model weight** (`layers.0.attention.to_q`) — confirms mlx-rs `mx.quantize`/`quantized_matmul` reproduce the fork's exactly. |
| `z_image_img2img_golden.safetensors` (+ `*_init.png`, `*_out.png`) | `dump_z_image_img2img_golden.py` | `tests/img2img_real_weights.rs` | img2img (`Conditioning::Reference`). Env: `ZIMAGE_PROMPT/SEED/STEPS/W/H/STRENGTH/IW/IH`. Dumps `init_image_u8` so Rust uses byte-identical pixels. |

### Qwen-Image / Qwen-Image-Edit (`mlx-gen-qwen-image`)

| golden | dump script | consumed by |
|---|---|---|
| `qwen_image_golden.safetensors`, `qwen_image_q8_golden.safetensors` | `dump_qwen_image_golden.py` (Q8 via its suffix arg) | `tests/e2e_real_weights.rs` |
| `qwen_image_edit_golden.safetensors`, `qwen_image_edit_q8_golden.safetensors` | `dump_qwen_image_edit_golden.py` | `tests/edit_real_weights.rs` |
| `qwen_text_encoder_golden.safetensors` | `dump_qwen_text_encoder_golden.py` | `tests/text_encoder_real_weights.rs` |
| `qwen_transformer_golden.safetensors` | `dump_qwen_transformer_golden.py` | `tests/transformer_real_weights.rs` |
| `qwen_vae_golden.safetensors` | `dump_qwen_vae_golden.py` | `tests/vae_real_weights.rs` |
| `qwen_vision_golden.safetensors`, `qwen_vl_encoder_golden.safetensors`, `qwen_vl_tokenize_golden.safetensors` | `dump_qwen_vision_golden.py`, `dump_qwen_vl_encoder_golden.py`, `dump_qwen_vl_tokenize_golden.py` | `tests/vision_real_weights.rs` |
| `qwen_edit_rope_golden.safetensors`, `qwen_edit_tokenize_debug.safetensors`, `qwen_edit_vision_stages_debug.safetensors` | `dump_qwen_edit_rope_golden.py`, `dump_qwen_edit_tokenize_debug.py`, `dump_qwen_edit_vision_stages_debug.py` | `tests/edit_real_weights.rs` (debug/bisection gates) |

See each script's module docstring for its exact env vars / arguments.

### FLUX.2-klein (`mlx-gen-flux2`)

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `flux2_te_real.safetensors`, `flux2_te_real_f32.safetensors` | `dump_flux2_te_real_golden.py` (`FLUX2_TE_F32=1` for the f32 ref) | `tests/te_real_weights.rs` | sc-2346 S1 Qwen3 text encoder + tokenizer. The f32 golden is the **correctness** ref (Rust runs f32 activations → peak_rel ~1e-5); the bf16 golden is the fork's production precision (the residual there is bf16-vs-f32 over 36 layers). The committed `tests/fixtures/te_golden.safetensors` (tiny synthetic) proves the encoder math on CI without weights. |
| `flux2_vae.safetensors` | `dump_flux2_vae_golden.py` | `tests/vae_real_weights.rs` | sc-2346 S2 VAE: `decode_packed_latents` (BN-denorm + 2×2 unpatchify + decode) and `encode`. f32 golden (Rust VAE runs f32) → mean_rel ~2e-3. Tensors are NCHW; the test transposes to the Rust VAE's NHWC. |
| `flux2_e2e.safetensors` | `dump_flux2_e2e_golden.py` | `tests/e2e_real_weights.rs` | sc-2346 S4 txt2img e2e (256², 4 steps, guidance 1.0), f32. Gates: seeded noise byte-match, step-0 velocity (real-weights transformer, chaos-free) mean_rel ~4e-4, full `generate()` render ~0.9% px>8 vs the fork's f32 image (the residual is the NAX-vs-wheel build delta over the sampler). |
| `flux2_edit.safetensors` | `dump_flux2_edit_golden.py` | `tests/edit_real_weights.rs` | sc-2346 S5 single-reference edit e2e (256², 4 steps), f32. Gates: reference-encoding chain (VAE-encode → patchify → BN-normalize → pack) mean_rel ~4e-4, full edit `generate()` render **0.00% px>8** vs the fork's f32 image (the dense ref conditioning makes the sampler even more stable than txt2img). Includes the 256²-resized `ref_u8` so the Rust test feeds byte-identical reference pixels. |

### SDXL acceleration samplers (`mlx-gen-sdxl`, sc-2769)

The few-step samplers (LCM / SDXL-Lightning / Hyper-SD) exist only in **diffusers**, so unlike the
other SDXL goldens (vendored Apple `mlx_sd`) these are dumped from diffusers — run the script from a
torch+diffusers venv (e.g. `/Users/michael/Repos/mflux/.venv` after `uv pip install diffusers`).

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `sdxl_accel_sched_golden.safetensors` | `dump_sdxl_accel_golden.py` (default) | `tests/accel_sampler_parity.rs` (core crate) | **Scheduler-math isolation:** per-step deterministic outputs of `LCMScheduler` / `EulerDiscreteScheduler(trailing)` / `TCDScheduler` on fixed synthetic tensors. Validates the Rust `mlx_gen::sampler` port to ~1e-6 (torch-f32 vs MLX-f32), no model needed. Small + fast. |
| `sdxl_accel_render_{ancestral,lightning,hyper,lcm}.safetensors` (+ implied `.png` via the test) | `dump_sdxl_accel_golden.py render` | `mlx-gen-sdxl/tests/accel_real_weights.rs` (`lightning_hyper_match_torch_teacher_forced`) | **Deterministic e2e:** torch initial latent + final RGB8 per variant. The Rust test teacher-forces the init latent and reports px>8 vs the torch render (a *qualitative* torch↔MLX backend gap, NOT bit-exact). Needs the full fp16 SDXL pipeline + accel LoRAs. |

### PuLID-FLUX face-identity (`mlx-gen-pulid`, epic 3069)

The reference is the **vendored torch `pulid_flux`** (SceneWorks worker `_vendor/pulid_flux/`), so
these dump from a torch venv, not `mflux`. Run from the vendored reference dir under `pulidenv`:

```sh
cd /Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor/pulid_flux
HF_HUB_OFFLINE=1 PYTHONPATH=. /private/tmp/pulidenv/bin/python /path/to/mlx-gen/tools/dump_eva_clip_golden.py
```

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `eva_clip_golden.safetensors` | `dump_eva_clip_golden.py` | `mlx-gen-pulid/tests/eva_clip_parity.rs` | **EVA02-CLIP-L-14-336 visual tower (sc-3070).** f32 reference weights + `enc_in` + 5 hidden states + `id_cond_vit`, plus the `rope.freqs_*` buffers (weight-free RoPE-construction gate) and a 512²→336² resize/normalize case (`ffi_512`/`tf_*`). Gate is cosine-primary: torch-CPU-f32 golden vs MLX-Metal-f32 has a depth-accumulating mean-rel floor (~1e-2 by block 20), but the final `id_cond_vit` re-normalizes to cos 0.999997 (bf16 0.999945). The float antialiased bicubic matches torchvision to ~1e-6. |
| `idformer_golden.safetensors` | `dump_idformer_golden.py` | `mlx-gen-pulid/tests/idformer_parity.rs` | **IDFormer perceiver-resampler (sc-3071).** f32 `pulid_encoder.*` weights (from `pulid_flux_v0.9.1.safetensors`) + deterministic `id_cond` [1,1280] + 5 EVA hidden states → `id_embedding` [1,32,2048]. cos 1.000000 / mean-rel 1.3e-3 (bf16 0.999999). |
| `pulid_ca_golden.safetensors` | `dump_pulid_ca_golden.py` | `mlx-gen-pulid/tests/pulid_ca_parity.rs` | **PerceiverAttentionCA ×20 + injection schedule (sc-3072).** f32 `pulid_ca.{0..19}.*` weights + `id_embedding` [1,32,2048] + `img` [1,64,3072] + per-module outputs at ca indices {0,9,10,19}. Driving these through the `PulidCa` injector validates the CA math (cos ~1.0) and the double→single ca_idx schedule (double i→ca[i/2], single i→ca[10+i/4]) in one shot. |
| _(reuses goldens above)_ | — | `mlx-gen-pulid/tests/pulid_flux_e2e.rs` | **PuLID-FLUX e2e (sc-3074).** No new golden: reuses `eva_clip_golden` (EVA, prefix `w`) + `scrfd_10g`/`arcface_iresnet100`/`bisenet_parsing`/`face_align_goldens` (face stack + reference face), with FLUX.1-dev from the HF cache and `pulid_flux_v0.9.1.safetensors` from `guozinan/PuLID`. Validates id_weight=0 == plain-FLUX bit-identical, id injection changes the render, and ArcFace identity cosine (0.68 @ 20-step/512²; sc-2012 baseline ≈0.80). Heavy — loads the full stack. |
### InstantID (`mlx-gen-sdxl`, epic 3109)

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `instantid_kps_golden.safetensors` | `dump_instantid_kps_golden.py` | `tests/instantid_kps.rs` (mlx-gen-instantid) | sc-3111 kps control-image renderer. cv2 ground truth (OpenCV 4.13.0) for `draw_kps` across 4 (canvas, kps) cases — square+view-angle, non-square+detected, extreme profile, tiny 64². The Rust port must bit-match OpenCV's integer rasterization (`ellipse2Poly` + `fillConvexPoly` + filled `circle`). Small (committed-size, but gitignored per the dir rule). |
| `instantid_e2e_ref.safetensors` | `dump_instantid_e2e_ref.py` | `tests/instantid_e2e.rs` (mlx-gen-instantid) | sc-3115 T2I end-to-end + identity. Raw RGB of a single large face cropped (with margin) from insightface `t1.jpg` — the reference whose ArcFace embedding + 5 kps drive generation. The test detects it (native face stack), generates 1024²/30-step, re-detects the output, and gates on **ArcFace-cosine(ref, generated) ≈ 0.82** (sc-2009 torch baseline ≈0.876; directional, not bit-exact). Also needs `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors` (epic 3079 converters: `convert_scrfd.py` / `convert_glintr100.py`), the converted `instantid/ip-adapter.safetensors`, the SDXL base snapshot, and the InstantID `ControlNetModel`. Writes `instantid_e2e_out.png` for inspection. |
| `instantid/ip-adapter.safetensors` | `convert_instantid.py` | `tests/instantid_convert_smoke.rs` | sc-3112 weight conversion. Re-serializes `ip-adapter.bin` (pickle → safetensors) bundling `image_proj.*` (Resampler) + `ip_adapter.*` (70 decoupled-cross-attn K/V pairs), mirroring the h94 IP-Adapter namespace. The IdentityNet `ControlNetModel/diffusion_pytorch_model.safetensors` needs **no conversion** (stock SDXL ControlNet, loads via `ControlNet::from_weights` + `UNetConfig::sdxl_base()`). Source dtype (f32) preserved; loader casts. |
| `instantid_resampler_golden.safetensors` | `dump_instantid_resampler_golden.py` | `tests/instantid_resampler_real_weights.rs` | sc-3110 face Resampler. InstantID's `image_proj_model` is the *same* Tencent `Resampler` as the SDXL IP-Adapter (sc-3059), validated under `ResamplerConfig::instantid_face()` (embedding_dim=512) on a seeded `[1,1,512]` ArcFace embed → `[1,16,2048]` face tokens. **Bundles the f32 `image_proj.*` weights** (from `InstantX/InstantID` `ip-adapter.bin`) so the test needs no separate converted file — hence ~313 MB (larger than the other goldens). f32 vs torch CPU → peak_rel 5.3e-4 (the `norm_out` renormalizes, like the IP-Adapter Resampler's 4.9e-4). |

### SAM2 segmenter (`mlx-gen-sam2`, epic 3704)

Dumped from the **MLX-native reference** `avbiswas/sam2-mlx` (the impl this crate ports) — run from
the MLX venv with the reference checkout on `PYTHONPATH`, e.g.
`PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python tools/dump_<name>.py --size large`. Both
sides run MLX Metal, so parity is near-bit.

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `sam2_encoder_golden_large.safetensors` | `dump_sam2_encoder_golden.py` | `tests/encoder_parity.rs` | sc-3705 Hiera trunk + FPN neck — `enc_in` [1,3,1024,1024] → the 3 backbone-FPN maps + position encodings. |
| `sam2_segmenter_golden_large.safetensors` | `dump_sam2_segmenter_golden.py` | `tests/segmenter_parity.rs` | sc-3706 box-prompt decoder — encode→prompt-encode→two-way-transformer→mask. Bundles the full `trunk/neck/sam_prompt_encoder/sam_mask_decoder` weights + `enc_in`/`box_1024` + ref low-res masks/IoUs. |
| `sam2_photo_golden.safetensors` | `dump_sam2_photo_golden.py` | `tests/photo_parity.rs` | sc-3708 real-photo box→mask vs the spike baseline (zidane/bus). |
| `sam2_memory_golden_large.safetensors` | `dump_sam2_memory_golden.py` | `tests/memory_parity.rs` | sc-3713 Phase-B video layer — `memory_encoder.*`/`memory_attention.*` weights + two fixtures: the memory encoder (`mem_pix_feat`/`mem_masks` → 64-ch feature map + pos enc) and the memory attention (a 3-frame bank + 2 object pointers: `ma_curr`/`ma_mem`/… + `ma_num_obj` → conditioned tokens). Exercises the depthwise-conv ConvNeXt fuser and the interleaved axial RoPE self/cross attention with key-repeat + object-pointer RoPE exclusion. cos 1.0 (encoder mean-rel 0; attention mean-rel ~3e-5). |

### Kolors (`mlx-gen-kolors`, epic 3090)

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `kolors_tokenizer_golden.safetensors` | `build_kolors_tokenizer.py` | `tests/tokenizer_parity.rs` | sc-3092 ChatGLM3 tokenizer. The script ALSO materializes the fast `tokenizer.json` into the snapshot `tokenizer/` dir (LLaMA-style byte_fallback BPE replica). Golden = reference `ChatGLMTokenizer(prompt, padding="max_length", max_length=256, truncation=True)` input_ids/attention_mask/position_ids for a 5-prompt EN/EN-long(truncated)/CN/mixed/empty battery. Rust `KolorsTokenizer` matches byte-identical. |
| `kolors_chatglm_golden.safetensors` | `dump_kolors_chatglm_golden.py` | `tests/chatglm_parity.rs` | sc-3091 ChatGLM3-6B text encoder. Diffusers `KolorsPipeline` `ChatGLMModel` (`text_encoder/` fp16 shards, ~12.5 GB) run with `output_hidden_states=True` on two fixed inputs — `packed` (pure causal) and `padded` (right-pad → causal+padding mask) — for BOTH f32 and fp16 (`f16_` prefix). Bundles per case/dtype: `input_ids`/`attention_mask` + all 29 hidden states (permuted `[S,B,H]→[B,S,H]`) + `context` (`hidden_states[-2]`) + `pooled` (`hidden_states[-1]` last token). f32 worst hidden ~1.1e-3 (flat over depth = Metal-vs-CPU floor), fp16 worst ~1.7e-3. |

### Weight-independent

| golden | dump script | consumed by | notes |
|---|---|---|---|
| `pil_resize_golden.safetensors` | `dump_pil_resize_golden.py` | `src/image.rs` (`resize_bicubic_matches_pil`) | Only needs Python + PIL (no model weights). Candidate to shrink + promote into `tests/fixtures/` so its test can run un-`#[ignore]`d on any clone. |

## `CHECKSUMS.txt`

SHA-256 of the goldens the **currently committed tests were last validated against**, as a
regression tripwire: after regenerating, `shasum -a 256 -c CHECKSUMS.txt` flags an unexpected
change. Caveats — re-bless (regenerate the file) deliberately when:
- the MLX version changes (precision drift; see sc-2517), or a scheduler/port fix changes outputs;
- you regenerate at a different resolution/seed/steps than the committed baseline.

A mismatch across a *different machine/GPU* is possible (Metal float results aren't guaranteed
bit-identical cross-device) and isn't necessarily a bug — treat `CHECKSUMS.txt` as "what this
baseline produced," not a hard cross-machine contract. Only the goldens present when the file was
generated are listed.
