# Spike — MLX advanced video conditioning + SVD port (epic 3040 / sc-3050)

**Verdict: GO on all five modes.** The conditioning *mechanism* ports cleanly to the existing
token-based MLX LTX DiT and (for the latent-replacement family) the Wan mask-blend path. The
remaining risk is **weight provisioning** (the IC-LoRA control/replacement adapters) and the
**absence of any MLX reference / golden** — validation strategy is spelled out below.

Confidence: **high** on the mechanism feasibility (read the actual torch source end-to-end and the
MLX seams that receive it); **medium** on full pixel-parity-to-torch for the IC-LoRA modes (gated on
sourcing the exact IC-LoRA weights + matching torch tiling/RoPE byte-for-byte).

---

## Reference source of truth (torch — there is no MLX reference for these modes)

- `…/SceneWorks/apps/worker/scene_worker/video_adapters.py` — `LtxPipelinesVideoAdapter`
  (native LTX IC-LoRA), `DiffusersVideoAdapter` (SVD + WanVACE replace_person), dispatch.
- `…/Wan2GP/models/ltx2/ltx_pipelines/ic_lora.py` — `ICLoraPipeline.__call__` (the two-stage loop).
- `…/Wan2GP/models/ltx2/ltx_pipelines/utils/helpers.py` —
  `image_conditionings_by_replacing_latent`, `video_conditionings_by_keyframe`,
  `prepare_mask_injection` / `_apply_mask_injection`.
- `…/Wan2GP/models/ltx2/ltx_core/conditioning/types/{latent_cond,keyframe_cond}.py` — the two
  `ConditioningItem` mechanisms.
- SVD checkpoint already local: `~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt`
  (`unet/config.json` = `UNetSpatioTemporalConditionModel`, in=8/out=4, blocks [320,640,1280,1280],
  cross_attn 1024, 25 frames, image_encoder = OpenCLIP ViT-H, vae = temporal AutoencoderKL).

## The two conditioning mechanisms (the whole epic reduces to these)

### A. Replace-latent (in-place) — `VideoConditionByLatentIndex`
VAE-encode the conditioning image → patchify → **overwrite** the target tokens at the frame's token
slice, set `clean_latent` there too, and set `denoise_mask = 1 − strength` there. No extra tokens, no
LoRA. **This is already implemented in `mlx-gen-ltx/src/conditioning.rs` (`apply_conditioning`)** for
a single image; it generalizes trivially to a *list* of (latent, frame_idx, strength).

Used by: `images=` → first/last keyframes. Also re-applied in **stage 2** of every IC-LoRA run.

### B. Keyframe-append (in-context / IC-LoRA) — `VideoConditionByKeyframeIndex`
VAE-encode the conditioning **clip** → patchify → **append** those tokens to the end of the token
sequence with their own RoPE positions (frame axis offset by `frame_idx`, then `÷ fps`), append a
`denoise_mask = 1 − strength` block, and concat positions. The target tokens can attend to the
appended conditioning tokens; an **IC-LoRA** is what teaches the DiT to use them. The MLX DiT is
already token-based and `positions.rs::create_position_grid_with` already builds exactly this frame
offset + causal-fix + fps layout — appending is `concat(tokens)`, `concat(positions)`,
`concat(mask)`.

Used by: `video_conditioning=` → extend_clip, video_bridge, replace_person. **Stage 1 only**
(stage 2 keeps only the image/replace-latent conditioning).

### Replace-person extra: mask injection — `prepare_mask_injection` / `_apply_mask_injection`
On top of (B), for the first `ceil(num_steps · masking_strength)` steps, the masked-region tokens are
forced toward `source_latents` re-noised to the step's sigma (`source·(1−σ) + noise·σ` style), inside
the binary person-mask token slice. Person detect/track stays in onnx/Python and supplies
`mask` + `person_track_id` + the neutral-gray (118) masked control clip; MLX consumes them.

## Per-mode GO/NO-GO + difficulty

| Mode | Verdict | Difficulty | Mechanism | IC-LoRA weight needed? |
|---|---|---|---|---|
| **first_last_frame** | GO | **LOW** | (A) replace-latent at frame 0 and frame N−1 | No (matches torch: FLF uses 2-stage unless an IC-LoRA is present) |
| **extend_clip** | GO | **MEDIUM** | (B) append source-clip-tail latents at frame_idx 0 | Yes (control IC-LoRA) — mechanism runs without, quality needs it |
| **video_bridge** | GO | **MEDIUM** | (B) append left clip @0 + right clip @tail | Yes |
| **replace_person** | GO | **MED-HIGH** | (B) append masked control clip **+** mask injection | Yes (replacement IC-LoRA) |
| **SVD** | GO | **HIGH** | new `mlx-gen-svd` crate (full model port) | n/a |

## Wan vs LTX
The torch reference is **LTX-centric** for FLF/extend/bridge (the IC-LoRA pipeline). Wan only appears
for replace_person via diffusers `WanVACEPipeline`. Decision: **LTX is the faithful primary target.**
Wan gets the natural **replace-latent** variants (FLF, and clip pinning via the existing
`denoise_ti2v` mask-blend) — Wan has no IC-LoRA-append mechanism in the reference, so Wan does not get
the in-context extend/bridge/replace_person append path.

## SVD scope (sc-3054)
- `UNetSpatioTemporalConditionModel`: in=8 (4 latent + 4 image-latent concat), out=4,
  block_out_channels [320,640,1280,1280], cross_attention_dim 1024, num_attention_heads [5,10,20,20],
  addition_time_embed_dim 256, projection_class_embeddings_input_dim 768, num_frames 25,
  spatiotemporal down/up blocks (spatial conv/attn + temporal conv/attn + temporal mixing).
- Image encoder: OpenCLIP **ViT-H/14** (hidden 1024) → image embeds for cross-attn.
- VAE: temporal `AutoencoderKLTemporalDecoder` (spatial encode; temporal decode with chunking).
- Micro-conditioning: `fps_id`, `motion_bucket_id`, `noise_aug_strength` → `addition_time_embed` (256
  each → concat → 768) added to time embedding.
- Scheduler: EDM/Euler (Karras sigmas, `sigma_max` large). CFG with linearly-increasing guidance
  (`min_guidance_scale`→`max_guidance_scale`) across frames.
- Reuse: SDXL UNet patterns (`mlx-gen-sdxl`) for the spatial blocks + conv/attn; new = temporal
  layers + temporal VAE decoder + image-latent concat + micro-cond.

## Validation strategy (no MLX ref, no golden — this is the real constraint)
1. **Component-golden parity** for the NEW primitives, dumped deterministically from the torch
   `ConditioningItem.apply_to` / `prepare_mask_injection` / SVD submodules (these are pure tensor ops
   and need no IC-LoRA): multi-frame VAE-encode, replace-latent token overwrite + mask,
   keyframe-append tokens/positions/mask, mask-injection blend. Pattern = the existing
   `dump_ltx_*_golden.py` + `*_parity.rs`.
2. **Structural invariants** (the repo's `scale=0 ≡ baseline` idiom): strength=0 conditioning ≡ plain
   T2V bit-exact; token counts after append; RoPE position layout for offset frames.
3. **e2e with real weights** where available (LTX-Desktop-MPS has an IC-LoRA downloader; SVD weights
   are already local) — directional/coherence gate, not necessarily byte-parity (cross-backend).
4. **SVD** validates like SDXL: per-submodule golden parity (UNet block, temporal layer, VAE decode,
   image encoder) then e2e vs diffusers `StableVideoDiffusionPipeline`.

## Surfaced risks / dependencies (for Michael)
- **IC-LoRA weights** for extend/bridge/replace_person are a hard dependency for *quality* parity (not
  for the mechanism). They are standard LTXV LoRAs (loadable via the existing `mlx-gen-ltx` LoRA path +
  COMFY rename map). If the production weights aren't checked in, e2e parity for those three modes is
  gated; the mechanism + primitive-level parity still lands.
- **True pixel-parity to the full torch ICLoraPipeline** additionally requires matching torch VAE
  tiling + the exact two-stage sigma schedule — feasible (we already match LTX VAE + schedule
  elsewhere) but is the long pole.
- **sc-3055 cutover** edits the **SceneWorks worker repo** (routing + retiring torch adapters), not
  mlx-gen — that lands in a separate session against that repo.
