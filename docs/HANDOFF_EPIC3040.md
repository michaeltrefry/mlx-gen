# Handoff — epic 3040 (mlx-gen advanced video modes + SVD)

Status as of this handoff. Epic: https://app.shortcut.com/trefry/epic/3040

## Where the work lives
- **Branch:** `claude/busy-dhawan-6f3287`. **PR [#156](https://github.com/michaeltrefry/mlx-gen/pull/156) is MERGED** (it carried sc-3050/3051/3052/3053/3357 + SVD S0/S2). The branch now has **4 more commits (SVD S1/S3/S4), not yet pushed/PR'd** — `44eacad` (S1 VAE), `aac4974` (S3 UNet), `b394acd` (S4 pipeline), `6ba47e5` (S4 provider). Open a fresh PR for these.
- Worktree this was built in: `.claude/worktrees/busy-dhawan-6f3287` (a worktree of `mlx-gen`; main is `michaeltrefry/mlx-gen`).
- Reference repos: torch worker `~/Repos/SceneWorks/apps/worker/scene_worker/video_adapters.py`; LTX `ltx_core`/`ltx_pipelines` at `~/Repos/Wan2GP/models/ltx2`; diffusers/transformers source in the venv below.
- **Reference python env:** `~/repos/mflux/.venv-0312` (MLX 0.31.2-matched). This session **installed `einops`, `diffusers` (0.37.1), `transformers` (5.10.1)** into it for golden dumps. SVD + IC-LoRA + Wan/LTX checkpoints are all in the HF cache.

## Read these first
- `docs/SPIKE_ADVANCED_VIDEO_3040.md` — the GO/NO-GO spike + the two conditioning mechanisms.
- `docs/SVD_PORT_SPEC.md` — **exhaustive diffusers spec** (weight keys + building blocks) for the remaining SVD slices. This is the source of truth for S1/S3/S4.

## DONE + validated (in PR #156)
- **sc-3050** spike (Done). **sc-3051** conditioning framework (Done) — `Conditioning::{Keyframe,VideoClip,ControlClip}` + `ReplacementMode` + `GenerationRequest::{keyframes,video_clips,control_clip}` + `Conditioning::kind()`.
- **sc-3052** first_last_frame + extend_clip + video_bridge on **LTX** (In Review). FLF fully usable; extend/bridge = token-native IC-LoRA keyframe-append. The append op is **byte-exact vs torch `ltx_core` `VideoConditionByKeyframeIndex`** (`mlx-gen-ltx/tests/keyframe_cond_parity.rs`).
- **sc-3053** replace_person on **LTX** (In Review). Gray-118 mask op **byte-exact vs Pillow** (`replace_mask_parity.rs`); reuses the append path. Production LTX IC-LoRA confirmed loadable via the existing seam (`adapters::tests::ic_lora_union_control_keys_map_to_av_blocks`).
- **sc-3357** Wan-native first_last_frame on TI2V-5B (In Review) — `build_ti2v_multi_mask` + `build_ti2v_keyframe_z` + a `Conditioning::Keyframe` path; structural tests only (no Wan reference exists for these modes).
- **SVD S0 (sc-3371, Done)** — `mlx-gen-svd` crate + config + EDM scheduler. **Validated vs diffusers `EulerDiscreteScheduler`** (`scheduler_parity.rs`).
- **SVD S2 (sc-3373, Done)** — ViT-H image encoder (reuses sdxl `ClipVisionEncoder` + projection head). **Validated vs transformers** (`image_encoder_parity.rs`, `--ignored`, f32, 0.2% peak-rel).
- **SVD S1 (sc-3372, In Review — `44eacad`)** — `AutoencoderKLTemporalDecoder`, net-new `mlx-gen-svd::vae` (eps 1e-6 spatial / 1e-5 temporal — written net-new, NOT reusing the sdxl 2-D VAE, because that hardcodes 1e-5). **Validated f32** (`vae_parity.rs`): encode rel-L2 0.29%, decode rel-L2 0.11%.
- **SVD S3 (sc-3374, In Review — `aac4974`)** — `UNetSpatioTemporalConditionModel`, net-new `mlx-gen-svd::{embeddings,transformer,unet}`. Per-block eps matches the diffusers inconsistency (CrossAttnDown 1e-6, else 1e-5). **Validated f32** (`unet_parity.rs`): isolated resnet 0.014% / transformer 0.16% (tight structural guards); full forward rel-L2 1.44% = benign cross-backend accumulation.
- **SVD S4 (sc-3375, In Review — `b394acd` pipeline, `6ba47e5` provider)** — `SvdPipeline` (CFG frame-wise v-pred Euler + chunked temporal decode) + the registered `svd_xt` provider. **Validated** (`pipeline_parity.rs`): decode-on-golden 0.034%, denoise 4.2% (CFG-amplified per-step UNet gap); `--ignored` real-weights provider smoke runs e2e. **The whole SVD port (sc-3054) is engine-complete.**

## REMAINING WORK

### 1. Wan-VACE port (sc-3388) — the IC-LoRA-type pose/depth control on Wan
Full model port (VACE context/hint blocks on the Wan DiT). Covers pose/depth/sketch control + the **Wan** side of extend/bridge/replace_person. Torch ref = diffusers `WanVACEPipeline` (used by `video_adapters.py` for Wan replace_person). Pick the checkpoint (Wan2.1/2.2-VACE), consume `Conditioning::ControlClip` + reference images + `conditioning_scale`. **Slice it like the now-complete SVD port** (the SVD slices S1–S4 are the template: net-new blocks in their own modules, a `tools/dump_*_golden.py` + `tests/*_parity.rs` per slice gated in f32, isolated-component gates as the structural guards + a full-forward/e2e gate). This is a large port (~the size of the whole SVD effort) → its own session.

### 2. SVD CLIP preprocess — byte-exact antialiased resize (sc-3412)
The `svd_xt` provider resizes the CLIP image with the core PIL bicubic resampler, not diffusers' `_resize_with_antialiasing` (gaussian blur + align-corners bicubic). Conditioning is robust to it; port the antialiased path + a `_encode_image` parity test. Small, mlx-gen-local.

### 3. Gated / cross-repo (not mlx-gen code, or external deps)
- **extend/bridge/replace_person e2e parity** — the conditioning ops are byte-validated and the IC-LoRA loads, but a full-render byte-parity vs torch `ICLoraPipeline` is impractical (22B torch base; mlx-gen runs the AV-q4 base, not the Lightricks 22B distilled). A **directional** e2e render (load AV-q4 via `LTX_BASE_DIR` + the Union-Control IC-LoRA via `spec.adapters` + a `VideoClip` request) is possible if quality confirmation is wanted.
- **sc-3385** (chore) — investigate + fix the SceneWorks worker's "all advanced modes route to LTX" (`_uses_ic_lora_pipeline`); define the per-mode×per-model routing matrix. **SceneWorks worker repo**, own session.
- **sc-3055** (chore) — routing + cutover: route modes/SVD to the Rust worker, retire the torch video adapters. **SceneWorks worker repo**, own session. The SVD-port blocker is now cleared (`svd_xt` is engine-complete + registered); still wants the directional e2e parity pass.

## Gotchas / lessons
- **CI won't run while the PR conflicts with main** — GitHub can't build the `pull_request` merge ref, so it silently skips. If runs stop appearing, check `gh pr view <n> --json mergeable` and merge main.
- CI = `cargo fmt --all --check` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` (macOS/Metal, `RUST_TEST_THREADS=1`). Run all three locally before pushing; clippy `-D warnings` is strict (needless borrows of temporaries, doc lines starting with `+ `, needless lifetimes).
- LTX DiT forward is **fully token+positions driven** (RoPE from `positions`), so appended conditioning tokens need no grid — that's what makes the IC-LoRA append path work.
- Golden dumps live in `tools/dump_*_golden.py` (use `from _paths import fixture`); parity tests in each crate's `tests/`. `Weights::cast_all(Dtype::Float32)` to gate in f32.
- Memory file: `~/.claude/projects/-Users-michael-Repos-mlx-gen/memory/advanced-video-epic3040.md`.
