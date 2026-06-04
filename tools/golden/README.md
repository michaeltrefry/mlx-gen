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
