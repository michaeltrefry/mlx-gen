# FLUX.1 (schnell + dev) — divergence report

> ⚠️ **HISTORICAL — frozen single-session snapshot, not current parity status.** This report is
> scoped to the branch `codex/sc-2345-flux1-complete` and mflux **v0.17.5**; FLUX changes merged to
> `main` since 2026-06-02 are **not** reflected here. Do **not** read the residuals below as the
> current state of `main`. The **living record of FLUX parity is the parity test suite**
> (`mlx-gen-flux/tests/*`, e.g. `e2e_real_weights.rs`), which runs against the current code. Kept
> under `docs/` for the methodology and the one-time measurements; moved off the repo root so it is
> no longer mistaken for a status page (sc-4145 / F-149).

**Date:** 2026-06-02
**Branch:** `codex/sc-2345-flux1-complete`
**Purpose:** Document, without attributing cause, where and how much the Rust `mlx-gen-flux` port diverges from the Python fork (mflux v0.17.5), how each point was tested, and the measured results. Numbers are from a single reproducible session run.

---

## 1. Setup

| | schnell | dev |
|---|---|---|
| Steps | 4 | 20 |
| Guidance | 0.0 | 3.5 |
| Sigma shift | none | mu-shift |
| T5 sequence length | 256 | 512 |
| Resolutions tested | 256² | 256², 512² |

- Prompt `"a red fox"`, seed `7`, for every run.
- Fork reference = mflux v0.17.5 (`/Users/michael/Repos/mflux/.venv`), real `black-forest-labs/FLUX.1-{schnell,dev}` weights (BF16 checkpoints).
- Rust = `mlx-gen-flux` on `codex/sc-2345-flux1-complete`.

### Golden variants

The fork golden is dumped at two compute precisions and (optionally) quantized:

- **f32-precision** golden — `FLUX_PRECISION=f32` (forces `ModelConfig.precision = mx.float32`).
- **bf16-precision** golden — fork default (`ModelConfig.precision = bf16`).
- **Quantized** golden — `QUANTIZE=8` / `QUANTIZE=4` (`FluxInitializer(quantize=N)`), at either precision.

Implementation fact (not a cause claim): the Rust diffusion path runs **f32 activations**; quantized weights are stored with **bf16-derived scales**.

---

## 2. Test method

**Golden dumper** — `tools/dump_flux_golden.py`: drives the fork by hand (mirroring `Transformer.__call__` + `LinearScheduler.step` + `FluxLatentCreator`) and writes every intermediate to a safetensors: T5/CLIP input ids, `prompt_embeds`, `pooled_prompt_embeds`, init noise, `v0` (first velocity), per-substage transformer tensors (`hidden0`, `encoder0`, `text_embeddings0`, `block0_*`, `joint_hidden`, `encoder_joint`, `single_b0_img`, `single_img`, `rope0`), `final_latents`, `decoded`, `sigmas`. Env: `FLUX_VARIANT`, `FLUX_PRECISION`, `QUANTIZE`, `FLUX_W/H/STEPS/SEED/GUIDANCE`.

**Rust harness** — `mlx-gen-flux/tests/e2e_real_weights.rs` (`#[ignore]`d; needs the real weights + a local golden):

- **Stage tests** feed the fork golden's *own* intermediates into each Rust stage in isolation (e.g. inject the golden `prompt_embeds`/`init`/`sigmas`, run only the Rust transformer).
- **Full-pipeline test** drives the public `load(id, spec).generate(req)` — Rust computes T5, CLIP, transformer, denoise, and VAE itself.
- Selected by `FLUX_VARIANT` (default schnell); the golden path, model id, guidance, mu-shift, and T5 seq-length follow.

**Reproduce:**
```
FLUX_VARIANT=<schnell|dev> MLX_GEN_FLUX_SNAPSHOT=<snapshot dir> \
  cargo test -p mlx-gen-flux --test e2e_real_weights -- --ignored --nocapture
```

**Metrics**
- `peak_rel` = max|a−b| / max|b|
- `mean_rel` = mean|a−b| / mean|b|
- `px>8` = % of decoded RGB8 pixels (0–255) differing from the golden by more than 8.

Unless stated, comparisons are **vs the f32-precision golden**.

---

## 3. Component stages — Rust vs f32-precision golden (golden inputs injected)

Each stage is fed the fork golden's own inputs, so this isolates the stage from upstream accumulation.

| Stage | schnell @256² | dev @512² |
|---|---|---|
| init noise (RNG) | peak 0.000e0 | peak 0.000e0 |
| scheduler sigmas (max\|Δ\|) | 5.96e-8 (5 sigmas) | 5.96e-8 (21 sigmas) |
| RoPE cos / sin (peak) | 4.17e-7 / 4.77e-7 | 9.46e-7 / 9.54e-7 |
| T5 `prompt_embeds` | peak 1.764e-2, mean 2.974e-3 | peak 1.028e-2, mean 2.655e-3 |
| CLIP `pooled` | peak 0.000e0 | peak 0.000e0 |
| `text_embeddings0` (modulation) | peak 5.71e-7, mean 3.37e-7 | peak 1.236e-6, mean 2.318e-6 |
| `hidden0` (x_embedder) | 0.000e0 | 0.000e0 |
| `encoder0` (context_embedder) | 0.000e0 | 0.000e0 |
| `block0_encoder` (peak) | 6.115e-5 | 1.381e-4 |
| `block0_hidden` (peak) | 1.534e-4 | 2.041e-4 |
| `joint_hidden` | peak 4.931e-4, mean 2.521e-4 | peak 4.526e-4, mean 2.092e-4 |
| `encoder_joint` (peak) | 2.745e-4 | 2.575e-4 |
| single block[0] (injected, peak) | 4.383e-6 | 4.134e-7 |
| single stack (injected) | peak 3.054e-3, mean 3.206e-3 | peak 2.873e-4, mean 3.994e-4 |
| `single_img` (full fwd) | peak 1.357e-2, mean 1.670e-2 | peak 2.054e-3, mean 5.557e-4 |
| transformer `v0` (full forward) | peak 4.003e-2, mean 7.891e-3 | peak 1.882e-4, mean 1.478e-4 |
| VAE decode | peak 9.068e-3; **0 / 196608 px>8** | peak 2.969e-2; **0 / 786432 px>8** |

---

## 4. Denoise loop — golden embeds injected, Rust runs the full step loop + VAE (vs f32 golden)

| | schnell @256² (4 steps) | dev @512² (20 steps) |
|---|---|---|
| `final_latents` | peak 3.676e-1, mean 5.954e-2 | peak 1.092e-1, mean 2.550e-3 |
| decoded px>8 | 8.32% | 0.06% |

---

## 5. Full pipeline — Rust computes everything (px>8)

The full-pipeline test compares the Rust render against the fork golden. The fork's own two-precision renders are included as reference points (each is a render produced entirely by the fork).

| Comparison | schnell @256² | dev @256² | dev @512² |
|---|---|---|---|
| Rust vs fork **f32** golden | 35.26% | 61.62% | 41.38% |
| Rust vs fork **bf16** golden | 37.39% | 74.41% | 44.84% |
| fork **f32** golden vs fork **bf16** golden | 20.61% | 76.27% | 38.40% |

Prior-session measurements (not re-run this session): schnell @1024² fork f32-vs-bf16 = 4.4% px>8; Rust full pipeline @1024² ≈ 32%.

---

## 6. Quantization (Q8 / Q4)

### 6a. vs **f32-precision** Q golden, 256² (golden embeds+init injected → Rust quantized transformer)

| | schnell Q4 | schnell Q8 | dev Q4 | dev Q8 |
|---|---|---|---|---|
| `hidden0` (mean) | 9.76e-4 | 3.24e-3 | 9.74e-4 | 3.25e-3 |
| `text_embeddings0` (mean) | 4.03e-4 | 5.24e-4 | 8.52e-4 | 1.02e-3 |
| `single_img` (mean) | 3.51e-2 | 6.09e-2 | 7.01e-3 | 8.48e-3 |
| `v0` (step-0 velocity, mean) | 1.61e-2 | 3.10e-2 | 3.14e-3 | 3.43e-3 |
| final latents (mean) | 6.41e-2 | 1.63e-1 | 1.74e-2 | 6.10e-2 |
| decoded px>8 (gate inputs) | 9.49% | 22.04% | 1.36% | 5.94% |
| full public generate, px>8 vs fork-Q | 38.42% | 32.39% | 79.13% | 59.15% |

### 6b. dev Q8 vs **bf16-precision** Q golden, 256² (same Rust run, different reference golden)

| substage (mean_rel) | value |
|---|---|
| `hidden0` | 0.000e0 |
| `encoder0` | 0.000e0 |
| `text_embeddings0` | 3.78e-2 |
| `block0_encoder` | 1.374e-1 |
| `block0_hidden` | 1.153e-1 |
| `joint_hidden` | 1.150e-1 |
| `encoder_joint` | 1.746e-1 |
| `single_img` | 1.290e-1 |
| `v0` | 6.750e-2 |
| final latents | 7.562e-1 |
| decoded px>8 | 74.82% |

Direct fork measurement (no Rust involved): the fork's own `text_embeddings0`, **Q8 vs f32 compute precision**, with `pooled=0` (isolating the timestep+guidance path) = **mean_rel 1.277e-3**.

---

## 7. Visual observations (neutral)

- **schnell @256²:** Rust render and fork f32 render are both coherent red foxes with differing fine detail (in an earlier viewing, the Rust fox showed ~2 tails and the fork f32 fox ~5 legs — distinct samples, both coherent).
- **dev @256²:** Rust f32 render and fork **f32** render share the same composition (close-up fox portrait). The fork **bf16** render is a different composition (full-body fox seated on a rock).
- **dev @512²:** Rust f32 render and fork **f32** render share the same composition (seated fox, forest background).
- **dev Q8 / Q4 @256²:** Rust renders are coherent fox portraits, sharing composition with the fork f32-precision Q renders.

Saved renders: `tools/golden/rust_flux_{schnell,dev}.png`, `tools/golden/rust_flux_{schnell,dev}_q{4,8}.png`, and the matching `flux1_*_golden.png` fork renders.

---

## 8. Current test status

All component/stage/scheduler/VAE/denoise asserts pass for both variants. The full-pipeline test is a regression guard (bound 0.85 dev / 0.5 schnell), not a parity assert. The quantization test asserts on `v0` (< 6e-2); the 20-step latent and full-generate px>8 are printed but not asserted.

- schnell non-Q: 11/11 pass.
- dev non-Q: 11/11 pass.
- schnell Q4/Q8, dev Q4/Q8 (vs f32-precision golden): pass.

### Known unimplemented / untested surface
- **LoRA / LoKr** — `descriptor` advertises `supports_lora: true` but the loader rejects all adapters for both variants (tracked: sc-2657).
- **Bit-exact parity** vs the fork's renders is not achieved at any setting (see §3–6 for magnitudes).
- dev verified at 256²/512²; not re-measured at 1024² this session (schnell @1024² figures in §5 are from a prior session).
