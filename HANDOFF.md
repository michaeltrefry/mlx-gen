# Handoff — mlx-gen (post sc-2352)

**Read this cold.** Supersedes `PERF_HANDOFF.md` (that perf hunt is DONE — see §1). This tells you
the current state and what to pick up next.

---

## 1. What just happened (sc-2352 — DONE)

The "Rust MLX is ~2× slower than the Python fork" problem is **solved, shipped, and merged.**

**Root cause:** the from-source MLX build floored the macOS deployment target at 14 → Metal 310,
which trips MLX's kernel gate (`MLX_METAL_VERSION ≥ 400 AND MACOS_SDK_VERSION ≥ 26.2`) and defines
`-DMLX_METAL_NO_NAX`, compiling out Apple's M-series matrix-unit ("NAX") Metal kernels
(`steel_gemm_*_nax` / `steel_attention_nax`) — the fast GEMM/attention path.

**Fix (two parts, both required):**
1. **MLX 0.30.6 engine** — `Cargo.toml` pins `mlx-rs = { package = "pmetal-mlx-rs", git =
   "github.com/michaeltrefry/mlx-rs", rev = "4b43fff…" }` (0.25.1 had no NAX source at all).
   Plus 6-arg `scaled_dot_product_attention` (0.30 added a trailing `sinks` arg) at 3 call sites.
2. **Metal-400 target** — `MACOSX_DEPLOYMENT_TARGET = "26.0"` in `.cargo/config.toml [env]`
   (NOT forced, so CI can override down). Restores NAX kernels: metallib 0 → 2736.

**Result (idle GPU, real Z-Image-Turbo weights, 4-step bf16):**
| 1024² | before | after | Python fork (compiled) |
|---|---|---|---|
| DiT per-step | ~4.7 s | **1.88 s** | 2.78 s |
| e2e generate() | ~22.9 s | **10.57 s** | 11.53 s |

Native Rust now **beats** the fork at every resolution (256²/512²/1024²). **Output parity confirmed**
at 1024²: 2684/3.1M px differ by >8 (0.085%), visually identical — residual is sub-perceptual bf16
rounding over the 4-step loop. Requires **macOS 26** at runtime (same floor Apple's mlx wheel imposes).

**Shipped:** PRs #25 (Metal-400 config), #26 (pmetal/0.30.6 engine — the real fix), #27 (bench
harness), #28 (resolution-flexible parity). All merged to `main`, CI green.

## 2. Things you should NOT redo (settled, with evidence)

- **Don't chase the Apple-wheel op gap.** Our *isolated* ops are still ~1.3× slower than Apple's
  wheel (SDPA 7.2 vs 5.3 ms @ S=4096). Fully diagnosed: it's the **Metal compiler toolchain**
  (ours `metalfe-32023.883` vs Apple's `.850` — same MLX source, flags, kernel count). Confirmed by
  the E2 swap (Apple libmlx + OUR metallib = slow; + Apple metallib = fast → it's the metallib, not
  the host). **BUT** a cooled interleaved 1024² test (with an A-vs-fallback internal control) showed
  **zero full-model benefit** — the DiT step isn't bottlenecked by the big SDPA/GEMM the microbench
  measures. Vendoring Apple's 128 MB metallib was rejected: macOS-26-pinned, Metal-320-CI-incompatible,
  benchmark-only win. Don't reopen unless you first profile the real step.
- **`bench_ops_micro` is unrepresentative.** It measures isolated S=4096 SDPA+GEMM; the real DiT
  step is many smaller ops + host dispatch. It misled the whole wheel-gap chase. If you want more DiT
  speed, **profile the actual step first**, don't optimize isolated GEMMs.
- **MLX version is irrelevant** — Apple 0.30.6 wheel == 0.31.0 wheel (both 5.5 ms). Don't bump
  the fork chasing perf.

## 3. ⚠️ CI constraint you must respect

CI runs on GitHub-hosted **`macos-15`** (Xcode 16.4 / SDK 15). Two hard facts:
- `macos-14` (SDK 14) **cannot compile** MLX 0.30.6's CPU SIMD (`accelerate_fp16_simd.h` errors).
  Don't move CI back to 14.
- No hosted `macos-26` exists, so CI builds at **Metal 320** (via job-level `MACOSX_DEPLOYMENT_TARGET:
  "15.0"` overriding the config default). **CI verifies correctness but NOT the NAX fast path.** A perf
  regression in NAX kernels would pass CI silently. Closing that needs a **self-hosted macOS-26 runner**
  (see §5).

## 4. Current repo state (updated 2026-05-31 — cleanup pass)

- `main` @ `e9334cc` (merge #28) has everything. CI green.
- **Main checkout** `/Users/michael/Repos/mlx-gen` is now on a **clean `main`** (was
  `spike/pmetal-mlx-030` with redundant pmetal edits). The leftover spike edits (identical to
  `origin/main`) + two stale test stubs + the stale `PERF_HANDOFF.md` were **stashed, not deleted** —
  recover via `git stash list` → `stash@{0}`. Drop it once you're sure nothing's needed:
  `git -C ~/Repos/mlx-gen stash drop`.
- `PERF_HANDOFF.md` is gone from the working tree (in the stash above; it was stale — its hypothesis
  was resolved in §1).
- **Remote branches — STILL PENDING:** the two merged branches `sc-2352-mlx-metal400-nax` and
  `sc-2352-pmetal-migration` remain on the remote (both fully merged into `main`; verified ancestors).
  Deletion was blocked by the auto-mode permission classifier this session. Run when authorized:
  `git push origin --delete sc-2352-mlx-metal400-nax sc-2352-pmetal-migration`. GitHub auto-deleted
  the rest (#27/#28).
- **Worktree** `/Users/michael/Repos/mlx-gen/.claude/worktrees/busy-pascal-900e4a` is on local branch
  `sc-2352-output-parity` (remote merged/auto-deleted) — harmless; removable via
  `git worktree remove`. (This `HANDOFF.md` lives in the main checkout root, not in that worktree.)
- **Shortcut hygiene done:** the 9 merged foundational stories (sc-2338/2339/2340/2341/2342/2343/2344/
  2350/2373) are now **Done**; sc-2345 (FLUX.1) and sc-2351 (worker integration) were corrected from a
  false "In Progress" back to **To Do** (their "linked" PRs #4/#24 were mislabeled auto-links).
- Golden/render artifacts under `tools/golden/` are gitignored, local-only.

## 5. What to work on next (epic 2337 — "eradicate Python")

sc-2352 was the **last gate** of the image port. Logical next moves, roughly in order:

1. **Worker integration (plan item 14)** — wire `mlx-gen`'s `generate()` into the Rust worker,
   remove `/opt/mlx-flux-venv` + the Python `mlx_flux_runner` sidecar. This is the north-star payoff.
2. **Remaining model ports** — FLUX.1, FLUX.2-klein 9b (+ 9b-kv via `compile_with_state`),
   Qwen-Image (+ causal Conv3d VAE), then SDXL (sc-2400, U-Net not DiT). Z-Image is the proven
   template (provider-crate pattern: `mlx-gen-z-image` → core `mlx-gen`).
3. **Z-Image ControlNet** (plan item 12) + LoKr coverage (sc-2216; engine already done in sc-2343).

**Optional infra/cleanup (non-blocking):**
- Self-hosted macOS-26 CI runner so CI can exercise the NAX fast path (§3).
- Report the `metalfe-32023.883` kernel-perf regression upstream (ml-explore / Apple).

## 6. How to reproduce / run things

```
Repo:        /Users/michael/Repos/mlx-gen   (main @ origin/main)
Build needs: macOS 26 + Xcode 26 (for the NAX fast path). MACOSX_DEPLOYMENT_TARGET=26.0 is in
             .cargo/config.toml — a plain `cargo build` gets it. CI overrides to 15.0 on macos-15.
Rust bench:  cargo test -p mlx-gen-z-image --release --test bench_z_image -- --ignored --nocapture
               (bench_ops_micro [fast, but unrepresentative], bench_denoise_per_step, bench_generate_wall_clock)
Parity test: cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
               (needs the golden — regenerate it below first)
Fork golden: cd ~/repos/mflux && ZIMAGE_W=1024 ZIMAGE_H=1024 uv run python \
               ~/repos/mlx-gen/tools/dump_z_image_golden.py    (env vars: ZIMAGE_W/H/STEPS/SEED/PROMPT)
             → writes tools/golden/{z_image_golden.safetensors,.png}; e2e test reads its metadata.
Fork bench:  cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/bench_z_image_fork.py
Weights:     ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<hash>/
```

**Benchmark methodology that survived scrutiny** (learned the hard way — earlier runs were ruined by
GPU contention + thermal ordering): idle GPU only (check `ioreg -r -d 1 -c IOAccelerator | grep
"Device Utilization"`; note the Claude desktop app itself adds ~36% UI GPU load), cool 45 s,
**interleave** configs back-to-back, report **min** across rounds, and include a control config to
catch artifacts. Also confirm no other GPU jobs are running before trusting numbers.

## 7. Memory

Durable findings are in CodeGraph memory: `rust-mlx-slow-root-cause-nax` (full root cause + the
don't-vendor decision + CI constraint). Check memory at session start.
