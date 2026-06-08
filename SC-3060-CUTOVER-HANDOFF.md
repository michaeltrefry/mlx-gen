# Handoff — sc-3060: route SDXL advanced conditioning to mlx-gen, retire `SdxlDiffusersAdapter`

**Read this cold.** You are an agent working in the **SceneWorks** repo. This document gives you
everything from the *engine* side (the `mlx-gen` Rust crate) that you need; the *worker* side
(Python `scene_worker` + the Rust worker bridge) lives in your repo and you'll discover the exact
paths there. Confidence notes are explicit throughout. When in doubt, read the source — don't assume.

- **Story:** [sc-3060](https://app.shortcut.com/trefry/story/3060) — "Routing + cutover" (last story of epic 3041).
- **Epic:** [3041](https://app.shortcut.com/trefry/epic/3041) — *mlx-gen: SDXL advanced conditioning (IP-Adapter, inpaint/outpaint, tile-ControlNet detail)*.
- **Workflow:** move the story To Do → In Progress when you start; keep comments updated (author `claude`); only → Done when parity-validated and `SdxlDiffusersAdapter` is actually deleted.

---

## 1. Mission

`mlx-gen-sdxl` (Rust, Apple-MLX) now implements the **four** SDXL features that were keeping the
torch `SdxlDiffusersAdapter` alive:

| Feature | Torch path being retired (`SdxlDiffusersAdapter`) | mlx-gen story |
|---|---|---|
| **Reference** (image-prompt) | IP-Adapter plus-face + ViT-H image encoder | sc-3059 ✅ merged |
| **Inpaint** (edit + mask) | `StableDiffusionXLInpaintPipeline` | sc-3057 ✅ merged |
| **Outpaint** | inpaint + border mask | sc-3057 ✅ merged |
| **Detail** | `StableDiffusionXLControlNetImg2ImgPipeline` + xinsir tile-CN | sc-3058 ✅ merged |

The engine side of epic 3041 is **complete and merged** (mlx-gen PRs #137 + #138). sc-3060 is the
**integration cutover**: route these four SDXL job types from the torch adapter to the Rust mlx-gen
worker, validate parity, then **delete `SdxlDiffusersAdapter`** (`apps/worker/scene_worker/image_adapters.py:5382`)
and its torch/diffusers dependencies.

This is an **outward-facing, hard-to-reverse production change.** Do the routing + parity validation
first; only delete the torch adapter once all four Rust paths are proven at parity.

---

## 2. What you already have on the SceneWorks side (verify in-repo)

From the mlx-gen source comments, your repo already contains:

- A **Rust worker** that links the `mlx-gen` crates (this is the only place `mlx-gen` is consumed —
  the mlx-gen repo itself is a pure Rust *library* workspace: no bin, no FFI, no Python bindings).
- A Python **`MlxSdxlAdapter`** that already routes SDXL **txt2img / img2img / LoRA** to that worker
  (landed by sc-3026 / epic 3018). `payload.model == "sdxl"` selects it (`MODEL_TARGETS["sdxl"]`).
- The torch **`SdxlDiffusersAdapter`** (`image_adapters.py:5382`) still handling the four advanced
  paths above.

**First moves in your repo (discover, don't assume):**
1. Read `SdxlDiffusersAdapter` end-to-end. Inventory exactly which request shapes hit it (how it
   decides reference vs inpaint vs outpaint vs detail — likely `payload` flags like `use_inpaint`,
   `outpaint`/`fit_mode`, `use_ip_adapter`, a control/detail flag). These are your routing branches.
2. Read `MlxSdxlAdapter` and the worker bridge it calls. Find **how a request crosses Python→Rust**
   (subprocess + JSON? a pyo3/uniffi binding? an embedded worker protocol?) and **how the Rust side
   builds a `LoadSpec` + `GenerationRequest`**. That mapping layer is what you extend.
3. Find the **model cache / loader** in the worker. See §5 — this is the one real architectural
   subtlety.

---

## 3. The mlx-gen contract (the part I can give you exactly)

`mlx-gen` exposes models through a registry. The worker loads by id and calls `generate`:

```rust
// id == "sdxl" (mlx_gen_sdxl::MODEL_ID). The provider crate must be linked so `inventory` registers it.
let model: Box<dyn mlx_gen::Generator> = mlx_gen::registry::load("sdxl", &load_spec)?;
let out: mlx_gen::GenerationOutput = model.generate(&request, &mut on_progress)?;
// out == GenerationOutput::Images(Vec<Image>)
```

### `LoadSpec` (load-time — `src/runtime.rs`)
```rust
LoadSpec {
    weights: WeightsSource::Dir(<sdxl-base-1.0 snapshot dir>),   // required
    quantize: Option<Quant>,            // None | Q4 | Q8
    precision: Precision,               // MUST be Bf16 sentinel (SDXL runs fp16 internally; an override errors)
    control: Option<WeightsSource>,     // ControlNet checkpoint  → enables Detail
    ip_adapter: Option<WeightsSource>,  // h94/IP-Adapter dir     → enables Reference (IP mode)
    adapters: Vec<AdapterSpec>,         // LoRA/LoKr (already used by MlxSdxlAdapter)
}
```
Builders: `LoadSpec::new(weights).with_control(src).with_ip_adapter(dir).with_quant(q).with_adapters(v)`.

### `GenerationRequest` (per-job — `src/generator.rs`)
Defaultable struct. Relevant fields: `prompt`, `negative_prompt`, `width`, `height`, `count`,
`seed`, `steps`, `guidance` (CFG), `sampler`, `strength` (img2img default when a `Reference` has no
own strength), `conditioning: Vec<Conditioning>`, `cancel`.

### `Conditioning` variants you'll use
```rust
Conditioning::Reference { image, strength: Option<f32> }       // img2img init OR IP image-prompt (see §4)
Conditioning::Mask      { image }                              // inpaint mask: WHITE(255)=repaint, BLACK(0)=keep
Conditioning::Control   { image, kind: ControlKind, scale: f32 } // ControlNet (tile detail)
```

### Defaults baked into the SDXL model (so you only override when the job specifies)
- img2img strength `0.8`; **inpaint/outpaint strength `0.85`**; **IP-Adapter scale `0.6`**.
- Production txt2img: 30 steps, CFG 7.0, sampler `euler_ancestral`.
- `min_size 512`, `max_size 2048`, dims must be multiples of 8, `count ≤ 8`.

---

## 4. Feature-by-feature: exactly how to drive each path

> The SDXL model's request dispatch (`mlx-gen-sdxl/src/model.rs::generate`) decides the path from the
> *combination* of loaded weights + conditioning. Get the combination right and it routes itself.

### 4a. Reference → IP-Adapter (image-prompt identity/style)
- **Load:** `LoadSpec::new(base).with_ip_adapter(Dir(<h94/IP-Adapter snapshot>))`.
- **Request:** one `Conditioning::Reference { image: <ref image>, strength: Some(<ip_scale>) }`, **no**
  Mask, **no** Control, non-accel sampler.
- **Behavior:** "IP mode" — the Reference is the *image prompt* (txt2img + IP cross-attn), **not** an
  img2img init. The `strength` field carries `ip_adapter_scale` (default `0.6` if `None`). CFG batches
  the IP tokens with a zeros uncond row automatically.
- The loader prefers `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors`, falling back to
  `ip-adapter-plus_sdxl_vit-h.safetensors` (same Resampler arch). Image encoder at
  `models/image_encoder/model.safetensors`.

### 4b. Inpaint (masked edit)
- **Load:** plain base (no control/ip needed).
- **Request:** **both** `Conditioning::Reference { image: <init>, strength: Some(0.85 or job value) }`
  **and** `Conditioning::Mask { image: <mask> }`. Mask polarity: **white = repaint, black = keep**.
- **Behavior:** ancestral img2img with a per-step latent blend pinning the kept region. The model
  **errors** if a Mask is passed without a Reference (it needs an init to blend against).

### 4c. Outpaint (= inpaint with a border mask)
- **Load:** plain base.
- **Host prep (this is the `fit_mode == outpaint` work):**
  1. Contain-fit the source onto the target canvas and composite it (transparent/black border).
  2. Generate the **border mask** (white border to generate, black centered source to keep).
  3. If the job also has a user edit mask, **union** it with the border mask (white wins).
- **Request:** `Reference { image: <composited canvas>, strength: Some(0.85) }` + `Mask { image: <border∪user mask> }`.
- **Helpers already ported into mlx-gen core (`src/image.rs`)** — reuse if the worker can reach them,
  else replicate host-side (they mirror the worker's originals exactly):
  - `contain_box(src_w, src_h, w, h) -> (new_w, new_h, left, top)` — round-half-even, matches the worker's `_contain_box`.
  - `outpaint_border_mask(src_w, src_h, w, h) -> Image` — white=generate / black=keep, aligned to the contain fit. (The worker's optional gaussian feather is intentionally omitted — the inpaint pipeline binarizes the mask, so a symmetric feather is a sub-latent-pixel no-op. Feather post-decode if needed.)
  - `union_masks(a, b) -> Image` — per-pixel max (PIL `ImageChops.lighter`).

### 4d. Detail → tile-ControlNet (img2img + ControlNet)
- **Load:** `LoadSpec::new(base).with_control(<xinsir/controlnet-tile-sdxl-1.0 checkpoint>)`
  (a `WeightsSource::File` for a single `.safetensors`, or `Dir` for a `diffusion_pytorch_model.safetensors` tree).
- **Request:** **both** `Conditioning::Reference { image: <init>, strength: Some(<detail strength>) }`
  **and** `Conditioning::Control { image: <tile control image>, kind: ControlKind::Other("tile".into()), scale: <conditioning_scale> }`.
  (`kind` is not inspected by the SDXL control path — any value works; pick a descriptive one.)
- **Behavior:** img2img init (VAE-encode the Reference) **with** ControlNet residuals injected each
  step — i.e. the diffusers `ControlNetImg2Img` path. Passing Control **without** a loaded `control`
  checkpoint errors; combining Control **with** a Mask errors (not supported in this build).

### Path-selection truth table (how the model decides)
| ip_adapter loaded | control loaded | Reference | Mask | Control | → path |
|:-:|:-:|:-:|:-:|:-:|---|
| ✓ | – | ✓ | – | – | **IP-Adapter** (image prompt) |
| – | – | ✓ | ✓ | – | **Inpaint / Outpaint** |
| – | ✓ | ✓ | – | ✓ | **Detail** (img2img + ControlNet) |
| – | – | ✓ | – | – | plain img2img |
| – | – | – | – | – | plain txt2img |

---

## 5. ⚠️ The one real architectural subtlety: load-time vs per-request

`control` and `ip_adapter` are **load-time graph components** — they add K/V projections / a control
branch to the U-Net *at load*. You **cannot** bolt them onto an already-loaded txt2img model per
request. Consequences for the worker's model cache/loader:

- The cached SDXL model must be **keyed on `(quant, has_ip_adapter, has_control, adapter set)`**, not
  just `payload.model`. A reference (IP) job and a detail (control) job need **different loaded models**.
- Decide the policy: load-on-demand per feature (simpler; a load cost per first-use), or pre-load the
  common variants. Check how `MlxSdxlAdapter` currently caches the txt2img/LoRA model and extend that
  keying — don't silently reuse a txt2img model for an IP job (it has no IP K/V → no effect).
- LoRA/accel samplers: the few-step accel samplers (`lcm`/`lightning`/`hyper`) **reject** reference,
  mask, and control in this build — keep advanced conditioning on the default `euler_ancestral` path.

---

## 6. Weights on disk (confirmed present during epic validation)
- `h94/IP-Adapter` — `sdxl_models/ip-adapter-plus[-face]_sdxl_vit-h.safetensors` + `models/image_encoder/` (ViT-H/14).
- `xinsir/controlnet-tile-sdxl-1.0` — the tile ControlNet checkpoint.
- `stabilityai/stable-diffusion-xl-base-1.0` — base snapshot (also serves inpaint/outpaint; no separate inpaint checkpoint — it's masked-latent blending on the base UNet).

Confirm the worker's weight-resolution config points at these (paths/HF cache) for each feature.

---

## 7. Parity validation plan (do this BEFORE deleting the torch adapter)

The engine is already validated **internally** byte-exact where a byte-exact invariant exists:
IP `scale=0 ≡ plain txt2img` (0/786432 px), inpaint white-mask ≡ img2img and black-mask ≡ VAE
roundtrip, ControlNet `scale=0 ≡ no-control`, all byte-exact (mlx-gen `tests/*_real_weights.rs` /
`ip_adapter_decoupled.rs`). So the *mechanism* is proven.

**Cross-impl (torch vs Rust) is NOT byte-exact and must not be expected to be.** torch-CPU vs
MLX-Metal GEMM/SDPA accumulate a characterized **~1–2% peak-relative** cross-backend floor in deep
pre-LN stacks — this is a backend difference, not a port bug. Acceptance = **visually equivalent**
output per feature, not pixel-identity.

Suggested gate (run each through *both* `SdxlDiffusersAdapter` and the Rust path, same seed/prompt/inputs):
1. **Reference/IP** — same ref image + prompt; identity/style transfer present and comparable. (Quant
   parity numbers — CLIP-preprocess vs torch `CLIPImageProcessor`, ArcFace-cosine identity vs
   `SdxlDiffusersAdapter._use_ip_adapter` — are the directional follow-ups noted on sc-3059; capture
   them here as the quantitative parity evidence.)
2. **Inpaint** — kept (black) region must be untouched outside the mask; repaint region coherent.
3. **Outpaint** — border filled, original center preserved; seam acceptable.
4. **Detail** — tile-CN sharpens/details the init while preserving structure.

Record the comparison artifacts on the story before cutover.

---

## 8. Cutover sequence
1. Extend the Python `MlxSdxlAdapter` (+ the Rust bridge's request mapping) to build the
   `LoadSpec`/`GenerationRequest` of §4 for each of the four job shapes that currently route to
   `SdxlDiffusersAdapter`. Implement the `fit_mode == outpaint` host prep (§4c).
2. Route the four shapes to the Rust path (a flag/config toggle is fine for the transition).
3. Run the §7 parity gate; attach evidence to sc-3060.
4. **Delete `SdxlDiffusersAdapter`** (`image_adapters.py:5382`) and prune now-unused torch/diffusers
   imports, the `StableDiffusionXL*Pipeline` deps, and any registration/`MODEL_TARGETS` entry that
   pointed at it. Grep the repo for every reference before deleting.
5. Run the worker's test/lint suite; smoke-test all four job types end-to-end.

---

## 9. Gotchas / do-not-trip-on
- **Mask polarity:** white = repaint, black = keep (binarized at ≥0.5, then 8× nearest downsample to latent res).
- **Inpaint needs an init** (Reference) alongside the Mask, or it errors.
- **Control + Mask not combinable**; **multiple** References / Masks / Controls each error (single of each).
- **IP `strength` field = the IP scale** (not an img2img strength) — that's deliberate, it's how the scale knob rides the existing request shape.
- **VAE runs f32** even though UNet/TEs are fp16 (SDXL VAE is fp16-unstable) — already handled in the loader; just don't override precision.
- **`precision` must stay the `Bf16` sentinel** — SDXL's dense path *is* fp16 internally; a precision override is rejected.
- Don't route advanced conditioning through `lcm`/`lightning`/`hyper` — txt2img-only in this build.

---

## 10. Definition of done
- All four SDXL advanced paths served by the Rust mlx-gen worker; `fit_mode == outpaint` host prep wired.
- Parity evidence (§7) attached to sc-3060.
- `SdxlDiffusersAdapter` **deleted**; torch/diffusers SDXL deps pruned; no dangling references.
- Worker tests/lints green; all four job types smoke-tested.
- sc-3060 → Done with outcomes + any residual limitations captured. (This fully retires the SDXL
  branch of epic 3018 and closes epic 3041.)

---

## 11. Open questions to resolve in your repo (I can't see them from mlx-gen)
- Exact Python→Rust bridge mechanism and where request mapping lives.
- Current model-cache keying in `MlxSdxlAdapter` (extend per §5).
- How the worker configures the IP-Adapter / tile-CN weight locations per request.
- Whether to ship behind a feature flag during the transition or hard-cut.
- plus vs plus-face IP weights — the worker's prior default (`SdxlDiffusersAdapter` used plus-face);
  the loader prefers plus-face automatically if present.

*Engine reference (read-only, for the contract): `mlx-gen-sdxl/src/model.rs` (`generate` dispatch +
defaults), `src/runtime.rs` (`LoadSpec`), `src/generator.rs` (`GenerationRequest`/`Conditioning`),
`src/image.rs` (outpaint helpers). mlx-gen is at `~/Repos/mlx-gen`, branch `main` after PR #138.*
