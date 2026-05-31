# mlx-gen — generator contract & per-model packaging

**Status: proposal for review.** Defines the public interface every model implements and how
models are packaged so consumers pull in only what they need. Once agreed, we walk each model
through the *fitment checklist* (§8) before changing code.

Mental model (C# terms): `mlx-gen` is the **abstractions + shared-runtime** package; each model
is a **provider** package implementing a common interface; a **registry** (link-time, the Rust
stand-in for a DI container's assembly scan) resolves a model by id to a `dyn` interface object.
"Brought in as needed" = a build-time dependency/feature choice (Rust has no runtime
assembly-load DI; if you don't depend on a provider crate, its code isn't compiled or shipped).

Grounded in two proven, cross-family contracts:
- **SceneWorks worker** `ImageRequest` / `VideoRequest` — the request schema that already
  dispatches 12+ image families + 4 video families (model chosen via `payload.model` →
  `MODEL_TARGETS`).
- **mflux adapters** — per-family `generate()` signatures + `ModelConfig` capability flags
  (`supports_guidance`, `requires_sigma_shift`, `supports_kv_cache`, …).

---

## 1. Goals & constraints

1. **Public crate.** Third parties may want one model and must not compile or ship the others.
   Opt-in is **compile-time** (crate deps / Cargo features); no runtime `dlopen` plugins.
2. **A stable contract all models satisfy**, hiding internals (FLUX dual-stream, Z-Image
   refiners, conv VAE — private behind the trait).
3. **The core (`mlx-gen`) stays fixed as models are added.** A new model is an additive provider
   crate; nothing in `mlx-gen` or other providers is edited. That is the sense in which
   `mlx-gen` "stops getting bigger."
4. Reversible: a provider can later split further, or be feature-gated, without touching the
   contract.

---

## 2. Crate topology — `mlx-gen` is the core; models are providers

No separate "core" or "facade" crate: **`mlx-gen` itself is the abstractions + runtime.**

```
mlx-gen              THE CORE (abstractions + shared runtime; NO model-specific code)
  ├─ generator.rs    Generator trait, GenerationRequest, GenerationOutput, ModelDescriptor,
  │                  Capabilities, GenerationError
  ├─ transform.rs    Transform trait (non-prompt media→media; see §3.3)
  ├─ registry.rs     model registration + lookup by id (inventory-style, see §4)
  ├─ nn/             shared primitives: attention, rope, rms_norm, swiglu, conv2d, group_norm,
  │                  linear, silu   (today under models/z_image — these MOVE here)
  ├─ adapters/       LoRA + LoKr: Adapter, AdaptableLinear (Vec of mixed adapters),
  │                  AdaptableHost, install + file loaders   (§3.4)
  ├─ weights.rs · quant.rs · tokenizer.rs · scheduler/      shared
  └─ (no model code)

mlx-gen-z-image      provider: depends ONLY on `mlx-gen`; implements Generator; registers a loader
mlx-gen-qwen-image   provider
mlx-gen-flux         provider  (FLUX.1 + FLUX.2-klein as variants/configs in one family crate)
mlx-gen-wan / -ltx   provider  (video; follow-on)
mlx-gen-seedvr2      provider  (Transform, not Generator; see §3.3)
```

- **SceneWorks worker** depends on every provider crate it serves (it needs all families).
- **A third party** depends on `mlx-gen` + the one provider crate they want (e.g.
  `mlx-gen-z-image`). None of the other models' code is compiled or shipped.
- A **family** gets one crate; intra-family variants (FLUX.1 vs FLUX.2-klein-9b vs -9b-kv;
  Z-Image-turbo vs -ControlNet) are configs/sub-modules inside it (§3.2).

> `mlx-sys` builds MLX from source (~5 min) and is shared regardless of the split, so the win is
> code/dependency isolation + third-party opt-in, not core build time.

**Migration:** extract today's shared `nn` primitives out of `models/z_image` into `mlx-gen`'s
`nn/`, then move the Z-Image model into `mlx-gen-z-image` as the first provider — proving the
contract on a nearly-complete pipeline before the others.

---

## 3. The contracts

### 3.1 `Generator` — prompt-conditioned synthesis (image **or** video, multi-modal)

Categorize by **interaction**, not output label. Image/Video/Utility as *trait categories*
breaks on multi-modal models; instead, modality is a **property** + an **output variant**.

```rust
pub trait Generator {
    fn descriptor(&self) -> &ModelDescriptor;              // identity + Capabilities + modality
    fn validate(&self, req: &GenerationRequest) -> Result<(), GenerationError>;
    fn generate(&self, req: &GenerationRequest,
                on_progress: &mut dyn FnMut(Progress)) -> Result<GenerationOutput, GenerationError>;
}

pub enum GenerationOutput {
    Images(Vec<Image>),
    Video(VideoClip { frames: Vec<Image>, fps: u32, audio: Option<AudioTrack> }),
}
```

One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX (text→video+audio),
even a model that does both image and video — the request is shared, the output enum + a
`descriptor().modality` of `Image | Video | Both` carry the variance. `generate` is **sync**
(long/blocking; the worker runs each job on its own thread); the request carries a cancel flag and
`on_progress` streams step/decode progress.

**`GenerationRequest`** — the union (lifted from the worker's `ImageRequest`/`VideoRequest`);
most fields optional, a model reads what it supports, `validate()` rejects the rest:

| Group | Fields |
|---|---|
| Core | `prompt`, `negative_prompt: Option`, `width`, `height`, `count` (1–8) |
| Sampling | `seed`/`seeds`, `steps`, `guidance`, `true_cfg`, `sampler`, `scheduler`, `scheduler_shift` (all `Option`) |
| Conditioning | `conditioning: Vec<Conditioning>`, `strength: Option<f32>` |
| Adapters | `adapters: Vec<AdapterSpec>` (§3.4) |
| Video | `frames`, `fps`, `duration`, `video_mode`, source-clip handles (all `Option`) |
| Control | cancel flag |

```rust
pub enum Conditioning {
    Reference { image: Image, strength: Option<f32> },        // img2img / IP-Adapter / identity
    MultiReference { images: Vec<Image> },                    // Qwen-Image-Edit
    ReduxRefs { refs: Vec<(Image, f32)> },                    // FLUX.1-Redux
    Control { image: Image, kind: ControlKind, scale: f32 },  // ControlNet / pose
    Depth { image: Image },                                   // FLUX.1-Depth
    Mask { image: Image },                                    // FIBO-Edit / inpaint
    // VideoClip { clips: Vec<(MediaRef, f32)> },             // video port (follow-on): extend/bridge/replace-person
}
```

**`ModelDescriptor` + `Capabilities`** drive `validate()` and a consumer's UI introspection:
modality; `supports_negative_prompt`/`supports_guidance`/`supports_true_cfg`; accepted
`Conditioning` kinds; `supports_lora`/`supports_lokr`; supported samplers/schedulers; resolution
& count bounds (+ frames/fps for video); `mac_only`; plus loader hints (`supports_kv_cache`,
`requires_sigma_shift`).

### 3.2 Config unifies *within* a family, not across

FLUX dual-stream MMDiT ≠ Z-Image single-stream ≠ Qwen — they don't share a forward, so each
family crate owns its blocks. Config (the pattern proven on `VaeDecoderConfig` /
`ZImageTransformerConfig`) collapses **variants within a family** (turbo vs ControlNet; 9b vs
9b-kv), not families.

### 3.3 `Transform` — non-prompt image→image (designed around SeedVR2)

Restorers/upscalers are **not** `Generator`s — no prompt; the input image *is* the subject.
SeedVR2 (the in-scope utility) is a diffusion-based **single-image** super-resolution model:
`seed` + input image + target size + `softness` → restored image (1-step, its own
VAE+transformer, fixed precomputed text embedding — no user prompt). So the contract is
image→image:

```rust
pub trait Transform {
    fn descriptor(&self) -> &TransformDescriptor;        // identity + TransformCapabilities
    fn validate(&self, req: &TransformRequest) -> Result<(), GenerationError>;
    fn apply(&self, req: &TransformRequest,
             on_progress: &mut dyn FnMut(Progress)) -> Result<Image, GenerationError>;
}

pub struct TransformRequest {            // Default-able, like GenerationRequest
    pub image: Image,
    pub target: TargetSize,
    pub seed: Option<u64>,               // diffusion restorers (SeedVR2); ignored by deterministic ones
    pub strength: Option<f32>,           // model-defined restoration knob (SeedVR2 "softness", 0..1)
    pub steps: Option<u32>,              // SeedVR2 is 1-step; override only if the model allows
    pub adapters: Vec<AdapterSpec>,      // uniform with Generator; usually empty
    pub cancel: CancelFlag,
}

pub enum TargetSize {
    Scale(f32),                          // SeedVR2 "2x"/"3x" (× the min edge); ESRGAN-style factor
    MinEdge(u32),                        // SeedVR2 `resolution: int` (target for min(w,h))
    Resolution { width: u32, height: u32 },
}
```

`TransformCapabilities`: supported `TargetSize` modes, max scale, `is_diffusion` (uses seed),
`supports_strength`, quantization, `mac_only`. Same `LoadSpec` + `inventory` registry as
`Generator`, into a parallel transform registry.

**Scope now = image→image** (SeedVR2 is single-image in this port; the existing worker upscale
paths are image too). A video restorer would extend this later (a media-enum or `VideoTransform`)
— not designed speculatively. **Latent-space** resizing stays an *internal pipeline step*, not a
public model.

### 3.4 Adapters — LoRA **and** LoKr, multiples + mixed (core, already built)

Not optional, not per-model-to-decide: the **core supports LoRA + LoKr, multiples of each, and
mixed stacks** — and it's already implemented. `AdaptableLinear` holds a `Vec` of mixed adapters;
the loaders stack a new adapter onto whatever is already installed (sc-2339 / sc-2343). The
request expresses it directly:

```rust
pub struct AdapterSpec { pub path: PathBuf, pub scale: f32, pub kind: AdapterKind /* Lora | Lokr */ }
// request.adapters: Vec<AdapterSpec>   → multiples + mixed by construction
```

The only *per-model* question is whether a model's blocks expose adapter **injection points**
(e.g. InstantID originally had none) — captured by `Capabilities.supports_lora/lokr`, not by any
doubt about framework support.

---

## 4. Loading & registration (the "DI" equivalent)

Construction is per-model (weight layouts differ); discovery is uniform.

```rust
pub struct LoadSpec { pub weights: WeightsSource, pub quantize: Option<Quant>, pub dtype: Dtype, pub device: Device }

pub struct ModelRegistration {                 // ≈ services.AddKeyedSingleton<IGenerator>("id", …)
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>, GenerationError>,
}
inventory::submit! { ModelRegistration { descriptor: z_image::descriptor, load: z_image::load } }
```

`mlx-gen` exposes `registry()` and `load(model_id, &LoadSpec) -> Box<dyn Generator>` via an
`inventory`/`linkme` link-time collection: **a provider crate self-registers just by being
linked** — `mlx-gen` has no central match to edit (additive). A third party linking one provider
sees exactly one registration. (`Transform`s register the same way into a parallel registry.)
Maps onto the worker's `payload.model` → `MODEL_TARGETS` → load.

> **Linkage nuance (verified in the z-image split):** "linked" means the provider's objects are
> actually pulled into the final binary. A dependency that is declared but *never referenced*
> can have its link-section statics dropped by the linker, so its `inventory::submit!` never
> runs. A consumer that uses the provider (constructs requests, names a type, etc.) links it
> automatically; to depend on a provider purely for its registration side-effect, force the link
> with `use mlx_gen_z_image as _;`. This is the "the DI container has to know about the assembly"
> detail — in Rust it's a link-time, not runtime, fact.

---

## 5. How consumers use it

**SceneWorks worker** (all models): depend on every provider crate → registry is populated → on a
job, map the `ImageRequest`/`VideoRequest` payload → `GenerationRequest`,
`load(payload.model, spec)`, `generate`, map `GenerationOutput` → asset writes.

**Third party** (one model):
```rust
let model = mlx_gen_z_image::load(&LoadSpec { weights, quantize: Some(Quant::Q8), .. })?;
let out = model.generate(&GenerationRequest { prompt: "a fox".into(), ..Default::default() }, &mut |_p| {})?;
```

---

## 6. Adding a new model (additive)

1. `cargo new --lib mlx-gen-<x>`, depend on `mlx-gen`.
2. Build with `mlx-gen`'s `nn`/`weights`/`quant`/`tokenizer`/`adapters` primitives.
3. `impl Generator` (or `Transform`) + `descriptor()`/`Capabilities` + `load()`;
   `inventory::submit!` it.

No edits to `mlx-gen` or any other provider crate.

---

## 7. Where the contracts strain — settle per model

- **Upscalers / SeedVR2:** `Transform`, not `Generator` (no prompt) — image→image, shape locked
  in §3.3. A video restorer or latent-space model, if one lands, extends it then.
- **Conditioning variety:** single-ref img2img vs Qwen multi-ref edit vs Redux refs+strengths vs
  ControlNet+scale vs Depth vs Mask → the `Conditioning` enum, gated by `Capabilities`.
- **Guidance-distilled vs CFG:** Z-Image-turbo (no guidance) vs FLUX/Qwen → `supports_guidance` +
  `validate()`.
- **Video:** modes (i2v, first-last, extend, bridge, replace-person) + audio (LTX) → `video_mode`
  + source-clip handles; `GenerationOutput::Video { audio }`.
- **Quantization at load:** `LoadSpec.quantize` + a capability for which models support Q4/Q8.

---

## 8. Per-model fitment — checklist + vetted results

A model "fits" if it implements `Generator`/`Transform` with no contract change; else note the
extension (new `Conditioning` variant, new modality, `Transform` shape change). The template is
below; the filled pass is §8.1 and the verdict §8.2.

| Field | Notes |
|---|---|
| Model id / family / variants | |
| Contract | Generator / Transform |
| Modality | Image / Video / Both / (Transform: media→media) |
| Required + accepted inputs | prompt? neg? refs? control? depth? mask? init+strength? |
| Guidance / true-CFG | |
| Conditioning variants used | Reference / MultiReference / ReduxRefs / Control / Depth / Mask |
| Samplers / schedulers (+ sigma shift) | |
| Adapter injection points wired? | LoRA / LoKr seams present in the model's blocks |
| Quantization | Q4 / Q8 / none |
| Output | images / video(+audio) / transformed media |
| Special construction | KV-cache, dual text encoders, multi-stage, … |
| **Verdict** | Fits as-is / needs `<extension>` |

### 8.1 Vetted results (read-only pass against the fork)

**Scope rule discovered:** `mlx-gen` is MLX-only, so it only has to fit **MLX-native** models.
Torch/diffusers models stay in the Python worker and are **out of scope** for the contract.

MLX-native candidates:

| Model (family) | Contract | Modality | Conditioning used | Guidance | Verdict |
|---|---|---|---|---|---|
| **Z-Image-turbo** | Generator | Image | Reference (img2img) | optional (distilled) | ✅ fits as-is |
| **Z-Image-ControlNet** | Generator | Image | Control{scale} (`control_context_scale`) | optional | ✅ fits |
| **FLUX.1** | Generator | Image | Reference (img2img) | 4.0 | ✅ fits |
| **FLUX.1-Depth** | Generator | Image | Depth | 4.0 | ✅ fits |
| **FLUX.1-Redux** | Generator | Image | ReduxRefs (refs+strengths) | 4.0 | ✅ fits |
| **FLUX.1-ControlNet** | Generator | Image | Control; no neg/img2img (caps) | 4.0 | ✅ fits |
| **FLUX.2-klein 9b / 9b-kv** | Generator | Image | Reference; kv-cache = load/caps flag | 1.0 | ✅ fits |
| **Qwen-Image** | Generator | Image | Reference (img2img) | 4.0 | ✅ fits |
| **Qwen-Image-Edit** | Generator | Image | MultiReference (image list, no strength) | 4.0 | ✅ fits |
| **FIBO** | Generator | Image | Reference (img2img) | 4.0 | ✅ fits¹ |
| **FIBO-Edit** | Generator | Image | Mask (image+mask) | 4.0 | ✅ fits |
| **SDXL-MLX** (sc-1975) | Generator | Image | none (txt2img v1) | ~7.0 CFG | ✅ fits (trivially) |
| **SeedVR2** | **Transform** | Image→Image | — (`TargetSize`, softness, seed, 1-step) | n/a | ✅ fits (contract designed around it) |
| **Wan2.2** | Generator | Video | Reference/MultiReference/Control + video clips | 5.0 (+MoE 2nd) | ⚠️ fits w/ extensions² |
| **LTX-2.3** | Generator | Video **+ audio** | Reference/MultiReference + video clips | per-modality guiders | ⚠️ fits w/ extensions² |

¹ FIBO uses a JSON-structured prompt + SmolLM3 encoder — that's a **model-internal** concern; the
contract still takes a `String` prompt.
² See §8.2.

**Out of scope — torch/diffusers, NOT MLX-native (stay in the Python worker):** InstantID, Kolors,
PuLID-FLUX, SenseNova-U1, Chroma, SDXL-diffusers, FLUX.1-dev (diffusers path).

### 8.2 Findings & verdict

**The `Generator` / `Transform` / `Conditioning` contract holds across the entire MLX-native
scope.** Concretely:
- **All image generators fit as-is.** The `Conditioning` variants
  Reference / MultiReference / ReduxRefs / Control / Depth / Mask exactly cover what the image
  families need (img2img, Qwen multi-ref edit, FLUX Redux/Depth/ControlNet, Z-Image-ControlNet,
  FIBO-Edit mask). Guidance-distilled vs CFG is handled by `supports_guidance` + `validate()`.
- **SeedVR2 fits `Transform`** — the contract was designed around it.
- **Video (Wan2.2, LTX-2.3) fit the `Generator` shape but need two *additive* extensions, taken
  at the video port (follow-on) — neither touches the core or any image model:**
  1. a **`Conditioning::VideoClip { clips: Vec<(MediaRef, f32)> }`** variant for
     extend-clip / video-bridge / replace-person;
  2. **structured guider params** for the video families (LTX's per-modality video+audio guiders;
     Wan A14B's `guidance_scale_2` + `boundary_ratio` + high/low-noise dual-LoRA sets) — a small
     typed `video` block beyond the scalar `guidance`.
  Frame alignment (Wan 4n+1, LTX 8n+1) is internal per-model snapping in `validate()`, not a
  contract change. `GenerationOutput::Video { audio: Option<AudioTrack> }` already handles
  LTX (always `Some`) vs Wan (`None`) — no change needed.
- **No image-scope model bends the contract.** The architecture is validated for the build order.

---

## 9. Decisions

**Locked:**
- `mlx-gen` *is* the core (abstractions + runtime); models are provider crates depending on it.
  No facade.
- Categorize by interaction: one **`Generator`** for all prompt→media (multi-modal via output
  enum + `modality`); a separate **`Transform`** for non-prompt image→image, designed around
  SeedVR2 (image→image, §3.3).
- Registry via `inventory`/`linkme` link-time auto-registration (the DI-container equivalent).
- Adapters: LoRA + LoKr, multiples + mixed, are core + already built; per-model only asks whether
  injection points are wired.
- `Conditioning` is a **typed enum**.
- `GenerationRequest` / `TransformRequest` are single **`Default`-able structs** (no builders).
- **Scope: MLX-native models only.** Torch/diffusers models (InstantID, Kolors, PuLID, SenseNova,
  Chroma, SDXL-diffusers, FLUX.1-dev) are out of `mlx-gen` and stay in the Python worker (§8.1).
- **Contract validated** by the §8 fitment pass: all image generators + SeedVR2 fit as-is.
- **Migration order (Michael):** split out **`mlx-gen-z-image`** first, then **`mlx-gen-qwen-image`**,
  then build the others. (Extract shared `nn` → `mlx-gen` core as part of the z-image split.)

**Known additive extensions (no core change; taken when that work lands):**
- Video port: a `Conditioning::VideoClip` variant + a typed video/audio guider block (§8.2).
- Video `Transform` (media-enum vs `VideoTransform`) — only if a video restorer is ported.
```
