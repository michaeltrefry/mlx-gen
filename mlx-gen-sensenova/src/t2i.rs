//! Text-to-image generation (sc-3188) — the `t2i_generate` spine end to end.
//!
//! Ports `modeling_neo_chat.py::t2i_generate` for the dense 8B-MoT checkpoint. The flow:
//!
//! 1. Build the `neo1_0` query ([`build_neo1_query`] + [`SYSTEM_MESSAGE_FOR_GEN`] + the think
//!    sentinel), tokenize, and **prefill** it into a KV cache on the understanding path
//!    ([`Qwen3Backbone::forward_cached`] append). With CFG (`cfg_scale > 1`) a second, *uncondition*
//!    prefix (`<img>` after an empty prompt) is prefilled into its own cache.
//! 2. (think-mode) run the [`Qwen3Backbone::generate_think`] rollout, extending the cache and
//!    placing the image block after the appended `\n\n<img>`.
//! 3. **Denoise** for `num_steps` over the standard flow-matching schedule
//!    ([`apply_time_schedule`]): each step embeds the current noisy image through the gen-path
//!    [`NeoVisionEmbedder`] (channel-first patches) + the timestep (and noise-scale) embedding, runs
//!    the **generation** path over `[cached prefix ++ image block]` via `forward_cached`
//!    **use-only** (`update_cache=False`), maps the image hidden states through the [`FmHead`] to a
//!    patch latent `x_pred`, forms the [`velocity`], and takes an [`euler_step`]. CFG blends the
//!    condition/uncondition velocities ([`CfgNorm`] variants).
//! 4. [`unpatchify`] the final latent → RGB `[1, 3, H, W]`.
//!
//! The pixel path is the `fm_head` → unpatchify (`use_pixel_head = false`); the conv decoders and the
//! dynamic-μ schedule are dead code for this checkpoint.

use mlx_rs::ops::{add, divide, matmul, minimum, multiply, subtract, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Progress, Result};

use crate::config::NeoChatConfig;
use crate::fm::{
    apply_time_schedule, euler_step, patchify, patchify_channel_first, unpatchify, velocity,
    FmHead, TimestepEmbedder,
};
use crate::qwen3::{KvCache, Path, Qwen3Backbone};
use crate::runtime::{argmax, Sampler, SplitMix64, SPLITMIX64_INCREMENT};
use crate::text::{build_neo1_query, image_indexes, text_indexes, tokens, SYSTEM_MESSAGE_FOR_GEN};
use crate::vision::NeoVisionEmbedder;

/// Per-step cancellation + progress for the denoise loops, matching the SDXL family's `&CancelFlag` +
/// `on_progress` threading. `None` ⇒ uncancellable with no progress (diagnostic/parity callers); the
/// production [`SenseNova`](crate::SenseNova) path passes the request's flag and the worker's progress
/// callback so a multi-minute 8B run is cancellable and reports **denoise steps**, not the image index
/// (F-128).
pub struct StepReporter<'a> {
    cancel: &'a CancelFlag,
    on_progress: &'a mut dyn FnMut(Progress),
}

impl<'a> StepReporter<'a> {
    pub fn new(cancel: &'a CancelFlag, on_progress: &'a mut dyn FnMut(Progress)) -> Self {
        Self {
            cancel,
            on_progress,
        }
    }

    /// Abort with a typed error if the request was cancelled (checked before each denoise step).
    fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(Error::Msg("sensenova: generation cancelled".into()));
        }
        Ok(())
    }

    /// Report one completed denoise step (`current` is 1-based).
    fn step(&mut self, current: usize, total: usize) {
        (self.on_progress)(Progress::Step {
            current: current as u32,
            total: total as u32,
        });
    }
}

/// Classifier-free-guidance velocity-blend normalisation (`t2i_generate`'s `cfg_norm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CfgNorm {
    /// Plain blend `v_uncond + cfg·(v_cond − v_uncond)`.
    #[default]
    None,
    /// Rescale the blended velocity to the condition velocity's global norm.
    Global,
    /// Per-token rescale to the condition velocity's per-token norm.
    Channel,
    /// CFG-Zero* (optimised-scale uncondition + step-0 zeroing).
    CfgZeroStar,
}

/// Knobs for [`T2iModel::generate`] (the `t2i_generate` arguments).
#[derive(Clone, Copy, Debug)]
pub struct T2iOptions {
    pub cfg_scale: f32,
    /// Image-guidance scale for it2i (`img_cfg_scale`): edit ≈ 1.0, character ≈ 1.5. Unused by T2I.
    pub img_cfg_scale: f32,
    pub cfg_norm: CfgNorm,
    pub cfg_interval: (f32, f32),
    pub num_steps: usize,
    pub timestep_shift: f32,
    pub enable_timestep_shift: bool,
    pub t_eps: f32,
    pub seed: u64,
    pub think_mode: bool,
    pub max_think_tokens: usize,
}

impl Default for T2iOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            img_cfg_scale: 1.0,
            cfg_norm: CfgNorm::None,
            cfg_interval: (0.0, 1.0),
            num_steps: 30,
            timestep_shift: 1.0,
            enable_timestep_shift: true,
            t_eps: 0.02,
            seed: 0,
            think_mode: false,
            max_think_tokens: 1024,
        }
    }
}

/// The result of a [`T2iModel::interleave_gen`] run: the composed text (with `<image>`
/// placeholders where images were generated) and the generated images in order.
pub struct InterleaveOutput {
    pub text: String,
    pub images: Vec<Array>,
}

/// The interleave resolution buckets (`examples/interleave/inference.py::SUPPORTED_RESOLUTIONS`) —
/// `(width, height)` per aspect ratio. Document Studio picks one of these. Default `"16:9"`.
pub const INTERLEAVE_RESOLUTIONS: &[(&str, (i32, i32))] = &[
    ("1:1", (1536, 1536)),
    ("16:9", (2048, 1152)),
    ("9:16", (1152, 2048)),
    ("3:2", (1888, 1248)),
    ("2:3", (1248, 1888)),
    ("4:3", (1760, 1312)),
    ("3:4", (1312, 1760)),
    ("1:2", (1088, 2144)),
    ("2:1", (2144, 1088)),
    ("1:3", (864, 2592)),
    ("3:1", (2592, 864)),
];

/// Look up an interleave resolution bucket by aspect-ratio key (e.g. `"16:9"`).
pub fn interleave_resolution_for(ratio: &str) -> Option<(i32, i32)> {
    INTERLEAVE_RESOLUTIONS
        .iter()
        .find(|(r, _)| *r == ratio)
        .map(|(_, wh)| *wh)
}

/// The result of a [`T2iModel::generate`] run.
pub struct T2iOutput {
    /// The generated image `[1, 3, H, W]` (model space, roughly `[-1, 1]`).
    pub image: Array,
    /// The decoded think-block text, when `think_mode` was set.
    pub think_text: Option<String>,
}

/// The T2I model: the backbone plus the flow-matching generation modules.
pub struct T2iModel {
    backbone: Qwen3Backbone,
    gen_vision: NeoVisionEmbedder,
    /// The **understanding**-path vision embedder (`vision_model.embeddings`) used to embed source /
    /// reference images for it2i (sc-3189). `None` for fixtures that omit `vision_model.*`.
    und_vision: Option<NeoVisionEmbedder>,
    fm_head: FmHead,
    timestep_embedder: TimestepEmbedder,
    noise_scale_embedder: Option<TimestepEmbedder>,
    patch_size: i32,
    merge_size: i32,
    noise_scale: f32,
    noise_scale_mode: String,
    noise_scale_base_image_seq_len: f32,
    noise_scale_max_value: f32,
    /// `<IMG_CONTEXT>` / `<img>` / `</img>` ids (the checkpoint constants; overridable for tiny test
    /// fixtures whose vocab can't hold the real ids).
    img_context_id: i32,
    img_start_id: i32,
    img_end_id: i32,
}

impl T2iModel {
    /// Build from a loaded checkpoint (`language_model.*` + `fm_modules.*`).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig) -> Result<Self> {
        let noise_scale_embedder = if cfg.add_noise_scale_embedding {
            Some(TimestepEmbedder::from_weights(
                w,
                "fm_modules.noise_scale_embedder",
            )?)
        } else {
            None
        };
        // The understanding-path vision embedder is only needed for it2i; gate on its presence so
        // T2I-only fixtures (no `vision_model.*`) still load.
        let und_vision = if w
            .get("vision_model.embeddings.patch_embedding.weight")
            .is_some()
        {
            Some(NeoVisionEmbedder::from_weights(
                w,
                cfg,
                "vision_model.embeddings",
            )?)
        } else {
            None
        };
        Ok(Self {
            backbone: Qwen3Backbone::from_weights(w, cfg, "language_model")?,
            gen_vision: NeoVisionEmbedder::from_weights(
                w,
                cfg,
                "fm_modules.vision_model_mot_gen.embeddings",
            )?,
            und_vision,
            fm_head: FmHead::from_weights(w, "fm_modules.fm_head")?,
            timestep_embedder: TimestepEmbedder::from_weights(w, "fm_modules.timestep_embedder")?,
            noise_scale_embedder,
            patch_size: cfg.patch_size as i32,
            merge_size: (1.0 / cfg.downsample_ratio).round() as i32,
            noise_scale: cfg.noise_scale,
            noise_scale_mode: cfg.noise_scale_mode.clone(),
            noise_scale_base_image_seq_len: cfg.noise_scale_base_image_seq_len as f32,
            noise_scale_max_value: cfg.noise_scale_max_value,
            img_context_id: tokens::IMG_CONTEXT,
            img_start_id: tokens::IMG_START,
            img_end_id: tokens::IMG_END,
        })
    }

    /// Quantize the backbone decoder stack to Q4/Q8 (sc-3193) — the bulk of the 8B params
    /// (attention projections + SwiGLU on both paths). The vision embedders, FM head, and
    /// timestep/noise embedders stay dense (small; precision-sensitive flow-matching head).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.backbone.quantize(bits)
    }

    /// Merge the 8-step distill LoRA (sc-3192) for the `sensenova_u1_8b_fast` variant: the backbone
    /// generation-path projections (`7 · layers`) + the two FM-head Linears. The understanding path,
    /// vision embedders, and timestep/noise embedders are untouched (the LoRA carries no targets for
    /// them). Must run on the dense model **before** [`T2iModel::quantize`] — the merge seam errors
    /// on a quantized base, matching the reference which merges into the dense weight pre-quant.
    /// Returns the total number of Linears merged (so the loader can assert full coverage).
    pub fn merge_distill_lora(&mut self, lora: &Weights) -> Result<usize> {
        let n = self.backbone.merge_distill_lora(lora, "language_model")?;
        Ok(n + self
            .fm_head
            .merge_distill_lora(lora, "fm_modules.fm_head")?)
    }

    /// Override the `<IMG_CONTEXT>` / `<img>` / `</img>` token ids (for tiny test fixtures whose
    /// vocab cannot hold the real checkpoint ids). Production callers never need this.
    pub fn with_image_token_ids(
        mut self,
        img_context_id: i32,
        img_start_id: i32,
        img_end_id: i32,
    ) -> Self {
        self.img_context_id = img_context_id;
        self.img_start_id = img_start_id;
        self.img_end_id = img_end_id;
        self
    }

    /// The resolution-mode noise scale for a `grid_h × grid_w` patch grid (the `t2i_generate`
    /// formula). For non-resolution modes the bare `noise_scale` is used; both are clamped to
    /// `noise_scale_max_value`.
    fn noise_scale_for(&self, grid_h: i32, grid_w: i32) -> f32 {
        let mut scale = self.noise_scale;
        if matches!(
            self.noise_scale_mode.as_str(),
            "resolution" | "dynamic" | "dynamic_sqrt"
        ) {
            let seq = (grid_h * grid_w) as f32 / (self.merge_size * self.merge_size) as f32;
            scale = (seq / self.noise_scale_base_image_seq_len).sqrt() * self.noise_scale;
            if self.noise_scale_mode == "dynamic_sqrt" {
                scale = scale.sqrt();
            }
        }
        scale.min(self.noise_scale_max_value)
    }

    /// Run the gen-path velocity prediction for one diffusion step against a prefilled cache
    /// (`_t2i_predict_v`): `forward_cached` (Gen path, use-only) over the image block, `fm_head` →
    /// `x_pred`, then the flow-matching velocity. `image_embeds` is the vision+timestep conditioned
    /// image block `[1, L, hidden]`; `text_len` is the prefix length the block sits after.
    #[allow(clippy::too_many_arguments)]
    fn predict_v(
        &self,
        image_embeds: &Array,
        token_h: i32,
        token_w: i32,
        text_len: usize,
        cache: &mut KvCache,
        z: &Array,
        t: f32,
        t_eps: f32,
    ) -> Result<Array> {
        let (it, ih, iw) = image_indexes(token_h as usize, token_w as usize, text_len);
        let hidden =
            self.backbone
                .forward_cached(image_embeds, &it, &ih, &iw, Path::Gen, cache, false)?;
        let x_pred = self.fm_head.forward(&hidden)?;
        velocity(&x_pred, z, t, t_eps)
    }

    /// Prefill a text query into a fresh cache on the understanding path. Returns the cache, the
    /// last-position logits (for think-mode), and the prefix token length.
    fn prefill(&self, ids: &[i32]) -> Result<(KvCache, Array, usize)> {
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.backbone.embed(&ids_arr)?;
        let (t, h, wid) = text_indexes(ids.len());
        let mut cache = self.backbone.new_cache();
        let hidden =
            self.backbone
                .forward_cached(&embeds, &t, &h, &wid, Path::Und, &mut cache, true)?;
        // Only the last position's logits are kept, so slice the hidden state to `[1, 1, 4096]`
        // *before* `lm_head` — applying it over the whole `[1, S, 4096]` prefix would materialize an
        // `[1, S, vocab]` (~GB) tensor and an `S×4096×vocab` matmul just to drop all but one row (F-129).
        let last_hidden = hidden.take_axis(Array::from_slice(&[n - 1], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        let last = logits.reshape(&[vocab])?;
        Ok((cache, last, ids.len()))
    }

    /// Generate an image for `prompt` at `width × height` (both multiples of `patch·merge = 32`).
    /// `init_noise`, when supplied, is a standard-normal tensor `[1, 3, H, W]` used in place of
    /// fresh sampling (for cross-build parity); it is scaled by the resolution-mode `noise_scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<T2iOutput> {
        let (traj, think_text) =
            self.t2i_run(tokenizer, prompt, width, height, opts, init_noise, reporter)?;
        let image = traj.into_iter().last().expect("at least one step");
        Ok(T2iOutput { image, think_text })
    }

    /// Diagnostic (sc-3192 parity): the full per-step denoise **trajectory** for a T2I run — every
    /// step's decoded image, not just the final. Identical setup to [`generate`]; lets a test compare
    /// the port's trajectory to the reference step by step (e.g. to show that a distilled few-step
    /// run agrees early and diverges only on the big decisive final steps, i.e. compounding precision
    /// chaos rather than a per-step bug).
    pub fn t2i_trajectory(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
    ) -> Result<Vec<Array>> {
        Ok(self
            .t2i_run(tokenizer, prompt, width, height, opts, init_noise, None)?
            .0)
    }

    /// Shared T2I body: condition/uncondition prefixes (+ optional think rollout) and the denoise
    /// loop. Returns the per-step trajectory and any think text. [`generate`] keeps the last frame;
    /// [`t2i_trajectory`](Self::t2i_trajectory) returns them all.
    #[allow(clippy::too_many_arguments)]
    fn t2i_run(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<(Vec<Array>, Option<String>)> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova t2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }

        // ---- Condition prefix ----
        let think_sentinel = if opts.think_mode {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n<img>"
        };
        let query_cond = format!(
            "{}{}",
            build_neo1_query(prompt, SYSTEM_MESSAGE_FOR_GEN),
            think_sentinel
        );
        let ids_cond = tokenizer.encode_ids(&query_cond, true)?;
        let (mut cache_cond, last_logits, prefix_len) = self.prefill(&ids_cond)?;

        // think-mode: roll out the reasoning block, then append `\n\n<img>`.
        let mut think_text = None;
        let mut text_len = prefix_len;
        if opts.think_mode {
            let append_ids = tokenizer.encode_ids("\n\n<img>", false)?;
            let roll = self.backbone.generate_think(
                last_logits.as_slice::<f32>(),
                &mut cache_cond,
                (prefix_len - 1) as i32,
                tokens::THINK_END,
                tokens::IM_END,
                &append_ids,
                opts.max_think_tokens,
            )?;
            let ids_u32: Vec<u32> = roll.think_token_ids.iter().map(|&i| i as u32).collect();
            think_text = Some(tokenizer.decode(&ids_u32, false)?);
            text_len = (roll.t_idx + 1) as usize;
        }

        // ---- Uncondition prefix (CFG) ----
        let needs_cfg = opts.cfg_scale > 1.0;
        let mut cache_uncond = None;
        if needs_cfg {
            let query_uncond = format!("{}<img>", build_neo1_query("", ""));
            let ids_uncond = tokenizer.encode_ids(&query_uncond, true)?;
            let (cache, _, plen) = self.prefill(&ids_uncond)?;
            cache_uncond = Some((cache, plen));
        }

        let base_noise = match init_noise {
            Some(n) => n.as_dtype(Dtype::Float32)?,
            None => gaussian(&[1, 3, height, width], opts.seed)?,
        };
        let cond_u = cache_uncond.as_mut().map(|(c, l)| (c, *l));
        let traj = self.denoise(
            &mut cache_cond,
            text_len,
            cond_u,
            width,
            height,
            &base_noise,
            opts,
            reporter,
        )?;
        Ok((traj, think_text))
    }

    /// Prefill `ids` into a fresh understanding-path cache; returns the cache and prefix length.
    /// Exposed for tests/callers that drive [`T2iModel::denoise`] with an explicit prefix.
    pub fn prefill_ids(&self, ids: &[i32]) -> Result<(KvCache, usize)> {
        let (cache, _, len) = self.prefill(ids)?;
        Ok((cache, len))
    }

    /// The flow-matching denoise loop. `cache_cond` (and the optional `(cache_uncond, text_len)` for
    /// CFG) are prefilled understanding-path caches; `base_noise` is a standard-normal `[1,3,H,W]`
    /// tensor (scaled here by the resolution-mode noise scale). Returns the per-step image
    /// trajectory `[1,3,H,W]` (the last entry is the final image).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        cache_cond: &mut KvCache,
        text_len: usize,
        mut cache_uncond: Option<(&mut KvCache, usize)>,
        width: i32,
        height: i32,
        base_noise: &Array,
        opts: &T2iOptions,
        mut reporter: Option<StepReporter>,
    ) -> Result<Vec<Array>> {
        let cell = self.patch_size * self.merge_size;
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let mut image = multiply(
            &base_noise.as_dtype(Dtype::Float32)?,
            Array::from_f32(noise_scale),
        )?;

        let steps = opts.num_steps;
        // `steps == 0` yields an empty trajectory (and a 0/0 NaN schedule), so the callers' final
        // `.last().expect("at least one step")` would panic. Surface a typed error instead (F-125);
        // the registered `Generator` rejects this upstream, but `interleave_gen`/`vqa` reach here too.
        if steps == 0 {
            return Err(Error::Msg("sensenova: num_steps must be >= 1".into()));
        }
        let lin: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
        let lin_arr = Array::from_slice(&lin, &[(steps + 1) as i32]);
        let ts_arr = if opts.enable_timestep_shift {
            apply_time_schedule(&lin_arr, opts.timestep_shift)?
        } else {
            lin_arr
        };
        let timesteps = ts_arr.as_slice::<f32>().to_vec();

        // Constant noise-scale conditioning token (added to every step's timestep embedding).
        let noise_embed = if let Some(emb) = &self.noise_scale_embedder {
            let ns = vec![noise_scale / self.noise_scale_max_value; l as usize];
            Some(
                emb.forward(&Array::from_slice(&ns, &[l]))?
                    .reshape(&[1, l, -1])?,
            )
        } else {
            None
        };

        let needs_cfg = opts.cfg_scale > 1.0 && cache_uncond.is_some();
        let mut traj = Vec::with_capacity(steps);
        for i in 0..steps {
            if let Some(r) = reporter.as_ref() {
                r.check_cancel()?;
            }
            let t = timesteps[i];
            let t_next = timesteps[i + 1];

            let (z, cond) = self.step_cond_embeds(&image, grid_h, grid_w, l, t, &noise_embed)?;

            let v_cond = self.predict_v(
                &cond, token_h, token_w, text_len, cache_cond, &z, t, opts.t_eps,
            )?;

            // CFG-interval gate for the T2I path: **inclusive** both ends, a faithful port of the
            // reference `modeling_neo_chat.py:1799`
            // `if t >= cfg_interval[0] and t <= cfg_interval[1] and cfg_scale > 1`. The it2i/edit path
            // (`it2i_denoise`) deliberately uses a DIFFERENT gate — exclusive `(i0, i1)` plus an
            // `i0 == 0` always-on override — because ITS reference (lines 901/1246/1557) does; see
            // there. At the default `(0.0, 1.0)` both are always-on, so the divergence only surfaces
            // for a custom `cfg_interval` (F-130).
            let v_pred = if needs_cfg && t >= opts.cfg_interval.0 && t <= opts.cfg_interval.1 {
                let (cache_u, tlu) = cache_uncond.as_mut().unwrap();
                let v_uncond =
                    self.predict_v(&cond, token_h, token_w, *tlu, cache_u, &z, t, opts.t_eps)?;
                cfg_blend(&v_cond, &v_uncond, opts.cfg_scale, opts.cfg_norm, i)?
            } else {
                v_cond
            };

            image = unpatchify(
                &euler_step(&v_pred, &z, t, t_next)?,
                cell,
                Some(token_h),
                Some(token_w),
            )?;
            traj.push(image.clone());
            if let Some(r) = reporter.as_mut() {
                r.step(i + 1, steps);
            }
        }
        Ok(traj)
    }

    /// Build one denoise step's latent `z` (channel-last patchify at `cell`) and the conditioned
    /// image block `cond = gen_vision(image) + timestep_embed(t) [+ noise_scale_embed]` `[1, L,
    /// hidden]`. Shared by the T2I and it2i denoise loops.
    fn step_cond_embeds(
        &self,
        image: &Array,
        grid_h: i32,
        grid_w: i32,
        l: i32,
        t: f32,
        noise_embed: &Option<Array>,
    ) -> Result<(Array, Array)> {
        let cell = self.patch_size * self.merge_size;
        let z = patchify(image, cell)?;
        let image_input =
            patchify_channel_first(image, self.patch_size)?.reshape(&[grid_h * grid_w, -1])?;
        let vis = self
            .gen_vision
            .forward(&image_input, &[(grid_h as usize, grid_w as usize)])?
            .reshape(&[1, l, -1])?;
        let t_tok = self
            .timestep_embedder
            .forward(&Array::from_slice(&vec![t; l as usize], &[l]))?
            .reshape(&[1, l, -1])?;
        let mut cond = add(&vis, &t_tok)?;
        if let Some(ne) = noise_embed {
            cond = add(&cond, ne)?;
        }
        Ok((z, cond))
    }

    // ===================== it2i (instruction edit + Character Studio) =====================

    /// ImageNet-normalise a source RGB image `[3, H, W]` (f32 in `[0, 1]`) and channel-first
    /// patchify to `[grid_h·grid_w, 3·ps²]` for the understanding vision embedder. Returns the
    /// patches and the `(grid_h, grid_w)` patch grid. `H`/`W` must be multiples of `patch·merge`
    /// (use [`smart_resize`] upstream). Mirrors the reference `load_image_native` transform
    /// (ToTensor + Normalize) + `preprocess_pixel_values`.
    pub fn preprocess_image(&self, rgb: &Array) -> Result<(Array, (i32, i32))> {
        let sh = rgb.shape();
        if sh.len() != 3 || sh[0] != 3 {
            return Err(Error::Msg(format!(
                "expected source image [3,H,W], got {sh:?}"
            )));
        }
        let (h, w) = (sh[1], sh[2]);
        let cell = self.patch_size * self.merge_size;
        if h % cell != 0 || w % cell != 0 {
            return Err(Error::Msg(format!(
                "source image H/W must be multiples of {cell}, got {h}x{w}"
            )));
        }
        let mean = Array::from_slice(&[0.485f32, 0.456, 0.406], &[3, 1, 1]);
        let std = Array::from_slice(&[0.229f32, 0.224, 0.225], &[3, 1, 1]);
        let norm = divide(&subtract(&rgb.as_dtype(Dtype::Float32)?, &mean)?, &std)?;
        let (gh, gw) = (h / self.patch_size, w / self.patch_size);
        let patches = patchify_channel_first(&norm.reshape(&[1, 3, h, w])?, self.patch_size)?
            .reshape(&[gh * gw, -1])?;
        Ok((patches, (gh, gw)))
    }

    /// Understanding-path vision features for source-image patches (diagnostic / it2i internals).
    pub fn und_vision_features(&self, pixel_values: &Array, grids: &[(i32, i32)]) -> Result<Array> {
        let und = self
            .und_vision
            .as_ref()
            .ok_or_else(|| Error::Msg("vision_model not loaded".into()))?;
        let g: Vec<(usize, usize)> = grids
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect();
        und.forward(pixel_values, &g)
    }

    /// (t, h, w) position rows for a prefix containing source-image blocks (the reference
    /// `get_thw_indexes`): text tokens advance temporal by one; an image-context block shares one
    /// temporal index and carries its **merged-grid** `(row, col)` as `(h, w)`. `grids` are the full
    /// patch grids `(grid_h, grid_w)` per image, in order.
    fn get_thw_indexes(&self, ids: &[i32], grids: &[(i32, i32)]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let n = ids.len();
        let mut t = Vec::with_capacity(n);
        let mut acc = 0i32;
        for i in 0..n {
            let shift = i32::from(i > 0 && ids[i - 1] == self.img_start_id);
            let not_img = i32::from(ids[i] != self.img_context_id);
            acc += shift + not_img;
            t.push(acc - 1);
        }
        // Merged-grid (row=y, col=x) coordinates, concatenated across images in order.
        let merge = self.merge_size;
        let mut abs = Vec::new();
        for &(gh, gw) in grids {
            let (mh, mw) = (gh / merge, gw / merge);
            for idx in 0..(mh * mw) {
                abs.push((idx / mw, idx % mw));
            }
        }
        let mut h = vec![0i32; n];
        let mut w = vec![0i32; n];
        let mut k = 0usize;
        for i in 0..n {
            if ids[i] == self.img_context_id {
                let (y, x) = abs[k];
                h[i] = y;
                w[i] = x;
                k += 1;
            }
        }
        (t, h, w)
    }

    /// Embed `ids` and splice the understanding vision features into the `<IMG_CONTEXT>` positions
    /// (the reference `_build_it2i_inputs`). Returns the prefix embeds `[1, S, hidden]` and its
    /// `(t, h, w)` rows. `pixel_values` is the concatenated per-image patch list; `grids` the full
    /// patch grids. Scatter is done with a one-hot selection matmul (no in-place index assignment).
    #[allow(clippy::type_complexity)]
    fn build_it2i_prefix(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(Array, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let s = ids.len();
        let ids_arr = Array::from_slice(ids, &[1, s as i32]);
        let mut embeds = self.backbone.embed(&ids_arr)?; // [1, S, H]
        let (t, h, w) = self.get_thw_indexes(ids, grids);

        if let Some(pv) = pixel_values {
            let und = self.und_vision.as_ref().ok_or_else(|| {
                Error::Msg("it2i needs the understanding vision embedder (vision_model.*)".into())
            })?;
            let grids_us: Vec<(usize, usize)> = grids
                .iter()
                .map(|&(a, b)| (a as usize, b as usize))
                .collect();
            let vit = und.forward(pv, &grids_us)?; // [n_ctx, H]
            let hidden = embeds.shape()[2];
            let dt = embeds.dtype();
            let ctx: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|(_, &id)| id == self.img_context_id)
                .map(|(i, _)| i)
                .collect();
            let n_ctx = vit.shape()[0];
            if ctx.len() as i32 != n_ctx {
                return Err(Error::Msg(format!(
                    "it2i: {} <IMG_CONTEXT> tokens but {n_ctx} vision tokens",
                    ctx.len()
                )));
            }
            // P [S, n_ctx] one-hot: row = sequence position, col = vision-token index.
            let mut p = vec![0f32; s * n_ctx as usize];
            let mut mask = vec![0f32; s];
            for (k, &pos) in ctx.iter().enumerate() {
                p[pos * n_ctx as usize + k] = 1.0;
                mask[pos] = 1.0;
            }
            let p_arr = Array::from_slice(&p, &[s as i32, n_ctx]).as_dtype(dt)?;
            let vit_full = matmul(&p_arr, &vit.as_dtype(dt)?)?; // [S, H], 0 off-context
            let keep_mask = Array::from_slice(
                &mask.iter().map(|m| 1.0 - m).collect::<Vec<_>>(),
                &[s as i32, 1],
            )
            .as_dtype(dt)?;
            let e2d = embeds.reshape(&[s as i32, hidden])?;
            embeds =
                add(&multiply(&e2d, &keep_mask)?, &vit_full)?.reshape(&[1, s as i32, hidden])?;
        }
        Ok((embeds, t, h, w))
    }

    /// Prefill a prepared prefix (embeds + positions) on the understanding path. Returns the cache,
    /// the last-position logits (for think-mode), and the image-block temporal index (`max(t) + 1`).
    fn prefill_prefix(
        &self,
        embeds: &Array,
        t: &[i32],
        h: &[i32],
        w: &[i32],
    ) -> Result<(KvCache, Array, usize)> {
        let mut cache = self.backbone.new_cache();
        let hidden = self
            .backbone
            .forward_cached(embeds, t, h, w, Path::Und, &mut cache, true)?;
        // Slice the last hidden row before `lm_head` — only its logits are used, and the prefix here
        // includes image-context blocks, so the full `[1, S, vocab]` projection is the worst case of
        // F-129 (an `S×4096×vocab` matmul + ~GB tensor) repeated per CFG cache.
        let s = t.len() as i32;
        let last_hidden = hidden.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?;
        let vocab = logits.shape()[2];
        let last = logits.reshape(&[vocab])?;
        let img_temporal = (*t.iter().max().unwrap_or(&0) + 1) as usize;
        Ok((cache, last, img_temporal))
    }

    /// The understanding-path prefill hidden states `[1, S, hidden]` for an it2i prefix (diagnostic).
    pub fn prefill_it2i_hidden(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<Array> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let mut cache = self.backbone.new_cache();
        self.backbone
            .forward_cached(&embeds, &t, &h, &w, Path::Und, &mut cache, true)
    }

    /// Build + prefill an it2i prefix from explicit token ids, source-image patches, and per-image
    /// patch grids (the tokenizer-free entry the parity test drives). Returns the cache and the
    /// image-block temporal index. `pixel_values` is the concatenated per-image patch list.
    pub fn prefill_it2i(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(KvCache, usize)> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let (cache, _, img_temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
        Ok((cache, img_temporal))
    }

    /// Build + prefill an it2i/VQA prefix (image-conditioned) and return the cache, the
    /// last-position logits, and the next-token temporal index (`max(t)` — decode starts at `+1`).
    /// The tokenizer-free entry the VQA path and its parity test share.
    pub fn prefill_it2i_logits(
        &self,
        ids: &[i32],
        pixel_values: Option<&Array>,
        grids: &[(i32, i32)],
    ) -> Result<(KvCache, Vec<f32>, usize)> {
        let (embeds, t, h, w) = self.build_it2i_prefix(ids, pixel_values, grids)?;
        let (cache, last, img_temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
        Ok((cache, last.as_slice::<f32>().to_vec(), img_temporal - 1))
    }

    /// Greedy/sampled understanding-path text decode from a prefilled cache (wraps
    /// [`Qwen3Backbone::generate`]). `first_logits` are the prefix's last-position logits; `t_idx`
    /// the prefix's max temporal index. Returns the generated token ids (stop ids excluded).
    pub fn decode_text(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: usize,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> Result<Vec<i32>> {
        self.backbone.generate(
            first_logits,
            cache,
            t_idx as i32,
            eos,
            max_new_tokens,
            sampler,
        )
    }

    /// Build the it2i condition prefix from a prompt + source images (replacing `<image>` markers
    /// with `<img><IMG_CONTEXT>×n</img>` per image, auto-prepending markers like `it2i_generate`).
    /// Returns the cache, last logits, image-block temporal index, and the per-image patch grids.
    fn build_it2i_query_ids(
        &self,
        tokenizer: &TextTokenizer,
        base_query: &str,
        grids: &[(i32, i32)],
    ) -> Result<Vec<i32>> {
        let mut query = base_query.to_string();
        for &(gh, gw) in grids {
            let n = (gh / self.merge_size) * (gw / self.merge_size);
            let block = format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n as usize));
            query = query.replacen("<image>", &block, 1);
        }
        tokenizer.encode_ids(&query, true)
    }

    /// Image-conditioned generation (`it2i_generate`): edit / Character-Studio reference. `images`
    /// are decoded source RGB tensors `[3, H, W]` in `[0, 1]` (smart-resized to multiples of
    /// `patch·merge`); `prompt` may contain `<image>` markers (auto-prepended otherwise). Dual
    /// guidance: `opts.cfg_scale` (text) + `opts.img_cfg_scale` (image; edit ≈ 1.0, character ≈
    /// 1.5). `init_noise` (optional) is a standard-normal `[1,3,H,W]` for cross-build parity.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn it2i_generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
        reporter: Option<StepReporter>,
    ) -> Result<T2iOutput> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova it2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }
        if images.is_empty() {
            return Err(Error::Msg("it2i requires at least one source image".into()));
        }

        // Preprocess source images → concatenated patches + per-image grids.
        let mut pv_parts = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for img in images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pv_refs: Vec<&Array> = pv_parts.iter().collect();
        let pixel_values = mlx_rs::ops::concatenate_axis(&pv_refs, 0)?;

        // Auto-prepend `<image>` markers when the prompt has fewer than the image count.
        let count = prompt.matches("<image>").count();
        let mut question = prompt.to_string();
        if images.len() > count {
            let extra = images.len() - count;
            let pre = if count == 0 && images.len() > 1 {
                (0..images.len())
                    .map(|i| format!("Image-{}:<image>\n", i + 1))
                    .collect::<String>()
            } else {
                "<image>\n".repeat(extra)
            };
            question = format!("{pre}{question}");
        }

        let think_sentinel = if opts.think_mode {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n<img>"
        };

        // Guidance plan (matches it2i_generate's needs_* flags).
        let needs_cfg = !(opts.cfg_scale == 1.0 && opts.img_cfg_scale == 1.0);
        let needs_img =
            needs_cfg && (opts.img_cfg_scale == 1.0 || opts.cfg_scale != opts.img_cfg_scale);
        let needs_uncond = needs_cfg && opts.img_cfg_scale != 1.0;

        // Condition: prompt + images.
        let cond_query = format!(
            "{}{}",
            build_neo1_query(&question, SYSTEM_MESSAGE_FOR_GEN),
            think_sentinel
        );
        let cond_ids = self.build_it2i_query_ids(tokenizer, &cond_query, &grids)?;
        let (cond_embeds, ct, ch, cw) =
            self.build_it2i_prefix(&cond_ids, Some(&pixel_values), &grids)?;
        let (mut cache_cond, last_logits, mut cond_temporal) =
            self.prefill_prefix(&cond_embeds, &ct, &ch, &cw)?;

        // think-mode rollout (extends cache + advances the image-block temporal index).
        let mut think_text = None;
        if opts.think_mode {
            let append_ids = tokenizer.encode_ids("\n\n<img>", false)?;
            let roll = self.backbone.generate_think(
                last_logits.as_slice::<f32>(),
                &mut cache_cond,
                (cond_temporal - 1) as i32,
                tokens::THINK_END,
                tokens::IM_END,
                &append_ids,
                opts.max_think_tokens,
            )?;
            let u32s: Vec<u32> = roll.think_token_ids.iter().map(|&i| i as u32).collect();
            think_text = Some(tokenizer.decode(&u32s, false)?);
            cond_temporal = (roll.t_idx + 1) as usize;
        }

        // Image-condition: images only, empty prompt.
        let mut cache_img = None;
        if needs_img {
            let q = format!(
                "{}<img>",
                build_neo1_query(&"<image>".repeat(images.len()), "")
            );
            let ids = self.build_it2i_query_ids(tokenizer, &q, &grids)?;
            let (embeds, t, h, w) = self.build_it2i_prefix(&ids, Some(&pixel_values), &grids)?;
            let (cache, _, temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
            cache_img = Some((cache, temporal));
        }

        // Uncondition: empty prompt, no images.
        let mut cache_uncond = None;
        if needs_uncond {
            let q = format!("{}<img>", build_neo1_query("", ""));
            let ids = tokenizer.encode_ids(&q, true)?;
            let (embeds, t, h, w) = self.build_it2i_prefix(&ids, None, &[])?;
            let (cache, _, temporal) = self.prefill_prefix(&embeds, &t, &h, &w)?;
            cache_uncond = Some((cache, temporal));
        }

        let base_noise = match init_noise {
            Some(n) => n.as_dtype(Dtype::Float32)?,
            None => gaussian(&[1, 3, height, width], opts.seed)?,
        };
        let img_ref = cache_img.as_mut().map(|(c, l)| (c, *l));
        let un_ref = cache_uncond.as_mut().map(|(c, l)| (c, *l));
        let traj = self.it2i_denoise(
            (&mut cache_cond, cond_temporal),
            img_ref,
            un_ref,
            width,
            height,
            &base_noise,
            opts,
            reporter,
        )?;
        let image = traj.into_iter().last().expect("at least one step");
        Ok(T2iOutput { image, think_text })
    }

    /// VQA / understanding (`chat` / `answer_question`): image(s) + question → understanding-path
    /// AR text generation → answer. The question prefix (empty system message, `<image>` markers
    /// auto-prepended) is built and prefilled exactly like the it2i condition prefix (vision features
    /// spliced into `<IMG_CONTEXT>`), then [`Qwen3Backbone::generate`] decodes to the `<|im_end|>`
    /// stop. `images` are decoded RGB `[3,H,W]` in `[0,1]` (sized to multiples of `patch·merge`);
    /// pass an empty slice for a text-only question. Returns the decoded answer (special tokens
    /// stripped, trimmed).
    pub fn vqa(
        &self,
        tokenizer: &TextTokenizer,
        question: &str,
        images: &[Array],
        max_new_tokens: usize,
        sampler: Sampler,
    ) -> Result<String> {
        // Preprocess any source images.
        let mut pv_parts = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for img in images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Array> = pv_parts.iter().collect();
            Some(mlx_rs::ops::concatenate_axis(&refs, 0)?)
        };

        // Auto-prepend `<image>` markers (the reference `chat` prepends one per missing image).
        let count = question.matches("<image>").count();
        let mut q = question.to_string();
        if images.len() > count {
            let pre = "<image>\n".repeat(images.len() - count);
            q = format!("{pre}{q}");
        }
        // VQA uses the neo1_0 default (empty) system message, not SYSTEM_MESSAGE_FOR_GEN. Prime an
        // empty `<think></think>` block so the model answers directly without a chain-of-thought —
        // matching the reference `chat(think=False)` (`modeling_neo_chat.py`:
        // `get_prompt() + '<think>\n\n</think>\n\n'`) and the engine's own no-think interleave path.
        // Without it the greedy decode spends its whole budget reasoning and never emits the answer.
        let base = format!("{}<think>\n\n</think>\n\n", build_neo1_query(&q, ""));
        let ids = if images.is_empty() {
            tokenizer.encode_ids(&base, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &base, &grids)?
        };

        let (mut cache, last_logits, t_idx) =
            self.prefill_it2i_logits(&ids, pixel_values.as_ref(), &grids)?;
        let tokens = self.decode_text(
            &last_logits,
            &mut cache,
            t_idx,
            &[tokens::IM_END],
            max_new_tokens,
            sampler,
        )?;
        let u32s: Vec<u32> = tokens.iter().map(|&i| i as u32).collect();
        Ok(tokenizer.decode(&u32s, true)?.trim().to_string())
    }

    /// Re-encode a generated image through the **understanding** vision embedder and append it (plus
    /// the `</img>` token) to a text cache, so subsequent text generation attends to it (the
    /// reference `interleave_gen`'s inner `append_image_to_cache`). The generated `image` `[1,3,H,W]`
    /// is mapped model-space→`[0,1]` (`·0.5+0.5`) then ImageNet-normalised. Image tokens take temporal
    /// `t_idx+1` with their merged-grid `(h,w)`; `</img>` takes `t_idx+2`. The block-causal mask makes
    /// image tokens attend bidirectionally to each other + all past but **not** to the `</img>`
    /// position, while `</img>` sees them — exactly what [`cached_block_mask`](crate::qwen3) yields
    /// for that temporal layout. Returns the next-token logits and the advanced `t_idx` (`+2`).
    /// Public so callers can compose custom interleave loops and so the parity test can drive it.
    pub fn append_generated_image(
        &self,
        image: &Array,
        token_h: i32,
        token_w: i32,
        t_idx: usize,
        cache: &mut KvCache,
    ) -> Result<(Vec<f32>, usize)> {
        let sh = image.shape();
        let (h, w) = (sh[2], sh[3]);
        let raw = add(
            &multiply(image, Array::from_f32(0.5))?,
            Array::from_f32(0.5),
        )?
        .reshape(&[3, h, w])?;
        let (patches, (gh, gw)) = self.preprocess_image(&raw)?;
        let vit = self.und_vision_features(&patches, &[(gh, gw)])?; // [n_img, hidden]
        let n_img = vit.shape()[0];
        let hidden = vit.shape()[1];
        let end = self
            .backbone
            .embed(&Array::from_slice(&[self.img_end_id], &[1, 1]))?
            .reshape(&[1, hidden])?;
        let embeds =
            mlx_rs::ops::concatenate_axis(&[&vit, &end], 0)?.reshape(&[1, n_img + 1, hidden])?;

        let ti = t_idx as i32;
        let mut t = vec![ti + 1; n_img as usize];
        t.push(ti + 2);
        let mut hh = Vec::with_capacity(n_img as usize + 1);
        let mut ww = Vec::with_capacity(n_img as usize + 1);
        for i in 0..n_img {
            hh.push(i / token_w);
            ww.push(i % token_w);
        }
        hh.push(0);
        ww.push(0);
        let hs = self
            .backbone
            .forward_cached(&embeds, &t, &hh, &ww, Path::Und, cache, true)?;
        // Same one-row-kept pattern as the prefill paths (F-129): slice the kept hidden row (the `end`
        // token at index `n_img`) before `lm_head` instead of projecting all `n_img + 1` rows.
        let last_hidden = hs.take_axis(Array::from_slice(&[n_img], &[1]), 1)?;
        let logits = self.backbone.lm_head(&last_hidden)?;
        let vocab = logits.shape()[2];
        let last = logits.reshape(&[vocab])?.as_slice::<f32>().to_vec();
        let _ = (token_h, token_w);
        Ok((last, t_idx + 2))
    }

    /// Interleaved text-image generation (`interleave_gen`) — the **Document Studio** deliverable. A
    /// single rollout that alternates understanding-path text generation and gen-path flow-matching
    /// image generation: text streams until the model emits `<img>`, an image is generated (3-cache
    /// CFG: condition / text-uncondition / image-uncondition) and re-encoded back into the text
    /// caches, then text resumes — all as **single-path** forwards over growing caches (the upstream
    /// mixed-token attention is never issued). `input_images` are optional source images;
    /// `system_message` is normally [`INTERLEAVE_SYSTEM_MESSAGE`]; `init_noises`, when supplied, are
    /// per-image standard-normal `[1,3,H,W]` tensors for cross-build parity. Returns the composed text
    /// (with `<image>` placeholders) and the generated images in order.
    #[allow(clippy::too_many_arguments)]
    pub fn interleave_gen(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        input_images: &[Array],
        width: i32,
        height: i32,
        opts: &T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        init_noises: Option<&[Array]>,
    ) -> Result<InterleaveOutput> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova interleave: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }
        let token_h = height / cell;
        let token_w = width / cell;

        // Source images (optional).
        let mut pv_parts = Vec::with_capacity(input_images.len());
        let mut grids = Vec::with_capacity(input_images.len());
        for img in input_images {
            let (p, g) = self.preprocess_image(img)?;
            pv_parts.push(p);
            grids.push(g);
        }
        let pixel_values = if pv_parts.is_empty() {
            None
        } else {
            let refs: Vec<&Array> = pv_parts.iter().collect();
            Some(mlx_rs::ops::concatenate_axis(&refs, 0)?)
        };

        // ---- Three prefixes / caches ----
        let mut cond_query = build_neo1_query(prompt, system_message);
        if !opts.think_mode {
            cond_query.push_str("<think>\n\n</think>\n\n");
        }
        let cond_ids = if input_images.is_empty() {
            tokenizer.encode_ids(&cond_query, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &cond_query, &grids)?
        };
        let (mut cache_cond, cond_logits, mut t_cond) =
            self.prefill_it2i_logits(&cond_ids, pixel_values.as_ref(), &grids)?;

        let tu_query = build_neo1_query(&"<image>".repeat(input_images.len()), "");
        let tu_ids = if input_images.is_empty() {
            tokenizer.encode_ids(&tu_query, true)?
        } else {
            self.build_it2i_query_ids(tokenizer, &tu_query, &grids)?
        };
        let (mut cache_tu, _, mut t_tu) =
            self.prefill_it2i_logits(&tu_ids, pixel_values.as_ref(), &grids)?;

        let iu_query = format!("{}<img>", build_neo1_query("", ""));
        let iu_ids = tokenizer.encode_ids(&iu_query, true)?;
        let (mut cache_iu, _, iu_max) = self.prefill_it2i_logits(&iu_ids, None, &[])?;

        let mut text = String::new();
        let mut images = Vec::new();
        let mut total_tokens = 0usize;
        let mut next = argmax(&cond_logits);

        loop {
            // ---- Text generation on the condition cache ----
            let mut gen_tokens = Vec::new();
            let mut hit_max = false;
            loop {
                if next == tokens::IM_END || next == self.img_start_id {
                    break;
                }
                gen_tokens.push(next);
                total_tokens += 1;
                let logits =
                    self.backbone
                        .decode_logits(next, (t_cond + 1) as i32, &mut cache_cond)?;
                t_cond += 1;
                next = argmax(&logits);
                if total_tokens >= max_new_tokens {
                    hit_max = true;
                    break;
                }
            }
            if !gen_tokens.is_empty() {
                let u32s: Vec<u32> = gen_tokens.iter().map(|&i| i as u32).collect();
                text.push_str(&tokenizer.decode(&u32s, true)?);
            }
            if next == tokens::IM_END || hit_max || images.len() >= max_images {
                break;
            }
            if next != self.img_start_id {
                break;
            }

            // ---- Image generation ----
            text.push_str("<image>");
            // Append `<img>` to the condition + text-uncondition caches.
            self.backbone
                .decode_logits(self.img_start_id, (t_cond + 1) as i32, &mut cache_cond)?;
            t_cond += 1;
            self.backbone
                .decode_logits(self.img_start_id, (t_tu + 1) as i32, &mut cache_tu)?;
            t_tu += 1;

            let base_noise = match init_noises {
                Some(ns) if images.len() < ns.len() => ns[images.len()].as_dtype(Dtype::Float32)?,
                _ => gaussian(
                    &[1, 3, height, width],
                    opts.seed.wrapping_add(images.len() as u64),
                )?,
            };
            let traj = self.it2i_denoise(
                (&mut cache_cond, t_cond + 1),
                Some((&mut cache_tu, t_tu + 1)),
                Some((&mut cache_iu, iu_max + 1)),
                width,
                height,
                &base_noise,
                opts,
                None,
            )?;
            let image = traj.into_iter().last().expect("at least one step");

            // Re-encode the generated image back into the condition + text-uncondition caches.
            let (cond_next, nt_cond) =
                self.append_generated_image(&image, token_h, token_w, t_cond, &mut cache_cond)?;
            t_cond = nt_cond;
            let (_, nt_tu) =
                self.append_generated_image(&image, token_h, token_w, t_tu, &mut cache_tu)?;
            t_tu = nt_tu;
            images.push(image);
            next = argmax(&cond_next);
        }

        Ok(InterleaveOutput { text, images })
    }

    /// The dual-guidance denoise loop (`it2i_generate`'s body). `cond`/`img`/`uncond` are
    /// `(cache, image-block temporal index)`; the per-step blend follows the reference's
    /// `cfg_scale`/`img_cfg_scale` cases, then optional `cfg_norm`. Returns the image trajectory.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn it2i_denoise(
        &self,
        cond: (&mut KvCache, usize),
        mut img: Option<(&mut KvCache, usize)>,
        mut uncond: Option<(&mut KvCache, usize)>,
        width: i32,
        height: i32,
        base_noise: &Array,
        opts: &T2iOptions,
        mut reporter: Option<StepReporter>,
    ) -> Result<Vec<Array>> {
        let cell = self.patch_size * self.merge_size;
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;
        let (cache_cond, cond_t) = cond;

        // `cfg_scale`/`img_cfg_scale` > 1 demand the corresponding caches; this is a `pub` method, so
        // an external caller can ask for guidance without supplying them. Fail fast with a typed error
        // (mirroring `it2i_generate`'s `needs_*` flags) instead of unwrap-panicking mid-loop (F-126).
        let (needs_img, needs_uncond) = it2i_cache_requirements(opts.cfg_scale, opts.img_cfg_scale);
        if needs_img && img.is_none() {
            return Err(img_cache_err());
        }
        if needs_uncond && uncond.is_none() {
            return Err(uncond_cache_err());
        }
        // Reject CFG-Zero* up front (before any forward), mirroring the reference `it2i_generate`'s
        // `assert cfg_norm in ['none','global','channel']` — it is a T2I-only blend mode (F-131).
        if opts.cfg_norm == CfgNorm::CfgZeroStar {
            return Err(cfg_zero_star_it2i_err());
        }

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let mut image = multiply(
            &base_noise.as_dtype(Dtype::Float32)?,
            Array::from_f32(noise_scale),
        )?;

        let steps = opts.num_steps;
        // `steps == 0` yields an empty trajectory (and a 0/0 NaN schedule), so the callers' final
        // `.last().expect("at least one step")` would panic. Surface a typed error instead (F-125);
        // the registered `Generator` rejects this upstream, but `interleave_gen`/`vqa` reach here too.
        if steps == 0 {
            return Err(Error::Msg("sensenova: num_steps must be >= 1".into()));
        }
        let lin: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
        let lin_arr = Array::from_slice(&lin, &[(steps + 1) as i32]);
        let ts_arr = if opts.enable_timestep_shift {
            apply_time_schedule(&lin_arr, opts.timestep_shift)?
        } else {
            lin_arr
        };
        let timesteps = ts_arr.as_slice::<f32>().to_vec();

        let noise_embed = if let Some(emb) = &self.noise_scale_embedder {
            let ns = vec![noise_scale / self.noise_scale_max_value; l as usize];
            Some(
                emb.forward(&Array::from_slice(&ns, &[l]))?
                    .reshape(&[1, l, -1])?,
            )
        } else {
            None
        };

        let (cfg, img_cfg) = (opts.cfg_scale, opts.img_cfg_scale);
        let (i0, i1) = opts.cfg_interval;
        let mut traj = Vec::with_capacity(steps);
        for i in 0..steps {
            if let Some(r) = reporter.as_ref() {
                r.check_cancel()?;
            }
            let t = timesteps[i];
            let t_next = timesteps[i + 1];
            // CFG-interval gate for the it2i/edit path: **exclusive** `(i0, i1)` OR an `i0 == 0`
            // always-on override — a faithful port of the reference `modeling_neo_chat.py` lines
            // 901/1246/1557: `use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or
            // cfg_interval[0] == 0`. Note the reference's own quirk: `i1` is ignored whenever
            // `i0 == 0`. The T2I `denoise` uses the DIFFERENT inclusive form (ref line 1799); they
            // diverge only for a custom `cfg_interval` (F-130).
            let use_cfg = (t > i0 && t < i1) || i0 == 0.0;

            let (z, cond_emb) =
                self.step_cond_embeds(&image, grid_h, grid_w, l, t, &noise_embed)?;
            let out_cond = self.predict_v(
                &cond_emb, token_h, token_w, cond_t, cache_cond, &z, t, opts.t_eps,
            )?;

            let mut v_pred = if !use_cfg || (cfg == 1.0 && img_cfg == 1.0) {
                out_cond.clone()
            } else if img_cfg == 1.0 {
                let (c, tl) = img.as_mut().ok_or_else(img_cache_err)?;
                let oi = self.predict_v(&cond_emb, token_h, token_w, *tl, c, &z, t, opts.t_eps)?;
                add(
                    &oi,
                    &multiply(&subtract(&out_cond, &oi)?, Array::from_f32(cfg))?,
                )?
            } else if cfg == img_cfg {
                let (c, tl) = uncond.as_mut().ok_or_else(uncond_cache_err)?;
                let ou = self.predict_v(&cond_emb, token_h, token_w, *tl, c, &z, t, opts.t_eps)?;
                add(
                    &ou,
                    &multiply(&subtract(&out_cond, &ou)?, Array::from_f32(cfg))?,
                )?
            } else {
                let oi = {
                    let (c, tl) = img.as_mut().ok_or_else(img_cache_err)?;
                    self.predict_v(&cond_emb, token_h, token_w, *tl, c, &z, t, opts.t_eps)?
                };
                let ou = {
                    let (c, tl) = uncond.as_mut().ok_or_else(uncond_cache_err)?;
                    self.predict_v(&cond_emb, token_h, token_w, *tl, c, &z, t, opts.t_eps)?
                };
                let a = multiply(&subtract(&out_cond, &oi)?, Array::from_f32(cfg))?;
                let b = multiply(&subtract(&oi, &ou)?, Array::from_f32(img_cfg))?;
                add(&add(&ou, &a)?, &b)?
            };

            if (cfg > 1.0 || img_cfg > 1.0) && use_cfg {
                v_pred = apply_cfg_norm(v_pred, &out_cond, opts.cfg_norm)?;
            }

            image = unpatchify(
                &euler_step(&v_pred, &z, t, t_next)?,
                cell,
                Some(token_h),
                Some(token_w),
            )?;
            traj.push(image.clone());
            if let Some(r) = reporter.as_mut() {
                r.step(i + 1, steps);
            }
        }
        Ok(traj)
    }
}

/// `(needs_img, needs_uncond)`: which extra caches `it2i_denoise` requires for the given guidance
/// scales. Mirrors the `needs_*` flags `it2i_generate` uses to decide which caches to build, so the
/// denoise guard and the cache construction agree (F-126).
fn it2i_cache_requirements(cfg_scale: f32, img_cfg_scale: f32) -> (bool, bool) {
    let needs_cfg = !(cfg_scale == 1.0 && img_cfg_scale == 1.0);
    let needs_img = needs_cfg && (img_cfg_scale == 1.0 || cfg_scale != img_cfg_scale);
    let needs_uncond = needs_cfg && img_cfg_scale != 1.0;
    (needs_img, needs_uncond)
}

fn img_cache_err() -> Error {
    Error::Msg(
        "it2i_denoise: image-CFG guidance needs an image-conditioned cache (`img`), but none was \
         supplied"
            .into(),
    )
}

fn uncond_cache_err() -> Error {
    Error::Msg(
        "it2i_denoise: guidance needs an uncond cache (`uncond`), but none was supplied".into(),
    )
}

/// Blend condition/uncondition velocities under the chosen [`CfgNorm`].
fn cfg_blend(
    v_cond: &Array,
    v_uncond: &Array,
    scale: f32,
    norm: CfgNorm,
    step: usize,
) -> Result<Array> {
    if norm == CfgNorm::CfgZeroStar {
        // CFG-Zero*: project uncond onto cond (optimised scale), zero step 0.
        if step == 0 {
            return multiply(v_cond, Array::from_f32(0.0)).map_err(Error::from);
        }
        let alpha = optimized_scale(v_cond, v_uncond)?;
        let scaled_u = multiply(v_uncond, Array::from_f32(alpha))?;
        let guided = multiply(&subtract(v_cond, &scaled_u)?, Array::from_f32(scale))?;
        return add(&scaled_u, &guided).map_err(Error::from);
    }

    let diff = subtract(v_cond, v_uncond)?;
    let blended = add(v_uncond, &multiply(&diff, Array::from_f32(scale))?)?;
    match norm {
        CfgNorm::Global => {
            let nc = frobenius(v_cond)?;
            let nb = frobenius(&blended)?;
            let s = (nc / (nb + 1e-8)).clamp(0.0, 1.0);
            multiply(&blended, Array::from_f32(s)).map_err(Error::from)
        }
        CfgNorm::Channel => {
            // Per-token (last-axis) norm rescale, clamped to ≤ 1 (norms are ≥ 0).
            let nc = l2_last(v_cond)?;
            let nb = l2_last(&blended)?;
            let ratio = divide(&nc, &add(&nb, Array::from_f32(1e-8))?)?;
            let s = minimum(&ratio, Array::from_f32(1.0))?;
            multiply(&blended, &s).map_err(Error::from)
        }
        _ => Ok(blended),
    }
}

/// `cfg_norm=cfg_zero_star` is a T2I-only blend mode (optimized-scale projection + step-0 zeroing,
/// done inside [`cfg_blend`]), NOT a post-rescale — so it has no place on the it2i/edit path. The
/// reference `it2i_generate` asserts `cfg_norm in ['none','global','channel']`; we mirror that with a
/// typed error rather than letting it silently degrade to plain CFG (F-131).
fn cfg_zero_star_it2i_err() -> Error {
    Error::Msg(
        "sensenova it2i: cfg_norm=cfg_zero_star is T2I-only — the it2i/edit path supports only \
         none/global/channel (matching the reference it2i_generate assert)"
            .into(),
    )
}

/// Apply the post-blend `cfg_norm` rescale (`it2i_generate`): clamp the guided velocity's norm to
/// the condition velocity's (global = whole-tensor, channel = per-token). `None` is a no-op;
/// `CfgZeroStar` is rejected (it is a T2I-only blend mode, not a post-rescale — see
/// [`cfg_zero_star_it2i_err`]).
fn apply_cfg_norm(v: Array, out_cond: &Array, norm: CfgNorm) -> Result<Array> {
    match norm {
        CfgNorm::None => Ok(v),
        CfgNorm::Global => {
            let nc = frobenius(out_cond)?;
            let nv = frobenius(&v)?;
            let s = (nc / (nv + 1e-8)).clamp(0.0, 1.0);
            multiply(&v, Array::from_f32(s)).map_err(Error::from)
        }
        CfgNorm::Channel => {
            let nc = l2_last(out_cond)?;
            let nv = l2_last(&v)?;
            let ratio = divide(&nc, &add(&nv, Array::from_f32(1e-8))?)?;
            let s = minimum(&ratio, Array::from_f32(1.0))?;
            multiply(&v, &s).map_err(Error::from)
        }
        CfgNorm::CfgZeroStar => Err(cfg_zero_star_it2i_err()),
    }
}

/// `‖x‖₂` over the whole tensor (the reference `torch.norm(v, dim=(1,2))` for batch 1).
fn frobenius(x: &Array) -> Result<f32> {
    Ok(sum_all(&multiply(x, x)?)?.sqrt())
}

/// Per-token L2 norm over the last axis, keeping dims: `[1,L,D] → [1,L,1]`.
fn l2_last(x: &Array) -> Result<Array> {
    let rank = x.shape().len() as i32;
    sum_axes(&multiply(x, x)?, &[rank - 1], true)?
        .sqrt()
        .map_err(Error::from)
}

/// Sum every element to a scalar.
fn sum_all(x: &Array) -> Result<f32> {
    let axes: Vec<i32> = (0..x.shape().len() as i32).collect();
    Ok(sum_axes(x, &axes, false)?.item::<f32>())
}

/// CFG-Zero* optimised scale `⟨cond,uncond⟩ / ‖uncond‖²` (computed in f32).
fn optimized_scale(v_cond: &Array, v_uncond: &Array) -> Result<f32> {
    let dot = sum_all(&multiply(v_cond, v_uncond)?)?;
    let nrm = sum_all(&multiply(v_uncond, v_uncond)?)?;
    Ok(dot / (nrm + 1e-8))
}

/// `smart_resize` (Qwen2.5-VL, the vendored `utils.smart_resize`): round `height`/`width` to
/// multiples of `factor` (use `patch·merge = 32`) with total pixels held in `[min_pixels,
/// max_pixels]`. Returns `(height, width)`.
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i64,
    max_pixels: i64,
) -> (i32, i32) {
    let round_by = |n: f64| ((n / factor as f64).round() as i32) * factor;
    let floor_by = |n: f64| ((n / factor as f64).floor() as i32) * factor;
    let ceil_by = |n: f64| ((n / factor as f64).ceil() as i32) * factor;
    let (hf, wf) = (height as f64, width as f64);
    let mut h_bar = factor.max(round_by(hf));
    let mut w_bar = factor.max(round_by(wf));
    let area = (h_bar as i64) * (w_bar as i64);
    if area > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max(floor_by(hf / beta));
        w_bar = factor.max(floor_by(wf / beta));
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_by(hf * beta);
        w_bar = ceil_by(wf * beta);
    }
    (h_bar, w_bar)
}

/// Standard-normal `[shape]` via Box–Muller over a SplitMix64 stream (deterministic per `seed`).
fn gaussian(shape: &[i32], seed: u64) -> Result<Array> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    // Reuse the shared `SplitMix64` so the scramble constants live in one place (F-133). The original
    // inline form pre-incremented `seed` once before producing any value, so seed the RNG offset by
    // one increment to keep the produced stream byte-identical. The f64 (0, 1] mapping below is
    // gaussian-specific (53-bit mantissa, biased off zero to keep `ln()` finite) and differs from
    // `SplitMix64::next_f32`, so it stays here.
    let mut rng = SplitMix64::new(seed.wrapping_add(SPLITMIX64_INCREMENT));
    let mut next_f = || ((rng.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_f();
        let u2 = next_f();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    Ok(Array::from_slice(&out, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(a: &Array) -> Vec<f32> {
        let n = a.shape().iter().product::<i32>();
        a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
    }

    #[test]
    fn gaussian_matches_inline_reference() {
        // Guard the F-133 dedup: the refactored `gaussian` (now reusing `runtime::SplitMix64`) must
        // produce a byte-identical stream to the original inline SplitMix64 Box–Muller. Reconstruct
        // the original algorithm here independently so future drift in either copy is caught.
        fn reference(shape: &[i32], seed: u64) -> Vec<f32> {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut next_f = || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64)
            };
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let u1 = next_f();
                let u2 = next_f();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                out.push((r * theta.cos()) as f32);
                if out.len() < n {
                    out.push((r * theta.sin()) as f32);
                }
            }
            out
        }
        // Cover both even and odd element counts (the trailing-sin branch).
        for shape in [[3, 5].as_slice(), [1, 7].as_slice()] {
            for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
                let got = slice(&gaussian(shape, seed).unwrap());
                assert_eq!(got, reference(shape, seed), "shape={shape:?} seed={seed}");
            }
        }
    }

    #[test]
    fn step_reporter_cancels_and_reports_steps() {
        use mlx_gen::CancelFlag;

        // A cancelled flag makes the pre-step check error (the denoise loop aborts here).
        let cancelled = CancelFlag::new();
        cancelled.cancel();
        let mut sink = |_p: Progress| {};
        let r = StepReporter::new(&cancelled, &mut sink);
        assert!(r.check_cancel().is_err());

        // A live flag passes the check; step() forwards 1-based denoise progress (not the image index).
        let live = CancelFlag::new();
        let seen = std::cell::RefCell::new(Vec::new());
        let mut rec = |p: Progress| {
            if let Progress::Step { current, total } = p {
                seen.borrow_mut().push((current, total));
            }
        };
        let mut r = StepReporter::new(&live, &mut rec);
        assert!(r.check_cancel().is_ok());
        r.step(1, 4);
        r.step(4, 4);
        assert_eq!(*seen.borrow(), vec![(1u32, 4u32), (4, 4)]);
    }

    #[test]
    fn it2i_cache_requirements_match_guidance_scales() {
        // F-126: the denoise guard's cache requirements must match it2i_generate's needs_* flags.
        assert_eq!(it2i_cache_requirements(1.0, 1.0), (false, false)); // no guidance
        assert_eq!(it2i_cache_requirements(4.0, 1.0), (true, false)); // image-CFG only
        assert_eq!(it2i_cache_requirements(4.0, 4.0), (false, true)); // uncond only
        assert_eq!(it2i_cache_requirements(4.0, 2.0), (true, true)); // dual guidance
                                                                     // The error builders name the missing cache so a caller knows what to pass.
        assert!(img_cache_err().to_string().contains("img"));
        assert!(uncond_cache_err().to_string().contains("uncond"));
    }

    #[test]
    fn cfg_blend_none_is_linear_extrapolation() {
        // v_uncond + scale·(v_cond − v_uncond): [1,1,2] tensors.
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::None, 1).unwrap();
        // 1 + 3·(2−1) = 4 ; 1 + 3·(4−1) = 10
        assert_eq!(slice(&out), vec![4.0, 10.0]);
    }

    #[test]
    fn cfg_zero_star_zeroes_first_step() {
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::CfgZeroStar, 0).unwrap();
        assert_eq!(slice(&out), vec![0.0, 0.0]);
    }

    /// F-131: `apply_cfg_norm` (the it2i/edit post-blend rescale) rejects `CfgZeroStar` rather than
    /// silently no-op'ing it (the old `_ => Ok(v)` arm); None/Global/Channel still apply.
    #[test]
    fn it2i_cfg_norm_rejects_cfg_zero_star() {
        let v = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let cond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        assert!(apply_cfg_norm(v.clone(), &cond, CfgNorm::CfgZeroStar).is_err());
        assert!(apply_cfg_norm(v.clone(), &cond, CfgNorm::None).is_ok());
        assert!(apply_cfg_norm(v, &cond, CfgNorm::Global).is_ok());
    }

    #[test]
    fn global_norm_never_amplifies() {
        // Blended norm > cond norm → scale clamps to keep ‖blended‖ ≤ ‖cond‖.
        let v_cond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[-2.0f32, -2.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 4.0, CfgNorm::Global, 1).unwrap();
        let on = (slice(&out).iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cn = (2.0f32).sqrt();
        assert!(
            on <= cn + 1e-4,
            "global-norm output {on} exceeds cond norm {cn}"
        );
    }

    #[test]
    fn interleave_resolution_lookup() {
        assert_eq!(interleave_resolution_for("16:9"), Some((2048, 1152)));
        assert_eq!(interleave_resolution_for("1:1"), Some((1536, 1536)));
        assert_eq!(interleave_resolution_for("nope"), None);
        for (_, (w, h)) in INTERLEAVE_RESOLUTIONS {
            assert_eq!(w % 32, 0, "interleave bucket width not 32-aligned");
            assert_eq!(h % 32, 0, "interleave bucket height not 32-aligned");
        }
    }

    #[test]
    fn smart_resize_upscales_to_min_pixels() {
        // 100×100 rounds to 96×96 (< 65536 px) → upscaled to the 256×256 bucket.
        assert_eq!(smart_resize(100, 100, 32, 65536, 4_194_304), (256, 256));
    }

    #[test]
    fn smart_resize_keeps_in_range_multiple() {
        assert_eq!(smart_resize(512, 512, 32, 65536, 4_194_304), (512, 512));
        // Non-multiples round to the nearest factor.
        assert_eq!(smart_resize(500, 500, 32, 65536, 4_194_304), (512, 512));
    }
}
