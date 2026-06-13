//! Token-axis packed conditioning + the 7-mode guided-velocity dispatch (`wan_diffusion.py`
//! `GEN_Wanx22.sample`, lines 336-559).
//!
//! Each conditioning source and the noisy target are patch-embedded separately (each with its own
//! source-id RoPE) and concatenated on the token axis with the **target last**; at batch 1 the
//! reference's varlen attention is one `cu_seqlens` segment, i.e. plain full self-attention, so the
//! whole packed sequence runs through [`mlx_gen_wan::WanTransformer::forward_packed`] and the target
//! tokens are sliced back out and unpatchified to a `[16, T, H/8, W/8]` velocity.
//!
//! [`guided_velocity`] runs the per-mode forward passes over the right conditioning combos and
//! combines them — either a plain weighted velocity sum (`t2v`, `v2v`, `v2v_chain`, `rv2v`) or APG in
//! x-space (`t2v_apg`, `v2v_apg`, `r2v_apg`; see [`crate::guidance`]).

use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;
use mlx_gen_wan::patchify::unpatchify;
use mlx_gen_wan::{RopeTable, WanTransformer};

use crate::guidance::{normalized_guidance, normalized_guidance_chain, MomentumBuffer};
use crate::rope::{apply_source_id, assign_source_ids};
use crate::vit_guidance::{rv2v_chain, vae_txt_vit};

/// One renderer guidance mode (the renderer half of `cli.GUIDANCE_MODES`; the two `*_wapg` modes are
/// full-Bernini ViT-planner only and out of scope here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    T2v,
    T2vApg,
    V2v,
    V2vChain,
    V2vApg,
    R2vApg,
    Rv2v,
}

impl Mode {
    pub fn from_name(s: &str) -> Option<Mode> {
        Some(match s {
            "t2v" => Mode::T2v,
            "t2v_apg" => Mode::T2vApg,
            "v2v" => Mode::V2v,
            "v2v_chain" => Mode::V2vChain,
            "v2v_apg" => Mode::V2vApg,
            "r2v_apg" => Mode::R2vApg,
            "rv2v" => Mode::Rv2v,
            _ => return None,
        })
    }

    /// Whether this mode routes through APG (x-space) vs a plain weighted velocity sum.
    pub fn is_apg(self) -> bool {
        matches!(self, Mode::T2vApg | Mode::V2vApg | Mode::R2vApg)
    }
}

/// The packed-forward engine: holds the spatial RoPE table + the patch geometry so it can patch-embed
/// the target and each conditioning source with their source-id RoPE and run one packed forward.
pub struct PackedForward {
    rope: RopeTable,
    head_dim: usize,
    out_dim: usize,
    patch_size: (usize, usize, usize),
    max_trained_src_id: f64,
    interpolate_src_id: bool,
}

/// The four conditioning combos (each a list of `(latent, source_id)`); the target is added per
/// forward with source_id 0.
struct Combos {
    none: Vec<(Array, f64)>,
    v: Vec<(Array, f64)>,
    i: Vec<(Array, f64)>,
    vi: Vec<(Array, f64)>,
}

impl PackedForward {
    pub fn new(
        head_dim: usize,
        out_dim: usize,
        patch_size: (usize, usize, usize),
        max_trained_src_id: f64,
        interpolate_src_id: bool,
    ) -> Self {
        Self {
            rope: RopeTable::new(head_dim),
            head_dim,
            out_dim,
            patch_size,
            max_trained_src_id,
            interpolate_src_id,
        }
    }

    /// Patch-embed one latent `[16, T, H8, W8]` to `(tokens [1,L,dim], cos, sin, grid)` with the
    /// source-id RoPE folded in. `cos`/`sin` are f32 here (concatenated + cast to bf16 once before the
    /// forward).
    #[allow(clippy::type_complexity)]
    fn embed_segment(
        &self,
        dit: &WanTransformer,
        latent: &Array,
        source_id: f64,
    ) -> Result<(Array, Array, Array, (usize, usize, usize))> {
        let (tokens, grid) = dit.patch_embed_tokens(latent)?;
        let (cos, sin) = self.rope.precompute_cos_sin(grid)?;
        let (cos, sin) = apply_source_id(&cos, &sin, source_id, self.head_dim)?;
        Ok((tokens, cos, sin, grid))
    }

    /// One packed forward: conditioning `sources` (each `(latent, source_id)`) + the noisy `target`
    /// (source_id 0), returning the **target** velocity `[16, T, H8, W8]` (the reference's
    /// `pred[:, target_mask, :]` then unpatchify). The target is concatenated last.
    pub fn velocity(
        &self,
        dit: &WanTransformer,
        target: &Array,
        sources: &[(Array, f64)],
        t: f32,
        cross_kv: &[(Array, Array)],
    ) -> Result<Array> {
        let mut toks = Vec::with_capacity(sources.len() + 1);
        let mut coss = Vec::with_capacity(sources.len() + 1);
        let mut sins = Vec::with_capacity(sources.len() + 1);
        for (lat, sid) in sources {
            let (tk, c, s, _) = self.embed_segment(dit, lat, *sid)?;
            toks.push(tk);
            coss.push(c);
            sins.push(s);
        }
        let (tk_t, c_t, s_t, grid_t) = self.embed_segment(dit, target, 0.0)?;
        let l_t = (grid_t.0 * grid_t.1 * grid_t.2) as i32;
        toks.push(tk_t);
        coss.push(c_t);
        sins.push(s_t);

        let tok_refs: Vec<&Array> = toks.iter().collect();
        let cos_refs: Vec<&Array> = coss.iter().collect();
        let sin_refs: Vec<&Array> = sins.iter().collect();
        let tokens = concatenate_axis(&tok_refs, 1)?;
        let cos = concatenate_axis(&cos_refs, 0)?.as_dtype(Dtype::Bfloat16)?;
        let sin = concatenate_axis(&sin_refs, 0)?.as_dtype(Dtype::Bfloat16)?;

        let out = dit.forward_packed(&tokens, t, cross_kv, &cos, &sin)?; // [1, total, out·∏patch]
        let total = out.shape()[1];
        let op = out.shape()[2];
        // Slice the target tokens (last l_t) and unpatchify to [16, T, H8, W8].
        let idx = Array::from_slice(&((total - l_t)..total).collect::<Vec<i32>>(), &[l_t]);
        let target_tokens = out.take_axis(&idx, 1)?.reshape(&[l_t, op])?;
        unpatchify(&target_tokens, grid_t, self.out_dim, self.patch_size)
    }

    fn build_combos(&self, videos: &[Array], images: &[Array]) -> Combos {
        let (nv, ni) = (videos.len(), images.len());
        let vi_sids = assign_source_ids(nv + ni, self.max_trained_src_id, self.interpolate_src_id);
        let i_sids = assign_source_ids(ni, self.max_trained_src_id, self.interpolate_src_id);
        let v = if nv > 0 {
            vec![(videos[0].clone(), vi_sids[0])]
        } else {
            vec![]
        };
        let i = images
            .iter()
            .enumerate()
            .map(|(j, im)| (im.clone(), i_sids[j]))
            .collect();
        let mut vi = Vec::with_capacity(nv + ni);
        for (k, v) in videos.iter().enumerate() {
            vi.push((v.clone(), vi_sids[k]));
        }
        for (j, im) in images.iter().enumerate() {
            vi.push((im.clone(), vi_sids[nv + j]));
        }
        Combos {
            none: vec![],
            v,
            i,
            vi,
        }
    }
}

/// All the per-step guidance knobs (the omegas are already `omega_scale`-rescaled when the low-noise
/// expert is active — done by the caller).
#[derive(Clone)]
pub struct GuidanceParams {
    pub omega_vid: f32,
    pub omega_img: f32,
    pub omega_txt: f32,
    pub eta: f32,
    /// Per-term norm thresholds (`r2v_apg` uses two; the single-cond modes use index 0).
    pub norm_threshold: [f32; 2],
}

/// `x = noisy − σ·v` (velocity → x-space) and back. APG operates in x-space.
fn to_x(noisy: &Array, sigma: f32, v: &Array) -> Result<Array> {
    Ok(subtract(noisy, &multiply(v, Array::from_f32(sigma))?)?)
}
fn from_x(noisy: &Array, sigma: f32, x: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::divide(
        &subtract(noisy, x)?,
        Array::from_f32(sigma),
    )?)
}

/// Compute the guided velocity `[16, T, H8, W8]` for one denoise step (`GEN_Wanx22.sample`'s per-mode
/// body). `cross_kv_cond`/`cross_kv_uncond` are this expert's prepared text K/V (cond / empty-neg);
/// `videos`/`images` are the VAE-encoded source latents; `mbufs` are the APG momentum buffers
/// (persisting across steps — one for the single-cond `*_apg` modes, two for `r2v_apg`). `sigma` is
/// this step's flow sigma (for the x-space conversion).
#[allow(clippy::too_many_arguments)]
pub fn guided_velocity(
    pf: &PackedForward,
    mode: Mode,
    dit: &WanTransformer,
    noisy: &Array,
    videos: &[Array],
    images: &[Array],
    t: f32,
    sigma: f32,
    cross_kv_cond: &[(Array, Array)],
    cross_kv_uncond: &[(Array, Array)],
    g: &GuidanceParams,
    mbufs: &mut [MomentumBuffer],
) -> Result<Array> {
    let c = pf.build_combos(videos, images);
    let v = |sources: &[(Array, f64)], cond: bool| -> Result<Array> {
        let kv = if cond { cross_kv_cond } else { cross_kv_uncond };
        pf.velocity(dit, noisy, sources, t, kv)
    };
    // Weighted velocity sum for a list of (vel, weight) deltas: base + Σ w·(cur − prev).
    let chain = |terms: &[(&Array, f32)]| -> Result<Array> {
        // terms[0] is the base (weight ignored); each subsequent is (cur, weight) diffing the prev.
        let mut acc = terms[0].0.clone();
        for w in 1..terms.len() {
            let delta = subtract(terms[w].0, terms[w - 1].0)?;
            acc = add(&acc, &multiply(&delta, Array::from_f32(terms[w].1))?)?;
        }
        Ok(acc)
    };

    match mode {
        Mode::T2v => {
            let e0 = v(&c.none, false)?;
            let et = v(&c.none, true)?;
            chain(&[(&e0, 0.0), (&et, g.omega_txt)])
        }
        Mode::V2v => {
            let e_vi = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[(&e_vi, 0.0), (&e_vti, g.omega_txt)])
        }
        Mode::V2vChain => {
            let e0 = v(&c.none, false)?;
            let ev = v(&c.v, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[(&e0, 0.0), (&ev, g.omega_vid), (&e_vti, g.omega_txt)])
        }
        Mode::Rv2v => {
            let e0 = v(&c.none, false)?;
            let ev = v(&c.v, false)?;
            let e_vi = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            chain(&[
                (&e0, 0.0),
                (&ev, g.omega_vid),
                (&e_vi, g.omega_img),
                (&e_vti, g.omega_txt),
            ])
        }
        Mode::T2vApg => {
            let e0 = v(&c.none, false)?;
            let et = v(&c.none, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xt = to_x(noisy, sigma, &et)?;
            let xg = normalized_guidance(
                &xt,
                &x0,
                g.omega_txt,
                Some(&mut mbufs[0]),
                g.eta,
                g.norm_threshold[0],
            )?;
            from_x(noisy, sigma, &xg)
        }
        Mode::V2vApg => {
            let e0 = v(&c.vi, false)?;
            let e_vti = v(&c.vi, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xvti = to_x(noisy, sigma, &e_vti)?;
            let xg = normalized_guidance(
                &xvti,
                &x0,
                g.omega_txt,
                Some(&mut mbufs[0]),
                g.eta,
                g.norm_threshold[0],
            )?;
            from_x(noisy, sigma, &xg)
        }
        Mode::R2vApg => {
            let e0 = v(&c.none, false)?;
            let ei = v(&c.i, false)?;
            let eti = v(&c.i, true)?;
            let x0 = to_x(noisy, sigma, &e0)?;
            let xi = to_x(noisy, sigma, &ei)?;
            let xti = to_x(noisy, sigma, &eti)?;
            let xg = normalized_guidance_chain(
                &x0,
                &[xi, xti],
                &[g.omega_img, g.omega_txt],
                mbufs,
                g.eta,
                &g.norm_threshold,
            )?;
            from_x(noisy, sigma, &xg)
        }
    }
}

/// Number of APG momentum buffers a mode needs (0 for the plain modes, 1 for the single-cond `*_apg`
/// modes, 2 for the chained `r2v_apg`).
pub fn num_momentum_buffers(mode: Mode) -> usize {
    match mode {
        Mode::T2vApg | Mode::V2vApg => 1,
        Mode::R2vApg => 2,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// sc-5142: the full-Bernini ViT-conditioned per-step velocity (`sample_one_step`).
// ---------------------------------------------------------------------------

/// One full-Bernini ViT-conditioned guidance mode (`BerniniPipeline`'s `sample_one_step` modes; the
/// renderer-only [`Mode`] modes are disjoint from these).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VitMode {
    /// `vae_txt_vit` — plain 4-prediction combine.
    VaeTxtVit,
    /// `vae_txt_vit_wapg` — the same combine with `apg_delta` on each delta (the primary t2i/edit).
    VaeTxtVitWapg,
    /// `rv2v_wapg` — the 5-prediction vid/img/txt/ViT chain, plain deltas.
    Rv2vWapg,
    /// `r2v_wapg` — the same chain with `apg_delta` deltas.
    R2vWapg,
    /// `v2v_apg` — x-space [`normalized_guidance`] over (uncond, wtxt_wvit) (the renderer family).
    V2vApg,
}

impl VitMode {
    pub fn from_name(s: &str) -> Option<VitMode> {
        Some(match s {
            "vae_txt_vit" => VitMode::VaeTxtVit,
            "vae_txt_vit_wapg" => VitMode::VaeTxtVitWapg,
            "rv2v_wapg" => VitMode::Rv2vWapg,
            "r2v_wapg" => VitMode::R2vWapg,
            "v2v_apg" => VitMode::V2vApg,
            _ => return None,
        })
    }

    fn apg(self) -> bool {
        matches!(self, VitMode::VaeTxtVitWapg | VitMode::R2vWapg)
    }
}

/// The planner's 4 prepared cross-attn K/V streams, in the reference's naming. Each is the prepared
/// `(K, V)` list for one prompt-embed stream (`UMT5(prompt)` ⧺ the planner ViT-context, sc-5140's
/// `concat_with_zero_init`).
pub struct VitStreams<'a> {
    pub wtxt_wvit: &'a [(Array, Array)],
    pub wtxt_wovit: &'a [(Array, Array)],
    pub wotxt_wvit: &'a [(Array, Array)],
    pub wotxt_wovit: &'a [(Array, Array)],
}

/// Per-step ViT-conditioned guidance knobs (already `omega_scale`-rescaled by the caller when the
/// low-noise expert is active).
#[derive(Clone)]
pub struct VitGuidanceParams {
    pub omega_txt: f32,
    pub omega_img: f32,
    pub omega_vid: f32,
    pub omega_tgt: f32,
    /// x-space APG knobs for `v2v_apg` (eta, norm_threshold); unused by the v-space modes.
    pub eta: f32,
    pub norm_threshold: f32,
}

/// `sample_one_step` (`wan_diffusion.py` 795-1049): the full-Bernini per-step velocity over the 4
/// prompt streams × the packed-latent variants. Each prediction is one [`PackedForward::velocity`]
/// (≡ the reference `shared_step`) over a source subset + a chosen prompt stream:
///   - `wvae` = image ⧺ video sources; `wimgvae` = image; `wvidvae` = video; `wovae` = target only.
///   - `images`/`videos` are `(latent, source_id)` source lists (the target is added with id 0 inside
///     `velocity`).
///
/// The combine is the slice-A math ([`vae_txt_vit`] / [`rv2v_chain`]) on the target-sliced spatial
/// velocities (a leading batch axis is added so `apg_delta` reduces over the whole tensor). `v2v_apg`
/// routes through the x-space [`normalized_guidance`] (momentum 0). Returns `[16, T, H8, W8]`.
#[allow(clippy::too_many_arguments)]
pub fn vit_one_step(
    pf: &PackedForward,
    dit: &WanTransformer,
    mode: VitMode,
    noisy: &Array,
    images: &[(Array, f64)],
    videos: &[(Array, f64)],
    t: f32,
    sigma: f32,
    streams: &VitStreams,
    g: &VitGuidanceParams,
) -> Result<Array> {
    let wvae: Vec<(Array, f64)> = images.iter().chain(videos).cloned().collect();
    let v =
        |sources: &[(Array, f64)], kv: &[(Array, Array)]| pf.velocity(dit, noisy, sources, t, kv);
    let shape = noisy.shape().to_vec();
    let b = |a: &Array| -> Result<Array> { Ok(a.expand_dims(0)?) }; // [16,T,H8,W8] -> [1,16,...]
    let unb = |a: Array| -> Result<Array> { Ok(a.reshape(&shape)?) }; // back to [16,T,H8,W8]

    match mode {
        VitMode::VaeTxtVit | VitMode::VaeTxtVitWapg => {
            let base = v(&[], streams.wotxt_wovit)?; // wovae · wotxt_wovit
            let img = v(&wvae, streams.wotxt_wovit)?; // wvae  · wotxt_wovit
            let txt = v(&wvae, streams.wtxt_wovit)?; // wvae  · wtxt_wovit
            let vit = v(&wvae, streams.wtxt_wvit)?; // wvae  · wtxt_wvit
            let out = vae_txt_vit(
                &b(&base)?,
                &b(&img)?,
                &b(&txt)?,
                &b(&vit)?,
                g.omega_img,
                g.omega_txt,
                g.omega_tgt,
                mode.apg(),
            )?;
            unb(out)
        }
        VitMode::Rv2vWapg | VitMode::R2vWapg => {
            let base = v(&[], streams.wotxt_wovit)?;
            // `if cur_omega_X > 0` short-circuits (reuse the previous prediction, no extra forward).
            let eps_v = if g.omega_vid > 0.0 {
                v(videos, streams.wotxt_wovit)? // wvidvae · wotxt_wovit
            } else {
                base.clone()
            };
            let eps_vi = if g.omega_img > 0.0 {
                v(&wvae, streams.wotxt_wovit)? // wvae · wotxt_wovit
            } else {
                eps_v.clone()
            };
            let eps_vti = if g.omega_txt > 0.0 {
                v(&wvae, streams.wtxt_wovit)? // wvae · wtxt_wovit
            } else {
                eps_vi.clone()
            };
            let eps_vtic = if g.omega_tgt > 0.0 {
                v(&wvae, streams.wtxt_wvit)? // wvae · wtxt_wvit
            } else {
                eps_vti.clone()
            };
            let out = rv2v_chain(
                &b(&base)?,
                &b(&eps_v)?,
                &b(&eps_vi)?,
                &b(&eps_vti)?,
                &b(&eps_vtic)?,
                g.omega_vid,
                g.omega_img,
                g.omega_txt,
                g.omega_tgt,
                mode.apg(),
            )?;
            unb(out)
        }
        VitMode::V2vApg => {
            let eps_uncond = v(&[], streams.wotxt_wovit)?; // wovae · wotxt_wovit
            let eps_t = v(&wvae, streams.wtxt_wvit)?; // wvae · wtxt_wvit
            let x0 = to_x(noisy, sigma, &eps_uncond)?;
            let xt = to_x(noisy, sigma, &eps_t)?;
            let mut buf = MomentumBuffer::new(0.0);
            let xg = normalized_guidance(
                &xt,
                &x0,
                g.omega_txt,
                Some(&mut buf),
                g.eta,
                g.norm_threshold,
            )?;
            from_x(noisy, sigma, &xg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::weights::Weights;
    use mlx_gen_wan::config::WanModelConfig;

    fn tiny_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_1_3b();
        c.dim = 128;
        c.num_heads = 1;
        c.num_layers = 2;
        c.ffn_dim = 256;
        c.freq_dim = 256;
        c.text_dim = 32;
        c.text_len = 8;
        c.in_dim = 16;
        c.out_dim = 16;
        c.vae_z_dim = 16;
        c.boundary = 0.875;
        c.num_train_timesteps = 1000;
        c
    }

    fn load(name: &str) -> Weights {
        let path = format!(
            "{}/../mlx-gen-wan/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        Weights::from_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    fn max_abs(a: &Array, b: &Array) -> f32 {
        mlx_rs::ops::max(subtract(a, b).unwrap().abs().unwrap(), None)
            .unwrap()
            .item::<f32>()
    }

    /// `t2v` (plain CFG over the target-only combo) computed by [`guided_velocity`] must equal the
    /// hand-written `uncond + ω·(cond − uncond)` over two [`PackedForward::velocity`] forwards — pins
    /// the mode-dispatch plumbing to the validated forward seam (no real weights / no conditioning).
    #[test]
    fn t2v_mode_matches_manual_cfg() {
        let w = load("s5_low.safetensors");
        let cfg = tiny_cfg();
        let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");
        let pf = PackedForward::new(
            cfg.dim / cfg.num_heads,
            cfg.out_dim,
            cfg.patch_size,
            5.0,
            true,
        );

        let noisy = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
        let ctx_c = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
        let ctx_u = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
        let kv_c = dit.prepare_cross_kv(&ctx_c).unwrap();
        let kv_u = dit.prepare_cross_kv(&ctx_u).unwrap();
        let t = 833.0f32;
        let omega = 4.0f32;

        let g = GuidanceParams {
            omega_vid: 1.0,
            omega_img: 1.0,
            omega_txt: omega,
            eta: 0.5,
            norm_threshold: [50.0, 50.0],
        };
        let mut mbufs: Vec<MomentumBuffer> = Vec::new();
        let got = guided_velocity(
            &pf,
            Mode::T2v,
            &dit,
            noisy,
            &[],
            &[],
            t,
            1.0,
            &kv_c,
            &kv_u,
            &g,
            &mut mbufs,
        )
        .unwrap();

        // Manual: uncond + ω·(cond − uncond) over the target-only packed forward.
        let e_u = pf.velocity(&dit, noisy, &[], t, &kv_u).unwrap();
        let e_c = pf.velocity(&dit, noisy, &[], t, &kv_c).unwrap();
        let want = add(
            &e_u,
            multiply(subtract(&e_c, &e_u).unwrap(), Array::from_f32(omega)).unwrap(),
        )
        .unwrap();
        assert_eq!(got.shape(), noisy.shape());
        assert_eq!(max_abs(&got, &want), 0.0, "t2v must equal manual CFG");
    }

    /// `vae_txt_vit` dispatch: [`vit_one_step`] must equal the manual 4-forward combine via
    /// [`crate::vit_guidance::vae_txt_vit`] over the right (source-subset × prompt-stream) pairs —
    /// pins the `sample_one_step` stream/source routing to the validated `velocity` seam + the
    /// (separately torch-validated) combine math. Distinct streams make a wrong routing show up.
    #[test]
    fn vae_txt_vit_dispatch_matches_manual() {
        let w = load("s5_low.safetensors");
        let cfg = tiny_cfg();
        let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");
        let pf = PackedForward::new(
            cfg.dim / cfg.num_heads,
            cfg.out_dim,
            cfg.patch_size,
            5.0,
            true,
        );
        let noisy = w.require("init_noise").unwrap(); // [16, 2, 2, 2]

        // Four distinct prompt streams (scale the two fixture contexts to get 4 unique K/V).
        let mk = |ctx: &Array, s: f32| {
            let scaled = multiply(ctx, Array::from_f32(s)).unwrap();
            dit.prepare_cross_kv(&dit.embed_text(&scaled).unwrap())
                .unwrap()
        };
        let cc = w.require("ctx_cond").unwrap();
        let cu = w.require("ctx_uncond").unwrap();
        let s = VitStreams {
            wtxt_wvit: &mk(cc, 1.0),
            wtxt_wovit: &mk(cu, 1.0),
            wotxt_wvit: &mk(cc, 0.5),
            wotxt_wovit: &mk(cu, 0.5),
        };
        let img = Array::zeros::<f32>(&[16, 1, 2, 2]).unwrap();
        let images = [(img, 1.0)];
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 3.0,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let t = 833.0f32;
        let got = vit_one_step(
            &pf,
            &dit,
            VitMode::VaeTxtVit,
            noisy,
            &images,
            &[],
            t,
            1.0,
            &s,
            &g,
        )
        .unwrap();

        // Manual: the four shared_step forwards, batched, combined, unbatched.
        let base = pf.velocity(&dit, noisy, &[], t, s.wotxt_wovit).unwrap();
        let im = pf.velocity(&dit, noisy, &images, t, s.wotxt_wovit).unwrap();
        let tx = pf.velocity(&dit, noisy, &images, t, s.wtxt_wovit).unwrap();
        let vi = pf.velocity(&dit, noisy, &images, t, s.wtxt_wvit).unwrap();
        let want = vae_txt_vit(
            &base.expand_dims(0).unwrap(),
            &im.expand_dims(0).unwrap(),
            &tx.expand_dims(0).unwrap(),
            &vi.expand_dims(0).unwrap(),
            g.omega_img,
            g.omega_txt,
            g.omega_tgt,
            false,
        )
        .unwrap()
        .reshape(noisy.shape())
        .unwrap();
        assert_eq!(got.shape(), noisy.shape());
        assert_eq!(
            max_abs(&got, &want),
            0.0,
            "vae_txt_vit dispatch must equal manual combine"
        );
    }

    /// A conditioning source extends the packed sequence but the sliced target velocity keeps the
    /// target's shape — pins the assemble/slice geometry with a source present.
    #[test]
    fn conditioning_source_preserves_target_shape() {
        let w = load("s5_low.safetensors");
        let cfg = tiny_cfg();
        let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");
        let pf = PackedForward::new(
            cfg.dim / cfg.num_heads,
            cfg.out_dim,
            cfg.patch_size,
            5.0,
            true,
        );
        let noisy = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
        let ctx = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
        let kv = dit.prepare_cross_kv(&ctx).unwrap();

        // One image source (single frame, [16, 1, H8, W8]) with source_id 1.
        let img = Array::zeros::<f32>(&[16, 1, 2, 2]).unwrap();
        let vel = pf.velocity(&dit, noisy, &[(img, 1.0)], 833.0, &kv).unwrap();
        assert_eq!(
            vel.shape(),
            noisy.shape(),
            "target velocity keeps target shape"
        );
    }
}
