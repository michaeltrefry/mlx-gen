//! SeedVR2 dual-stream MMDiT — native MLX port (sc-4813), image-mode (B=1) parity.
//!
//! Port of `mflux.models.seedvr2.model.seedvr2_transformer`. Flow:
//! `txt_in` (proj precomputed neg-prompt embed) + `vid_in` (3-D patchify) + `emb_in` (sinusoidal
//! timestep → AdaLN params) → N dual-stream blocks → `vid_out_norm`+out-AdaLN → `vid_out` (unpatchify).
//!
//! Each block: plain-RMSNorm → AdaLN-in → windowed joint attention (QK-norm + 3-D axial RoPE) →
//! AdaLN-out → residual → RMSNorm → AdaLN-in → SwiGLU → AdaLN-out → residual (txt frozen on the
//! last layer when `last_layer_vid_only`). Layers ≥ `mm_layers` share the MLP/AdaLN (`.all`); the
//! attention always keeps separate `_vid`/`_txt` projections.
//!
//! Window attention: the partition is data-independent (depends only on the patch grid + window +
//! shift), so the forward/reverse permutations and per-window shapes are computed on the host.
//! Each window jointly attends over its video tokens + all text tokens; video is scattered back,
//! text is averaged across the windows it appeared in.

use mlx_gen::nn::{gelu_tanh, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, cos, matmul, multiply, quantize as quantize_affine,
    quantized_matmul, sin,
};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::config::DitConfig;

// ---------------------------------------------------------------------------
// small leaves
// ---------------------------------------------------------------------------

/// A dense or group-wise-affine-quantized `[out,in]` weight (sc-5198). Quantization is **Linear-only**
/// (the VAE convs stay dense) and skips any Linear whose `in_features` is not a multiple of the group
/// size — matching the reference predicate (which leaves e.g. `vid_in.proj`, in=132, dense).
enum LinearWeight {
    Dense(Array),
    Quant {
        wq: Array,
        scales: Array,
        biases: Array,
        group: i32,
        bits: i32,
    },
}

struct Linear {
    w: LinearWeight,
    b: Option<Array>,
}
impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        Ok(Self {
            w: LinearWeight::Dense(w.require(&format!("{prefix}.weight"))?.clone()),
            b: if bias {
                Some(w.require(&format!("{prefix}.bias"))?.clone())
            } else {
                None
            },
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = match &self.w {
            LinearWeight::Dense(w) => matmul(x, w.t())?,
            LinearWeight::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            } => quantized_matmul(x, wq, scales, biases, true, *group, *bits)?,
        };
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }

    /// Quantize the dense weight to `bits` (group-wise affine, with the fork's bf16-parity cast).
    /// No-op when already quantized or when `in_features % group != 0` (the reference predicate
    /// leaves those dense — group-wise quantization requires the contraction dim divisible by `group`).
    /// Evals the packed tensors so the dense weight is freed promptly.
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        if let LinearWeight::Dense(w) = &self.w {
            if *w.shape().last().unwrap() % group != 0 {
                return Ok(());
            }
            let (wq, scales, biases) = quantize_affine(&w.as_dtype(Dtype::Bfloat16)?, group, bits)?;
            eval([&wq, &scales, &biases])?;
            self.w = LinearWeight::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            };
        }
        Ok(())
    }
}

fn arange_f32(n: i32) -> Array {
    Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<f32>>(), &[n])
}

/// fast RMSNorm with a unit (`ones`) weight — the block pre-norms have no learnable scale.
fn rms_plain(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x.shape().last().unwrap();
    Ok(rms_norm(x, &Array::ones::<f32>(&[dim])?, eps)?)
}

// ---------------------------------------------------------------------------
// time embedding
// ---------------------------------------------------------------------------

struct TimeEmbedding {
    proj_in: Linear,
    proj_hid: Linear,
    proj_out: Linear,
    sinusoidal_dim: i32,
}
impl TimeEmbedding {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj_in: Linear::load(w, &format!("{prefix}.proj_in"), true)?,
            proj_hid: Linear::load(w, &format!("{prefix}.proj_hid"), true)?,
            proj_out: Linear::load(w, &format!("{prefix}.proj_out"), true)?,
            sinusoidal_dim: 256,
        })
    }
    fn forward(&self, timestep: &Array) -> Result<Array> {
        // timestep: scalar or (B,). sinusoidal embedding.
        let ts = if timestep.ndim() == 0 {
            timestep.reshape(&[1])?
        } else {
            timestep.clone()
        };
        let half = self.sinusoidal_dim / 2;
        let scale = -(10000f64.ln()) / half as f64;
        let freqs = Array::from_slice(
            &(0..half)
                .map(|i| (i as f64 * scale).exp() as f32)
                .collect::<Vec<f32>>(),
            &[half],
        );
        // args = ts[:,None] * freqs -> (B, half)
        let args = multiply(
            &ts.reshape(&[-1, 1])?.as_dtype(Dtype::Float32)?,
            &freqs.reshape(&[1, half])?,
        )?;
        let emb = concatenate_axis(&[&sin(&args)?, &cos(&args)?], -1)?; // (B, 256)
        let emb = silu(&self.proj_in.forward(&emb)?)?;
        let emb = silu(&self.proj_hid.forward(&emb)?)?;
        self.proj_out.forward(&emb)
    }
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        self.proj_in.quantize(bits, group)?;
        self.proj_hid.quantize(bits, group)?;
        self.proj_out.quantize(bits, group)
    }
}

// ---------------------------------------------------------------------------
// patch in / out
// ---------------------------------------------------------------------------

struct PatchIn {
    proj: Linear,
    pt: i32,
    ph: i32,
    pw: i32,
}
impl PatchIn {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(w, &format!("{prefix}.proj"), true)?,
            pt: cfg.patch_t,
            ph: cfg.patch_h,
            pw: cfg.patch_w,
        })
    }
    /// `(B,C,T,H,W)` → tokens `(B, Tp*Hp*Wp, dim)` + `(Tp,Hp,Wp)`.
    fn forward(&self, vid: &Array) -> Result<(Array, (i32, i32, i32))> {
        let sh = vid.shape();
        let (b, c, t, h, wd) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (tp, hp, wp) = (t / self.pt, h / self.ph, wd / self.pw);
        let x = vid
            .reshape(&[b, c, tp, self.pt, hp, self.ph, wp, self.pw])?
            .transpose_axes(&[0, 2, 4, 6, 3, 5, 7, 1])?
            .reshape(&[b, tp, hp, wp, self.pt * self.ph * self.pw * c])?;
        let x = self.proj.forward(&x)?;
        let dim = *x.shape().last().unwrap();
        Ok((x.reshape(&[b, -1, dim])?, (tp, hp, wp)))
    }
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        self.proj.quantize(bits, group) // in=patch·channels=132 → left dense by the predicate
    }
}

struct PatchOut {
    proj: Linear,
    pt: i32,
    ph: i32,
    pw: i32,
}
impl PatchOut {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(w, &format!("{prefix}.proj"), true)?,
            pt: cfg.patch_t,
            ph: cfg.patch_h,
            pw: cfg.patch_w,
        })
    }
    fn forward(&self, vid: &Array, shape: (i32, i32, i32)) -> Result<Array> {
        let (tp, hp, wp) = shape;
        let x = self.proj.forward(vid)?;
        let b = x.shape()[0];
        let cc = *x.shape().last().unwrap() / (self.pt * self.ph * self.pw);
        Ok(x.reshape(&[b, tp, hp, wp, self.pt, self.ph, self.pw, cc])?
            .transpose_axes(&[0, 7, 1, 4, 2, 5, 3, 6])?
            .reshape(&[b, cc, tp * self.pt, hp * self.ph, wp * self.pw])?)
    }
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        self.proj.quantize(bits, group)
    }
}

// ---------------------------------------------------------------------------
// window partition (host-side, data-independent)
// ---------------------------------------------------------------------------

/// python `round` (round-half-to-even).
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if (diff - 0.5).abs() < 1e-9 {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    } else {
        x.round() as i64
    }
}
fn ceil_div_f(a: f64, b: f64) -> i64 {
    (a / b).ceil() as i64
}

/// Replicates `WindowPartitioner._make_windows`. Returns each window's `(t0,t1,h0,h1,w0,w1)`.
fn make_windows(
    t: i32,
    h: i32,
    w: i32,
    window: (i32, i32, i32),
    shift: bool,
) -> Vec<(i32, i32, i32, i32, i32, i32)> {
    let (nt_w, nh_w, nw_w) = window;
    let scale = ((45.0 * 80.0) / (h as f64 * w as f64)).sqrt();
    let resized_h = py_round(h as f64 * scale) as f64;
    let resized_w = py_round(w as f64 * scale) as f64;
    let wh = ceil_div_f(resized_h, nh_w as f64);
    let ww = ceil_div_f(resized_w, nw_w as f64);
    let wt = ceil_div_f(t.min(30) as f64, nt_w as f64);

    let (st, sh_, sw_, nt, nh, nw);
    if shift {
        st = if wt < t as i64 { 0.5 } else { 0.0 };
        sh_ = if wh < h as i64 { 0.5 } else { 0.0 };
        sw_ = if ww < w as i64 { 0.5 } else { 0.0 };
        nt = if st > 0.0 {
            ceil_div_f(t as f64 - st, wt as f64) + 1
        } else {
            1
        };
        nh = if sh_ > 0.0 {
            ceil_div_f(h as f64 - sh_, wh as f64) + 1
        } else {
            1
        };
        nw = if sw_ > 0.0 {
            ceil_div_f(w as f64 - sw_, ww as f64) + 1
        } else {
            1
        };
    } else {
        st = 0.0;
        sh_ = 0.0;
        sw_ = 0.0;
        nt = ceil_div_f(t as f64, wt as f64);
        nh = ceil_div_f(h as f64, wh as f64);
        nw = ceil_div_f(w as f64, ww as f64);
    }

    let mut out = Vec::new();
    for iw in 0..nw {
        let w0 = (((iw as f64 - sw_) * ww as f64) as i64).max(0) as i32;
        let w1 = (((iw as f64 - sw_ + 1.0) * ww as f64) as i64).min(w as i64) as i32;
        if w1 <= w0 {
            continue;
        }
        for ih in 0..nh {
            let h0 = (((ih as f64 - sh_) * wh as f64) as i64).max(0) as i32;
            let h1 = (((ih as f64 - sh_ + 1.0) * wh as f64) as i64).min(h as i64) as i32;
            if h1 <= h0 {
                continue;
            }
            for it in 0..nt {
                let t0 = (((it as f64 - st) * wt as f64) as i64).max(0) as i32;
                let t1 = (((it as f64 - st + 1.0) * wt as f64) as i64).min(t as i64) as i32;
                if t1 <= t0 {
                    continue;
                }
                out.push((t0, t1, h0, h1, w0, w1));
            }
        }
    }
    out
}

/// forward permutation (windowed order → original token index), per-window shapes.
struct WindowPlan {
    forward_idx: Vec<i32>,
    reverse_idx: Vec<i32>,
    window_shapes: Vec<(i32, i32, i32)>, // (f,h,w) per window
}
fn window_plan(tp: i32, hp: i32, wp: i32, window: (i32, i32, i32), shift: bool) -> WindowPlan {
    let wins = make_windows(tp, hp, wp, window, shift);
    let mut forward_idx = Vec::new();
    let mut window_shapes = Vec::new();
    for (t0, t1, h0, h1, w0, w1) in &wins {
        window_shapes.push((t1 - t0, h1 - h0, w1 - w0));
        for t in *t0..*t1 {
            for h in *h0..*h1 {
                for w in *w0..*w1 {
                    forward_idx.push((t * hp + h) * wp + w);
                }
            }
        }
    }
    let mut reverse_idx = vec![0i32; forward_idx.len()];
    for (i, &orig) in forward_idx.iter().enumerate() {
        reverse_idx[orig as usize] = i as i32;
    }
    WindowPlan {
        forward_idx,
        reverse_idx,
        window_shapes,
    }
}

// ---------------------------------------------------------------------------
// 3-D axial RoPE
// ---------------------------------------------------------------------------

/// `axial_1d(freqs, pos)`: for positions `pos` (len,), returns `(len, 2*nfreq)` =
/// `[p·f0, p·f0, p·f1, p·f1, …]` (each base freq duplicated).
fn axial_1d(freqs: &Array, pos: &Array) -> Result<Array> {
    let len = pos.shape()[0];
    let nf = freqs.shape()[0];
    let outer = multiply(&pos.reshape(&[len, 1])?, &freqs.reshape(&[1, nf])?)?; // (len, nf)
    let dup = broadcast_to(&outer.reshape(&[len, nf, 1])?, &[len, nf, 2])?;
    Ok(dup.reshape(&[len, nf * 2])?)
}

/// 1-D axis positions for 3-D RoPE. **lang mode** (3B): integer `arange(n) + offset` (the reference
/// `_get_axial_freqs` non-pixel branch; vid temporal is offset by `txt_len`). **pixel mode** (7B):
/// normalized `linspace(-1, 1, n)` (the `freqs_for="pixel"` branch; the offset is unused because the
/// 7B attention takes the non-mm rope path — `rope_on_text=false` — with no temporal offset). mlx
/// `linspace(-1,1,1)` returns `[-1]`.
fn axis_pos(n: i32, pixel: bool, offset: i32) -> Result<Array> {
    if pixel {
        let data: Vec<f32> = if n <= 1 {
            vec![-1.0]
        } else {
            let step = 2.0 / (n - 1) as f32;
            (0..n).map(|i| -1.0 + step * i as f32).collect()
        };
        Ok(Array::from_slice(&data, &[n]))
    } else if offset == 0 {
        Ok(arange_f32(n)) // bit-identical to the original plain `arange` for the h/w axes
    } else {
        Ok(add(arange_f32(n), Array::from_f32(offset as f32))?)
    }
}

/// Per-window video freqs `(f*h*w, nf2*3)`; temporal positions offset by `txt_off` (lang mode only —
/// pixel mode normalizes each window's f/h/w independently with no offset).
fn vid_freq_block(
    freqs: &Array,
    f: i32,
    h: i32,
    w: i32,
    txt_off: i32,
    pixel: bool,
) -> Result<Array> {
    let nf2 = freqs.shape()[0] * 2;
    let axt = broadcast_to(
        &axial_1d(freqs, &axis_pos(f, pixel, txt_off)?)?.reshape(&[f, 1, 1, nf2])?,
        &[f, h, w, nf2],
    )?;
    let axh = broadcast_to(
        &axial_1d(freqs, &axis_pos(h, pixel, 0)?)?.reshape(&[1, h, 1, nf2])?,
        &[f, h, w, nf2],
    )?;
    let axw = broadcast_to(
        &axial_1d(freqs, &axis_pos(w, pixel, 0)?)?.reshape(&[1, 1, w, nf2])?,
        &[f, h, w, nf2],
    )?;
    Ok(concatenate_axis(&[&axt, &axh, &axw], -1)?.reshape(&[f * h * w, nf2 * 3])?)
}

/// Text freqs `(txt_len, 126)` = the 1-D axial freqs tiled across the 3 axis slots.
fn txt_freq_block(freqs: &Array, txt_len: i32) -> Result<Array> {
    let a = axial_1d(freqs, &arange_f32(txt_len))?; // (txt_len, 42)
    Ok(concatenate_axis(&[&a, &a, &a], -1)?)
}

/// rotate_half on `(..., 2k)`: pairs `(x0,x1) -> (-x1, x0)`.
fn rotate_half(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let last = *sh.last().unwrap();
    let mut head: Vec<i32> = sh[..sh.len() - 1].to_vec();
    head.push(last / 2);
    head.push(2);
    let xr = x.reshape(&head)?; // (..., k, 2)
    let nd = xr.ndim() as i32;
    let x1 = xr.take_axis(Array::from_int(0), nd - 1)?; // (..., k)
    let x2 = xr.take_axis(Array::from_int(1), nd - 1)?;
    let neg = multiply(&x2, Array::from_f32(-1.0))?;
    let mut tshape: Vec<i32> = sh[..sh.len() - 1].to_vec();
    tshape.push(last / 2);
    tshape.push(1);
    let a = neg.reshape(&tshape)?;
    let b = x1.reshape(&tshape)?;
    Ok(concatenate_axis(&[&a, &b], -1)?.reshape(sh)?)
}

/// apply RoPE to `t` `(N, heads, head_dim)` with `freqs` `(N, rot)` (rot ≤ head_dim).
fn apply_rope(t: &Array, freqs: &Array) -> Result<Array> {
    let sh = t.shape();
    let (n, _heads, hd) = (sh[0], sh[1], sh[2]);
    let rot = freqs.shape()[1];
    let t_mid = t.take_axis(
        Array::from_slice(&(0..rot).collect::<Vec<i32>>(), &[rot]),
        2,
    )?; // (N,heads,rot)
    let cosf = cos(&freqs.as_dtype(Dtype::Float32)?)?.reshape(&[n, 1, rot])?;
    let sinf = sin(&freqs.as_dtype(Dtype::Float32)?)?.reshape(&[n, 1, rot])?;
    let mid_f = t_mid.as_dtype(Dtype::Float32)?;
    let rotated = add(
        &multiply(&mid_f, &cosf)?,
        &multiply(&rotate_half(&mid_f)?, &sinf)?,
    )?
    .as_dtype(t.dtype())?;
    if rot < hd {
        let right = t.take_axis(
            Array::from_slice(&(rot..hd).collect::<Vec<i32>>(), &[hd - rot]),
            2,
        )?;
        Ok(concatenate_axis(&[&rotated, &right], -1)?)
    } else {
        Ok(rotated)
    }
}

// ---------------------------------------------------------------------------
// AdaLN modulation
// ---------------------------------------------------------------------------

struct AdaParams {
    attn_shift: Array,
    attn_scale: Array,
    attn_gate: Array,
    mlp_shift: Array,
    mlp_scale: Array,
    mlp_gate: Array,
}
impl AdaParams {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            attn_shift: g("attn_shift")?,
            attn_scale: g("attn_scale")?,
            attn_gate: g("attn_gate")?,
            mlp_shift: g("mlp_shift")?,
            mlp_scale: g("mlp_scale")?,
            mlp_gate: g("mlp_gate")?,
        })
    }
}

/// emb (B, vid_dim, 2, 3); layer_idx 0=attn,1=mlp; comp 0=shift,1=scale,2=gate -> (B,1,vid_dim).
fn emb_param(emb: &Array, layer_idx: i32, comp: i32) -> Result<Array> {
    let m = emb.take_axis(Array::from_int(layer_idx), 2)?; // (B,vid_dim,3)
    let c = m.take_axis(Array::from_int(comp), 2)?; // (B,vid_dim)
    Ok(c.expand_dims(1)?) // (B,1,vid_dim)
}

fn modulate_in(
    hidden: &Array,
    emb: &Array,
    layer_idx: i32,
    p_shift: &Array,
    p_scale: &Array,
) -> Result<Array> {
    let shift = add(&emb_param(emb, layer_idx, 0)?, p_shift)?;
    let scale = add(&emb_param(emb, layer_idx, 1)?, p_scale)?;
    Ok(add(&multiply(hidden, &scale)?, &shift)?)
}
fn modulate_out(hidden: &Array, emb: &Array, layer_idx: i32, p_gate: &Array) -> Result<Array> {
    let gate = add(&emb_param(emb, layer_idx, 2)?, p_gate)?;
    Ok(multiply(hidden, &gate)?)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

enum Mlp {
    SwiGlu {
        proj_in: Linear,
        gate: Linear,
        proj_out: Linear,
    },
    Gelu {
        proj_in: Linear,
        proj_out: Linear,
    },
}
impl Mlp {
    fn load(w: &Weights, prefix: &str, swiglu: bool) -> Result<Self> {
        if swiglu {
            Ok(Mlp::SwiGlu {
                proj_in: Linear::load(w, &format!("{prefix}.proj_in"), false)?,
                gate: Linear::load(w, &format!("{prefix}.proj_in_gate"), false)?,
                proj_out: Linear::load(w, &format!("{prefix}.proj_out"), false)?,
            })
        } else {
            Ok(Mlp::Gelu {
                proj_in: Linear::load(w, &format!("{prefix}.proj_in"), true)?,
                proj_out: Linear::load(w, &format!("{prefix}.proj_out"), true)?,
            })
        }
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Mlp::SwiGlu {
                proj_in,
                gate,
                proj_out,
            } => {
                let g = silu(&gate.forward(x)?)?;
                proj_out.forward(&multiply(&g, &proj_in.forward(x)?)?)
            }
            Mlp::Gelu { proj_in, proj_out } => proj_out.forward(&gelu_tanh(&proj_in.forward(x)?)?),
        }
    }
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        match self {
            Mlp::SwiGlu {
                proj_in,
                gate,
                proj_out,
            } => {
                proj_in.quantize(bits, group)?;
                gate.quantize(bits, group)?;
                proj_out.quantize(bits, group)?;
            }
            Mlp::Gelu { proj_in, proj_out } => {
                proj_in.quantize(bits, group)?;
                proj_out.quantize(bits, group)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// per-forward window/RoPE cache (sc — speed): the window partition + RoPE frequency blocks depend
// only on (grid, window, shift-parity, freqs, txt_len) — identical across all blocks of a given
// shift parity — so they are computed twice per forward (shift on/off) and shared, instead of being
// rebuilt in every one of the 32/36 layers. Output is bit-identical (the parity gates verify).
// ---------------------------------------------------------------------------

struct WindowCache {
    fwd: Array,                          // (L,) windowed-order permutation
    rev: Array,                          // (L,) inverse permutation
    window_shapes: Vec<(i32, i32, i32)>, // (f,h,w) per window
    vid_freqs: Array,                    // (L, nf2*3) video RoPE freqs
    txt_freqs: Option<Array>,            // (Lt, nf2*3) text RoPE freqs (lang mode only)
}

/// Build the (shift-specific) window partition + RoPE freq blocks once. `freqs` is any block's RoPE
/// buffer (identical across blocks); `lt` the text-token count.
#[allow(clippy::too_many_arguments)]
fn build_window_cache(
    freqs: &Array,
    vid_shape: (i32, i32, i32),
    window: (i32, i32, i32),
    shift: bool,
    pixel: bool,
    rope_on_text: bool,
    lt: i32,
) -> Result<WindowCache> {
    let plan = window_plan(vid_shape.0, vid_shape.1, vid_shape.2, window, shift);
    let l = plan.forward_idx.len() as i32;
    let fwd = Array::from_slice(&plan.forward_idx, &[l]);
    let rev = Array::from_slice(&plan.reverse_idx, &[l]);
    let txt_off = if rope_on_text { lt } else { 0 };
    let mut blocks = Vec::with_capacity(plan.window_shapes.len());
    for (f, wh, ww) in &plan.window_shapes {
        blocks.push(vid_freq_block(freqs, *f, *wh, *ww, txt_off, pixel)?);
    }
    let refs: Vec<&Array> = blocks.iter().collect();
    let vid_freqs = concatenate_axis(&refs, 0)?;
    let txt_freqs = if rope_on_text {
        Some(txt_freq_block(freqs, lt)?)
    } else {
        None
    };
    Ok(WindowCache {
        fwd,
        rev,
        window_shapes: plan.window_shapes,
        vid_freqs,
        txt_freqs,
    })
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

struct MMAttention {
    qkv_vid: Linear,
    out_vid: Linear,
    nq_vid: Array,
    nk_vid: Array,
    qkv_txt: Linear,
    out_txt: Linear,
    nq_txt: Array,
    nk_txt: Array,
    freqs: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    window: (i32, i32, i32),
    rope_on_text: bool,
    rope_pixel: bool,
}
impl MMAttention {
    fn load(w: &Weights, prefix: &str, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            qkv_vid: Linear::load(w, &format!("{prefix}.proj_qkv_vid"), false)?,
            out_vid: Linear::load(w, &format!("{prefix}.proj_out_vid"), true)?,
            nq_vid: w.require(&format!("{prefix}.norm_q_vid.weight"))?.clone(),
            nk_vid: w.require(&format!("{prefix}.norm_k_vid.weight"))?.clone(),
            qkv_txt: Linear::load(w, &format!("{prefix}.proj_qkv_txt"), false)?,
            out_txt: Linear::load(w, &format!("{prefix}.proj_out_txt"), true)?,
            nq_txt: w.require(&format!("{prefix}.norm_q_txt.weight"))?.clone(),
            nk_txt: w.require(&format!("{prefix}.norm_k_txt.weight"))?.clone(),
            freqs: w.require(&format!("{prefix}.rope.freqs"))?.clone(),
            heads: cfg.heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
            eps: cfg.norm_eps,
            window: cfg.window,
            rope_on_text: cfg.rope_on_text,
            rope_pixel: cfg.rope_pixel,
        })
    }

    /// vid (1,L,vid_dim), txt (1,Lt,vid_dim) -> (vid_out (1,L,vid_dim), txt_out (1,Lt,vid_dim)). B=1.
    /// `cache` carries the shift-specific window partition + RoPE freqs (shared across same-parity
    /// blocks).
    fn forward(&self, vid: &Array, txt: &Array, cache: &WindowCache) -> Result<(Array, Array)> {
        let (h, hd) = (self.heads, self.head_dim);
        let l = vid.shape()[1];
        let lt = txt.shape()[1];
        let vid_dim = *vid.shape().last().unwrap();
        let txt_dim = *txt.shape().last().unwrap();

        let qkv_v = self
            .qkv_vid
            .forward(&vid.reshape(&[l, vid_dim])?)?
            .reshape(&[l, 3, h, hd])?;
        let qkv_t = self
            .qkv_txt
            .forward(&txt.reshape(&[lt, txt_dim])?)?
            .reshape(&[lt, 3, h, hd])?;

        let qkv_v = qkv_v.take_axis(&cache.fwd, 0)?; // windowed order (cached permutation)

        let q_v = rms_norm(
            &qkv_v.take_axis(Array::from_int(0), 1)?,
            &self.nq_vid,
            self.eps,
        )?; // (L,h,hd)
        let k_v = rms_norm(
            &qkv_v.take_axis(Array::from_int(1), 1)?,
            &self.nk_vid,
            self.eps,
        )?;
        let v_v = qkv_v.take_axis(Array::from_int(2), 1)?;
        let q_t = rms_norm(
            &qkv_t.take_axis(Array::from_int(0), 1)?,
            &self.nq_txt,
            self.eps,
        )?; // (Lt,h,hd)
        let k_t = rms_norm(
            &qkv_t.take_axis(Array::from_int(1), 1)?,
            &self.nk_txt,
            self.eps,
        )?;
        let v_t = qkv_t.take_axis(Array::from_int(2), 1)?;

        // RoPE: cached vid/txt freq blocks (shared across same-parity blocks).
        let q_v = apply_rope(&q_v, &cache.vid_freqs)?;
        let k_v = apply_rope(&k_v, &cache.vid_freqs)?;
        let (q_t, k_t) = match &cache.txt_freqs {
            Some(tf) => (apply_rope(&q_t, tf)?, apply_rope(&k_t, tf)?),
            None => (q_t, k_t),
        };

        // per-window joint attention; vid scattered back, txt averaged across windows
        let nwin = cache.window_shapes.len();
        let mut vid_out_parts: Vec<Array> = Vec::with_capacity(nwin);
        let mut txt_acc: Option<Array> = None;
        let mut start = 0i32;
        for (f, wh, ww) in &cache.window_shapes {
            let vlen = f * wh * ww;
            let idx = Array::from_slice(&(start..start + vlen).collect::<Vec<i32>>(), &[vlen]);
            let qv = q_v.take_axis(&idx, 0)?;
            let kv = k_v.take_axis(&idx, 0)?;
            let vv = v_v.take_axis(&idx, 0)?;
            // concat vid window + all txt -> (vlen+Lt, h, hd)
            let q = concatenate_axis(&[&qv, &q_t], 0)?;
            let k = concatenate_axis(&[&kv, &k_t], 0)?;
            let v = concatenate_axis(&[&vv, &v_t], 0)?;
            // -> (1, h, S, hd)
            let to_bhsd = |x: &Array| -> Result<Array> {
                Ok(x.reshape(&[1, vlen + lt, h, hd])?
                    .transpose_axes(&[0, 2, 1, 3])?)
            };
            let o = scaled_dot_product_attention(
                &to_bhsd(&q)?,
                &to_bhsd(&k)?,
                &to_bhsd(&v)?,
                self.scale,
                None,
                None,
            )?;
            let o = o
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[vlen + lt, h, hd])?;
            let v_part = o.take_axis(
                Array::from_slice(&(0..vlen).collect::<Vec<i32>>(), &[vlen]),
                0,
            )?;
            let t_part = o.take_axis(
                Array::from_slice(&(vlen..vlen + lt).collect::<Vec<i32>>(), &[lt]),
                0,
            )?;
            vid_out_parts.push(v_part);
            txt_acc = Some(match txt_acc {
                Some(a) => add(&a, &t_part)?,
                None => t_part,
            });
            start += vlen;
        }

        // vid: concat windows -> (L,h,hd) -> reshape (L, h*hd) -> reverse permutation (cached)
        let vid_refs: Vec<&Array> = vid_out_parts.iter().collect();
        let vid_cat = concatenate_axis(&vid_refs, 0)?.reshape(&[l, h * hd])?;
        let vid_unwin = vid_cat.take_axis(&cache.rev, 0)?;
        let vid_out = self
            .out_vid
            .forward(&vid_unwin)?
            .reshape(&[1, l, vid_dim])?;

        // txt: mean over windows
        let txt_mean = multiply(txt_acc.unwrap(), Array::from_f32(1.0 / nwin as f32))?
            .reshape(&[lt, h * hd])?;
        let txt_out = self
            .out_txt
            .forward(&txt_mean)?
            .reshape(&[1, lt, txt_dim])?;

        Ok((vid_out, txt_out))
    }

    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        self.qkv_vid.quantize(bits, group)?;
        self.out_vid.quantize(bits, group)?;
        self.qkv_txt.quantize(bits, group)?;
        self.out_txt.quantize(bits, group) // norm_q/k are RMSNorm weights, not Linear
    }
}

// ---------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------

struct Block {
    attn: MMAttention,
    mlp_vid: Mlp,
    mlp_txt: Option<Mlp>, // None when shared (uses mlp_vid as `.all`) handled via shared flag
    mlp_all: Option<Mlp>,
    ada_vid: AdaParams,
    ada_txt: Option<AdaParams>,
    ada_all: Option<AdaParams>,
    shared: bool,
    is_last: bool,
    eps: f32,
}
impl Block {
    fn load(w: &Weights, idx: i32, cfg: &DitConfig) -> Result<Self> {
        let prefix = format!("blocks.{idx}");
        let shared = idx >= cfg.mm_layers;
        let is_last = cfg.last_layer_vid_only && idx == cfg.num_layers - 1;
        let attn = MMAttention::load(w, &format!("{prefix}.attn"), cfg)?;
        let (mlp_vid, mlp_txt, mlp_all) = if shared {
            (
                Mlp::load(w, &format!("{prefix}.mlp.all"), cfg.swiglu_mlp)?,
                None,
                Some(Mlp::load(w, &format!("{prefix}.mlp.all"), cfg.swiglu_mlp)?),
            )
        } else {
            let txt = if is_last {
                None
            } else {
                Some(Mlp::load(w, &format!("{prefix}.mlp.txt"), cfg.swiglu_mlp)?)
            };
            (
                Mlp::load(w, &format!("{prefix}.mlp.vid"), cfg.swiglu_mlp)?,
                txt,
                None,
            )
        };
        let (ada_vid, ada_txt, ada_all) = if shared {
            (
                AdaParams::load(w, &format!("{prefix}.ada.params_all"))?,
                None,
                Some(AdaParams::load(w, &format!("{prefix}.ada.params_all"))?),
            )
        } else {
            let txt = if is_last {
                None
            } else {
                Some(AdaParams::load(w, &format!("{prefix}.ada.params_txt"))?)
            };
            (
                AdaParams::load(w, &format!("{prefix}.ada.params_vid"))?,
                txt,
                None,
            )
        };
        Ok(Self {
            attn,
            mlp_vid,
            mlp_txt,
            mlp_all,
            ada_vid,
            ada_txt,
            ada_all,
            shared,
            is_last,
            eps: cfg.norm_eps,
        })
    }

    fn ada_v(&self) -> &AdaParams {
        if self.shared {
            self.ada_all.as_ref().unwrap()
        } else {
            &self.ada_vid
        }
    }
    fn ada_t(&self) -> &AdaParams {
        if self.shared {
            self.ada_all.as_ref().unwrap()
        } else {
            self.ada_txt.as_ref().unwrap()
        }
    }
    fn mlp_v(&self) -> &Mlp {
        if self.shared {
            self.mlp_all.as_ref().unwrap()
        } else {
            &self.mlp_vid
        }
    }
    fn mlp_t(&self) -> &Mlp {
        if self.shared {
            self.mlp_all.as_ref().unwrap()
        } else {
            self.mlp_txt.as_ref().unwrap()
        }
    }

    fn forward(
        &self,
        vid: &Array,
        txt: &Array,
        emb: &Array,
        cache: &WindowCache,
    ) -> Result<(Array, Array)> {
        // attention
        let av = self.ada_v();
        let mut vid_attn = modulate_in(
            &rms_plain(vid, self.eps)?,
            emb,
            0,
            &av.attn_shift,
            &av.attn_scale,
        )?;
        let mut txt_attn = rms_plain(txt, self.eps)?;
        if !self.is_last {
            let at = self.ada_t();
            txt_attn = modulate_in(&txt_attn, emb, 0, &at.attn_shift, &at.attn_scale)?;
        }
        let (va, ta) = self.attn.forward(&vid_attn, &txt_attn, cache)?;
        vid_attn = modulate_out(&va, emb, 0, &av.attn_gate)?;
        let vid = add(vid, &vid_attn)?;
        let txt = if self.is_last {
            txt.clone()
        } else {
            let ta = modulate_out(&ta, emb, 0, &self.ada_t().attn_gate)?;
            add(txt, &ta)?
        };

        // mlp
        let mut vid_mlp = modulate_in(
            &rms_plain(&vid, self.eps)?,
            emb,
            1,
            &av.mlp_shift,
            &av.mlp_scale,
        )?;
        let txt_mlp_in = if self.is_last {
            txt.clone()
        } else {
            let at = self.ada_t();
            modulate_in(
                &rms_plain(&txt, self.eps)?,
                emb,
                1,
                &at.mlp_shift,
                &at.mlp_scale,
            )?
        };
        vid_mlp = self.mlp_v().forward(&vid_mlp)?;
        vid_mlp = modulate_out(&vid_mlp, emb, 1, &av.mlp_gate)?;
        let vid = add(&vid, &vid_mlp)?;
        let txt = if self.is_last {
            txt
        } else {
            let mut tm = self.mlp_t().forward(&txt_mlp_in)?;
            tm = modulate_out(&tm, emb, 1, &self.ada_t().mlp_gate)?;
            add(&txt, &tm)?
        };
        Ok((vid, txt))
    }

    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        self.attn.quantize(bits, group)?;
        self.mlp_vid.quantize(bits, group)?;
        if let Some(m) = &mut self.mlp_txt {
            m.quantize(bits, group)?;
        }
        if let Some(m) = &mut self.mlp_all {
            m.quantize(bits, group)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// transformer
// ---------------------------------------------------------------------------

pub struct Seedvr2Transformer {
    vid_in: PatchIn,
    txt_in: Linear,
    emb_in: TimeEmbedding,
    blocks: Vec<Block>,
    vid_out_norm: Option<Array>,
    out_shift: Option<Array>,
    out_scale: Option<Array>,
    vid_out: PatchOut,
    vid_dim: i32,
    eps: f32,
    use_output_ada: bool,
}

impl Seedvr2Transformer {
    pub fn from_weights(w: &Weights, cfg: &DitConfig) -> Result<Self> {
        let blocks = (0..cfg.num_layers)
            .map(|i| Block::load(w, i, cfg))
            .collect::<Result<Vec<_>>>()?;
        let (vid_out_norm, out_shift, out_scale) = if cfg.use_output_ada {
            (
                Some(w.require("vid_out_norm.weight")?.clone()),
                Some(w.require("out_shift")?.clone()),
                Some(w.require("out_scale")?.clone()),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            vid_in: PatchIn::load(w, "vid_in", cfg)?,
            txt_in: Linear::load(w, "txt_in", true)?,
            emb_in: TimeEmbedding::load(w, "emb_in")?,
            blocks,
            vid_out_norm,
            out_shift,
            out_scale,
            vid_out: PatchOut::load(w, "vid_out", cfg)?,
            vid_dim: cfg.vid_dim,
            eps: cfg.norm_eps,
            use_output_ada: cfg.use_output_ada,
        })
    }

    /// vid `(1,33,T,H,W)`, txt `(1,Lt,5120)`, timestep scalar -> `(1,16,T,H,W)`.
    pub fn forward(&self, vid: &Array, txt: &Array, timestep: &Array) -> Result<Array> {
        let txt = self.txt_in.forward(txt)?;
        let (mut vid, vid_shape) = self.vid_in.forward(vid)?;
        let emb = self
            .emb_in
            .forward(timestep)?
            .reshape(&[-1, self.vid_dim, 2, 3])?;
        let mut txt = txt;

        // Build the window partition + RoPE freqs once per shift parity (shared across blocks). The
        // RoPE `freqs` buffer is identical across blocks (a fixed base-frequency vector), so block 0's
        // is representative; the grid is constant through the stack.
        let lt = txt.shape()[1];
        let a0 = &self.blocks[0].attn;
        let cache_even = build_window_cache(
            &a0.freqs,
            vid_shape,
            a0.window,
            false,
            a0.rope_pixel,
            a0.rope_on_text,
            lt,
        )?;
        let cache_odd = build_window_cache(
            &a0.freqs,
            vid_shape,
            a0.window,
            true,
            a0.rope_pixel,
            a0.rope_on_text,
            lt,
        )?;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = if i % 2 == 1 { &cache_odd } else { &cache_even };
            let (v, t) = block.forward(&vid, &txt, &emb, cache)?;
            vid = v;
            txt = t;
        }
        if self.use_output_ada {
            vid = rms_norm(&vid, self.vid_out_norm.as_ref().unwrap(), self.eps)?;
            let shift_a = emb_param(&emb, 0, 0)?; // (1,1,vid_dim)
            let scale_a = emb_param(&emb, 0, 1)?;
            let scale = add(&scale_a, self.out_scale.as_ref().unwrap())?;
            let shift = add(&shift_a, self.out_shift.as_ref().unwrap())?;
            vid = add(&multiply(&vid, &scale)?, &shift)?;
        }
        self.vid_out.forward(&vid, vid_shape)
    }

    /// Quantize every Linear to `bits` (group-wise affine, group 64) — sc-5198. Linear-only by
    /// construction (the DiT has no convs); `vid_in.proj` (in=132) is auto-skipped by the
    /// in-features-divisibility predicate, matching the reference. Idempotent / safe to call once.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        let group = mlx_gen::quant::DEFAULT_GROUP_SIZE;
        self.vid_in.quantize(bits, group)?;
        self.txt_in.quantize(bits, group)?;
        self.emb_in.quantize(bits, group)?;
        for block in &mut self.blocks {
            block.quantize(bits, group)?;
        }
        self.vid_out.quantize(bits, group)
    }
}

#[cfg(test)]
mod stage_tests {
    use super::*;

    fn gdir() -> std::path::PathBuf {
        std::env::var("SEEDVR2_GOLDEN_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(&std::env::var("HOME").unwrap())
                    .join(".cache/mlx-gen-seedvr2-golden")
            })
    }

    fn cmp(label: &str, got: &Array, exp: &Array) -> f32 {
        assert_eq!(
            got.shape(),
            exp.shape(),
            "{label} shape {:?} vs {:?}",
            got.shape(),
            exp.shape()
        );
        let g = got
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap();
        let e = exp.reshape(&[-1]).unwrap();
        let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
        let (mut dot, mut na, mut nb, mut maxd, mut maxr) = (0f64, 0f64, 0f64, 0f32, 0f32);
        for (a, b) in gs.iter().zip(es.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
            maxd = maxd.max((a - b).abs());
            maxr = maxr.max(b.abs());
        }
        let cos = (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32;
        eprintln!(
            "[{label}] {:?} cosine={cos:.6} peak_rel={:.3e}",
            got.shape(),
            maxd / maxr.max(1e-12)
        );
        cos
    }

    #[test]
    fn dit_stage_localize() {
        let dir = gdir();
        if !dir.join("dit_io_f32.safetensors").exists() {
            eprintln!("SKIP: no dit goldens");
            return;
        }
        let w = Weights::from_file(dir.join("dit_f32.safetensors")).unwrap();
        let io = Weights::from_file(dir.join("dit_io_f32.safetensors")).unwrap();
        let cfg = DitConfig::seedvr2_3b();
        let m = Seedvr2Transformer::from_weights(&w, &cfg).unwrap();

        cmp(
            "txt_proj",
            &m.txt_in.forward(io.require("txt").unwrap()).unwrap(),
            io.require("txt_proj").unwrap(),
        );
        let (vid_tok, _) = m.vid_in.forward(io.require("vid").unwrap()).unwrap();
        cmp("vid_tok", &vid_tok, io.require("vid_tok").unwrap());
        cmp(
            "emb",
            &m.emb_in.forward(io.require("timestep").unwrap()).unwrap(),
            io.require("emb").unwrap(),
        );

        let out = m
            .forward(
                io.require("vid").unwrap(),
                io.require("txt").unwrap(),
                io.require("timestep").unwrap(),
            )
            .unwrap();
        let cos = cmp("dit_out", &out, io.require("dit_out").unwrap());
        assert!(cos > 0.999, "dit_out cosine {cos} too low");
    }
}
