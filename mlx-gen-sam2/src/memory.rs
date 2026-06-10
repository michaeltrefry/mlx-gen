//! SAM2 memory bank — **memory encoder** + **memory attention** — port of `mlx_sam/models/memory.py`
//! (`avbiswas/sam2-mlx`). This is the Phase-B video layer (sc-3713) that turns the Phase-A image
//! segmenter into a temporally-consistent video predictor:
//!
//!   * [`MemoryEncoder`] — encodes the current frame's `(vision_features, predicted mask)` into a
//!     64-channel **memory feature map** + its sinusoidal position encoding. These are what get
//!     stored in the memory bank and replayed on later frames.
//!   * [`MemoryAttention`] — conditions the current frame's flattened image tokens on the memory
//!     bank: 4 transformer layers of RoPE self-attention + RoPE cross-attention (current tokens →
//!     memory) + a feed-forward, then a final norm.
//!
//! The propagation loop that *drives* these (assembles the bank across frames, threads object
//! pointers, runs `init_state`/`propagate`) is the next story (sc-3714); this slice is the two
//! standalone forward passes plus their parity vs the MLX reference.
//!
//! Conventions mirror the reference exactly so the golden parity is near-bit (both run MLX Metal):
//!   * Feature maps are **NCHW** at the module boundary (transposed to MLX's NHWC for convs).
//!   * Token tensors are **sequence-first** `[seq, batch, dim]` (the reference's transformer layout)
//!     at the [`MemoryAttention`] boundary; the layers transpose to batch-first internally.
//!   * `nn.LayerNorm` uses eps `1e-5`; `LayerNorm2d` uses `1e-6`.

use std::cell::RefCell;
use std::collections::HashMap;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{
    self, add, broadcast_to, concatenate_axis, multiply, sigmoid, stack_axis, subtract,
};
use mlx_rs::Array;

use mlx_gen::nn::{gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::image_encoder::position_encoding;

/// `nn.LayerNorm` default epsilon (memory-attention norms).
const LN_EPS: f32 = 1e-5;
/// `LayerNorm2d` epsilon (memory-encoder norms).
const LN2D_EPS: f32 = 1e-6;
/// Memory feature channels (`out_proj` output / cross-attn `kv_in_dim`).
const MEM_DIM: i32 = 64;
/// Transformer working dim.
const D_MODEL: i32 = 256;
/// Axial-RoPE base.
const ROPE_THETA: f64 = 10000.0;

fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

fn weight_bias(w: &Weights, prefix: &str) -> Result<(Array, Array)> {
    Ok((
        w.require(&join(prefix, "weight"))?.clone(),
        w.require(&join(prefix, "bias"))?.clone(),
    ))
}

/// 2-D conv over an **NCHW** input with an OHWI weight (+ bias) and a `groups` factor (the reference
/// `CXBlock.dwconv` is depthwise, `groups = channels`; everything else is `groups = 1`).
fn conv2d_nchw(
    x: &Array,
    w: &Array,
    b: &Array,
    stride: i32,
    pad: i32,
    groups: i32,
) -> Result<Array> {
    let y = ops::conv2d(
        x.transpose_axes(&[0, 2, 3, 1])?,
        w,
        (stride, stride),
        (pad, pad),
        (1, 1),
        groups,
    )?;
    let y = add(&y, b)?; // bias over the last (channel) axis, NHWC
    Ok(y.transpose_axes(&[0, 3, 1, 2])?)
}

/// `LayerNorm2d`: normalize an **NCHW** tensor over the channel axis (per spatial position).
fn layer_norm_2d(x: &Array, weight: &Array, bias: &Array) -> Result<Array> {
    let mean = ops::mean_axes(x, &[1], true)?;
    let centered = subtract(x, &mean)?;
    let var = ops::mean_axes(&ops::square(&centered)?, &[1], true)?;
    let normed = multiply(
        &centered,
        &ops::rsqrt(&add(&var, Array::from_f32(LN2D_EPS))?)?,
    )?;
    let wt = weight.reshape(&[1, -1, 1, 1])?;
    let bs = bias.reshape(&[1, -1, 1, 1])?;
    Ok(add(&multiply(&normed, &wt)?, &bs)?)
}

// ---------------------------------------------------------------------------------------------
// Memory encoder
// ---------------------------------------------------------------------------------------------

/// `MaskDownSampler`: 4× stride-2 conv+LayerNorm2d+GELU stages (1→4→16→64→256) then a 1×1 conv,
/// shrinking a `[B,1,4S,4S]` mask to `[B,256,S,S]` aligned with the image feature grid.
struct MaskDownSampler {
    /// `(weight, bias, stride, pad)` per stage conv (`conv0..conv4`).
    convs: Vec<(Array, Array, i32, i32)>,
    /// `(weight, bias)` per `LayerNorm2d` (`norm0..norm3`).
    norms: Vec<(Array, Array)>,
}

impl MaskDownSampler {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let convs = ["conv0", "conv1", "conv2", "conv3", "conv4"]
            .iter()
            .enumerate()
            .map(|(i, name)| -> Result<(Array, Array, i32, i32)> {
                let (cw, cb) = weight_bias(w, &p(name))?;
                // conv0..conv3 are k3/s2/p1; conv4 is k1/s1/p0.
                let (stride, pad) = if i < 4 { (2, 1) } else { (1, 0) };
                Ok((cw, cb, stride, pad))
            })
            .collect::<Result<Vec<_>>>()?;
        let norms = ["norm0", "norm1", "norm2", "norm3"]
            .iter()
            .map(|name| weight_bias(w, &p(name)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { convs, norms })
    }

    /// `x`: NCHW `[B,1,H,W]` → `[B,256,H/16,W/16]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for i in 0..4 {
            let (cw, cb, stride, pad) = &self.convs[i];
            x = conv2d_nchw(&x, cw, cb, *stride, *pad, 1)?;
            let (nw, nb) = &self.norms[i];
            x = gelu_exact(&layer_norm_2d(&x, nw, nb)?)?;
        }
        let (cw, cb, stride, pad) = &self.convs[4];
        conv2d_nchw(&x, cw, cb, *stride, *pad, 1)
    }
}

/// `CXBlock`: a ConvNeXt-style residual block — 7×7 depthwise conv, `LayerNorm2d`, a channel MLP
/// (1×1 expand→GELU→1×1 project) gated by a learned `gamma`, added to the input.
struct CxBlock {
    gamma: Array,           // [256]
    dwconv: (Array, Array), // depthwise k7/p3, groups=256
    norm: (Array, Array),   // LayerNorm2d(256)
    pwconv1: (Array, Array),
    pwconv2: (Array, Array),
}

impl CxBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            gamma: w.require(&p("gamma"))?.clone(),
            dwconv: weight_bias(w, &p("dwconv"))?,
            norm: weight_bias(w, &p("norm"))?,
            pwconv1: weight_bias(w, &p("pwconv1"))?,
            pwconv2: weight_bias(w, &p("pwconv2"))?,
        })
    }

    /// `x`: NCHW `[B,256,H,W]` → same shape.
    fn forward(&self, x: &Array) -> Result<Array> {
        let residual = x;
        // Depthwise conv keeps channels; groups == channels (256).
        let h = conv2d_nchw(x, &self.dwconv.0, &self.dwconv.1, 1, 3, D_MODEL)?;
        let h = layer_norm_2d(&h, &self.norm.0, &self.norm.1)?;
        // Channel MLP runs in NHWC (last-dim linears), gated by gamma, then back to NCHW.
        let h = h.transpose_axes(&[0, 2, 3, 1])?;
        let h = linear(
            &gelu_exact(&linear(&h, &self.pwconv1.0, &self.pwconv1.1)?)?,
            &self.pwconv2.0,
            &self.pwconv2.1,
        )?;
        let h = multiply(&self.gamma, &h)?;
        Ok(add(residual, &h.transpose_axes(&[0, 3, 1, 2])?)?)
    }
}

/// SAM2 memory encoder output: a 64-channel memory feature map + its position encoding (both NCHW).
pub struct MemoryEncoderOutput {
    pub vision_features: Array,
    /// One position-encoding level (matches the reference's single-element list).
    pub vision_pos_enc: Array,
}

/// `MemoryEncoder`: downsample the predicted mask, project + fuse it with the frame's image
/// features, and project to the 64-channel memory feature map stored in the memory bank.
pub struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    pix_feat_proj: (Array, Array), // 1×1 256→256
    fuser: Vec<CxBlock>,
    out_proj: (Array, Array), // 1×1 256→64
}

impl MemoryEncoder {
    /// Build from a converted SAM2 checkpoint (`memory_encoder.*` keys).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            mask_downsampler: MaskDownSampler::from_weights(w, &p("mask_downsampler"))?,
            pix_feat_proj: weight_bias(w, &p("pix_feat_proj"))?,
            fuser: vec![
                CxBlock::from_weights(w, &p("fuser.0"))?,
                CxBlock::from_weights(w, &p("fuser.1"))?,
            ],
            out_proj: weight_bias(w, &p("out_proj"))?,
        })
    }

    /// `pix_feat`: NCHW `[B,256,S,S]` image features; `masks_high_res`: NCHW `[B,1,16S,16S]` mask
    /// logits. `skip_mask_sigmoid` mirrors the reference flag (the predictor pre-scales the mask and
    /// passes `true`). Returns the memory feature map `[B,64,S,S]` + its position encoding.
    pub fn forward(
        &self,
        pix_feat: &Array,
        masks_high_res: &Array,
        skip_mask_sigmoid: bool,
    ) -> Result<MemoryEncoderOutput> {
        let masks = if skip_mask_sigmoid {
            masks_high_res.clone()
        } else {
            sigmoid(masks_high_res)?
        };
        let masks = self.mask_downsampler.forward(&masks)?; // [B,256,S,S]

        let mut x = conv2d_nchw(
            pix_feat,
            &self.pix_feat_proj.0,
            &self.pix_feat_proj.1,
            1,
            0,
            1,
        )?;
        x = add(&x, &masks)?;
        for layer in &self.fuser {
            x = layer.forward(&x)?;
        }
        let x = conv2d_nchw(&x, &self.out_proj.0, &self.out_proj.1, 1, 0, 1)?; // [B,64,S,S]

        // Sinusoidal PE over the feature grid (num_pos_feats=64), NCHW to match the reference.
        let pos = position_encoding(&x.transpose_axes(&[0, 2, 3, 1])?, MEM_DIM)?;
        let vision_pos_enc = pos.transpose_axes(&[0, 3, 1, 2])?.as_dtype(x.dtype())?;
        Ok(MemoryEncoderOutput {
            vision_features: x,
            vision_pos_enc,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Axial RoPE
// ---------------------------------------------------------------------------------------------

/// Axial rotary position embedding tables (`compute_axial_rope`). Returns `(cos, sin)`, each
/// `[end_x·end_y, head_dim/2]`: the first `head_dim/4` columns encode the x-position, the next
/// `head_dim/4` the y-position. Built on the host (f32) so the values are deterministic.
fn axial_rope_tables(head_dim: i32, end_x: i32, end_y: i32) -> (Array, Array) {
    let quarter = (head_dim / 4) as usize; // arange(0, dim, 4)[: dim // 4]
                                           // freqs[j] = theta ^ -(4j / head_dim)
    let freqs: Vec<f64> = (0..quarter)
        .map(|j| ROPE_THETA.powf(-((4 * j) as f64) / head_dim as f64))
        .collect();
    let n = (end_x * end_y) as usize;
    let cols = 2 * quarter;
    let mut cos = vec![0f32; n * cols];
    let mut sin = vec![0f32; n * cols];
    for t in 0..n {
        let tx = (t % end_x as usize) as f64;
        let ty = (t / end_x as usize) as f64; // floor(t / end_x)
        for (j, &f) in freqs.iter().enumerate() {
            let px = tx * f;
            let py = ty * f;
            cos[t * cols + j] = px.cos() as f32;
            sin[t * cols + j] = px.sin() as f32;
            cos[t * cols + quarter + j] = py.cos() as f32;
            sin[t * cols + quarter + j] = py.sin() as f32;
        }
    }
    let shape = [n as i32, cols as i32];
    (
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    )
}

/// Repeat each row of a `[N, C]` table `repeat` times consecutively → `[N·repeat, C]`
/// (`mx.repeat(t[:, None, :], repeat, axis=1).reshape(-1, C)`).
fn repeat_rows(t: &Array, repeat: i32) -> Result<Array> {
    let sh = t.shape();
    let (n, c) = (sh[0], sh[1]);
    let t = broadcast_to(&t.reshape(&[n, 1, c])?, &[n, repeat, c])?;
    Ok(t.reshape(&[n * repeat, c])?)
}

/// `(cos, sin)` RoPE tables, keyed by grid shape in the caches below.
type RopeTables = (Array, Array);

thread_local! {
    /// Axial RoPE tables keyed by `(head_dim, end_x, end_y)`. Every caller in the per-frame video
    /// loop hits a fixed 64×64 grid with a constant `head_dim`, so the ~1M host trig evaluations +
    /// ~4 MB upload that `axial_rope_tables` performs run **once** instead of 8× per frame (F-167).
    /// MLX `Array`s are ref-counted handles, so returning a cache hit is a cheap clone. Thread-local
    /// so the cache needs no `Send`/`Sync` guarantees on `Array`.
    static ROPE_TABLE_CACHE: RefCell<HashMap<(i32, i32, i32), RopeTables>> =
        RefCell::new(HashMap::new());
    /// The cross-attention `repeat_rows` variant keyed by `(head_dim, end_x, end_y, repeat)` — the
    /// tiled tables are also constant across the video, so cache them too rather than re-tiling per
    /// layer per frame.
    static ROPE_REPEAT_CACHE: RefCell<HashMap<(i32, i32, i32, i32), RopeTables>> =
        RefCell::new(HashMap::new());
}

/// Memoized [`axial_rope_tables`] (F-167): the tables are a pure function of the key, so the first
/// call for a given grid builds them and every later call returns a cheap handle clone.
fn axial_rope_tables_cached(head_dim: i32, end_x: i32, end_y: i32) -> (Array, Array) {
    let key = (head_dim, end_x, end_y);
    if let Some(hit) = ROPE_TABLE_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit;
    }
    let tables = axial_rope_tables(head_dim, end_x, end_y);
    ROPE_TABLE_CACHE.with(|c| c.borrow_mut().insert(key, tables.clone()));
    tables
}

/// The `repeat`-tiled axial RoPE tables for cross-attention, memoized (F-167). `repeat == 1` is the
/// un-tiled base.
fn axial_rope_tables_repeated(
    head_dim: i32,
    end_x: i32,
    end_y: i32,
    repeat: i32,
) -> Result<(Array, Array)> {
    if repeat == 1 {
        return Ok(axial_rope_tables_cached(head_dim, end_x, end_y));
    }
    let key = (head_dim, end_x, end_y, repeat);
    if let Some(hit) = ROPE_REPEAT_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(hit);
    }
    let (cos, sin) = axial_rope_tables_cached(head_dim, end_x, end_y);
    let tiled = (repeat_rows(&cos, repeat)?, repeat_rows(&sin, repeat)?);
    ROPE_REPEAT_CACHE.with(|c| c.borrow_mut().insert(key, tiled.clone()));
    Ok(tiled)
}

/// Apply interleaved axial RoPE to `x` `[b, heads, tokens, head_dim]`; `cos`/`sin` are
/// `[tokens, head_dim/2]` (broadcast over batch/heads). Matches `apply_rope` (even/odd interleave).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, n, d) = (sh[0], sh[1], sh[2], sh[3]);
    let pair = x.reshape(&[b, h, n, d / 2, 2])?;
    let even = pair.take_axis(Array::from_int(0), 4)?; // [b,h,n,d/2]
    let odd = pair.take_axis(Array::from_int(1), 4)?;
    let cos = cos.reshape(&[1, 1, n, d / 2])?;
    let sin = sin.reshape(&[1, 1, n, d / 2])?;
    let out_even = subtract(&multiply(&even, &cos)?, &multiply(&odd, &sin)?)?;
    let out_odd = add(&multiply(&even, &sin)?, &multiply(&odd, &cos)?)?;
    Ok(stack_axis(&[&out_even, &out_odd], 4)?.reshape(&[b, h, n, d])?)
}

/// `x[:, :, start..end, :]` over the token axis (axis 2).
fn take_token_range(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
    Ok(x.take_axis(&idx, 2)?)
}

// ---------------------------------------------------------------------------------------------
// Memory attention
// ---------------------------------------------------------------------------------------------

/// A RoPE attention head bank (`RoPEAttention`): q/k/v/out projections + axial RoPE on q and the
/// rope-eligible part of k. `num_heads` is 1 throughout SAM2's memory attention.
struct RoPEAttention {
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    out: (Array, Array),
    num_heads: i32,
    /// Tile the q-grid RoPE across the (longer) memory key sequence (cross-attention).
    rope_k_repeat: bool,
}

impl RoPEAttention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        rope_k_repeat: bool,
    ) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            q: weight_bias(w, &p("q_proj"))?,
            k: weight_bias(w, &p("k_proj"))?,
            v: weight_bias(w, &p("v_proj"))?,
            out: weight_bias(w, &p("out_proj"))?,
            num_heads,
            rope_k_repeat,
        })
    }

    /// Split `[b, n, c]` into heads `[b, num_heads, n, c/num_heads]`.
    fn sep_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n, c) = (sh[0], sh[1], sh[2]);
        Ok(x.reshape(&[b, n, self.num_heads, c / self.num_heads])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }

    /// `num_k_exclude_rope` trailing key tokens (object pointers) skip RoPE.
    fn forward(&self, q: &Array, k: &Array, v: &Array, num_k_exclude_rope: i32) -> Result<Array> {
        let q = self.sep_heads(&linear(q, &self.q.0, &self.q.1)?)?;
        let k = self.sep_heads(&linear(k, &self.k.0, &self.k.1)?)?;
        let v = self.sep_heads(&linear(v, &self.v.0, &self.v.1)?)?;

        let q_len = q.shape()[2];
        let head_dim = q.shape()[3];
        let k_len = k.shape()[2];
        // A square spatial grid (end_x == end_y) when q_len is a perfect square, else a 1-D line.
        let side = (q_len as f64).sqrt().floor() as i32;
        let (end_x, end_y) = if side * side == q_len {
            (side, side)
        } else {
            (q_len, 1)
        };
        let (cos, sin) = axial_rope_tables_cached(head_dim, end_x, end_y);
        let repeat = if self.rope_k_repeat && q_len != 0 {
            k_len / q_len
        } else {
            1
        };

        let q = apply_rope(&q, &cos, &sin)?;
        let rope_len = k_len - num_k_exclude_rope;
        let k = if rope_len > 0 {
            let (ck, sk) = axial_rope_tables_repeated(head_dim, end_x, end_y, repeat)?;
            let k_rope = apply_rope(&take_token_range(&k, 0, rope_len)?, &ck, &sk)?;
            if num_k_exclude_rope > 0 {
                concatenate_axis(&[&k_rope, &take_token_range(&k, rope_len, k_len)?], 2)?
            } else {
                k_rope
            }
        } else {
            k
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        // Recombine heads → [b, n, num_heads·head_dim].
        let sh = out.shape();
        let (b, h, n, c) = (sh[0], sh[1], sh[2], sh[3]);
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * c])?;
        linear(&out, &self.out.0, &self.out.1)
    }
}

/// One `MemoryAttentionLayer`: RoPE self-attention, RoPE cross-attention (tokens → memory), FFN.
struct MemoryAttentionLayer {
    self_attn: RoPEAttention,
    cross_attn: RoPEAttention,
    linear1: (Array, Array),
    linear2: (Array, Array),
    norm1: (Array, Array),
    norm2: (Array, Array),
    norm3: (Array, Array),
}

impl MemoryAttentionLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            self_attn: RoPEAttention::from_weights(w, &p("self_attn"), 1, false)?,
            cross_attn: RoPEAttention::from_weights(w, &p("cross_attn_image"), 1, true)?,
            linear1: weight_bias(w, &p("linear1"))?,
            linear2: weight_bias(w, &p("linear2"))?,
            norm1: weight_bias(w, &p("norm1"))?,
            norm2: weight_bias(w, &p("norm2"))?,
            norm3: weight_bias(w, &p("norm3"))?,
        })
    }

    /// `tgt`/`memory`/`memory_pos` are batch-first `[b, seq, dim]`. (The reference layer also accepts
    /// a `query_pos`, but this RoPE variant never uses it — positional info is injected via RoPE.)
    fn forward(
        &self,
        tgt: &Array,
        memory: &Array,
        memory_pos: &Array,
        num_obj_ptr_tokens: i32,
    ) -> Result<Array> {
        let ln = |x: &Array, n: &(Array, Array)| layer_norm(x, Some(&n.0), Some(&n.1), LN_EPS);

        let tgt2 = ln(tgt, &self.norm1)?;
        let tgt = add(tgt, &self.self_attn.forward(&tgt2, &tgt2, &tgt2, 0)?)?;

        let tgt2 = ln(&tgt, &self.norm2)?;
        let key = add(memory, memory_pos)?;
        let tgt = add(
            &tgt,
            &self
                .cross_attn
                .forward(&tgt2, &key, memory, num_obj_ptr_tokens)?,
        )?;

        let tgt2 = ln(&tgt, &self.norm3)?;
        let ff = linear(
            &relu(&linear(&tgt2, &self.linear1.0, &self.linear1.1)?)?,
            &self.linear2.0,
            &self.linear2.1,
        )?;
        Ok(add(&tgt, &ff)?)
    }
}

/// `MemoryAttention`: 4 [`MemoryAttentionLayer`]s + a final norm. Conditions the current frame's
/// image tokens on the assembled memory bank, producing the memory-conditioned image embeddings the
/// mask decoder consumes on a video frame.
pub struct MemoryAttention {
    layers: Vec<MemoryAttentionLayer>,
    norm: (Array, Array),
}

impl MemoryAttention {
    /// Build from a converted SAM2 checkpoint (`memory_attention.*` keys).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let layers = (0..4)
            .map(|i| MemoryAttentionLayer::from_weights(w, &p(&format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            norm: weight_bias(w, &p("norm"))?,
        })
    }

    /// `curr`/`curr_pos`: current-frame image tokens + position encoding, sequence-first
    /// `[seq, batch, 256]`. `memory`/`memory_pos`: the assembled memory bank, sequence-first
    /// `[mem, batch, 64]`. `num_obj_ptr_tokens` is the count of trailing object-pointer tokens in
    /// `memory` (they skip RoPE). Returns the conditioned tokens, sequence-first `[seq, batch, 256]`.
    pub fn forward(
        &self,
        curr: &Array,
        curr_pos: &Array,
        memory: &Array,
        memory_pos: &Array,
        num_obj_ptr_tokens: i32,
    ) -> Result<Array> {
        // output = curr + 0.1 * curr_pos, then batch-first for the layers.
        let mut output =
            add(curr, &multiply(curr_pos, Array::from_f32(0.1))?)?.transpose_axes(&[1, 0, 2])?;
        let memory = memory.transpose_axes(&[1, 0, 2])?;
        let memory_pos = memory_pos.transpose_axes(&[1, 0, 2])?;
        for layer in &self.layers {
            output = layer.forward(&output, &memory, &memory_pos, num_obj_ptr_tokens)?;
        }
        let output = layer_norm(&output, Some(&self.norm.0), Some(&self.norm.1), LN_EPS)?;
        Ok(output.transpose_axes(&[1, 0, 2])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, max};

    fn maxabs(a: &Array) -> f32 {
        max(abs(a).unwrap(), None).unwrap().item::<f32>()
    }

    /// RoPE is a per-pair rotation, so it preserves the L2 norm of every `[b,h,n,head_dim]` token.
    #[test]
    fn apply_rope_preserves_token_norm() {
        let (b, h, n, d) = (1, 1, 4, 8); // n=4 → side 2, square grid
        let x = Array::from_slice(
            &(0..b * h * n * d)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            &[b, h, n, d],
        );
        let (cos, sin) = axial_rope_tables(d, 2, 2);
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let norm_sq = |t: &Array| ops::sum_axes(ops::square(t).unwrap(), &[3], false).unwrap();
        let diff = maxabs(&subtract(norm_sq(&x), norm_sq(&y)).unwrap());
        assert!(diff < 1e-4, "rope changed token norm by {diff:e}");
    }

    /// `axial_rope_tables` yields `[N, head_dim/2]` tables whose cos/sin agree pairwise (cos²+sin²=1).
    #[test]
    fn axial_rope_tables_are_unit_circle() {
        let (cos, sin) = axial_rope_tables(16, 3, 3); // head_dim 16 → 8 cols, N=9
        assert_eq!(cos.shape(), &[9, 8]);
        let sumsq = add(ops::square(&cos).unwrap(), ops::square(&sin).unwrap()).unwrap();
        let one = Array::from_slice(&vec![1f32; 72], &[9, 8]);
        assert!(maxabs(&subtract(&sumsq, &one).unwrap()) < 1e-5);
    }

    /// The memoized RoPE-table accessors (F-167) must return tables bit-identical to the direct
    /// `axial_rope_tables` / `repeat_rows` computations they cache — both on a cold miss and a warm hit.
    #[test]
    fn cached_rope_tables_match_direct_compute() {
        let (cos_d, sin_d) = axial_rope_tables(16, 4, 4);
        for _ in 0..2 {
            // First iteration is a cache miss, second a hit — both must equal the direct compute.
            let (cos_c, sin_c) = axial_rope_tables_cached(16, 4, 4);
            assert!(maxabs(&subtract(&cos_c, &cos_d).unwrap()) == 0.0);
            assert!(maxabs(&subtract(&sin_c, &sin_d).unwrap()) == 0.0);
        }
        // The repeated variant matches `repeat_rows` of the base, and `repeat == 1` is the base.
        let (cos_r, sin_r) = axial_rope_tables_repeated(16, 4, 4, 3).unwrap();
        assert!(maxabs(&subtract(&cos_r, repeat_rows(&cos_d, 3).unwrap()).unwrap()) == 0.0);
        assert!(maxabs(&subtract(&sin_r, repeat_rows(&sin_d, 3).unwrap()).unwrap()) == 0.0);
        let (cos_1, _) = axial_rope_tables_repeated(16, 4, 4, 1).unwrap();
        assert!(maxabs(&subtract(&cos_1, &cos_d).unwrap()) == 0.0);
    }

    /// `repeat_rows` tiles each row consecutively: row `r` lands at output rows `r*repeat .. (r+1)*repeat`.
    #[test]
    fn repeat_rows_tiles_consecutively() {
        let t = Array::from_slice(&[0f32, 1.0, 2.0, 3.0], &[2, 2]); // rows [0,1] and [2,3]
        let r = repeat_rows(&t, 3).unwrap();
        assert_eq!(r.shape(), &[6, 2]);
        let v = r.as_slice::<f32>();
        // First 3 rows == original row 0; next 3 == row 1.
        assert_eq!(&v[0..6], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&v[6..12], &[2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
    }

    /// A depthwise conv (`groups == channels`) must not mix channels: zeroing one channel's kernel
    /// zeroes only that channel's output.
    #[test]
    fn depthwise_conv_does_not_mix_channels() {
        let x = Array::from_slice(&[1f32; 2 * 3 * 3], &[1, 2, 3, 3]); // 2 channels of ones
                                                                      // weight [O=2, kH=3, kW=3, C_in/groups=1]; channel 0 all-ones kernel, channel 1 all-zero.
        let mut wv = vec![1f32; 9];
        wv.extend(vec![0f32; 9]);
        let w = Array::from_slice(&wv, &[2, 3, 3, 1]);
        let b = Array::from_slice(&[0f32, 0.0], &[2]);
        let y = conv2d_nchw(&x, &w, &b, 1, 1, 2).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3, 3]);
        let ch0 = y.take_axis(Array::from_int(0), 1).unwrap();
        let ch1 = y.take_axis(Array::from_int(1), 1).unwrap();
        assert!(maxabs(&ch0) > 0.0, "channel 0 should be nonzero");
        assert!(
            maxabs(&ch1) < 1e-6,
            "channel 1 (zero kernel) leaked: {}",
            maxabs(&ch1)
        );
    }

    fn zeros(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        Array::from_slice(&vec![0f32; n as usize], shape)
    }

    /// Build a zero-filled checkpoint with every memory-encoder + memory-attention key, then run
    /// both forward passes and assert the output shapes — exercises the whole graph, no download.
    #[test]
    fn forward_passes_emit_expected_shapes() {
        let mut t: Vec<(String, Array)> = Vec::new();
        let me = "memory_encoder";
        // mask_downsampler: conv0..3 (k3) + conv4 (k1), norm0..3.
        let ds_ch = [(1, 4), (4, 16), (16, 64), (64, 256)];
        for (i, (ci, co)) in ds_ch.iter().enumerate() {
            t.push((
                format!("{me}.mask_downsampler.conv{i}.weight"),
                zeros(&[*co, 3, 3, *ci]),
            ));
            t.push((format!("{me}.mask_downsampler.conv{i}.bias"), zeros(&[*co])));
            t.push((
                format!("{me}.mask_downsampler.norm{i}.weight"),
                zeros(&[*co]),
            ));
            t.push((format!("{me}.mask_downsampler.norm{i}.bias"), zeros(&[*co])));
        }
        t.push((
            format!("{me}.mask_downsampler.conv4.weight"),
            zeros(&[256, 1, 1, 256]),
        ));
        t.push((format!("{me}.mask_downsampler.conv4.bias"), zeros(&[256])));
        t.push((
            format!("{me}.pix_feat_proj.weight"),
            zeros(&[256, 1, 1, 256]),
        ));
        t.push((format!("{me}.pix_feat_proj.bias"), zeros(&[256])));
        t.push((format!("{me}.out_proj.weight"), zeros(&[64, 1, 1, 256])));
        t.push((format!("{me}.out_proj.bias"), zeros(&[64])));
        for f in 0..2 {
            let p = format!("{me}.fuser.{f}");
            t.push((format!("{p}.gamma"), zeros(&[256])));
            t.push((format!("{p}.dwconv.weight"), zeros(&[256, 7, 7, 1])));
            t.push((format!("{p}.dwconv.bias"), zeros(&[256])));
            t.push((format!("{p}.norm.weight"), zeros(&[256])));
            t.push((format!("{p}.norm.bias"), zeros(&[256])));
            t.push((format!("{p}.pwconv1.weight"), zeros(&[1024, 256])));
            t.push((format!("{p}.pwconv1.bias"), zeros(&[1024])));
            t.push((format!("{p}.pwconv2.weight"), zeros(&[256, 1024])));
            t.push((format!("{p}.pwconv2.bias"), zeros(&[256])));
        }
        let ma = "memory_attention";
        for l in 0..4 {
            let p = format!("{ma}.layers.{l}");
            for (attn, kv_in) in [("self_attn", 256), ("cross_attn_image", 64)] {
                t.push((format!("{p}.{attn}.q_proj.weight"), zeros(&[256, 256])));
                t.push((format!("{p}.{attn}.q_proj.bias"), zeros(&[256])));
                t.push((format!("{p}.{attn}.k_proj.weight"), zeros(&[256, kv_in])));
                t.push((format!("{p}.{attn}.k_proj.bias"), zeros(&[256])));
                t.push((format!("{p}.{attn}.v_proj.weight"), zeros(&[256, kv_in])));
                t.push((format!("{p}.{attn}.v_proj.bias"), zeros(&[256])));
                t.push((format!("{p}.{attn}.out_proj.weight"), zeros(&[256, 256])));
                t.push((format!("{p}.{attn}.out_proj.bias"), zeros(&[256])));
            }
            t.push((format!("{p}.linear1.weight"), zeros(&[2048, 256])));
            t.push((format!("{p}.linear1.bias"), zeros(&[2048])));
            t.push((format!("{p}.linear2.weight"), zeros(&[256, 2048])));
            t.push((format!("{p}.linear2.bias"), zeros(&[256])));
            for n in ["norm1", "norm2", "norm3"] {
                t.push((format!("{p}.{n}.weight"), zeros(&[256])));
                t.push((format!("{p}.{n}.bias"), zeros(&[256])));
            }
        }
        t.push((format!("{ma}.norm.weight"), zeros(&[256])));
        t.push((format!("{ma}.norm.bias"), zeros(&[256])));

        let path = std::env::temp_dir().join("mlx_gen_sam2_synth_memory.safetensors");
        let refs: Vec<(&str, &Array)> = t.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        let w = Weights::from_file(&path).unwrap();

        // Memory encoder: pix_feat [1,256,4,4] + mask [1,1,64,64] → features/pos [1,64,4,4].
        let enc = MemoryEncoder::from_weights(&w, "memory_encoder").unwrap();
        let out = enc
            .forward(&zeros(&[1, 256, 4, 4]), &zeros(&[1, 1, 64, 64]), true)
            .unwrap();
        assert_eq!(out.vision_features.shape(), &[1, 64, 4, 4]);
        assert_eq!(out.vision_pos_enc.shape(), &[1, 64, 4, 4]);

        // Memory attention: 16 current tokens, a 2-frame (+1 obj-ptr) memory bank of 36 tokens.
        let attn = MemoryAttention::from_weights(&w, "memory_attention").unwrap();
        let curr = zeros(&[16, 1, 256]);
        let mem = zeros(&[36, 1, 64]); // 2*16 spatial + 4 obj-ptr
        let conditioned = attn.forward(&curr, &curr, &mem, &mem, 4).unwrap();
        assert_eq!(conditioned.shape(), &[16, 1, 256]);

        let _ = std::fs::remove_file(&path);
    }
}
