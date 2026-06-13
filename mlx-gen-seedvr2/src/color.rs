//! SeedVR2 post-decode color correction (sc-4813).
//!
//! Faithful port of `SeedVR2Util.apply_color_correction` (`variants/upscale/seedvr2_util.py`):
//! a wavelet reconstruction (content high-freq + style low-freq) followed by a LAB color transfer
//! (histogram-match the a/b chroma channels and partially the L channel to the LR "style"). The
//! reference runs this in host numpy, so it is ported as plain-Rust f32 arithmetic (B=1) rather than
//! MLX ops. Operates on `(1,3,H,W)` in [-1,1] and returns `(1,3,H,W)` in [-1,1].

use mlx_gen::Result;
use mlx_rs::{Array, Dtype};

const KERNEL: [[f32; 3]; 3] = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625],
];

#[inline]
fn clampi(v: i64, lo: i64, hi: i64) -> usize {
    v.clamp(lo, hi) as usize
}

/// Dilated 3×3 wavelet blur of one `H×W` channel (edge/clamp padding), dilation = `radius`.
fn wavelet_blur(img: &[f32], h: i32, w: i32, radius: i32) -> Vec<f32> {
    let mut r = radius.max(1);
    let max_safe = (h.min(w) / 8).max(1);
    if r > max_safe {
        r = max_safe;
    }
    let (hh, ww) = (h as i64, w as i64);
    let mut out = vec![0f32; (h * w) as usize];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut acc = 0f32;
            for (ky, dy) in [-1i64, 0, 1].iter().enumerate() {
                let yy = clampi(y + dy * r as i64, 0, hh - 1);
                for (kx, dx) in [-1i64, 0, 1].iter().enumerate() {
                    let xx = clampi(x + dx * r as i64, 0, ww - 1);
                    acc += KERNEL[ky][kx] * img[yy * w as usize + xx];
                }
            }
            out[(y * ww + x) as usize] = acc;
        }
    }
    out
}

/// 5-level wavelet decomposition of one channel → `(high_freq, low_freq)`.
fn wavelet_decomp(img: &[f32], h: i32, w: i32) -> (Vec<f32>, Vec<f32>) {
    let n = (h * w) as usize;
    let mut high = vec![0f32; n];
    let mut cur = img.to_vec();
    for i in 0..5 {
        let radius = 1 << i; // 1,2,4,8,16
        let low = wavelet_blur(&cur, h, w, radius);
        for k in 0..n {
            high[k] += cur[k] - low[k];
        }
        cur = low;
    }
    (high, cur)
}

fn srgb_to_linear(x: f32) -> f32 {
    if x > 0.04045 {
        ((x + 0.055) / 1.055).powf(2.4)
    } else {
        x / 12.92
    }
}
fn linear_to_srgb(x: f32) -> f32 {
    if x > 0.0031308 {
        1.055 * x.max(0.0).powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * x
    }
}

const EPS: f32 = 6.0 / 29.0;
fn lab_f(t: f32) -> f32 {
    let eps3 = EPS * EPS * EPS;
    if t > eps3 {
        t.cbrt()
    } else {
        let kappa = (29.0f32 / 3.0).powi(3);
        (kappa * t + 16.0) / 116.0
    }
}
fn lab_finv(f: f32) -> f32 {
    if f > EPS {
        f * f * f
    } else {
        let kappa = (29.0f32 / 3.0).powi(3);
        (116.0 * f - 16.0) / kappa
    }
}

/// per-pixel sRGB[0,1] → CIE-LAB.
fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (rl, gl, bl) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let mut x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let mut z = 0.0193339 * rl + 0.119_192 * gl + 0.9503041 * bl;
    x /= 0.95047;
    z /= 1.08883;
    let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// per-pixel CIE-LAB → sRGB[0,1].
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = lab_finv(fx) * 0.95047;
    let y = lab_finv(fy);
    let z = lab_finv(fz) * 1.08883;
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.969_266 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl))
}

/// Histogram match `source` onto `reference` (equal length) by rank — numpy stable-argsort port.
fn hist_match(source: &[f32], reference: &[f32]) -> Vec<f32> {
    let n = source.len();
    let mut src_idx: Vec<usize> = (0..n).collect();
    src_idx.sort_by(|&i, &j| source[i].partial_cmp(&source[j]).unwrap()); // stable
    let mut ref_sorted = reference.to_vec();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // inv[p] = rank of pixel p in sorted source order
    let mut inv = vec![0usize; n];
    for (rank, &p) in src_idx.iter().enumerate() {
        inv[p] = rank;
    }
    (0..n).map(|p| ref_sorted[inv[p]]).collect()
}

/// `content`/`style`: `(1,3,H,W)` in [-1,1]. Returns the corrected `(1,3,H,W)` in [-1,1].
pub fn apply_color_correction(
    content: &Array,
    style: &Array,
    luminance_weight: f32,
) -> Result<Array> {
    let sh = content.shape();
    let (h, w) = (sh[2], sh[3]);
    let n = (h * w) as usize;
    let c = content.as_dtype(Dtype::Float32)?.reshape(&[3 * h * w])?;
    let s = style.as_dtype(Dtype::Float32)?.reshape(&[3 * h * w])?;
    let c = c.as_slice::<f32>();
    let s = s.as_slice::<f32>();

    // 1. wavelet reconstruction: content high-freq + style low-freq, per channel
    let mut recon = vec![0f32; 3 * n];
    for ch in 0..3 {
        let (chigh, _) = wavelet_decomp(&c[ch * n..(ch + 1) * n], h, w);
        let (_, slow) = wavelet_decomp(&s[ch * n..(ch + 1) * n], h, w);
        for k in 0..n {
            recon[ch * n + k] = (chigh[k] + slow[k]).clamp(-1.0, 1.0);
        }
    }

    // 2. to LAB (content from `recon`, style from the original)
    let to01 = |v: f32| ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    let (mut c_l, mut c_a, mut c_b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let (mut s_l, mut s_a, mut s_b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for p in 0..n {
        let (l, a, b) = rgb_to_lab(to01(recon[p]), to01(recon[n + p]), to01(recon[2 * n + p]));
        c_l[p] = l;
        c_a[p] = a;
        c_b[p] = b;
        let (l, a, b) = rgb_to_lab(to01(s[p]), to01(s[n + p]), to01(s[2 * n + p]));
        s_l[p] = l;
        s_a[p] = a;
        s_b[p] = b;
    }

    // 3. histogram-match chroma; partial L blend
    let matched_a = hist_match(&c_a, &s_a);
    let matched_b = hist_match(&c_b, &s_b);
    let out_l: Vec<f32> = if luminance_weight < 1.0 {
        let matched_l = hist_match(&c_l, &s_l);
        (0..n)
            .map(|p| luminance_weight * c_l[p] + (1.0 - luminance_weight) * matched_l[p])
            .collect()
    } else {
        c_l
    };

    // 4. back to RGB → [-1,1], (3,H,W)
    let mut out = vec![0f32; 3 * n];
    for p in 0..n {
        let (r, g, b) = lab_to_rgb(out_l[p], matched_a[p], matched_b[p]);
        out[p] = r.clamp(0.0, 1.0) * 2.0 - 1.0;
        out[n + p] = g.clamp(0.0, 1.0) * 2.0 - 1.0;
        out[2 * n + p] = b.clamp(0.0, 1.0) * 2.0 - 1.0;
    }
    Ok(Array::from_slice(&out, &[1, 3, h, w]).as_dtype(content.dtype())?)
}
