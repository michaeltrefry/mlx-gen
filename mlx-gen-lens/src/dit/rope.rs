//! Lens complex axial RoPE (`LensEmbedRope`, `theta=10000`, `axes_dim=(8,28,28)`, `scale_rope=True`).
//! Identical pos/neg-frequency construction to `mlx-gen-qwen-image`'s `QwenRope3d` (only the axes
//! differ: 8+28+28 = 64 = head_dim → 4+14+14 = 32 complex pairs). Produces **interleaved** cos/sin
//! tables for the image and text streams; `freqs_cis = polar(1, angle) = cos + i·sin`, so the
//! reference's complex `view_as_complex` apply is reproduced by the interleaved real rotation in
//! [`super::attention`].
//!
//! Image stream: frame axis at positions `0..frame`, height/width at **centered** positions
//! `[-(N - N/2), …, -1, 0, …, N/2 - 1]` (`scale_rope`). Text stream: a single scalar position
//! `max(H/2, W/2) + t` across all 32 pair-frequencies.

use mlx_rs::Array;

use mlx_gen::Result;

/// `(img_cos, img_sin, txt_cos, txt_sin)`: image tables `[frame·h·w, head_dim/2]`, text tables
/// `[txt_seq, head_dim/2]`.
pub type RopeTables = (Array, Array, Array, Array);

/// Lens 3-axis (frame/height/width) RoPE table builder.
pub struct LensRope3d {
    theta: f32,
    axes_dim: [i32; 3],
}

impl LensRope3d {
    pub fn new(theta: f32, axes_dim: [i32; 3]) -> Self {
        Self { theta, axes_dim }
    }

    /// The Lens default: θ=10000, axes `(8, 28, 28)` (Σ = 64 = head_dim, Σ/2 = 32 pairs).
    pub fn lens() -> Self {
        Self::new(10000.0, [8, 28, 28])
    }

    fn omega(&self, dim: i32) -> Vec<f32> {
        (0..dim / 2)
            .map(|k| 1.0 / self.theta.powf((2 * k) as f32 / dim as f32))
            .collect()
    }

    /// Build the `(img_cos, img_sin, txt_cos, txt_sin)` tables for a single `(frame, h, w)` image grid
    /// and a text sequence of `txt_seq` tokens.
    pub fn forward(&self, frame: usize, h: usize, w: usize, txt_seq: usize) -> Result<RopeTables> {
        let (o0, o1, o2) = (
            self.omega(self.axes_dim[0]),
            self.omega(self.axes_dim[1]),
            self.omega(self.axes_dim[2]),
        );
        let half = o0.len() + o1.len() + o2.len(); // 4 + 14 + 14 = 32

        let total_seq = frame * h * w;
        let mut img_cos = vec![0f32; total_seq * half];
        let mut img_sin = vec![0f32; total_seq * half];
        // height/width centered positions (scale_rope): hp ∈ [-(h - h/2), …, h/2 - 1].
        let h_off = (h - h / 2) as i32;
        let w_off = (w - w / 2) as i32;
        for f in 0..frame {
            for hi in 0..h {
                let hp = hi as i32 - h_off;
                for wi in 0..w {
                    let wp = wi as i32 - w_off;
                    let row = (f * h * w + hi * w + wi) * half;
                    let mut j = 0;
                    for &fr in &o0 {
                        let a = f as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &fr in &o1 {
                        let a = hp as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &fr in &o2 {
                        let a = wp as f32 * fr;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                }
            }
        }

        // text stream: scalar position max(H/2, W/2) + t across all 32 pair-frequencies.
        let txt_base = (h / 2).max(w / 2) as i32;
        let all_omega: Vec<f32> = o0.iter().chain(&o1).chain(&o2).copied().collect();
        let mut txt_cos = vec![0f32; txt_seq * half];
        let mut txt_sin = vec![0f32; txt_seq * half];
        for t in 0..txt_seq {
            let p = (txt_base + t as i32) as f32;
            let row = t * half;
            for (j, &fr) in all_omega.iter().enumerate() {
                let a = p * fr;
                txt_cos[row + j] = a.cos();
                txt_sin[row + j] = a.sin();
            }
        }

        let hh = half as i32;
        Ok((
            Array::from_slice(&img_cos, &[total_seq as i32, hh]),
            Array::from_slice(&img_sin, &[total_seq as i32, hh]),
            Array::from_slice(&txt_cos, &[txt_seq as i32, hh]),
            Array::from_slice(&txt_sin, &[txt_seq as i32, hh]),
        ))
    }
}
