//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers two on-disk conventions, both family-agnostic:
//! - **PEFT/diffusers** (`‹prefix›‹path›.lora_A/B.weight` + optional `‹path›.alpha`): dotted module
//!   paths resolve directly via [`AdaptableHost::adaptable_mut`] ([`apply_lora_peft`]).
//! - **kohya / sd-scripts** (`lora_unet_‹path, .→_›.lora_down/up.weight` + optional `.alpha`,
//!   sc-2618): the flattened module path can't be re-split blindly, so it resolves through a
//!   `flattened → dotted` table built from [`AdaptableHost::adaptable_paths`]
//!   ([`apply_lora_kohya`]). kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`, so both feed the
//!   shared [`install_lora_groups`] and a kohya file yields the identical adapter to its PEFT twin.
//!
//! - **BFL / ComfyUI** (`lora_unet_double_blocks_*` / `diffusion_model.…` / `base_model.model.…`,
//!   sc-2743): a *fused* source linear (`…img_attn.qkv`, `…linear1`) is row-sliced into the model's
//!   *split* targets (`attn.to_q/to_k/to_v`, …) via per-target [`LoraRowSlice`] transforms, with BFL
//!   module renames (`img_in`→`x_embedder`). This is fused→split weight surgery, orthogonal to the
//!   kohya underscore form; the host supplies its table via [`AdaptableHost::bfl_targets`]
//!   ([`apply_lora_bfl`]). Only FLUX.2/FLUX.1 expose one; for other hosts a BFL file's keys surface as
//!   unmatched (loud), never silently dropped.

use std::collections::{BTreeMap, BTreeSet};

use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use super::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use crate::runtime::{AdapterKind, AdapterSpec};
use crate::weights::Weights;
use crate::Result;

/// PEFT LoKr per-module factor suffixes; each factor is full (`lokr_w1`/`lokr_w2`) or
/// low-rank (`_a` @ `_b`). Exact-suffix matched, so order is for readability only.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    w.metadata("networkType")
        .map(|s| s.trim().eq_ignore_ascii_case("lokr"))
        .unwrap_or(false)
}

/// A parsed LoKr file: the global `(alpha, rank)` from metadata plus every module's Kronecker
/// factors grouped by path. The factor map is keyed by the bare factor name (`lokr_w1`,
/// `lokr_w1_a`, `lokr_w1_b`, `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`); a module is full or low-rank.
///
/// This is the format-parsing half of a LoKr install, factored out of [`apply_lokr`] so the video
/// providers (LTX/Wan) — which install onto their crate-local `Linear`s as a forward-time residual
/// or an in-place weight merge, rather than the core [`AdaptableHost`] — reuse the exact same factor
/// grouping + metadata read and differ only in the install step. Each provider then maps the bare
/// module `path` through its own key→module table and calls [`reconstruct_lokr_delta`].
#[derive(Debug)]
pub struct LokrFile {
    pub alpha: f32,
    pub rank: f32,
    /// `module path → { factor name → tensor }`.
    pub groups: BTreeMap<String, BTreeMap<String, Array>>,
}

impl LokrFile {
    /// `alpha/rank` — the scale the fork bakes into the reconstructed delta (PEFT default `alpha=rank`
    /// ⇒ 1.0). The per-adapter user `strength` multiplies this separately at the residual/merge site.
    pub fn delta_scale(&self) -> f32 {
        self.alpha / self.rank
    }

    /// Reconstruct one module's `[out,in]` delta at `out_dtype` from its grouped factors, baking in
    /// `alpha/rank` (the user `strength` is applied separately). `base_shape` is the target linear's
    /// logical weight shape. Returns the [`reconstruct_lokr_delta`] result.
    pub fn delta(
        &self,
        factors: &BTreeMap<String, Array>,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Array> {
        reconstruct_lokr_delta(
            self.alpha,
            self.rank,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }
}

/// Parse a LoKr `.safetensors` into [`LokrFile`]: read `rank`/`alpha` from metadata (alpha defaults
/// to rank, i.e. scale 1.0, matching PEFT) and group every `‹path›.lokr_*` tensor by module path.
/// Shared by [`apply_lokr`] (core `AdaptableHost` install) and the video providers' crate-local
/// residual/merge installers.
pub fn parse_lokr(w: &Weights) -> Result<LokrFile> {
    let rank = w
        .metadata("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    // alpha defaults to rank (scale 1.0) when absent, matching PEFT.
    let alpha = w
        .metadata("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut groups: BTreeMap<String, BTreeMap<String, Array>> = BTreeMap::new();
    for key in &keys {
        for suffix in LOKR_SUFFIXES {
            if let Some(path) = key.strip_suffix(suffix) {
                let factor = suffix[1..].to_string(); // drop the leading '.'
                groups
                    .entry(path.to_string())
                    .or_default()
                    .insert(factor, w.require(key)?.clone());
                break;
            }
        }
    }
    Ok(LokrFile {
        alpha,
        rank,
        groups,
    })
}

/// Read a scalar adapter value (an `alpha`) as `f32`, regardless of its on-disk dtype. Trained
/// adapters store `alpha` in their compute dtype: real kohya/BFL FLUX LoRAs ship it **bf16** (sc-2657),
/// and `Array::as_slice::<f32>()` `unwrap`s a hard dtype-mismatch (it never casts), so reading a bf16
/// scalar that way panics. Cast to f32 first (exact for the small integer alphas these files carry, and
/// a no-op when already f32); a `[]`- or `[1]`-shaped scalar both read as a one-element slice.
fn scalar_alpha(a: &Array) -> Result<Option<f32>> {
    Ok(a.as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .first()
        .copied())
}

/// Outcome of installing an adapter file: how many target modules were adapted, and any
/// adapter keys that matched no module in the host (surfaced, never silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub applied: usize,
    pub unmatched_paths: Vec<String>,
}

/// Install a LoKr adapter file onto `host`. `scale` is the user-facing strength (the
/// `alpha/rank` factor is baked into the reconstructed delta, mirroring the fork).
pub fn apply_lokr(host: &mut impl AdaptableHost, w: &Weights, scale: f32) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let mut report = ApplyReport::default();
    for (path, factors) in &file.groups {
        let parts: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                // Fork-parity residual path keeps the delta at bf16 (PARITY-BF16, sc-2609).
                let delta = file.delta(factors, &base_shape, Dtype::Bfloat16)?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path.clone()),
        }
    }
    Ok(report)
}

/// Install a PEFT/diffusers-format LoRA file onto `host`. The down/up factors carry the file's
/// namespace prefix on a **dotted** module path, in either of two interchangeable spellings:
/// - PEFT: `‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`;
/// - diffusers/ComfyUI/ai-toolkit (e.g. the lightx2v Qwen-Image-Lightning LoRAs, sc-2909):
///   `‹prefix›‹path›.lora_down.weight` / `.lora_up.weight` — `lora_down`==`lora_A`, `lora_up`==`lora_B`
///   (identical shapes), differing from the kohya format only in that the path stays dotted (no
///   `lora_unet_` flattening), so it routes here rather than to [`apply_lora_kohya`].
///
/// Both store the down factor as `[r, in]` and the up factor as `[out, r]`; we transpose to the
/// residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`) and fold `alpha/rank` into `B`, matching the
/// fork. `‹prefix›‹path›.alpha` is optional (and may be bare — see below). `strip_prefix` removes a
/// leading namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        // The down/up factors always carry the file's namespace prefix. `lora_A`/`lora_B` (PEFT) and
        // `lora_down`/`lora_up` (diffusers/ComfyUI) are interchangeable spellings of the same role.
        if let Some(rest) = key.strip_prefix(prefix) {
            if let Some(path) = rest
                .strip_suffix(".lora_A.weight")
                .or_else(|| rest.strip_suffix(".lora_down.weight"))
            {
                groups.entry(path.to_string()).or_default().a = Some(w.require(&key)?.clone());
                continue;
            }
            if let Some(path) = rest
                .strip_suffix(".lora_B.weight")
                .or_else(|| rest.strip_suffix(".lora_up.weight"))
            {
                groups.entry(path.to_string()).or_default().b = Some(w.require(&key)?.clone());
                continue;
            }
        }
        // `alpha` may be prefixed (`<prefix><path>.alpha`) OR bare (`<path>.alpha`): some trainers
        // pair prefixed `lora_A/B` with a bare `alpha` — notably the fork's `QwenLoRAMapping`, whose
        // alpha patterns are bare-only. Resolve to the same `<path>` either way (rather than
        // stripping the A/B prefix off the alpha key and dropping a bare one) so the `alpha/rank`
        // fold is kept; a prefixed and a bare alpha that *disagree* for one path is a hard error (no
        // silent pick). Without this, a prefixed-A/B + bare-alpha file applied at the wrong
        // (unscaled) strength while reporting success (sc-2528 adversarial review).
        if let Some(path) = key
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(".alpha"))
            .or_else(|| key.strip_suffix(".alpha"))
        {
            if let Some(new) = scalar_alpha(w.require(&key)?)? {
                let slot = &mut groups.entry(path.to_string()).or_default().alpha;
                match *slot {
                    Some(existing) if existing != new => {
                        return Err(format!(
                            "LoRA alpha conflict for `{path}`: {existing} vs {new} \
                             (prefixed and bare alpha keys disagree)"
                        )
                        .into());
                    }
                    _ => *slot = Some(new),
                }
            }
        }
    }

    install_lora_groups(host, groups, scale)
}

/// Install grouped `(down=A, up=B, alpha)` LoRA factors onto `host`, one residual per resolved module
/// path. Shared by the PEFT/diffusers loader ([`apply_lora_peft`]) and the kohya loader
/// ([`apply_lora_kohya`]): both conventions agree on the math (`A: [r,in]`, `B: [out,r]`, transpose to
/// the residual form `x·A·B`, fold `alpha/rank` into `B`) and differ only in how keys map to `path`.
/// A path with a missing `down` or `up` half is skipped (its partner targeted a non-LoRA key);
/// a path that resolves to no module is surfaced in `unmatched_paths`, never silently dropped.
fn install_lora_groups(
    host: &mut impl AdaptableHost,
    groups: BTreeMap<String, LoraParts>,
    scale: f32,
) -> Result<ApplyReport> {
    let mut report = ApplyReport::default();
    for (path, parts) in groups {
        let (Some(a_raw), Some(b_raw)) = (parts.a, parts.b) else {
            continue;
        };
        let parents: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parents) {
            Some(lin) => {
                let a = a_raw.t(); // [r, in] -> [in, r]
                let mut b = b_raw.t(); // [out, r] -> [r, out]
                if let Some(alpha) = parts.alpha {
                    let rank = a.shape()[1] as f32; // r
                    b = b.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
                }
                lin.push(Adapter::Lora { a, b, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

#[derive(Default)]
struct LoraParts {
    a: Option<Array>,
    b: Option<Array>,
    alpha: Option<f32>,
}

/// kohya / sd-scripts key prefix: the trained diffusers module path with dots flattened to
/// underscores, denominated under the denoiser as `lora_unet_…`.
pub const KOHYA_PREFIX: &str = "lora_unet_";

/// kohya factor suffixes mapped to a [`LoraParts`] role. `lora_down`==PEFT `lora_A`,
/// `lora_up`==PEFT `lora_B`; the optional `.default` infix is the peft-export form some kohya
/// converters emit. Order is irrelevant (exact-suffix match).
const KOHYA_SUFFIXES: [(&str, KohyaRole); 5] = [
    (".lora_down.weight", KohyaRole::Down),
    (".lora_up.weight", KohyaRole::Up),
    (".lora_down.default.weight", KohyaRole::Down),
    (".lora_up.default.weight", KohyaRole::Up),
    (".alpha", KohyaRole::Alpha),
];

#[derive(Clone, Copy)]
enum KohyaRole {
    Down,
    Up,
    Alpha,
}

/// `true` if `w` is a kohya-format LoRA — any key carries the `lora_unet_` prefix. (kohya files are
/// the only convention that flattens the module path; PEFT/diffusers keep dots, LoKr is bare.)
pub fn is_kohya(w: &Weights) -> bool {
    w.keys().any(|k| k.starts_with(KOHYA_PREFIX))
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths
/// (`AdaptableHost::adaptable_paths`). The stem is the dotted path with `.`→`_` (the kohya
/// flattening), WITHOUT the `lora_unet_` prefix. Mirrors the SDXL matcher (sc-2639) and the fork's
/// explicit `lora_unet_…` patterns, generalized over any [`AdaptableHost`].
fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    paths
        .iter()
        .map(|p| (p.replace('.', "_"), p.clone()))
        .collect()
}

/// Install a kohya-format LoRA (`lora_unet_<flattened path>.lora_down/up.weight` + optional `.alpha`)
/// onto `host`. The flattened stem is resolved against `table` (built from
/// [`AdaptableHost::adaptable_paths`]) — blind `_`→`.` splitting is impossible because module names
/// contain underscores (`to_out.0`, `feed_forward.w1`, `img_mlp.net.0.proj`). Resolved factors are
/// installed through the same [`install_lora_groups`] path as PEFT, so a kohya file produces the
/// identical adapter to the equivalent PEFT file.
///
/// `lora_unet_` keys whose stem is NOT in the table (off-surface) are surfaced in `unmatched_paths`
/// so the strict policy fails loudly rather than silently dropping them. The BFL fused→split kohya
/// form (`lora_unet_double_blocks_*`, sc-2743) is routed to [`apply_lora_bfl`] *before* this loader
/// for a host that exposes [`AdaptableHost::bfl_targets`]; reaching here it has no table entry and is
/// likewise surfaced. Keys without the `lora_unet_` prefix (e.g. a bundled text-encoder `lora_te_…`)
/// are not denoiser targets and are ignored, matching the PEFT loader's treatment of out-of-namespace
/// keys.
pub fn apply_lora_kohya(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
) -> Result<ApplyReport> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(rem) = key.strip_prefix(KOHYA_PREFIX) else {
            continue; // not a denoiser kohya key (e.g. text-encoder `lora_te_…`) — ignore.
        };
        let Some((stem, role)) = KOHYA_SUFFIXES
            .iter()
            .find_map(|(suf, role)| rem.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // a `lora_unet_` key with an unrecognized suffix — ignore.
        };
        let Some(path) = table.get(stem) else {
            unresolved.insert(stem.to_string());
            continue;
        };
        let parts = groups.entry(path.clone()).or_default();
        match role {
            KohyaRole::Down => parts.a = Some(w.require(&key)?.clone()),
            KohyaRole::Up => parts.b = Some(w.require(&key)?.clone()),
            KohyaRole::Alpha => parts.alpha = scalar_alpha(w.require(&key)?)?,
        }
    }

    let mut report = install_lora_groups(host, groups, scale)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----------------------------------------------------

/// A row-slice over a raw LoRA factor (applied BEFORE the `[r,in]`/`[out,r]`→residual transpose),
/// porting the fork's `LoraTransforms` byte-for-byte (sc-2743). The BFL/ComfyUI checkpoints store a
/// block's q/k/v (and, for FLUX.1, the qkv+mlp) concatenated along dim-0 of a single *fused* linear;
/// each diffusers split target slices out its own rows. Indices/divisibility match `LoraTransforms`
/// exactly (verified against the fork venv).
#[derive(Clone, Debug)]
pub enum LoraRowSlice {
    /// Chunk `index` of `n` equal dim-0 chunks (`chunk = shape[0] / n`), ALWAYS sliced — the fork's
    /// `_split_qkv_up` (the up factor `[n·out, r]` → `[out, r]`). `n=3` for qkv.
    Chunk { n: i32, index: i32 },
    /// Chunk `index` of `n` equal dim-0 chunks IFF `shape[0] % n == 0`, else the whole tensor — the
    /// fork's `_split_qkv_down`/`_split_qkv_mlp_down` (the down factor is *shared* across q/k/v when
    /// the rank isn't divisible by `n`, which is the usual fused-qkv LoRA, and sliced when it is).
    ChunkIfDivisible { n: i32, index: i32 },
    /// The dim-0 slice `[Σdims[..index] .. Σdims[..=index]]` — the fork's `_split_qkv_mlp_up` with
    /// config-derived `dims` (FLUX.1 `linear1` = `[q,k,v,mlp]`, e.g. `[3072,3072,3072,12288]`). FLUX.2
    /// keeps qkv+mlp fused (`to_qkv_mlp_proj`) so it never uses this; FLUX.1 (sc-2657) will.
    Dims { dims: Vec<i32>, index: i32 },
}

impl LoraRowSlice {
    fn apply(&self, t: &Array) -> Result<Array> {
        let rows = t.shape()[0];
        let (start, end) = match self {
            LoraRowSlice::Chunk { n, index } => {
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::ChunkIfDivisible { n, index } => {
                if rows % n != 0 {
                    return Ok(t.clone());
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::Dims { dims, index } => {
                let i = *index as usize;
                let start: i32 = dims[..i].iter().sum();
                (start, start + dims[i])
            }
        };
        // `t[start:end, :]` — byte-identical to the fork's slicing.
        Ok(t.try_index((start..end, ..))?)
    }
}

/// One BFL/ComfyUI adapter target: a set of source key spellings (across the `lora_unet_` /
/// `diffusion_model.` / `base_model.model.` prefix conventions) mapping to a diffusers module
/// `target_path`, with an optional [`LoraRowSlice`] on the up/down factor. A *fused* source (BFL
/// `…img_attn.qkv`) is named by SEVERAL `BflTarget`s — one per split destination (`to_q`/`to_k`/`to_v`)
/// — that share its key spellings but slice different rows; the loader fans the one source tensor into
/// all of them. A plain rename (BFL `img_in` → `x_embedder`) is a `BflTarget` with no slice. Mirrors a
/// fork `LoRATarget` restricted to its BFL patterns + up/down transforms.
#[derive(Clone, Debug)]
pub struct BflTarget {
    /// Diffusers module path that [`AdaptableHost::adaptable_mut`] resolves (concrete, no `{block}`).
    pub target_path: String,
    /// Source up-factor (`lora_up`/`lora_B`) key spellings.
    pub up_keys: Vec<String>,
    /// Source down-factor (`lora_down`/`lora_A`) key spellings.
    pub down_keys: Vec<String>,
    /// Source `alpha` key spellings (no transform, no transpose).
    pub alpha_keys: Vec<String>,
    /// Row-slice applied to the up factor (the qkv split). `None` for a plain rename.
    pub up_slice: Option<LoraRowSlice>,
    /// Row-slice applied to the down factor (shared-or-split). `None` for a plain rename.
    pub down_slice: Option<LoraRowSlice>,
}

/// One contribution of a source key to a target: which target/role it feeds and how to slice it.
struct BflSlot {
    target: String,
    role: KohyaRole,
    slice: Option<LoraRowSlice>,
}

/// Invert `targets` into `source_key → [contribution, …]`. One fused source key (e.g. a qkv
/// `lora_up`) contributes to multiple targets (q/k/v) with different slices, so the value is a list.
fn bfl_index(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflSlot>> {
    let mut index: BTreeMap<String, Vec<BflSlot>> = BTreeMap::new();
    let mut push = |key: &str, target: &str, role: KohyaRole, slice: Option<LoraRowSlice>| {
        index.entry(key.to_string()).or_default().push(BflSlot {
            target: target.to_string(),
            role,
            slice,
        });
    };
    for t in targets {
        for k in &t.up_keys {
            push(k, &t.target_path, KohyaRole::Up, t.up_slice.clone());
        }
        for k in &t.down_keys {
            push(k, &t.target_path, KohyaRole::Down, t.down_slice.clone());
        }
        for k in &t.alpha_keys {
            push(k, &t.target_path, KohyaRole::Alpha, None);
        }
    }
    index
}

/// `true` if any key in `w` is a known BFL source key for `targets` — i.e. the file uses the BFL /
/// ComfyUI naming (`double_blocks`/`single_blocks`/`img_in`/… across the three prefix conventions),
/// which the diffusers/peft/standard-kohya paths cannot resolve. Precise: a standard diffusers/peft
/// or standard-kohya file shares none of these spellings, so it is never misrouted here.
pub fn is_bfl(w: &Weights, targets: &[BflTarget]) -> bool {
    if targets.is_empty() {
        return false;
    }
    let index = bfl_index(targets);
    w.keys().any(|k| index.contains_key(k))
}

/// Recognized LoRA factor suffixes — a key ending in one of these is adapter-shaped (vs. a base
/// weight or some bundled extra). Used to surface BFL-named keys that resolve to no target.
const LORA_FACTOR_SUFFIXES: [&str; 5] = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
];

/// Install a BFL / ComfyUI fused→split LoRA onto `host` (sc-2743). Each file key is matched against
/// the inverted [`BflTarget`] index; a matched *fused* source is row-sliced per destination and fanned
/// into the diffusers split targets (`…img_attn.qkv` → `attn.to_q/to_k/to_v`), a plain rename is copied
/// through. Resolved factors feed the same [`install_lora_groups`] path as PEFT/kohya (transpose +
/// `alpha/rank` fold), so a BFL file yields the byte-identical adapter to the equivalent diffusers
/// split-target LoRA.
///
/// An adapter-shaped key that matches NO target — an off-surface BFL key (e.g. a block out of range)
/// — is surfaced in `unmatched_paths` (loud, never silently dropped). A bundled text-encoder key
/// (`lora_te_…`/`text_encoder.…`) is not a denoiser target and is ignored, matching the PEFT/kohya
/// loaders' treatment of out-of-namespace keys.
pub fn apply_lora_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    targets: &[BflTarget],
) -> Result<ApplyReport> {
    let index = bfl_index(targets);
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(slots) = index.get(&key) else {
            // Not a BFL source key: surface it if it's an adapter-shaped denoiser key (off-surface),
            // ignore a bundled text-encoder adapter (`lora_te_…`/`…text_encoder.…`).
            let adapter_shaped = LORA_FACTOR_SUFFIXES.iter().any(|s| key.ends_with(s));
            let foreign_te = key.starts_with("lora_te") || key.contains("text_encoder");
            if adapter_shaped && !foreign_te {
                unresolved.insert(key);
            }
            continue;
        };
        let v = w.require(&key)?;
        for slot in slots {
            let parts = groups.entry(slot.target.clone()).or_default();
            match slot.role {
                KohyaRole::Down => {
                    parts.a = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Up => {
                    parts.b = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Alpha => parts.alpha = scalar_alpha(v)?,
            }
        }
    }

    let mut report = install_lora_groups(host, groups, scale)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

/// Load and install every adapter in `specs` onto `host`, stacking in order. Each spec's file is
/// read, dispatched to the LoKr or PEFT-LoRA loader by its [`AdapterKind`], applied at `spec.scale`,
/// and its [`ApplyReport`] merged into the combined result — unmatched target paths are surfaced,
/// never silently dropped. `lora_strip_prefix` is the per-family namespace stripped from PEFT-LoRA
/// keys (e.g. `"transformer."`); it does not apply to LoKr (whose keys are bare module paths).
///
/// This is the load-time seam (sc-2534): a provider calls it inside `load()` with its model's
/// [`AdaptableHost`] while the model is still mutable. Empty `specs` is a no-op (empty report).
pub fn apply_adapter_specs(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    lora_strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        let report = match spec.kind {
            AdapterKind::Lokr => apply_lokr(host, &w, spec.scale)?,
            AdapterKind::Lora => {
                // The file's metadata is authoritative; a kind/metadata mismatch is a caller error
                // (the PEFT-LoRA loader would find no `lora_A/B` keys and apply nothing) — surface it.
                if is_lokr(&w) {
                    return Err(format!(
                        "adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )
                    .into());
                }
                apply_lora_peft(host, &w, spec.scale, lora_strip_prefix)?
            }
        };
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// LoRA key namespace prefixes diffusers/peft adapter files use, tried in order; the first that any
/// key begins with is stripped. LoKr files are bare (no prefix). kohya `lora_unet_…` files flatten
/// the module dots to underscores and resolve through a separate flattened→dotted table
/// ([`apply_lora_kohya`], sc-2618), not this prefix strip. SceneWorks' trained LoRAs use
/// `transformer.` (peft `save_lora_weights`) or `diffusion_model.` (sd-scripts export) — both
/// observed on real files.
pub const COMMON_LORA_PREFIXES: [&str; 2] = ["transformer.", "diffusion_model."];

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`COMMON_LORA_PREFIXES`]).
pub fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    let keys: Vec<&str> = w.keys().collect();
    COMMON_LORA_PREFIXES
        .into_iter()
        .find(|&p| keys.iter().any(|k| k.starts_with(p)))
        .map(|v| v as _)
}

/// [`apply_adapter_specs`] with per-file LoRA-prefix **auto-detection** ([`detect_lora_prefix`])
/// instead of a fixed prefix — the common provider path, since LoRA files vary
/// (`transformer.` / `diffusion_model.` / bare) while LoKr keys are bare. The host's key→module map
/// must match the (prefix-stripped) diffusers module paths.
pub fn apply_adapter_specs_autoprefix(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    // The kohya `flattened → dotted` table and the BFL target list both walk the model tree, so
    // build each lazily and only once, the first time it is needed across `specs`.
    let mut kohya: Option<BTreeMap<String, String>> = None;
    let mut bfl: Option<Vec<BflTarget>> = None;
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        // BFL / ComfyUI fused→split naming (sc-2743) is the orthogonal axis to kohya flattening and
        // shares the `lora_unet_` prefix, so it must be detected BEFORE `is_kohya`. (LoKr first.)
        let is_bfl_file = if is_lokr(&w) {
            false
        } else {
            if bfl.is_none() {
                bfl = Some(host.bfl_targets());
            }
            is_bfl(&w, bfl.as_ref().unwrap())
        };
        let report = if is_bfl_file {
            apply_lora_bfl(host, &w, spec.scale, bfl.as_ref().unwrap())?
        } else if !is_lokr(&w) && is_kohya(&w) {
            // kohya LoRA: dots are flattened to underscores, so keys resolve through the table
            // rather than the prefix-strip path. (LoKr keeps dotted paths; checked first.)
            if kohya.is_none() {
                kohya = Some(kohya_table(&host.adaptable_paths()));
            }
            apply_lora_kohya(host, &w, spec.scale, kohya.as_ref().unwrap())?
        } else {
            let prefix = if is_lokr(&w) {
                None
            } else {
                detect_lora_prefix(&w)
            };
            apply_adapter_specs(host, std::slice::from_ref(spec), prefix)?
        };
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// Provider-facing load-time adapter install: [`apply_adapter_specs_autoprefix`] plus a strict
/// no-silent-drop policy — errors if a non-empty spec list matched nothing, or if any adapter
/// target resolved to no module. `model` names the model in the error (e.g. `"z_image_turbo"`).
/// Both Z-Image and Qwen providers call this; the only per-family piece is the model's
/// `AdaptableHost` key→module map.
pub fn apply_adapters_strict(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    let report = apply_adapter_specs_autoprefix(host, specs)?;
    if !specs.is_empty() && report.applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — or LoKr keys)",
            specs.len()
        )
        .into());
    }
    if !report.unmatched_paths.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )
        .into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AdaptableLinear, Adapter};
    use crate::runtime::{AdapterKind, AdapterSpec};
    use mlx_rs::ops::{all_close, array_eq};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A host whose modules live at arbitrary dotted paths — including segment names with internal
    /// underscores (`feed_forward`, `to_out.0`) so the kohya flattening is genuinely ambiguous and a
    /// blind `_`→`.` split would mis-route. `adaptable_paths` returns the registered paths, so it
    /// exercises the real `flattened → dotted` table path.
    struct MultiHost {
        mods: HashMap<String, AdaptableLinear>,
        paths: Vec<String>,
    }
    impl MultiHost {
        fn new(specs: &[(&str, Array)]) -> Self {
            let mut mods = HashMap::new();
            let mut paths = Vec::new();
            for (p, w) in specs {
                mods.insert((*p).to_string(), AdaptableLinear::dense(w.clone(), None));
                paths.push((*p).to_string());
            }
            Self { mods, paths }
        }
    }
    impl AdaptableHost for MultiHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            self.mods.get_mut(&path.join("."))
        }
        fn adaptable_paths(&self) -> Vec<String> {
            self.paths.clone()
        }
    }

    /// Minimal host with a single adaptable linear at path `["lin"]`.
    struct OneLinear {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for OneLinear {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["lin"] => Some(&mut self.lin),
                _ => None,
            }
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_loader_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn lora_peft_transposes_and_folds_alpha() {
        // base [out=4, in=3]; PEFT lora_A [r=2, in=3], lora_B [out=4, r=2], alpha=4 (rank=2).
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let path = tmp("lora.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: a = A^T [in,r], b = B^T * (alpha/rank=2.0) [r,out], scale 0.5.
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_bf16_scalar_alpha_reads_without_panic() {
        // sc-2657: real kohya/BFL FLUX LoRAs ship `alpha` as a **bf16 scalar of shape []**. The alpha
        // read used `as_slice::<f32>()`, which `unwrap`s a dtype mismatch and would panic on bf16 — a
        // latent bug masked by every prior test synthesizing f32 alpha. The fix casts to f32 first.
        // Here a bf16 `[]`-shaped alpha must load AND fold identically to its f32 `[1]`-shaped twin.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        // alpha = 16 (exactly representable in bf16), stored bf16 and 0-d — like the real file.
        let alpha_bf16 = Array::from_slice(&[16.0f32], &[1])
            .reshape(&[] as &[i32])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let path = tmp("lora_bf16_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha_bf16),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1, "bf16 alpha LoRA should apply, not panic");

        // Reference: identical fold with alpha=16, rank=2 → factor 8.
        let mut expected = AdaptableLinear::dense(weight, None);
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[8.0f32], &[1]))
                .unwrap(),
            scale: 0.5,
        });
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_peft_folds_bare_alpha_under_a_prefix() {
        // Prefixed `lora_A/B` (`transformer.lin.lora_{A,B}.weight`) + a BARE `lin.alpha` — the
        // fork's Qwen convention (bare-only alpha patterns). The bare alpha must NOT be dropped:
        // the residual folds alpha/rank into B exactly as the all-bare case does. (sc-2528 review.)
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r=2, in=3]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]); // rank=2 -> factor 2

        let path = tmp("lora_prefixed_bare_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha), // BARE — no `transformer.` prefix
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, Some("transformer.")).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: B scaled by alpha/rank = 2 (the bare alpha was honored).
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "bare alpha under a prefix was dropped or mis-folded"
        );
    }

    #[test]
    fn lora_peft_conflicting_alpha_errors() {
        // A prefixed alpha and a bare alpha that disagree for the same path -> hard error, no
        // silent pick.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("lora_conflicting_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("transformer.lin.alpha", &Array::from_slice(&[4.0f32], &[1])),
                ("lin.alpha", &Array::from_slice(&[8.0f32], &[1])), // disagrees
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        assert!(apply_lora_peft(&mut host, &w, 1.0, Some("transformer.")).is_err());
    }

    #[test]
    fn unmatched_paths_are_reported_not_dropped() {
        // A LoKr file targeting a path the host doesn't have -> applied 0, path reported.
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("lokr_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("missing.path.lokr_w1", &dummy),
                ("missing.path.lokr_w2", &dummy),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_lokr(&w));

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_lokr(&mut host, &w, 1.0).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["missing.path".to_string()]);
    }

    /// The load-time connector stacks a mixed LoRA + LoKr spec list and is equivalent to calling
    /// the underlying loaders directly, in order.
    #[test]
    fn apply_specs_stacks_mixed_lora_and_lokr() {
        // base [out=4, in=2].
        let base_vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let weight = Array::from_slice(&base_vals, &[4, 2]);

        // PEFT LoRA file targeting ["lin"]: lora_A [r=2, in=2], lora_B [out=4, r=2].
        let a_raw = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let lora_path = tmp("specs_lora.safetensors");
        Array::save_safetensors(
            vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
            None,
            &lora_path,
        )
        .unwrap();

        // LoKr file targeting ["lin"]: kron(w1[2,1], w2[2,2]) -> [4,2]; alpha==rank -> factor 1.
        let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let lokr_path = tmp("specs_lokr.safetensors");
        Array::save_safetensors(
            vec![("lin.lokr_w1", &w1), ("lin.lokr_w2", &w2)],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();

        let specs = vec![
            AdapterSpec {
                path: lora_path.clone(),
                scale: 0.5,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            },
            AdapterSpec {
                path: lokr_path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lokr,
                pass_scales: None,
                moe_expert: None,
            },
        ];

        let mut via_specs = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut via_specs, &specs, None).unwrap();
        assert_eq!(report.applied, 2);
        assert!(report.unmatched_paths.is_empty());

        // Reference: the same files through the underlying loaders directly, in order.
        let mut via_loaders = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_loaders,
            &Weights::from_file(&lora_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();
        apply_lokr(
            &mut via_loaders,
            &Weights::from_file(&lokr_path).unwrap(),
            1.0,
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0], &[1, 2]);
        let got = via_specs.lin.forward(&x).unwrap();
        let want = via_loaders.lin.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());

        // Both adapters actually moved the output off the bare base.
        let base = AdaptableLinear::dense(Array::from_slice(&base_vals, &[4, 2]), None)
            .forward(&x)
            .unwrap();
        assert!(!all_close(&got, &base, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_empty_is_noop() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut host, &[], None).unwrap();
        assert_eq!(report, ApplyReport::default());

        let x = Array::from_slice(&[1.0f32, -1.0], &[1, 2]);
        let got = host.lin.forward(&x).unwrap();
        let want = AdaptableLinear::dense(weight, None).forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_reports_unmatched_paths() {
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("specs_miss.safetensors");
        Array::save_safetensors(
            vec![("nope.here.lokr_w1", &dummy), ("nope.here.lokr_w2", &dummy)],
            Some(&meta),
            &path,
        )
        .unwrap();

        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_adapter_specs(&mut host, &specs, None).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["nope.here".to_string()]);
    }

    #[test]
    fn apply_specs_kind_metadata_mismatch_errors() {
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        let path = tmp("specs_mismatch.safetensors");
        Array::save_safetensors(vec![("lin.lokr_w1", &dummy)], Some(&meta), &path).unwrap();

        // Declared Lora but the file's metadata says LoKr -> a loud error, not a silent no-op.
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        assert!(apply_adapter_specs(&mut host, &specs, None).is_err());
    }

    #[test]
    fn detect_lora_prefix_variants() {
        let a = Array::from_slice(&[0.0f32], &[1, 1]);
        let bare = tmp("detect_bare.safetensors");
        Array::save_safetensors(vec![("lin.lora_A.weight", &a)], None, &bare).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&bare).unwrap()),
            None
        );

        let tf = tmp("detect_tf.safetensors");
        Array::save_safetensors(vec![("transformer.lin.lora_A.weight", &a)], None, &tf).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&tf).unwrap()),
            Some("transformer.")
        );

        let dm = tmp("detect_dm.safetensors");
        Array::save_safetensors(vec![("diffusion_model.lin.lora_A.weight", &a)], None, &dm)
            .unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&dm).unwrap()),
            Some("diffusion_model.")
        );
    }

    #[test]
    fn autoprefix_strips_detected_prefix_and_applies() {
        // base [out=2, in=2]; a `transformer.`-prefixed peft LoRA on path ["lin"].
        let weight = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let a = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]); // [r=2, in=2]
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]); // [out=2, r=2]
        let path = tmp("autoprefix_lora.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a),
                ("transformer.lin.lora_B.weight", &b),
            ],
            None,
            &path,
        )
        .unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let report = apply_adapter_specs_autoprefix(&mut host, &specs).unwrap();
        assert_eq!(
            report.applied, 1,
            "transformer.-prefixed key should resolve to lin"
        );
        assert!(report.unmatched_paths.is_empty());

        // Strict wrapper: a bare-but-unmatched target errors rather than silently dropping.
        let miss = tmp("autoprefix_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.nope.lora_A.weight", &a),
                ("transformer.nope.lora_B.weight", &b),
            ],
            None,
            &miss,
        )
        .unwrap();
        let mut host2 = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]), None),
        };
        let specs2 = vec![AdapterSpec {
            path: miss,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        assert!(apply_adapters_strict(&mut host2, &specs2, "test").is_err());
    }

    /// sc-2909: a diffusers/ComfyUI LoRA spelled with `lora_down`/`lora_up` factor suffixes on a
    /// **dotted, un-prefixed** path (the lightx2v Qwen-Image-Lightning format) routes through the
    /// PEFT loader (no `lora_unet_` prefix → not kohya) and installs the BYTE-IDENTICAL adapter to
    /// its `lora_A`/`lora_B` twin — and `apply_adapter_specs_autoprefix` resolves it end-to-end.
    #[test]
    fn diffusers_lora_down_up_equals_peft_ab() {
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r, in]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out, r]
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // down==A, up==B, bare alpha, no namespace prefix — exactly the lightx2v Lightning spelling.
        let down_path = tmp("diffusers_down_up.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_down.weight", &a_raw),
                ("lin.lora_up.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &down_path,
        )
        .unwrap();
        // Detected as un-prefixed (not kohya, not BFL) and resolved through the strict seam.
        let w = Weights::from_file(&down_path).unwrap();
        assert!(!is_kohya(&w), "dotted-path lora_down is NOT kohya");
        assert_eq!(detect_lora_prefix(&w), None, "no namespace prefix");

        let mut via_down = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs_autoprefix(
            &mut via_down,
            &[AdapterSpec::new(down_path, 0.5, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(report.applied, 1, "lora_down/up resolved to lin");
        assert!(report.unmatched_paths.is_empty());

        // The `lora_A`/`lora_B` twin must install the identical adapter.
        let ab_path = tmp("diffusers_ab_twin.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &ab_path,
        )
        .unwrap();
        let mut via_ab = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_ab,
            &Weights::from_file(&ab_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();

        let pull = |h: &mut OneLinear| match h.adaptable_mut(&["lin"]).unwrap().adapters() {
            [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
            _ => panic!("expected one LoRA"),
        };
        let (da, db, ds) = pull(&mut via_down);
        let (pa, pb, ps) = pull(&mut via_ab);
        assert_eq!(ds, ps);
        assert!(
            array_eq(&da, &pa, false).unwrap().item::<bool>()
                && array_eq(&db, &pb, false).unwrap().item::<bool>(),
            "lora_down/up and lora_A/B installed different adapters"
        );
    }

    // ---- kohya `lora_unet_` LoRA (sc-2618) ----

    /// Two modules whose flattened kohya stems are ambiguous under a blind `_`→`.` split: the
    /// segment `to_out.0` and the segment name `feed_forward` both contain the separator char.
    fn kohya_two_module_host() -> MultiHost {
        let w_out = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let w_ff = Array::from_slice(
            &(0..15).map(|i| i as f32 * 0.07).collect::<Vec<_>>(),
            &[5, 3],
        );
        MultiHost::new(&[
            ("blocks.0.attn.to_out.0", w_out),
            ("blocks.0.feed_forward.w1", w_ff),
        ])
    }

    /// The same (down, up, alpha) factors written in BOTH conventions and applied through the
    /// provider seam must yield byte-identical adapters — a kohya file is interchangeable with its
    /// PEFT twin. This is the sc-2618 gate at the core level (no model weights needed).
    #[test]
    fn kohya_equiv_to_peft_bit_exact() {
        // out=4/in=3 and out=5/in=3, rank=2; alpha=4 (≠ rank → exercises the alpha/rank fold).
        let a_out = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r,in]
        let b_out = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out,r]
        let a_ff = Array::from_slice(&[0.05f32, -0.15, 0.2, 0.3, -0.25, 0.1], &[2, 3]);
        let b_ff = Array::from_slice(
            &[0.2f32, -0.2, 0.1, 0.3, -0.1, 0.4, 0.15, -0.35, 0.05, 0.25],
            &[5, 2],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let kohya_path = tmp("equiv_kohya.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a_out),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b_out),
                ("lora_unet_blocks_0_attn_to_out_0.alpha", &alpha),
                ("lora_unet_blocks_0_feed_forward_w1.lora_down.weight", &a_ff),
                ("lora_unet_blocks_0_feed_forward_w1.lora_up.weight", &b_ff),
                ("lora_unet_blocks_0_feed_forward_w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &kohya_path,
        )
        .unwrap();

        let peft_path = tmp("equiv_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blocks.0.attn.to_out.0.lora_A.weight", &a_out),
                ("transformer.blocks.0.attn.to_out.0.lora_B.weight", &b_out),
                ("transformer.blocks.0.attn.to_out.0.alpha", &alpha),
                ("transformer.blocks.0.feed_forward.w1.lora_A.weight", &a_ff),
                ("transformer.blocks.0.feed_forward.w1.lora_B.weight", &b_ff),
                ("transformer.blocks.0.feed_forward.w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();

        let mut via_kohya = kohya_two_module_host();
        let rep_k = apply_adapters_strict(
            &mut via_kohya,
            &[AdapterSpec {
                path: kohya_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();
        assert_eq!(rep_k.applied, 2, "both kohya modules resolve");

        let mut via_peft = kohya_two_module_host();
        apply_adapters_strict(
            &mut via_peft,
            &[AdapterSpec {
                path: peft_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        for p in ["blocks.0.attn.to_out.0", "blocks.0.feed_forward.w1"] {
            let gk = via_kohya.mods.get(p).unwrap().forward(&x).unwrap();
            let gp = via_peft.mods.get(p).unwrap().forward(&x).unwrap();
            assert!(
                array_eq(&gk, &gp, false).unwrap().item::<bool>(),
                "kohya and peft adapters diverged at {p}"
            );
            // And both actually moved off the bare base.
            let base = AdaptableLinear::dense(
                via_kohya
                    .mods
                    .get(p)
                    .unwrap()
                    .dense_weight()
                    .unwrap()
                    .0
                    .clone(),
                None,
            )
            .forward(&x)
            .unwrap();
            assert!(
                !array_eq(&gk, &base, false).unwrap().item::<bool>(),
                "adapter at {p} was a no-op"
            );
        }
    }

    /// The flattened stem `blocks_0_feed_forward_w1` must resolve to `blocks.0.feed_forward.w1`
    /// (the table), NOT the blind split `blocks.0.feed.forward.w1` — proving the disambiguation does
    /// real work.
    #[test]
    fn kohya_table_disambiguates_underscore_segment_names() {
        let mut host = kohya_two_module_host();
        // The blind `_`→`.` split target does not exist; the correct dotted path does.
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed", "forward", "w1"])
            .is_none());
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed_forward", "w1"])
            .is_some());

        let table = kohya_table(&host.adaptable_paths());
        assert_eq!(
            table.get("blocks_0_feed_forward_w1").map(String::as_str),
            Some("blocks.0.feed_forward.w1")
        );
        assert_eq!(
            table.get("blocks_0_attn_to_out_0").map(String::as_str),
            Some("blocks.0.attn.to_out.0")
        );
    }

    /// A `lora_unet_` key whose stem is off-surface (e.g. FLUX.2 BFL `double_blocks_*`, sc-2743) is
    /// surfaced in `unmatched_paths` and fails the strict policy — loud, never silently dropped.
    #[test]
    fn kohya_offsurface_stem_surfaced_and_strict_errors() {
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("kohya_offsurface.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &a,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();

        let mut host = kohya_two_module_host();
        let table = kohya_table(&host.adaptable_paths());
        let report =
            apply_lora_kohya(&mut host, &Weights::from_file(&path).unwrap(), 1.0, &table).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(
            report.unmatched_paths,
            vec!["double_blocks_0_img_attn_qkv".to_string()]
        );

        // Through the strict provider seam it is a hard error.
        let mut host2 = kohya_two_module_host();
        assert!(apply_adapters_strict(
            &mut host2,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .is_err());
    }

    /// A kohya adapter at `scale = 0` is a bit-exact no-op (the scale-0 invariant), and `is_kohya`
    /// detects the format.
    #[test]
    fn kohya_scale_zero_is_bit_exact_noop() {
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("kohya_scale0.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_kohya(&w));

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = kohya_two_module_host();
        let base = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        let table = kohya_table(&host.adaptable_paths());
        apply_lora_kohya(&mut host, &w, 0.0, &table).unwrap();
        let out = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    // ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----

    /// The [`LoraRowSlice`] variants are byte-faithful to the fork's `LoraTransforms`. Expected values
    /// are pinned to the mflux venv (`LoraTransforms.split_*` on the same inputs, sc-2743): up always
    /// slices, down is shared unless the rank is divisible, and the qkv-mlp `dims` slice matches the
    /// `[3072,3072,3072,12288]` boundaries.
    #[test]
    fn lora_row_slice_matches_fork_transforms() {
        // arange(6,2): split_q_up = rows[0:2], split_v_up = rows[4:6].
        let t6 = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[6, 2]);
        let q_up = LoraRowSlice::Chunk { n: 3, index: 0 }.apply(&t6).unwrap();
        assert_eq!(q_up.shape(), &[2, 2]);
        assert_eq!(q_up.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
        let v_up = LoraRowSlice::Chunk { n: 3, index: 2 }.apply(&t6).unwrap();
        assert_eq!(v_up.as_slice::<f32>(), &[8.0, 9.0, 10.0, 11.0]);

        // down: ChunkIfDivisible — whole when rank%3!=0 (the usual fused-qkv LoRA), sliced when ==0.
        let d4 = Array::from_slice(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), &[4, 2]);
        let d4q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&d4)
            .unwrap();
        assert_eq!(d4q.shape(), &[4, 2], "rank 4 not ÷3 → shared whole");
        assert_eq!(d4q.as_slice::<f32>(), d4.as_slice::<f32>());
        let d6q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&t6)
            .unwrap();
        assert_eq!(
            d6q.as_slice::<f32>(),
            &[0.0, 1.0, 2.0, 3.0],
            "rank 6 ÷3 → sliced"
        );

        // qkv-mlp up `dims` (FLUX.1 `linear1`): q = rows[0:3072], mlp = rows[9216:21504].
        let dims = vec![3072, 3072, 3072, 12288];
        let total: i32 = dims.iter().sum();
        let big = Array::from_slice(
            &(0..total).map(|i| i as f32).collect::<Vec<_>>(),
            &[total, 1],
        );
        let q = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 0,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(q.shape(), &[3072, 1]);
        assert_eq!(q.as_slice::<f32>()[0], 0.0);
        let mlp = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 3,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(mlp.shape(), &[12288, 1]);
        assert_eq!(mlp.as_slice::<f32>()[0], 9216.0);
    }

    /// A host with three separate per-head linears at `blk.attn.to_{q,k,v}` (`[inner,in]` each).
    fn three_qkv_host(inner: i32, inp: i32) -> MultiHost {
        let zeros = || Array::from_slice(&vec![0.0f32; (inner * inp) as usize], &[inner, inp]);
        MultiHost::new(&[
            ("blk.attn.to_q", zeros()),
            ("blk.attn.to_k", zeros()),
            ("blk.attn.to_v", zeros()),
        ])
    }

    /// The sc-2743 gate at the core level: a BFL *fused* qkv LoRA, split via [`apply_lora_bfl`],
    /// installs the BYTE-IDENTICAL adapter at each of `to_q/to_k/to_v` as the equivalent *diffusers
    /// split-target* LoRA (the fork-verified PEFT path). The fused up `[3·inner, r]` is row-sliced into
    /// per-head `[inner, r]`; the down `[r, in]` (rank not ÷3) is shared. No model weights needed.
    #[test]
    fn bfl_fused_qkv_equals_diffusers_split() {
        let (inner, inp, r) = (4i32, 3i32, 2i32);
        // Per-head up factors, then the fused up = their dim-0 concat (row-major, so flat concat).
        let bq: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let bk: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.02)
            .collect();
        let bv: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.03)
            .collect();
        let mut fused = Vec::new();
        fused.extend_from_slice(&bq);
        fused.extend_from_slice(&bk);
        fused.extend_from_slice(&bv);
        let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let b_q = Array::from_slice(&bq, &[inner, r]);
        let b_k = Array::from_slice(&bk, &[inner, r]);
        let b_v = Array::from_slice(&bv, &[inner, r]);
        // Shared down [r, in] (rank 2 not ÷3 → shared across q/k/v) + alpha ≠ rank.
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // BFL file: one fused qkv linear (kohya `lora_unet_` spelling).
        let up_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
        let down_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight";
        let alpha_key = "lora_unet_double_blocks_0_img_attn_qkv.alpha";
        let bfl_path = tmp("bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![(up_key, &b_fused), (down_key, &a), (alpha_key, &alpha)],
            None as Option<&HashMap<String, String>>,
            &bfl_path,
        )
        .unwrap();
        let wb = Weights::from_file(&bfl_path).unwrap();

        let mk = |idx: i32, tgt: &str| BflTarget {
            target_path: tgt.to_string(),
            up_keys: vec![up_key.to_string()],
            down_keys: vec![down_key.to_string()],
            alpha_keys: vec![alpha_key.to_string()],
            up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
            down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
        };
        let targets = vec![
            mk(0, "blk.attn.to_q"),
            mk(1, "blk.attn.to_k"),
            mk(2, "blk.attn.to_v"),
        ];

        let mut host_bfl = three_qkv_host(inner, inp);
        let rep = apply_lora_bfl(&mut host_bfl, &wb, 0.7, &targets).unwrap();
        assert_eq!(rep.applied, 3, "all three split targets installed");
        assert!(rep.unmatched_paths.is_empty());

        // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
        let peft_path = tmp("bfl_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blk.attn.to_q.lora_B.weight", &b_q),
                ("transformer.blk.attn.to_q.lora_A.weight", &a),
                ("transformer.blk.attn.to_q.alpha", &alpha),
                ("transformer.blk.attn.to_k.lora_B.weight", &b_k),
                ("transformer.blk.attn.to_k.lora_A.weight", &a),
                ("transformer.blk.attn.to_k.alpha", &alpha),
                ("transformer.blk.attn.to_v.lora_B.weight", &b_v),
                ("transformer.blk.attn.to_v.lora_A.weight", &a),
                ("transformer.blk.attn.to_v.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();
        let wp = Weights::from_file(&peft_path).unwrap();
        let mut host_peft = three_qkv_host(inner, inp);
        apply_lora_peft(&mut host_peft, &wp, 0.7, Some("transformer.")).unwrap();

        for p in ["blk.attn.to_q", "blk.attn.to_k", "blk.attn.to_v"] {
            let pull = |h: &MultiHost| match h.mods.get(p).unwrap().adapters() {
                [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
                _ => panic!("expected one LoRA at {p}"),
            };
            let (ba, bb, bs) = pull(&host_bfl);
            let (pa, pb, ps) = pull(&host_peft);
            assert_eq!(bs, ps, "scale differs at {p}");
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "BFL split and diffusers split installed different adapters at {p}"
            );
        }
    }

    /// `is_bfl` detects a BFL file; an off-surface adapter-shaped key is surfaced (not dropped) while a
    /// bundled text-encoder key is ignored; and a scale-0 BFL adapter is a bit-exact no-op.
    #[test]
    fn bfl_detection_unmatched_and_scale_zero() {
        let up = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            &[4, 2],
        );
        let down = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2, 0.3, -0.3], &[2, 3]);
        let targets = vec![BflTarget {
            target_path: "blk.attn.to_out".to_string(),
            up_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_up.weight".to_string()],
            down_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight".to_string()],
            alpha_keys: vec![],
            up_slice: None,
            down_slice: None,
        }];

        let path = tmp("bfl_detect.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // off-surface BFL key (no target) → surfaced, not silently dropped.
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // bundled text-encoder key → ignored (out of denoiser namespace).
                ("lora_te_text_model_layer_0.lora_up.weight", &up),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_bfl(&w, &targets), "a BFL source key marks the file BFL");

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = MultiHost::new(&[(
            "blk.attn.to_out",
            Array::from_slice(
                &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                &[4, 3],
            ),
        )]);
        let base = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();

        // scale 0 → bit-exact no-op; the off-surface block-9 key is surfaced, the TE key ignored.
        let rep = apply_lora_bfl(&mut host, &w, 0.0, &targets).unwrap();
        assert_eq!(rep.applied, 1, "the on-surface proj target installed");
        // Both block-9 keys (up + down) are surfaced (sorted: down < up); the `lora_te_` key ignored.
        assert_eq!(
            rep.unmatched_paths,
            vec![
                "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight".to_string(),
                "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight".to_string(),
            ],
            "off-surface BFL keys surfaced; TE key ignored"
        );
        let out = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(
            array_eq(&out, &base, false).unwrap().item::<bool>(),
            "scale-0 BFL adapter must be a bit-exact no-op"
        );
    }
}
