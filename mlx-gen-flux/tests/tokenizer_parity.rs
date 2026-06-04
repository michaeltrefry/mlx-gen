//! sc-2787: multi-prompt CLIP+T5 tokenizer parity vs the fork.
//!
//! The CLIP-tokenizer bug (GPT-2 byte-level `from_clip_bpe` instead of CLIP word-BPE) survived
//! because every other test fed the golden's `clip_input_ids` straight into the encoder and the one
//! e2e prompt was plain ASCII. This gate tokenizes a battery of edge-case prompts with the Rust
//! loaders and asserts byte-equality against the fork's ids (`tools/dump_flux_tokenizer_battery.py`).
//!
//! Run: MLX_GEN_FLUX_SNAPSHOT=<snapshot> cargo test -p mlx-gen-flux --test tokenizer_parity \
//!        -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_flux::{load_clip_tokenizer, load_t5_tokenizer, FluxVariant};
use mlx_rs::{Array, Dtype};

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to the matching FLUX.1 snapshot directory"),
    )
}

fn battery_path() -> String {
    format!(
        "{}/../tools/golden/flux_tokenizer_battery.safetensors",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn ids(a: &Array) -> Vec<i32> {
    a.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec()
}

fn first_diff(rust: &[i32], golden: &[i32]) -> usize {
    rust.iter()
        .zip(golden)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| rust.len().min(golden.len()))
}

#[test]
#[ignore = "needs FLUX.1 snapshot + tools/dump_flux_tokenizer_battery.py golden"]
fn clip_and_t5_tokenizer_match_fork_battery() {
    let g = Weights::from_file(battery_path()).unwrap();
    let count: usize = g.metadata("count").unwrap().parse().unwrap();
    let clip_tok = load_clip_tokenizer(&snapshot()).unwrap();
    let t5_tok = load_t5_tokenizer(&snapshot(), FluxVariant::Schnell).unwrap();

    let mut failures = 0;
    for i in 0..count {
        let prompt = g.metadata(&format!("prompt_{i}")).unwrap().to_string();
        let r_clip = ids(&clip_tok.tokenize(&prompt).unwrap().input_ids);
        let r_t5 = ids(&t5_tok.tokenize(&prompt).unwrap().input_ids);
        let g_clip = ids(g.require(&format!("clip_{i}")).unwrap());
        let g_t5 = ids(g.require(&format!("t5_{i}")).unwrap());
        let clip_ok = r_clip == g_clip;
        let t5_ok = r_t5 == g_t5;
        let shown = if prompt.len() > 42 {
            format!("{:.39}...", prompt)
        } else {
            prompt.clone()
        };
        if clip_ok && t5_ok {
            println!("[{i:2}] ✓ {shown:?}");
        } else {
            failures += 1;
            println!("[{i:2}] ✗ {shown:?}  clip_ok={clip_ok} t5_ok={t5_ok}");
            if !clip_ok {
                let j = first_diff(&r_clip, &g_clip);
                let e = |v: &[i32]| v[j..(j + 6).min(v.len())].to_vec();
                println!(
                    "       clip len rust={} golden={}, first diff @ {j}: rust={:?} golden={:?}",
                    r_clip.len(),
                    g_clip.len(),
                    e(&r_clip),
                    e(&g_clip)
                );
            }
            if !t5_ok {
                let j = first_diff(&r_t5, &g_t5);
                let e = |v: &[i32]| v[j..(j + 6).min(v.len())].to_vec();
                println!(
                    "       t5 len rust={} golden={}, first diff @ {j}: rust={:?} golden={:?}",
                    r_t5.len(),
                    g_t5.len(),
                    e(&r_t5),
                    e(&g_t5)
                );
            }
        }
    }
    assert_eq!(
        failures, 0,
        "{failures}/{count} prompts diverge from the fork CLIP/T5 tokenizer"
    );
    println!("✓ all {count} prompts match the fork CLIP+T5 tokenizer byte-for-byte");
}
