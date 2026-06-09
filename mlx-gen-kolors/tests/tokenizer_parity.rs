//! Kolors (ChatGLM3) tokenizer parity vs the diffusers reference (sc-3092).
//!
//! `#[ignore]`d: needs the materialized `tokenizer.json` in the Kolors snapshot + the golden from
//! `tools/build_kolors_tokenizer.py` (writes both). Asserts the Rust `KolorsTokenizer` produces
//! input_ids / attention_mask / position_ids **byte-identical** to
//! `ChatGLMTokenizer(prompt, padding="max_length", max_length=256, truncation=True)` across an EN /
//! EN-long(truncated) / CN / mixed / empty battery.
//!
//! Run: `cargo test -p mlx-gen-kolors --test tokenizer_parity -- --ignored --nocapture`

use mlx_rs::ops::all_close;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_kolors::tokenizer::KolorsTokenizer;

// Kept in sync with BATTERY in tools/build_kolors_tokenizer.py (same order as the golden p0..p4).
fn battery() -> Vec<String> {
    vec![
        "A cat playing a grand piano on a city rooftop at sunset.".into(),
        "a serene mountain lake at dawn, ".repeat(40),
        "夕阳下，一只猫在城市楼顶弹钢琴。".into(),
        "A red 熊猫 sitting under a 樱花树 in 京都, 8k photo.".into(),
        String::new(),
    ]
}

fn tokenizer_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("KOLORS_TOKENIZER_DIR") {
        return d.into();
    }
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
        .join("tokenizer")
}

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_tokenizer_golden.safetensors"
);

fn assert_eq_ids(got: &Array, want: &Array, label: &str) {
    assert_eq!(
        got.shape(),
        want.shape(),
        "{label}: shape {:?} vs {:?}",
        got.shape(),
        want.shape()
    );
    // Integer ids — must be exactly equal.
    assert!(
        all_close(got, want, 0.0, 0.0, None).unwrap().item::<bool>(),
        "{label}: ids differ\n got={:?}\nwant={:?}",
        got.as_slice::<i32>(),
        want.as_slice::<i32>()
    );
}

#[test]
#[ignore = "needs snapshot tokenizer.json + tools/golden/kolors_tokenizer_golden.safetensors (build_kolors_tokenizer.py)"]
fn kolors_tokenizer_matches_reference() {
    let tok = KolorsTokenizer::from_dir(tokenizer_dir()).expect("load tokenizer");
    let g = Weights::from_file(GOLDEN).expect("tokenizer golden");

    for (i, prompt) in battery().iter().enumerate() {
        let out = tok.encode(prompt).expect("encode");
        assert_eq_ids(
            &out.input_ids,
            g.require(&format!("p{i}_input_ids")).unwrap(),
            &format!("p{i} input_ids"),
        );
        assert_eq_ids(
            &out.attention_mask,
            g.require(&format!("p{i}_attention_mask")).unwrap(),
            &format!("p{i} attention_mask"),
        );
        assert_eq_ids(
            &out.position_ids,
            g.require(&format!("p{i}_position_ids")).unwrap(),
            &format!("p{i} position_ids"),
        );
        eprintln!("p{i}: byte-identical (input_ids/attention_mask/position_ids)");
    }
}
