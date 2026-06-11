//! sc-3838: the Chroma text-mask construction matches `ChromaPipeline._get_t5_prompt_embeds`.
//! Golden = `tools/dump_chroma_text_mask_golden.py` (tokenizer-only — no T5 weights). Validates
//! (a) the vendored tokenizer encodes identical `input_ids`, and (b) the transformer mask reproduces
//! the keep-one-extra-pad quirk exactly. The T5 *numeric* masked-encode parity rides on the e2e
//! (sc-3839), where the real T5 weights load.

use mlx_gen::weights::Weights;
use mlx_gen_chroma::{loader, transformer_text_mask};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Dtype;

const PROMPTS: [&str; 2] = ["a photograph of an astronaut riding a horse", "a cat"];
const MAX_LEN: usize = 64; // must match the dump script

fn max_abs(a: &mlx_rs::Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

#[test]
fn text_mask_matches_diffusers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let io = Weights::from_file(format!("{dir}/chroma_text_mask.safetensors")).unwrap();
    let tok = loader::load_tokenizer_with_max_len(MAX_LEN).unwrap();

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let out = tok.tokenize(prompt).unwrap();
        let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&out);

        // (a) input_ids identical to the diffusers T5TokenizerFast.
        let ids_golden = io.require(&format!("input_ids_{i}")).unwrap();
        let ids_got = input_ids.as_dtype(Dtype::Int32).unwrap();
        assert_eq!(ids_got.shape(), ids_golden.shape(), "input_ids shape [{i}]");
        let id_diff = max_abs(
            &subtract(
                ids_got.as_dtype(Dtype::Float32).unwrap(),
                ids_golden.as_dtype(Dtype::Float32).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(id_diff, 0.0, "input_ids diverge for prompt {i}");

        // (b) transformer mask reproduces (arange <= seq_lengths), keep-one-extra-pad.
        let mask_got = transformer_text_mask(&input_ids, 0).unwrap();
        let mask_golden = io.require(&format!("attention_mask_{i}")).unwrap();
        let m_diff = max_abs(&subtract(&mask_got, mask_golden).unwrap());
        assert_eq!(m_diff, 0.0, "transformer mask diverges for prompt {i}");
    }
}
