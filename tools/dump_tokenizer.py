"""Generate the text-tokenizer parity fixture for the Rust port (sc-2341).

Uses the fork's actual Z-Image `LanguageTokenizer` (AutoTokenizer over the Qwen2 fast
tokenizer from tokenizer.json; Qwen3 chat template with enable_thinking=True; max_length=512,
padding='max_length', add_special_tokens=True). Dumps input_ids + attention_mask (int32,
padded to 512) for several prompts so the Rust crate can reproduce them exactly.

The Rust side loads the SAME tokenizer.json via the `tokenizers` crate, renders the
single-user-message chat template (which collapses to
`<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n`), encodes, then pads to 512.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_tokenizer.py
"""

import mlx.core as mx

from transformers import AutoTokenizer

from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer

TOK = (
    "/Users/michael/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/"
    "snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6/tokenizer"
)

raw = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
lt = LanguageTokenizer(
    tokenizer=raw,
    max_length=512,
    padding="max_length",
    use_chat_template=True,
    chat_template_kwargs={"enable_thinking": True},
    add_special_tokens=True,
)

PROMPTS = [
    "a red fox",
    "A serene mountain lake at sunset, photorealistic",
    "café — naïve façade, 日本語 prompt, emoji 🦊",  # unicode / byte-level BPE stress
]

out = {}
for i, p in enumerate(PROMPTS):
    o = lt.tokenize(p)
    ids = o.input_ids.astype(mx.int32)
    mask = o.attention_mask.astype(mx.int32)
    out[f"p{i}.input_ids"] = ids
    out[f"p{i}.attention_mask"] = mask
    n = int(mx.sum(mask).item())
    print(f"p{i}: shape={ids.shape} valid={n} prompt={p!r}")

path = "/Users/michael/repos/mlx-gen/tests/fixtures/tokenizer_zimage.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors); pad_token_id={raw.pad_token_id}")
