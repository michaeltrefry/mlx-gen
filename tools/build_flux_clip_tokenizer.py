"""Vendor an HF-faithful CLIP `tokenizer.json` for the FLUX.1 port (sc-2787).

The FLUX repo ships the CLIP tokenizer only as `vocab.json` + `merges.txt` (no `tokenizer.json`),
and the core `TextTokenizer::from_clip_bpe` mis-tokenizes it (GPT-2 byte-level BPE instead of CLIP's
lowercased word-BPE with `</w>` suffixes). This bakes the real CLIP fast tokenizer into the crate so
the loader can load it explicitly (`include_str!`) and never silently fall back to the broken helper.

Truncation is enabled (max_length=77, the fork's `truncation=True`) so over-length prompts truncate
HF-style keeping the EOS; padding is left OFF here — the Rust `tokenize_preformatted` pads to 77 with
the CLIP pad id (49407) and builds the attention mask. Run with the mflux venv (transformers):
  /Users/michael/Repos/mflux/.venv-0312/bin/python3 tools/build_flux_clip_tokenizer.py
"""

import os

from transformers import CLIPTokenizerFast

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/"
    "snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/tokenizer"
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlx-gen-flux", "assets", "clip_tokenizer.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

tok = CLIPTokenizerFast.from_pretrained(SNAP)
backend = tok.backend_tokenizer
backend.enable_truncation(max_length=77)
backend.no_padding()
backend.save(OUT)
print("wrote", OUT)

# Self-check: the vendored json (via the raw `tokenizers` lib) must match the slow CLIPTokenizer that
# mflux uses, for the BOS/EOS + content ids (pre-padding).
from tokenizers import Tokenizer  # noqa: E402

vend = Tokenizer.from_file(OUT)
from transformers import CLIPTokenizer  # noqa: E402

slow = CLIPTokenizer.from_pretrained(SNAP)
for p in ["a red fox", "A RED FOX", "a cat, sitting on a mat."]:
    v = vend.encode(p).ids
    s = slow(p, add_special_tokens=True)["input_ids"]
    print(f"{p!r:30} vend={v} slow={s} match={v == s}")
