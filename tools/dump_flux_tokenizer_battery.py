"""Multi-prompt FLUX tokenizer parity golden (sc-2787).

The CLIP-tokenizer bug hid because every test fed the golden's `clip_input_ids` into the encoder and
the one e2e prompt was plain ASCII. This dumps the fork's CLIP + T5 ids for a battery of edge-case
prompts (case, punctuation, whitespace collapse, digits, apostrophes, hyphens, accents/non-ASCII,
symbols, leading/trailing space, >77-token truncation, emoji) so the Rust tokenizer can be asserted
byte-equal. Mirrors mflux's `LanguageTokenizer.tokenize` exactly: HF tokenizer with
`padding="max_length"`, `truncation=True`, `add_special_tokens=True` (CLIP max 77, T5 max 256).

Run with the mflux venv: /Users/michael/Repos/mflux/.venv-0312/bin/python3 tools/dump_flux_tokenizer_battery.py
"""

import os

import mlx.core as mx
from transformers import AutoTokenizer, CLIPTokenizer

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/"
    "snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "flux_tokenizer_battery.safetensors")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PROMPTS = [
    "a red fox",
    "A RED FOX",
    "a cat, sitting on a mat.",
    "a   red    fox",
    "3 cats and 2 dogs",
    "the dog's tail",
    "a sci-fi cityscape, 8k",
    "café au lait",
    "100% wool sweater!",
    "  leading and trailing spaces  ",
    "a highly detailed " * 30 + "fox",  # > 77 tokens → exercises truncation
    "a fox \U0001F98A in the snow",  # emoji (multi-byte) → byte-level fallback
]

clip = CLIPTokenizer.from_pretrained(SNAP + "/tokenizer")
t5 = AutoTokenizer.from_pretrained(SNAP + "/tokenizer_2")

tensors = {}
for i, p in enumerate(PROMPTS):
    c = clip(p, padding="max_length", max_length=77, truncation=True, add_special_tokens=True, return_tensors="np")
    t = t5(p, padding="max_length", max_length=256, truncation=True, add_special_tokens=True, return_tensors="np")
    tensors[f"clip_{i}"] = mx.array(c["input_ids"].astype("int32"))
    tensors[f"t5_{i}"] = mx.array(t["input_ids"].astype("int32"))

meta = {"count": str(len(PROMPTS))}
meta.update({f"prompt_{i}": p for i, p in enumerate(PROMPTS)})
mx.save_safetensors(OUT, tensors, meta)
print(f"wrote {OUT} with {len(PROMPTS)} prompts")
for i, p in enumerate(PROMPTS):
    disp = p if len(p) < 40 else p[:37] + "..."
    print(f"  [{i:2}] {disp!r:44} clip[:6]={tensors[f'clip_{i}'].tolist()[0][:6]} t5[:6]={tensors[f't5_{i}'].tolist()[0][:6]}")
