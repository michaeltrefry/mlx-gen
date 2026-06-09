"""Kolors ChatGLM3 tokenizer — materialize the fast `tokenizer.json` + dump the parity golden (sc-3092).

ChatGLM3 ships only a **slow** SentencePiece tokenizer (`tokenizer/tokenizer.model`, a custom
`ChatGLMTokenizer(PreTrainedTokenizer)` — no fast `tokenizer.json`, no `backend_tokenizer`). The Rust
`mlx_gen::TextTokenizer` loads the HF `tokenizers` fast serialization, so this:

 1. Converts the SP model → a fast `tokenizer.json` via transformers' `SpmConverter` and writes it
    into the snapshot's `tokenizer/` dir. The fast tokenizer reproduces the SP **content** ids
    (`EncodeAsPieces`→`PieceToId`); it adds NO special tokens (the ChatGLM `[gMASK]`/`sop` prefix +
    left-pad + position_ids are applied by the Rust `KolorsTokenizer` wrapper, matching
    `build_inputs_with_special_tokens` + `_pad`).
 2. Validates the fast ids == `sp_model.encode(text)` across an EN+CN battery, and asserts the
    special-token ids (`[gMASK]`=64790, `sop`=64792, pad=unk=0).
 3. Dumps `tools/golden/kolors_tokenizer_golden.safetensors`: per battery prompt, the REFERENCE
    `ChatGLMTokenizer(prompt, padding="max_length", max_length=256, truncation=True)` input_ids /
    attention_mask / position_ids — the byte-identical gate for the Rust `KolorsTokenizer`.

Run (fork venv with transformers + sentencepiece):
    ~/repos/mflux/.venv-0312/bin/python tools/build_kolors_tokenizer.py
"""

import glob

import mlx.core as mx
import numpy as np
from sentencepiece import SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as spb
from tokenizers import Tokenizer, decoders, normalizers
from tokenizers.models import BPE

from _paths import fixture, hf_hub_cache

from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer
from transformers.tokenization_utils_base import generate_merges

MAX_LEN = 256

# EN, EN-long (>256 content → truncation), CN, mixed CN/EN, empty (the negative-prompt path).
BATTERY = [
    "A cat playing a grand piano on a city rooftop at sunset.",
    "a serene mountain lake at dawn, " * 40,
    "夕阳下，一只猫在城市楼顶弹钢琴。",
    "A red 熊猫 sitting under a 樱花树 in 京都, 8k photo.",
    "",
]


def tok_dir():
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-diffusers" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-diffusers snapshot not found in HF cache")
    return f"{snaps[-1]}/tokenizer"


def main():
    td = tok_dir()
    model_path = f"{td}/tokenizer.model"

    ref = ChatGLMTokenizer(vocab_file=model_path)
    sp = SentencePieceProcessor(model_file=model_path)

    # Special-token ids (appended after the SP vocab). Assert the constants the Rust wrapper bakes.
    sp_vocab = sp.vocab_size()
    gmask = ref.get_command("[gMASK]")
    sop = ref.get_command("sop")
    pad = ref.pad_token_id
    print(f"sp_vocab={sp_vocab} [gMASK]={gmask} sop={sop} pad={pad}")
    assert (gmask, sop, pad) == (64790, 64792, 0), (gmask, sop, pad)

    # 1. Build the fast tokenizer.json from the SP model — a faithful replica of transformers'
    # `LlamaConverter` (ChatGLM3's SP is LLaMA-style: byte_fallback BPE, 256 `<0xXX>` byte pieces,
    # identity normalizer, dummy ▁ prefix). transformers' `SpmConverter`/`SpmExtractor` is broken in
    # this version (extract() signature regression), so we build the BPE directly: vocab from the
    # proto pieces, merges via `generate_merges` (score-ordered, reproducing SP's merge priority).
    proto = spb.ModelProto()
    proto.ParseFromString(open(model_path, "rb").read())
    vocab_scores = [(p.piece, p.score) for p in proto.pieces]
    vocab_dict = {piece: i for i, (piece, _) in enumerate(vocab_scores)}
    if "\t" not in vocab_dict:  # "<0x09>" is the byte fallback for tab; needed for merges
        vocab_dict["\t"] = vocab_dict.get("<0x09>")
    merges = generate_merges(vocab_dict, vocab_scores)

    bpe = BPE(vocab=vocab_dict, merges=merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True)
    fast = Tokenizer(bpe)
    # LLaMA-style legacy normalizer: prepend ▁, map space→▁ (add_dummy_prefix + identity norm).
    fast.normalizer = normalizers.Sequence(
        [normalizers.Prepend(prepend="▁"), normalizers.Replace(pattern=" ", content="▁")]
    )
    fast.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1)]
    )
    out = f"{td}/tokenizer.json"
    fast.save(out)
    print(f"wrote {out}")

    # 2. Validate fast content ids == sp.encode across the battery.
    for s in BATTERY:
        if not s:
            continue
        want = sp.encode(s)
        got = fast.encode(s).ids
        assert got == want, f"fast != sp for {s[:30]!r}: {got[:8]} vs {want[:8]}"
    print("fast tokenizer content ids == sp.encode ✓")

    # 3. Dump the reference (input_ids/attention_mask/position_ids) golden.
    tensors = {}
    for i, s in enumerate(BATTERY):
        enc = ref(s, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="np")
        tensors[f"p{i}_input_ids"] = mx.array(enc["input_ids"].astype(np.int32))
        tensors[f"p{i}_attention_mask"] = mx.array(enc["attention_mask"].astype(np.int32))
        tensors[f"p{i}_position_ids"] = mx.array(enc["position_ids"].astype(np.int32))
        nz = int(enc["attention_mask"].sum())
        print(f"p{i}: {s[:30]!r:34} valid={nz}/{MAX_LEN}")
    mx.eval(list(tensors.values()))
    meta = {"max_len": str(MAX_LEN), "n_prompts": str(len(BATTERY)),
            "gmask": str(gmask), "sop": str(sop), "pad": str(pad)}
    gpath = fixture("tools/golden/kolors_tokenizer_golden.safetensors")
    mx.save_safetensors(gpath, tensors, metadata=meta)
    print(f"wrote {gpath} ({len(tensors)} tensors)")
    # The prompts the Rust test feeds (kept in sync with the golden order).
    print("BATTERY:", BATTERY)


if __name__ == "__main__":
    main()
