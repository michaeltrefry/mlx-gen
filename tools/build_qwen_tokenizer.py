"""Materialize the Qwen-Image fast `tokenizer.json` for the Rust port (sc-2348, slice 4).

The `Qwen/Qwen-Image` snapshot ships the Qwen2 BPE tokenizer as `vocab.json` + `merges.txt` only
(no `tokenizer.json`); the Python fork builds the *fast* tokenizer at runtime via `transformers`.
The Rust `mlx_gen::TextTokenizer` loads the HF `tokenizers` fast serialization (`tokenizer.json`),
so this writes that file once into the snapshot's `tokenizer/` dir (the byte-identical fast
tokenizer the fork uses — same vocab, merges, NFC + ByteLevel pipeline, and special tokens).

Run (fork venv, which has `transformers`):
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/build_qwen_tokenizer.py
Override the snapshot with QWEN_IMAGE_SNAPSHOT.
"""

import glob
import os

from transformers import AutoTokenizer


def snapshot_dir() -> str:
    if env := os.environ.get("QWEN_IMAGE_SNAPSHOT"):
        return env
    home = os.path.expanduser("~")
    snaps = sorted(
        glob.glob(f"{home}/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/*/")
    )
    if not snaps:
        raise SystemExit("no Qwen-Image snapshot found; set QWEN_IMAGE_SNAPSHOT")
    return snaps[0]


def main() -> None:
    tok_dir = os.path.join(snapshot_dir(), "tokenizer")
    out = os.path.join(tok_dir, "tokenizer.json")
    tk = AutoTokenizer.from_pretrained(tok_dir)
    if not tk.is_fast:
        raise SystemExit("loaded a slow tokenizer; cannot serialize a fast tokenizer.json")
    tk.backend_tokenizer.save(out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes); is_fast={tk.is_fast}")


if __name__ == "__main__":
    main()
