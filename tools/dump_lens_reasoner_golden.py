#!/usr/bin/env python
"""Dump a greedy-generation golden for the Lens PromptReasoner (mlx-gen sc-3176).

Runs the authoritative `LensGptOssEncoder` (the vendor `GptOssForCausalLM` subclass) as a **generating**
model — `generate(do_sample=False)` — over the harmony reasoner prompt (the rewriter system instruction
+ `reasoning_effort="low"` + the generation prompt), and dumps the prompt `input_ids` + the greedily
generated token ids. The Rust gate (`tests/reasoner_parity.rs`) reproduces the template byte-exactly,
runs its own KV-cache greedy decode, and checks the token stream against torch.

Generation is `do_sample=False` (greedy / argmax) for determinism. `max_new_tokens` is small (the
reasoner output is one sentence; the gate compares the leading greedy tokens — bf16 MLX-vs-torch
argmax can diverge on a late near-tie, exactly as the encoder e2e, so the gate is prefix-based +
teacher-forced).

Run from the reference venv (loads the ~40 GB bf16 model):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_reasoner_golden.py
Writes `tools/golden/lens_reasoner_golden.safetensors` (gitignored).
"""

from __future__ import annotations

import datetime
import glob
import importlib.util
import os

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Mxfp4Config

HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*"
VENDOR_TE = os.path.expanduser(
    "~/Repos/SceneWorks/apps/worker/scene_worker/_vendor/lens/text_encoder.py"
)
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_reasoner_golden.safetensors")

PROMPT = "a cat on a skateboard"
MAX_NEW_TOKENS = 24

# The vendor PromptReasoner system prompt (verbatim) + the local-path suffix.
SYSTEM_PROMPT = """
You are a prompt rewriter for a text-to-image model.
Your task is to convert the user's input into a single, precise, descriptive image prompt suitable for a text-to-image model.
Follow these rules strictly:

1. The output must be a clear and accurate description of a single image scene, written in the style of a text-to-image prompt.
  - Do not include explanations, reasoning, commentary, or meta text.
  - Do not ask questions.
  - Do not output multiple options.
  - Do not use uncertain, speculative, or alternative wording such as "maybe", "possibly", "perhaps", "or", "might", or "could".

2. Preserve the user's intended scene faithfully.
  - Do not change the objects, entities, attributes, actions, relationships, or core setting explicitly described by the user.
  - You may add reasonable visual details only when they help make the image concrete and coherent.
  - Any added details must be consistent with the user's description and must not introduce new important objects or alter the meaning.

3. If the image contains many main subjects of the same kind, describe each subject in detail, including humans, animals, objects, and any other prominent elements.
  - For each subject, include its appearance, color, size, shape, material, pose, expression, and position if applicable in the scene.
  - Make sure every main subject is clearly distinguishable from the others, such as in a scene with "4 dogs," describing each dog separately.

4. The output must fully cover the scene implied by the user's input.
  - Include the main subjects, relevant attributes, actions, spatial relationships, environment, and visible details necessary to render the scene.
  - If the user input is already sufficiently detailed and already suitable for image generation, keep it unchanged or only make minimal edits for fluency and clarity.

5. Resolve content that requires simple inference into explicit visual results when the result is unambiguous and visually representable.
  - Example: if the user says "the answer to 2+2 is written on the blackboard", output should explicitly describe "the blackboard shows 2+2=4".
  - Use only direct, necessary inference that is clearly implied by the user input.
  - Do not invent hidden facts, backstory, or ambiguous details.

6. Language rule:
  - If the user input is not in English, output in the same language.
  - Otherwise, output in English.

7. Output format:
  - Output exactly one final rewritten prompt.
  - Do not use bullet points, numbering, JSON, XML, Markdown, or quotation marks unless they are part of the scene itself.

Your goal is to produce a prompt that is concrete, visual, faithful to the user intent, and directly usable as input to a text-to-image model.
""".strip()


def load_encoder_cls():
    spec = importlib.util.spec_from_file_location("lens_text_encoder", VENDOR_TE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensGptOssEncoder


@torch.no_grad()
def main() -> None:
    snaps = sorted(p for p in glob.glob(SNAP_GLOB) if os.path.isdir(p))
    if not snaps:
        raise SystemExit(f"no Lens-Turbo snapshot at {SNAP_GLOB}")
    snap = snaps[-1]

    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Keep any reasoning private. The visible answer must contain only the final rewritten prompt."
    )
    conversation = [
        {"role": "system", "content": system_prompt, "thinking": None},
        {"role": "user", "content": PROMPT, "thinking": None},
    ]
    text = tok.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
    )
    input_ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids

    cfg = AutoConfig.from_pretrained(os.path.join(snap, "text_encoder"))
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    print("loading text_encoder (MXFP4 → bf16, CPU)…", flush=True)
    model = load_encoder_cls().from_pretrained(
        os.path.join(snap, "text_encoder"),
        config=cfg,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    # NOTE: do NOT call set_selected_layers — the generate path must hit the stock LM forward, not the
    # feature-capture override.

    print(f"greedy generate ({MAX_NEW_TOKENS} tokens)…", flush=True)
    out_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out_ids[0, input_ids.shape[1]:]
    decoded = tok.decode(new_tokens, skip_special_tokens=False)
    print("generated:", repr(decoded), flush=True)

    tensors = {
        "input_ids": input_ids.to(torch.int32).cpu(),
        "new_tokens": new_tokens.to(torch.int32).cpu().reshape(1, -1),
    }
    meta = {
        "prompt": PROMPT,
        "current_date": datetime.date.today().isoformat(),
        "max_new_tokens": str(MAX_NEW_TOKENS),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}  (L={input_ids.shape[1]}, new={new_tokens.shape[0]})")


if __name__ == "__main__":
    main()
