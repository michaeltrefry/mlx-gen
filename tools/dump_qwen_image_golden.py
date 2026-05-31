"""Dump the Qwen-Image T2I end-to-end golden for the Rust port (sc-2348, slice 4).

Runs the frozen fork's full prompt→image pipeline at its real inference precision (txt2img keeps
the latents f32; only `prompt_embeds` is bf16 — MLX promotes the bf16 weights to f32 per-op), at a
small resolution so the Rust e2e test stays runnable. Dumps the tokenized inputs, the (bf16)
positive/negative prompt embeds, the seeded f32 noise, the latents after step 0 (for localization),
the final packed latents, and the VAE-decoded image — so the Rust test can validate the transformer
loader + scheduler + CFG + VAE loader against this golden (feeding the dumped noise + embeds to
remove cross-impl RNG / text-encoder precision from the comparison).

Run from the mflux fork venv (loads the full ~54 GB model — needs the RAM the fork already uses):
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_image_golden.py
Output (gitignored): tools/golden/qwen_image_golden.safetensors
"""

import os

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

# --- fixed generation config (mirror these constants in the Rust test) ---
SEED = 42
PROMPT = "a fox sitting in a forest, photorealistic"
NEGATIVE = ""  # -> the fork's single-space fallback
STEPS = 4
HEIGHT = 512
WIDTH = 512
GUIDANCE = 4.0

model = QwenImage(quantize=None)
config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=HEIGHT,
    width=WIDTH,
    guidance=GUIDANCE,
    scheduler="linear",
)

# 1. Seeded packed noise [1, (h/16)*(w/16), 64], f32.
noise = QwenLatentCreator.create_noise(SEED, HEIGHT, WIDTH)

# 2. Encode positive + negative prompts (drop-34, bf16). Tokenize separately to dump the ids.
pos_tok = model.tokenizers["qwen"].tokenize(PROMPT)
neg_tok = model.tokenizers["qwen"].tokenize(" ")  # fork's empty-negative fallback
prompt_embeds, prompt_mask, neg_embeds, neg_mask = QwenPromptEncoder.encode_prompt(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    prompt_cache={},
    qwen_tokenizer=model.tokenizers["qwen"],
    qwen_text_encoder=model.text_encoder,
)

# 3. Denoise loop with CFG (faithful to variants/txt2img/qwen_image.py).
latents = noise
latents_step1 = None
for t in config.time_steps:
    n_pos = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask)  # fmt: off
    n_neg = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=neg_embeds, encoder_hidden_states_mask=neg_mask)  # fmt: off
    guided = QwenImage.compute_guided_noise(n_pos, n_neg, config.guidance)
    latents = config.scheduler.step(noise=guided, timestep=t, latents=latents)
    mx.eval(latents)
    if t == 0:
        latents_step1 = latents

# 4. Unpack + VAE decode.
unpacked = QwenLatentCreator.unpack_latents(latents=latents, height=HEIGHT, width=WIDTH)
decoded = model.vae.decode(unpacked)
mx.eval(decoded)

out = {
    "input_ids_pos": pos_tok.input_ids.astype(mx.int32),
    "attention_mask_pos": pos_tok.attention_mask.astype(mx.int32),
    "input_ids_neg": neg_tok.input_ids.astype(mx.int32),
    "attention_mask_neg": neg_tok.attention_mask.astype(mx.int32),
    "prompt_embeds": prompt_embeds,  # bf16
    "negative_prompt_embeds": neg_embeds,  # bf16
    "noise": noise.astype(mx.float32),
    "latents_step1": latents_step1.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_image_golden.safetensors")
mx.save_safetensors(
    path_out,
    out,
    metadata={
        "seed": str(SEED),
        "steps": str(STEPS),
        "height": str(HEIGHT),
        "width": str(WIDTH),
        "guidance": str(GUIDANCE),
        "prompt": PROMPT,
    },
)
print(f"prompt_embeds={prompt_embeds.shape} neg={neg_embeds.shape} noise={noise.shape}")
print(f"final_latents={latents.shape} decoded={decoded.shape}")
print(f"wrote {path_out} ({len(out)} tensors)")
