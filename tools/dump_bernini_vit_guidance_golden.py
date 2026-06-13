"""sc-5142: golden for the renderer's ViT-conditioned guidance-combine modes (sample_one_step).

`apg_delta` + the velocity-combine formulas of `sample_one_step` (`wan_diffusion.py` 795-1049), the
full-Bernini `BerniniPipeline`-only modes:
  - `vae_txt_vit` / `vae_txt_vit_wapg` — 3 cumulative VAE-conditioned deltas (apg ref = the "to" pred).
  - `rv2v_wapg` / `r2v_wapg` — 5-prediction reference chain (apg ref = the "from" pred).

`apg_delta` is copied **verbatim** from the reference; the combine arms are transcribed line-for-line.
Inputs are random `[1, n_target, C]` target-sliced packed-token predictions (B=1), so the Rust port
must match to the f32 floor.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_vit_guidance_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/vit_guidance_golden.safetensors
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "vit_guidance_golden.safetensors")

N = 5    # target tokens
C = 8    # packed channels (ph*pw*c)
W_IMG, W_TXT, W_TGT, W_VID = 4.5, 4.0, 3.0, 1.25


# ===== verbatim reference: apg_delta (wan_diffusion.py) =====
def apg_delta(delta, ref, parallel_scale=0.2, orthogonal_scale=1.0, eps=1e-8):
    b = delta.shape[0]
    delta_f = delta.reshape(b, -1)
    ref_f = ref.reshape(b, -1)
    ref_norm_sq = (ref_f * ref_f).sum(dim=1, keepdim=True).clamp_min(eps)
    proj_coeff = (delta_f * ref_f).sum(dim=1, keepdim=True) / ref_norm_sq
    delta_parallel_f = proj_coeff * ref_f
    delta_orthogonal_f = delta_f - delta_parallel_f
    return (
        parallel_scale * delta_parallel_f.reshape_as(delta)
        + orthogonal_scale * delta_orthogonal_f.reshape_as(delta)
    )


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)

    def rnd():
        return torch.randn(1, N, C)

    # vae_txt_vit predictions
    base = rnd()           # cond_pred_wotxt_wovit_wovae
    img = rnd()            # cond_pred_wotxt_wovit_wvae
    txt = rnd()            # cond_pred_wtxt_wovit_wvae
    vit = rnd()            # cond_pred_wtxt_wvit_wvae

    # vae_txt_vit (plain)
    vtv_plain = (
        base
        + W_IMG * (img - base)
        + W_TXT * (txt - img)
        + W_TGT * (vit - txt)
    )
    # vae_txt_vit_wapg (apg, ref = the "to" pred)
    vtv_apg = (
        base
        + W_IMG * apg_delta(img - base, ref=img, parallel_scale=0.2, orthogonal_scale=1.0)
        + W_TXT * apg_delta(txt - img, ref=txt, parallel_scale=0.2, orthogonal_scale=1.0)
        + W_TGT * apg_delta(vit - txt, ref=vit, parallel_scale=0.2, orthogonal_scale=1.0)
    )

    # rv2v chain predictions
    eps_v = rnd()
    eps_vi = rnd()
    eps_vti = rnd()
    eps_vtic = rnd()

    # rv2v_wapg (plain deltas)
    rv2v_plain = (
        base
        + W_VID * (eps_v - base)
        + W_IMG * (eps_vi - eps_v)
        + W_TXT * (eps_vti - eps_vi)
        + W_TGT * (eps_vtic - eps_vti)
    )
    # r2v_wapg (apg deltas, ref = the "from" pred)
    r2v_apg = (
        base
        + W_VID * apg_delta(eps_v - base, ref=base)
        + W_IMG * apg_delta(eps_vi - eps_v, ref=eps_v)
        + W_TXT * apg_delta(eps_vti - eps_vi, ref=eps_vi)
        + W_TGT * apg_delta(eps_vtic - eps_vti, ref=eps_vti)
    )

    # a bare apg_delta for the projection-only check
    apg_only = apg_delta(img - base, ref=img, parallel_scale=0.2, orthogonal_scale=1.0)

    out = {
        "io.base": base, "io.img": img, "io.txt": txt, "io.vit": vit,
        "io.eps_v": eps_v, "io.eps_vi": eps_vi, "io.eps_vti": eps_vti, "io.eps_vtic": eps_vtic,
        "out.apg_only": apg_only,
        "out.vtv_plain": vtv_plain, "out.vtv_apg": vtv_apg,
        "out.rv2v_plain": rv2v_plain, "out.r2v_apg": r2v_apg,
    }
    out = {k: v.contiguous() for k, v in out.items()}
    meta = {"n": str(N), "c": str(C),
            "w_img": repr(W_IMG), "w_txt": repr(W_TXT), "w_tgt": repr(W_TGT), "w_vid": repr(W_VID)}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)  N={N} C={C}")


if __name__ == "__main__":
    main()
