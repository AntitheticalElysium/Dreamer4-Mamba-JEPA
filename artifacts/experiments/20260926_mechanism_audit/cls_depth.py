"""Controlled near/far zombie edit, read at every ViT layer (init and trained)."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/experiments/20260926_diagnosis")]
from twins import base_states, render_all, encoder_at, natural_frames, inverse_sqrt_cov


@torch.no_grad()
def all_layers(encoder, frames, device, batch=64):
    layers = None
    for i in range(0, len(frames), batch):
        f = frames[i:i + batch].to(device)
        pixels = f.permute(0, 3, 1, 2).float() / 255
        pixels = (pixels - encoder.pixel_mean) / encoder.pixel_std
        out = encoder.backbone(pixels, interpolate_pos_encoding=True, output_hidden_states=True)
        levels = list(out.hidden_states) + [out.last_hidden_state]
        if layers is None:
            layers = [[] for _ in levels]
        for bucket, x in zip(layers, levels):
            bucket.append(x.float().cpu())
    return [torch.cat(bucket) for bucket in layers]


def main():
    torch.set_num_threads(6)
    states = base_states()
    frames = render_all(states)
    natural = natural_frames()[:804]
    out = {"states": len(states), "natural": len(natural), "steps": {}}
    for step in (0, 10000):
        encoder = encoder_at(step, torch.device("cuda"))
        base = all_layers(encoder, frames[("day", "base")], "cuda")
        near = all_layers(encoder, frames[("day", "zombie")], "cuda")
        far = all_layers(encoder, frames[("day", "zombie_far")], "cuda")
        nat = all_layers(encoder, natural, "cuda")
        rows = []
        for layer, (b, n, f, v) in enumerate(zip(base, near, far, nat)):
            c0, cn, cf, cv = b[:, 0], n[:, 0], f[:, 0], v[:, 0]
            t0, tn, tf, tv = b[:, 1+3*9+5], n[:, 1+3*9+5], f[:, 1+3*9+5], v[:, 1+3*9+5]
            dn, df = (cn - c0).double().mean(0), (cf - c0).double().mean(0)
            rel = (cn - cf).norm(dim=1).mean() / (cv - cv.mean(0)).norm(dim=1).median().clamp_min(1e-8)
            wc = inverse_sqrt_cov(cv)
            wt = inverse_sqrt_cov(tv)
            rows.append({"layer": layer, "cls_near_far_cosine": float(F.cosine_similarity(dn[None], df[None])[0])
                         if dn.norm() > 0 and df.norm() > 0 else None,
                         "cls_near_far_over_natural_radius": float(rel),
                         "cls_near_far_dprime": float((wc @ (cn - cf).double().mean(0)).norm()),
                         "local_token_near_far_dprime": float((wt @ (tn - tf).double().mean(0)).norm())})
        out["steps"][str(step)] = rows
        print(step, [(r["layer"], round(r["cls_near_far_dprime"],3), round(r["local_token_near_far_dprime"],2)) for r in rows], flush=True)
        del encoder
        torch.cuda.empty_cache()
    (HERE / "cls_depth.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
