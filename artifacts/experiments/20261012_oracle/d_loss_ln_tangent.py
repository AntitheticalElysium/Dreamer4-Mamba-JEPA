"""CPU fixed-output loss descent through the world's terminal LayerNorm.

Gradient geometry is measured at saved emissions; it does not measure the
parameter update. The LayerNorm Jacobian numerator is computed at the emitted
output y: g - mean(g) - y mean(g*y). The missing positive pre-LN scale does not
change the sign of the resulting linear-reader derivative. Saved fp16 outputs
are used unchanged. The failed attempt that re-normalized them is archived.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from d_loss_fixed_outputs import GROUPS, OUT, gradients, objectives


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    panel = torch.load(OUT / "panel.pt", map_location="cpu", weights_only=False)
    state = torch.load(OUT / "snap_bank_readers.pt", map_location="cpu", weights_only=False)
    m = state["heads"]["mob"]
    rw = (m["w"][:-1] / m["sd"][:, None]).float()
    thr = torch.tensor(state["thresholds"])

    def scores(x):
        q = x.reshape(-1, 192)
        parts = []
        for i in range(0, len(q), 4096):
            f = ((q[i:i+4096].float() - m["mu"]) / m["sd"]).double()
            parts.append((torch.cat([f, torch.ones(len(f), 1, dtype=torch.float64)], 1) @ m["w"]).float())
        return torch.cat(parts).view(*x.shape[:-1], 4)

    result = {"schema": "d_loss_ln_tangent_v1", "unit": "output loss-descent direction before/through per-token LayerNorm",
              "pre_ln_scale": "unknown positive divisor omitted; sign is invariant to it",
              "inputs": {str(p.name): sha(p) for p in [OUT / "panel.pt", OUT / "snap_bank_readers.pt"]}, "worlds": {}}
    target = panel["tokens"][:, 16:32].float()
    target_sc = scores(target)
    for world in ("oracle_deltairis", "oracle_l1", "control_deltairis", "control_l1"):
        path = OUT / f"pred_{world}.pt"
        result["inputs"][path.name] = sha(path)
        st = torch.load(path, map_location="cpu", weights_only=False)
        wr = result["worlds"].setdefault(world, {})
        for mode in ("tf", "gen"):
            pred = st[mode].float()
            x = pred.flatten(0, 1)
            z = x
            e = x - target.flatten(0, 1)
            gs, _ = gradients(e)
            tg = {k: g - g.mean(-1, keepdim=True) - z * (g*z).mean(-1, keepdim=True) for k,g in gs.items()}
            pred_sc = scores(z.view_as(pred))
            mr = wr.setdefault(mode, {"output_reused_unchanged": bool(torch.equal(x,z)),
                                     "output_coordinate_mean_max_abs": float(x.mean(-1).abs().max()),
                                     "output_token_variance_min": float(x.var(-1,unbiased=False).min()), "groups": {}})
            for group, steps in GROUPS.items():
                ids = torch.zeros(200, 16, dtype=torch.bool)
                ids[:, steps] = True
                row = {}
                for mob, (lay_ch, read_ch) in {"cow": (1,2), "zombie": (0,0)}.items():
                    truth = panel["layouts"][:, 16:32, :, lay_ch] > 0
                    decode = pred_sc[:, :, :63, read_ch] > thr[read_ch]
                    truth[:, :, 31] = False; decode[:,:,31] = False
                    masks = {"fp_actual_target_reader_absent": decode & ~truth & (target_sc[:,:,:63,read_ch] <= thr[read_ch]),
                             "fn_actual_target_reader_present": ~decode & truth & (target_sc[:,:,:63,read_ch] > thr[read_ch])}
                    mobs = row.setdefault(mob, {})
                    for label, mask in masks.items():
                        mask = mask & ids[:,:,None]
                        n = int(mask.sum())
                        out = {"cells": n, "losses": {}}
                        for loss,g in gs.items():
                            direct = (-g @ rw[:,read_ch]).view(200,16,81)[:,:,:63]
                            tang = (-tg[loss] @ rw[:,read_ch]).view(200,16,81)[:,:,:63]
                            bad_direct = direct > 0 if label.startswith('fp') else direct < 0
                            bad_tang = tang > 0 if label.startswith('fp') else tang < 0
                            out["losses"][loss] = {"wrong_semantic_direction_direct": int(bad_direct[mask].sum()),
                                                  "wrong_semantic_direction_layernorm": int(bad_tang[mask].sum()),
                                                  "layernorm_signed_score_derivative_numerator_median": float(tang[mask].median()) if n else None}
                        mobs[label] = out
                mr["groups"][group] = row
            print(json.dumps({"world":world,"mode":mode,"output_reused_unchanged":mr["output_reused_unchanged"],"late":mr["groups"]["9-16"]}),flush=True)
    # Direct analytic LayerNorm Jacobian must agree with a literal autograd
    # computation, including the max term's selected coordinate.
    x = torch.randn(5,81,192,generator=torch.Generator().manual_seed(20261012),requires_grad=True)
    y = F.layer_norm(x,(192,)); target_test = torch.randn(y.shape,generator=torch.Generator().manual_seed(12))
    gs,_ = gradients((y-target_test).detach())
    sd = (x.detach().var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
    checks = {}
    for loss,value in objectives(y-target_test).items():
        ag = torch.autograd.grad(value.sum(),x,retain_graph=True)[0]
        g = gs[loss]; z=y.detach()
        formula = (g-g.mean(-1,keepdim=True)-z*(g*z).mean(-1,keepdim=True))/sd
        checks[loss] = float((ag-formula).abs().max())
        assert checks[loss] < 1e-6, checks
    result['autograd_max_difference'] = checks
    result['source_sha256'] = sha(__file__)
    dest = OUT / 'd_loss_ln_tangent.json'
    tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(dest)
    print(json.dumps({'done':str(dest),'autograd_max_difference':checks}),flush=True)


if __name__=='__main__':
    main()
