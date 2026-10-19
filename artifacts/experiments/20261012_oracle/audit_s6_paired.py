"""Audit (10-14), stage 6: the paired event study (other agent, PAIRED_RESULTS.md) - mechanisms behind its two failures.
The registered numbers and milestone trajectories are recomputed by audit_s5_direct.py paired (independent code).
  attrib   paired u3000, the 14 strict sources (audit_s5's selection), 256 fresh noise draws per root (seed 20261014, as the
           noise probe). The output token at each cross cell is LN(v), v = reference b + own-packet self edit + arrivals; LN has
           no affine, so given the token's own LN scale, the real reader's score is exactly additive over these components:
           score = bias' + sum_c (centered(c) / sigma(v) / sd) . w. Components: reference, self (the cell's own packet's
           -g(1-p_self)R write), from_source (the true source packet's arrival into this cell; 0 at the source), other (all
           other arrivals). Summarized per decoded event class, at the source cell and at decoded destinations (and, for absent
           draws, at the highest-scoring destination). Also source route entropy and route-argmax vs decoded direction on valid
           moves. Reconstruction of the cached-path output is asserted.
  entering paired u3000 deterministic reference on the 104 roots (batch 4 as evaluated): for every entering map cell (exact
           recorded camera offset, identical across the 32 real keys of each root), which copy candidate the scroll asks for
           (zero padding outside the 9x9 grid, or a HUD token in rows 7-8), the parent routing weight on it and on the
           generator, and the reference centered RMS; low-reference (<0.1, the other agent's cut) vs normal entering cells.
Usage: audit_s6_paired.py attrib|entering -> artifacts/eda/oracle_v1/paired_event/audit_<mode>.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_s5_direct as A5  # noqa: E402

OUT = A5.BASE / "paired_event"
OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def setup():
    import paired_model as PM
    import direct_train as T
    import direct_eval as E
    config = T.settings()
    data, _ = E.cohort("exploratory")
    world = PM.PairedWorld().cuda().eval()
    world.load_state_dict(torch.load(OUT / "training/paired_u3000.pt", map_location="cpu", weights_only=False)["world"])
    head = torch.load(A5.BASE / "learned_innovation/eval_u3000_h01/mob_heads.pt", map_location="cpu", weights_only=False)["heads"]["real"]
    return PM, config, data, world, head


@torch.no_grad()
def attrib():
    PM, config, data, world, head = setup()
    from d4mj.train import autocast_context
    decode = A5.decoder(head)
    mu, sd, w = head["mu"].double(), head["sd"].double(), head["w"].double()
    units = json.loads((A5.OUT / "audit_recompute.json").read_text())["sources_list"]
    K = 256
    g = torch.Generator().manual_seed(20261014)
    names = ("reference", "self", "from_source", "other")
    rows = {}                                                           # (event class, cell role) -> list of [K', 4] contributions

    def add(key, x):
        rows.setdefault(key, []).append(x)
    route_entropy, direction_match, n_moves = [], 0, 0
    for root, source, kindname in units:
        kind = A5.KINDS.index(kindname)
        wk = w[:-1, A5.READER_CHANNEL[kindname]]
        r, c = divmod(source, 9)
        pts = [(r + dr) * 9 + c + dc for dr, dc in A5.CROSS]
        ctx = data["context_tokens"][root:root + 1].cuda().float(); ac = data["actions"][root:root + 1].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1)
        ac = F.pad(ac, (0, 1))
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
        frames, ac, belief, ha = frames[:, 14:15], ac[:, 14:15], belief[:, 14:15], ha[:, 14:15]
        base = PM.draw_noise((K, 16), g)[0]; PM.draw_noise((K, 16), g)  # same stream as the noise probe (base, fresh)
        comps, toks, routes_s, gates = [], [], [], []
        for c0 in range(0, K, 64):
            k = min(64, K - c0)
            with autocast_context(config):
                out, routes, aux = world.complete(frames.expand(k, -1, -1, -1), ac.expand(k, -1), belief.expand(k, -1, -1, -1),
                                                  ha.expand(k, -1, -1), base[c0:c0 + k, 14:15].cuda(), trace=True)
            ref, selfe, arr = (aux[x][:, 0].double().cpu() for x in ("reference", "source_edits", "arrivals"))
            # arrivals[s, e] is written by source s along edge e (OFFSETS[e]) into cell s + OFFSETS[e]
            cell = torch.zeros(k, 5, 4, 192, dtype=torch.float64)
            for j, p in enumerate(pts):
                total_in = torch.zeros(k, 192, dtype=torch.float64); from_src = torch.zeros(k, 192, dtype=torch.float64)
                pr, pc = divmod(p, 9)
                for e, (dr, dc) in enumerate(OFFSETS[1:], 1):
                    sr, sc = pr - dr, pc - dc                           # the sender whose edge e lands on p
                    if 0 <= sr < 9 and 0 <= sc < 9:
                        contrib = arr[:, sr * 9 + sc, e]
                        total_in += contrib
                        if sr * 9 + sc == source:
                            from_src += contrib
                cell[:, j] = torch.stack([ref[:, p], selfe[:, p], from_src, total_in - from_src], 1)
            vv = cell.sum(2)                                            # [k,5,192]
            recon = F.layer_norm(vv, (192,))
            err = float((recon - out[:, 0, pts].double().cpu()).abs().max())
            assert err < 2e-3, err
            sigma = (vv.var(-1, unbiased=False) + 1e-5).sqrt()            # [k,5]
            centered = cell - cell.mean(-1, keepdim=True)
            comps.append(((centered / sigma[..., None, None]) / sd) @ wk)  # [k,5,4] additive score contributions
            toks.append(out[:, 0].float().cpu()); routes_s.append(routes[:, 0, source].float().cpu()); gates.append(aux["gate"][:, 0, source, 0].float().cpu())
        comps, toks, routes_s = torch.cat(comps), torch.cat(toks), torch.cat(routes_s)
        f = decode(toks)[:, pts, kind]; ev = A5.event(f)
        for i in range(K):
            e = int(ev[i]); cls = "stay" if e == 0 else "move" if e < 5 else "absent" if e == 5 else "multiple"
            add((cls, "source"), comps[i, 0])
            flagged = [j for j in range(1, 5) if f[i, j]]
            if flagged:
                for j in flagged:
                    add((cls, "flagged_destination"), comps[i, j])
            elif cls == "absent":
                j = 1 + int(comps[i, 1:].sum(-1).argmax())
                add((cls, "best_unflagged_destination"), comps[i, j])
            if cls == "move":
                n_moves += 1
                direction_match += int(int(routes_s[i].argmax()) == e)
                p = routes_s[i].clamp_min(1e-12); route_entropy.append(float(-(p * p.log()).sum()))
        print(json.dumps({"root": root, "source": source, "kind": kindname, "events": [int((ev == x).sum()) for x in range(7)]}), flush=True)
    bias = {k: float((-(mu / sd) @ w[:-1, A5.READER_CHANNEL[k]]) + w[-1, A5.READER_CHANNEL[k]]) for k in A5.KINDS}
    summary = {}
    for (cls, role), xs in sorted(rows.items()):
        x = torch.stack(xs)
        summary[f"{cls}/{role}"] = {"n": len(x), **{n: round(float(x[:, j].mean()), 3) for j, n in enumerate(names)},
                                    "sum_mean": round(float(x.sum(-1).mean()), 3)}
    res = {"draws_per_root": K, "sources": len(units), "score_bias_by_kind": {k: round(v, 3) for k, v in bias.items()},
           "thresholds_dev_f1[zombie,skeleton,cow,arrow]": head["thresholds"]["dev_f1"],
           "mean_score_contribution": summary,
           "valid_moves": n_moves, "route_argmax_matches_decoded_direction": round(direction_match / max(n_moves, 1), 4),
           "chance_if_uniform_over_4": .25, "source_route_entropy_on_valid_moves_mean": round(float(np.mean(route_entropy)), 4) if route_entropy else None,
           "max_entropy_5_routes": round(math.log(5), 4)}
    (OUT / "audit_attrib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


@torch.no_grad()
def entering():
    PM, config, data, world, head = setup()
    from d4mj.train import autocast_context
    meta = torch.load(A5.OUT / "eval_reference/metadata.pt", map_location="cpu", weights_only=False)
    shifts = meta["shifts"]
    assert bool((shifts == shifts[:, :1]).all())
    groups = {}
    n = len(data["roots"])
    for s0 in range(0, n, 4):
        ctx = data["context_tokens"][s0:s0 + 4].cuda().float(); ac = data["actions"][s0:s0 + 4].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            ref = world.reference(frames, ac, belief, ha)[:, 14].float().cpu()
            # the same routing weights reference() mixes with (copied from PairedWorld.reference)
            logits = world.choose(belief).float()
            moved = world.frame(ha).float()[:, :, None, :]
            target = torch.tensor([31, 30, 32, 22, 40], device=ac.device)[ac.clamp(max=4)]
            htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
            moved = moved + (((ac >= 1) & (ac <= 4)).float()[:, :, None, None] * world.target_gate(htarget).float()[:, :, None, :])
            weights = (logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])).softmax(-1)[:, 14].float().cpu()
        rms = (ref - ref.mean(-1, keepdim=True)).square().mean(-1).sqrt()
        for b in range(len(ctx)):
            i = s0 + b
            dr, dc = (int(x) for x in shifts[i, 0])
            if (dr, dc) == (0, 0):
                continue
            slot = OFFSETS.index((dr, dc))
            for cell in range(63):
                r, c = divmod(cell, 9)
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    continue                                            # not entering
                kind = "zero_pad" if not (0 <= r + dr < 9 and 0 <= c + dc < 9) else "HUD_token"
                low = bool(rms[b, cell] < .1)
                key = f"{'low' if low else 'normal'}_reference/{kind}"
                groups.setdefault(key, []).append([float(weights[b, cell, slot]), float(weights[b, cell, 5]), float(weights[b, cell, 0]), float(rms[b, cell])])
    res = {"definition": "entering = (r+dr, c+dc) outside the 7x9 map; candidate = the copy slot of the recorded offset; zero_pad = outside the 9x9 token grid",
           "groups": {k: {"cells": len(v), "mean_weight_on_scroll_candidate": round(float(np.mean([x[0] for x in v])), 4),
                          "mean_weight_on_generator": round(float(np.mean([x[1] for x in v])), 4),
                          "mean_weight_on_self": round(float(np.mean([x[2] for x in v])), 4),
                          "mean_reference_centered_rms": round(float(np.mean([x[3] for x in v])), 4)} for k, v in sorted(groups.items())}}
    (OUT / "audit_entering.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"attrib": attrib, "entering": entering}[sys.argv[1]]()
