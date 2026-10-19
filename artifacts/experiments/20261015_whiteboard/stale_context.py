"""Does exact copying freeze information that should evolve? (2026-10-15 night, user question; GPU inference, no training)

Substitution test with an IDEALIZED PI rollout. Seeded training-ledger windows with layouts (layouts diagnostic only), frames
f0..f15. PI's training-time inference on the REAL window gives, for every step j -> j+1: the label-free scroll est_j, the IC
posterior reassignment partner_j, the edit set E_j (posterior q > 0.5, or entering) and the real pre-chance target z_pre_j.
Stale context of depth m: frames f_{15-m}..f_14 are rebuilt as PI builds imagined frames, but with the TRUE transformations,
  c_j = (E_{j-1} ? z_pre_{j-1} : exact copy of c_{j-1} under est_{j-1}) reassigned by partner_{j-1},   c_{14-m} = f_{14-m}
so c_j has the real content (every changed cell is the real token) and differs from f_j only in that unchanged content keeps the
token it had when last encoded: stale rendering (daylight) and, after scrolls / moves, a stale position code (R2). The world
reads [f_0 .. f_{14-m}, c_{15-m} .. c_14] and its outputs at index 14 (the 14 -> 15 transition) are compared with the fresh
context (m = 0). Everything read is a function of the belief only:
  pursuit     chance head's canonical pair probability toward / away from the player for every layout zombie (afterstate)
  gate        AUC of sigmoid(gate) for the fresh posterior's real changes (non-entering, map / HUD), and mean probability
  gen         |LN(gen) - z_pre| at real-change cells (map / HUD): consequence content
  scroll      own scroll argmax == est_14 (blocked vs moved: reads the faced cell)
  belief      cosine(belief_stale, belief_fresh) at index 14, by cell class (zombie, player-adjacent, other map, HUD)
Split by the number of scrolls inside the stale span (0 = rendering staleness only; >= 1 adds stale position codes).
REFRESH mode (env STALE_REFRESH=1; R34 follow-up): preserve the content, re-derive the position code. Each slot's position offset
P[x] = mean real token at slot x minus the mean over map slots (400 seeded windows, seed 20261025; HUD slots 0). Every stale
cell's origin slot (where its token was last encoded) is tracked through copies and swaps; the world reads token + P[x] -
P[origin] instead of the stale token. Rendering stays stale (one change). Also a token-level check on REAL scrolled transitions
(unchanged, unswapped cells): |fresh - copy| vs |fresh - (copy + P[new] - P[old])|.
Env STALE_CKPT (default pi_seed{s}/pi_u3000.pt, relative to transport/) and STALE_TAG (output suffix) select another checkpoint.
Usage: stale_context.py -> artifacts/eda/whiteboard/stale_context[_refresh][<tag>].json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_model as DM
import direct_train as BASE
import transport_ic_train as IC
import transport_stage2 as S2
import pi_model as PI
import rep_layers as RL
from d4mj.train import autocast_context

OUT = HERE.parents[1] / 'eda/whiteboard'
T = A5.BASE / 'transport'
N_WINDOWS = int(os.environ.get("STALE_WINDOWS", 320))
DEPTHS = (0, 1, 4, 8, 14)
ADJ = (22, 30, 32, 40)
INTERIOR = set(r * 9 + c for r in range(1, 6) for c in range(1, 8))


def dist(x):
    r, c = divmod(x, 9)
    return abs(r - 3) + abs(c - 4)


def outputs(w, frames, ac, noise, config, est16):
    with autocast_context(config):
        belief, ha = w.backbone_full(frames, ac)
        copy, gen, glogit, enter, sl = w.afterstate_parts(frames, ac, belief, ha, scroll=est16)
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    return belief.float(), copy.float(), gen.float(), glogit.float(), enter, sl.float(), logits


def new_tally():
    return {'gate_s': [], 'gate_y': [], 'gate_hud_s': [], 'gate_hud_y': [], 'gen_err_map': [], 'gen_err_hud': [], 'scroll_ok': 0,
            'n': 0, 'toward': [], 'away': []}


def tally(a, i, o, change14, z_pre, est, L, scroll, pair_index):
    """one window's index-14 readings from outputs o = (belief, copy, gen, glogit, enter, sl, logits)"""
    _, _, gn, gl, _, sl_m, lg = o
    a['n'] += 1
    p = torch.sigmoid(gl[i, 14]).cpu(); y = change14[i].cpu()
    a['gate_s'].append(p[:PI.MAP]); a['gate_y'].append(y[:PI.MAP])
    a['gate_hud_s'].append(p[PI.MAP:]); a['gate_hud_y'].append(y[PI.MAP:])
    err = (F.layer_norm(gn[i, 14], (192,)) - z_pre[i, 14]).norm(dim=-1).cpu()
    a['gen_err_map'].append(err[:PI.MAP][y[:PI.MAP]]); a['gen_err_hud'].append(err[PI.MAP:][y[PI.MAP:]])
    a['scroll_ok'] += int(sl_m[i, 14].argmax() == est[i, 14])
    prob = torch.sigmoid(IC.pair_prior_logit(lg[i:i + 1, 14:15]))[0, 0].cpu()
    prev = S2.afterstate_layout(L[14], *scroll.SHIFTS[int(est[i, 14])])
    for x in range(63):
        if x not in INTERIOR or x == 31 or int(prev[x, 0]) != 1 or dist(x) > 4:
            continue
        tw = aw = 0.
        for yy in (x - 9, x + 9, x - 1, x + 1):
            pr = (min(x, yy), max(x, yy))
            if pr in pair_index:
                q = float(prob[pair_index[pr]])
                tw += q * (dist(yy) < dist(x)); aw += q * (dist(yy) > dist(x))
        a['toward'].append(tw); a['away'].append(aw)


def summarize(a):
    gs, gy = torch.cat(a['gate_s']), torch.cat(a['gate_y'])
    hs, hy = torch.cat(a['gate_hud_s']), torch.cat(a['gate_hud_y'])
    em, eh = torch.cat(a['gen_err_map']), torch.cat(a['gen_err_hud'])
    auc = lambda sc, lb: None if RL.auc(sc, lb) is None else round(float(RL.auc(sc, lb)), 4)
    return {'gate_auc_map': auc(gs, gy), 'gate_auc_hud': auc(hs, hy),
            'gate_prob_changed_map': round(float(gs[gy].mean()), 4) if gy.any() else None,
            'gate_prob_unchanged_map': round(float(gs[~gy].mean()), 4),
            'gate_prob_changed_hud': round(float(hs[hy].mean()), 4) if hy.any() else None,
            'gen_err_changed_map': round(float(em.mean()), 3) if len(em) else None, 'changed_map_cells': len(em),
            'gen_err_changed_hud': round(float(eh.mean()), 3) if len(eh) else None, 'changed_hud_cells': len(eh),
            'scroll_acc': round(a['scroll_ok'] / max(a['n'], 1), 4), 'zombies_d<=4': len(a['toward']),
            'pursuit_toward': round(float(np.mean(a['toward'])), 4) if a['toward'] else None,
            'pursuit_away': round(float(np.mean(a['away'])), 4) if a['away'] else None}


@torch.no_grad()
def main():
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261023))[:N_WINDOWS]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    scroll = A5.load_scroll()
    pair_index = {p: i for i, p in enumerate(IC.PAIRS)}
    refresh = os.environ.get('STALE_REFRESH') == '1'
    res = {}
    if refresh:
        psel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261025))[:400]
        tot = torch.zeros(81, 192, dtype=torch.float64); cnt = 0
        for b0 in range(0, 400, 8):
            fr, _, _ = BASE.batch(tokens, actions, windows, psel[b0:b0 + 8], device='cpu')
            tot += fr.double().sum((0, 1)); cnt += fr.shape[0] * fr.shape[1]
        mean = (tot / cnt).float()
        P = torch.zeros(81, 192); P[:PI.MAP] = mean[:PI.MAP] - mean[:PI.MAP].mean(0)
        P = P.cuda()
        src = torch.arange(81).repeat(5, 1)                                         # [5,81] copy source per scroll choice
        for sc, (dr, dc) in enumerate(PI.SCROLL_OFFSETS):
            for x in range(PI.MAP):
                r, c = divmod(x, 9)
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    src[sc, x] = (r + dr) * 9 + c + dc
        src = src.cuda()
        res['token_check'] = {'plain': [], 'refreshed': []}
    for seed in (1, 2):
        w = PI.PIWorld().cuda().eval()
        w.load_state_dict(torch.load(T / os.environ.get('STALE_CKPT', 'pi_seed{s}/pi_u3000.pt').format(s=seed), map_location='cpu', weights_only=False)['world'])
        g = torch.Generator().manual_seed(3)
        acc = {}
        for b0 in range(0, N_WINDOWS, 8):
            frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
            noise = DM.draw_noise(frames.shape[:2], g)[0].cuda()
            est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())      # [B,15]
            est16 = F.pad(est, (0, 1))
            belief, copy, gen, glogit, enter, sl, logits = outputs(w, frames, ac, noise, config, est16)
            z = frames[:, 1:].float()
            a0 = torch.where(((glogit[:, :15] > 0) | enter[:, :15])[..., None], gen[:, :15], copy[:, :15])
            partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits[:, :15]))
            z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
            res_copy = (F.layer_norm(copy[:, :15], (192,)) - z_pre).norm(dim=-1)
            gain = res_copy - (F.layer_norm(gen[:, :15], (192,)) - z_pre).norm(dim=-1)
            s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
            edit = (torch.sigmoid(gain / s + glogit[:, :15]) > .5) | enter[:, :15]          # [B,15,81]
            change14 = edit[:, 14] & ~enter[:, 14]                                         # real changes at 14 -> 15
            idx = torch.zeros(frames.shape[0], 15, 81, dtype=torch.long, device=frames.device)
            idx[..., :PI.MAP] = est[..., None]
            lays = [torch.from_numpy(np.array(lay[row, start:start + 16]) > 0) for row, start in choices.tolist()]
            if refresh and seed == 1:                                                  # token-level check on real transitions
                for j in range(15):
                    moved = est[:, j] != 0
                    if not bool(moved.any()):
                        continue
                    cand = w.candidates(frames[:, j:j + 1].float())[:, 0]
                    cp = cand.gather(-2, idx[:, j][..., None, None].expand(-1, -1, 1, 192))[..., 0, :]
                    keep = moved[:, None] & ~edit[:, j] & (partner[:, j] == torch.arange(81, device=frames.device))
                    keep[:, PI.MAP:] = False
                    sx = src[est[:, j]]                                                # [B,81]
                    rf = cp + P[None] - P[sx]
                    res['token_check']['plain'].append((frames[:, j + 1].float() - cp).norm(dim=-1)[keep].cpu())
                    res['token_check']['refreshed'].append((frames[:, j + 1].float() - rf).norm(dim=-1)[keep].cpu())
            fresh = None
            for m in DEPTHS:
                stale = frames.clone().float()
                orig = torch.arange(81, device=frames.device).repeat(len(frames), 1)
                feed = stale
                for j in range(15 - m, 15):
                    if m == 0:
                        break
                    cand = w.candidates(stale[:, j - 1:j])[:, 0]                           # [B,81,5,192]
                    cp = cand.gather(-2, idx[:, j - 1][..., None, None].expand(-1, -1, 1, 192))[..., 0, :]
                    after = torch.where(edit[:, j - 1][..., None], z_pre[:, j - 1], cp)
                    stale[:, j] = after.gather(1, partner[:, j - 1][..., None].expand(-1, -1, 192))
                    if refresh:
                        o_cp = orig.gather(1, src[est[:, j - 1]])
                        o_after = torch.where(edit[:, j - 1], torch.arange(81, device=frames.device).expand_as(o_cp), o_cp)
                        orig = o_after.gather(1, partner[:, j - 1])
                        if feed is stale:
                            feed = stale.clone()
                        feed[:, j] = stale[:, j] + P[None] - P[orig]
                outs = outputs(w, feed, ac, noise, config, est16)
                if m == 0:
                    fresh = outs
                    continue
                nscroll = (est[:, 15 - m:15] != 0).sum(1)
                for i in range(len(frames)):
                    key = f'm{m}_' + ('scrolls0' if int(nscroll[i]) == 0 else ('scrolls1-3' if int(nscroll[i]) <= 3 else 'scrolls4+'))
                    a = acc.setdefault(key, {'windows': 0, 'fresh': new_tally(), 'stale': new_tally(),
                                             'cos': {'zombie': [], 'adjacent': [], 'map_other': [], 'hud': []}})
                    a['windows'] += 1
                    for which, o in (('fresh', fresh), ('stale', outs)):
                        tally(a[which], i, o, change14, z_pre, est, lays[i], scroll, pair_index)
                    cs = F.cosine_similarity(outs[0][i, 14], fresh[0][i, 14], dim=-1).cpu()
                    L = lays[i]
                    zmask = torch.zeros(81, dtype=torch.bool); zmask[:63] = L[14][:, 0].bool()
                    amask = torch.zeros(81, dtype=torch.bool); amask[list(ADJ)] = True
                    hmask = torch.zeros(81, dtype=torch.bool); hmask[63:] = True
                    omask = ~(zmask | amask | hmask); omask[31] = False
                    for k_, mk in (('zombie', zmask), ('adjacent', amask), ('map_other', omask), ('hud', hmask)):
                        if mk.any():
                            a['cos'][k_].append(cs[mk])
            if b0 % 80 == 0:
                print(json.dumps({'seed': seed, 'windows_done': b0 + 8}), flush=True)
        out = {}
        for key, a in sorted(acc.items()):
            out[key] = {'windows': a['windows'], 'fresh': summarize(a['fresh']), 'stale': summarize(a['stale']),
                        'belief_cos': {k_: round(float(torch.cat(v).mean()), 4) for k_, v in a['cos'].items() if v}}
        res[f'seed{seed}'] = out
        print(json.dumps({f'seed{seed}': out}), flush=True)
        del w; torch.cuda.empty_cache()
    if refresh:
        pl, rf = torch.cat(res['token_check']['plain']), torch.cat(res['token_check']['refreshed'])
        res['token_check'] = {'cells': len(pl), 'plain_mean_dist': round(float(pl.mean()), 4), 'refreshed_mean_dist': round(float(rf.mean()), 4)}
        print(json.dumps({'token_check': res['token_check']}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'stale_context{"_refresh" if refresh else ""}{os.environ.get("STALE_TAG", "")}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
