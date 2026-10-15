"""Where does the chase reading live? Partial-freshness substitution (2026-10-16; R39 open question; GPU inference; exact
simulator state used only for the true transformations and evaluation labels)

recontext_stale.py's data and exact transformations (diagnosis_futures_v1; window = root + 15 future frames). Stale Raw-token
context of depth m = 8 (frames 7..14 rebuilt by exact copies under the true transformations, fresh tokens at changed cells),
restricted to windows with >= 1 camera scroll in the span. Variants restore the REAL (fresh) token in a subset of the stale span:
  stale            nothing restored (R34 / R39 T)
  zombie           cells holding a zombie in that frame (exact layout)
  zombie_nb        zombie cells and their 4 neighbours
  player_nb        the player cell 31 and its 4 neighbours
  map_not_zombie   every map cell except zombie cells (terrain and other mobs fresh, zombies stale)
  hud              the 18 HUD cells
  last_frame       all cells of the most recent context frame (14) only
  history          all cells of frames 7..13, frame 14 stale
  fresh            everything (the real context)
PI u3000 (both seeds) reads each; at index 14: chase toward-probability for exact zombies within distance 4 of the player,
gen error at exact terrain edits, gate health-drop AUC. Recovery = (variant - stale) / (fresh - stale).
Usage: chase_cue.py -> artifacts/eda/whiteboard/chase_cue.json   (env CHASE_ROOTS, default 240)
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

import recontext_stale as RS

M = 8
VARIANTS = ('stale', 'zombie', 'zombie_nb', 'player_nb', 'map_not_zombie', 'hud', 'last_frame', 'history', 'fresh')


def mask_for(variant, mobs):
    """[16,81] bool: which (frame, cell) of the stale span take the fresh token"""
    m = torch.zeros(16, 81, dtype=torch.bool)
    span = range(15 - M, 15)
    for j in span:
        z = torch.zeros(81, dtype=torch.bool); z[:63] = mobs[j][:, 0]
        if variant == 'zombie':
            m[j] = z
        elif variant == 'zombie_nb':
            nb = z.clone()
            for x in torch.where(z)[0].tolist():
                r, c = divmod(x, 9)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                        nb[(r + dr) * 9 + c + dc] = True
            m[j] = nb
        elif variant == 'player_nb':
            m[j, [31, 22, 40, 30, 32]] = True
        elif variant == 'map_not_zombie':
            m[j, :63] = ~z[:63]
        elif variant == 'hud':
            m[j, 63:] = True
        elif variant == 'last_frame':
            m[j] = j == 14
        elif variant == 'history':
            m[j] = j != 14
        elif variant == 'fresh':
            m[j] = True
    return m


@torch.no_grad()
def main():
    import direct_model as DM
    import direct_train as BASE
    import transport_ic_train as IC
    import pi_model as PI
    import rep_layers as RL
    import rollouts as Rr
    from d4mj.train import autocast_context
    config = BASE.settings()
    enc = RS.Encoder()
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261028))[:int(os.environ.get('CHASE_ROOTS', 240))].tolist()
    idx_all = torch.zeros(5, 81, dtype=torch.long)
    for s in range(5):
        idx_all[s, :PI.MAP] = s
    idx_all = idx_all.cuda()
    pair_index = {p: i for i, p in enumerate(IC.PAIRS)}
    prepared = []
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if bool(r['future_dead'][k][:15].any()):
                continue
            frames = torch.cat([r['context'][-1:], r['future_frames'][k][:15]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k][:15]])
            tiles, mobs, hud, facing = RS.facts(vis)
            trans = RS.transitions(tiles, mobs, hud, facing, frames)
            if all(t['scroll'] == 0 for t in trans[14 - M:14]):
                continue
            fresh = enc.recontext(enc.content(frames))
            stale = RS.rebuild(fresh, trans, M, idx_all)
            prepared.append({'ac': F.pad(r['future_actions'][:15], (0, 1)), 'trans': trans, 'fresh': fresh.half(), 'stale': stale.half(),
                             'mobs': mobs, 'hud': hud})
    print(json.dumps({'windows': len(prepared)}), flush=True)
    res = {'windows': len(prepared)}
    for seed in (1, 2):
        w = PI.PIWorld().cuda().eval()
        w.load_state_dict(torch.load(RS.T / f'pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        g = torch.Generator().manual_seed(5)
        acc = {v: {'toward': [], 'gen': [], 'hs': [], 'hy': []} for v in VARIANTS}
        for b0 in range(0, len(prepared), 8):
            batch = prepared[b0:b0 + 8]
            ac = torch.stack([p['ac'] for p in batch]).cuda()
            noise = DM.draw_noise((len(batch), 16), g)[0].cuda()
            for v in VARIANTS:
                fr = torch.stack([torch.where(mask_for(v, p['mobs']).cuda()[..., None], p['fresh'].float(), p['stale'].float()) for p in batch])
                with autocast_context(config):
                    belief, ha = w.backbone_full(fr, ac)
                    _, gen, glogit, _, _ = w.afterstate_parts(fr, ac, belief, ha)
                    logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
                logits = logits.masked_fill(~w.completion.valid, -torch.inf)
                prob = torch.sigmoid(IC.pair_prior_logit(logits[:, 14:15]))[:, 0].cpu()
                gen, glogit = gen[:, 14].float(), glogit[:, 14].float()
                for i, p in enumerate(batch):
                    tr14 = p['trans'][14]; prev = tr14['lay_prev']
                    for x in range(63):
                        if x == 31 or int(prev[x, 0]) != 1 or RS.dist(x) > 4 or x not in RS.INTERIOR:
                            continue
                        tw = 0.
                        for yy in (x - 9, x + 9, x - 1, x + 1):
                            pr = (min(x, yy), max(x, yy))
                            if pr in pair_index and RS.dist(yy) < RS.dist(x):
                                tw += float(prob[i, pair_index[pr]])
                        acc[v]['toward'].append(tw)
                    edits = [x for x in range(63) if bool(tr14['changed'][x]) and not bool(PI.entering_mask()[tr14['scroll']][x])
                             and x != 31 and not p['mobs'][15][x].any() and not prev[x].clamp_min(0).any()]
                    if edits:
                        acc[v]['gen'].append((F.layer_norm(gen[i, edits], (192,)) - p['fresh'][15].float()[edits]).norm(dim=-1).cpu())
                    acc[v]['hs'].append(float(torch.sigmoid(glogit[i, 63]))); acc[v]['hy'].append(float(p['hud'][15][0] - p['hud'][14][0]) < 0)
        out = {}
        for v, a in acc.items():
            hs, hy = torch.tensor(a['hs']), torch.tensor(a['hy'])
            out[v] = {'chase_toward': round(float(np.mean(a['toward'])), 4), 'zombies': len(a['toward']),
                      'gen_err_edits': round(float(torch.cat(a['gen']).mean()), 3) if a['gen'] else None,
                      'health_auc': None if RL.auc(hs, hy) is None else round(RL.auc(hs, hy), 4)}
        st, fr_ = out['stale']['chase_toward'], out['fresh']['chase_toward']
        for v in VARIANTS:
            out[v]['chase_recovery'] = round((out[v]['chase_toward'] - st) / (fr_ - st), 3) if fr_ != st else None
            if out[v]['gen_err_edits'] is not None and out['fresh']['gen_err_edits'] != out['stale']['gen_err_edits']:
                out[v]['gen_recovery'] = round((out['stale']['gen_err_edits'] - out[v]['gen_err_edits']) /
                                               (out['stale']['gen_err_edits'] - out['fresh']['gen_err_edits']), 3)
        res[f'seed{seed}'] = out
        print(json.dumps({f'seed{seed}': out}), flush=True)
        del w; torch.cuda.empty_cache()
    RS.OUT.mkdir(parents=True, exist_ok=True)
    (RS.OUT / 'chase_cue.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
