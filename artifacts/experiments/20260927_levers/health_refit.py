"""Which property of the output head blocks health? (2026-10-08, after health_chain_read: a fresh linear map from the frozen
world's h catches 75% of held hits / 63% of fresh hits at 7% false drops, while the trained corrt head emits 13% / 2%.)
Same subset, episodes, reader and cut as health_chain_read. One frozen world's saved h; every variant fits a token-63
generator on fit episodes (fit-internal episode validation picks the step), and is read on held episodes, token 63 replaced
in the world's own emitted HUD. Variants change ONE factor at a time from the trained head's situation:
  input       h63 only (the trained head's per-tile input) | h at 63, 31 and the player's 4 neighbours
  sampling    natural (the subset's class mix: 6,005 hits / 8,000 unchanged) | balanced 1:1
  map         dedicated linear | the world's own proj (trained weights, frozen) for reference | dedicated MLP (512)
  target      next token-63 (L1, as trained) | copy + residual: the map predicts t15[63] - t14[63] (L1)
NOTE the subset is hit-enriched vs training (6,005 hits : 8,000 unchanged vs ~1.4% hits in the TRAIN windows); `natural`
here therefore still over-represents hits. `train_rate` reweights unchanged x (6005/8000) * (98.6/1.4) = 52.9 to the TRAIN rate.
Reported: held drawn hits / fresh hits / unchanged false (1.5 cut), and mean L1 of token 63 on hits and on unchanged.
Usage: health_refit.py <world result .pt>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'


def main():
    import hashlib
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    x = torch.from_numpy(np.array(np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))[:, 14:16, 63]))
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    r = torch.load(sys.argv[1], weights_only=False)
    world, st = T.load_world(Path(r['checkpoint']), torch.device('cpu'))
    cls, fresh = sub['classes'], sub['fresh']
    fit = sub['fit']
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    held = ~fit
    hit, unch = cls == 1, cls == 2
    keep = hit | unch
    hud = r['hud'].float()
    read = lambda h: (P.hud(h.flatten(-2))[..., 0] * 9).float()
    t14_hud = torch.from_numpy(np.array(np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))[:, 14, 63:81])).float()
    cur = read(t14_hud)
    t14, t15 = x[:, 0].float(), x[:, 1].float()
    H = r['h'].float()
    feats = {'h63': H[:, 0], 'h6': H.flatten(1)}
    def evaluate(tok63):
        hh = hud.clone(); hh[:, 0] = tok63
        d = read(hh) < cur - 1.5
        m = lambda q: [int(d[q & held].sum()), int((q & held).sum())]
        l1 = (tok63 - t15).abs().mean(-1)
        return {'hits': m(hit), 'fresh': m(hit & fresh), 'unchanged_false': m(unch),
                'l1_hits': round(float(l1[hit & held].mean()), 4), 'l1_unchanged': round(float(l1[unch & held].mean()), 4)}
    res = {'world': st['name'], 'emitted': evaluate(hud[:, 0]), 'copy': evaluate(t14)}
    with torch.no_grad():                                              # the world's own (shared, trained) generator map on h63
        res['world_proj_on_h63'] = evaluate(F.layer_norm(world.proj(H[:, 0]), (192,)))
    def fit_map(X, target, sampling, mlp=False, residual=False, seed=0):
        mu, sd = X[fit].mean(0), X[fit].std(0).clamp_min(1e-4); Z = (X - mu) / sd
        torch.manual_seed(seed)
        net = (torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 192)) if mlp
               else torch.nn.Linear(Z.shape[1], 192))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(seed)
        tr_hit, tr_unch = torch.where(fit & ~inner & hit)[0], torch.where(fit & ~inner & unch)[0]
        tr_all = torch.where(fit & ~inner & keep)[0]
        w_unch = (6005 / 8000) * (98.6 / 1.4)                              # unchanged weight: hits are 1.4% as in TRAIN
        val = inner & keep
        y = (t15 - t14) if residual else t15
        best = (1e9, None)
        for step in range(4000):
            if sampling == 'balanced':
                b = torch.cat([tr_hit[torch.randint(len(tr_hit), (256,), generator=g)], tr_unch[torch.randint(len(tr_unch), (256,), generator=g)]])
                wt = torch.ones(len(b))
            else:
                b = tr_all[torch.randint(len(tr_all), (512,), generator=g)]
                wt = torch.where(unch[b], torch.tensor(w_unch if sampling == 'train_rate' else 1.0), torch.tensor(1.0))
            loss = ((net(Z[b]) - y[b]).abs().mean(-1) * wt).sum() / wt.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 250 == 0:
                with torch.no_grad():
                    vw = torch.where(unch[val], torch.tensor(w_unch if sampling == 'train_rate' else 1.0), torch.tensor(1.0))
                    v = float(((net(Z[val]) - y[val]).abs().mean(-1) * vw).sum() / vw.sum())
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in net.state_dict().items()})
        net.load_state_dict(best[1])
        with torch.no_grad():
            out = net(Z)
            out = t14 + out if residual else out
            return F.layer_norm(out, (192,))
    for fname in ('h63', 'h6'):
        for sampling in ('train_rate', 'natural', 'balanced'):
            res[f'linear|{fname}|{sampling}'] = evaluate(fit_map(feats[fname], t15, sampling))
        res[f'linear_residual|{fname}|train_rate'] = evaluate(fit_map(feats[fname], t15, 'train_rate', residual=True))
        res[f'mlp|{fname}|train_rate'] = evaluate(fit_map(feats[fname], t15, 'train_rate', mlp=True))
        print(json.dumps({k: v for k, v in res.items() if fname in k or k in ('emitted', 'copy', 'world_proj_on_h63')}), flush=True)
    (OUT / f"refit_{st['name']}.json").write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
