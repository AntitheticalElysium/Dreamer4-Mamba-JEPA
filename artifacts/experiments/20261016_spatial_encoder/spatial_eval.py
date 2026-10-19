"""Tier A readings for any encoder (SPATIAL_ENCODER_PREREG.md): content, nuisance, collapse of the PATCH STATE. Evaluation only;
exact simulator state labels the frames (diagnosis_futures_v1), never training.

Patch state per encoder:
  raw@<step>        the canonical Raw encoder at that joint step: LN(last-layer patch tokens), exactly what D / IC / PI use
  spatial:<ckpt>    a spatial_encoder.py checkpoint: patch_proj(last-layer patch tokens), eval-mode BatchNorm
Frames: diagnosis_futures_v1 roots (seeded 400; the first 280 by a seeded permutation fit the probes, the other 120 test; split
by root = by episode seed), each root's last context frame + its 5 futures x 16 frames (alive).
Probes saved for Tier B, with mob thresholds rate-matched on the FIT split (a ridge score on a 1.4%-positive label never reaches
0.5; the threshold makes the fit-split read rate equal the true rate).
A1 content: per-patch ridge-classifier tile accuracy (17 classes, map cells); zombie / cow AUC (ridge score, map cells);
   HUD R^2 of health, food, drink, energy (shown values) from the 18 HUD patch states (ridge).
A2 nuisance: persisting-content drift (consecutive frames, exact camera shift; tile + mob layer unchanged) divided by the mean
   distance between map cells of different tile class in the same frames, by light regime (day steady / day changing /
   night); k-means (K 1024, fitted on fit-split states) code flip rate on persisting cells; HUD salience = code change rate
   with a displayed value change minus without.
A3 collapse: effective rank (exp entropy of normalized singular values) of test map-patch states; mean per-dimension std.
Usage: spatial_eval.py <name>=<spec> ...  -> artifacts/eda/spatial_encoder_v1/tierA_<name>.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261015_whiteboard'))
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))

import torch
from torch.nn import functional as F

import objective_audit as OA

OUT = ROOT / 'artifacts/eda/spatial_encoder_v1'
DEV = os.environ.get('EVAL_DEVICE', 'cuda')                                   # 'cpu' for previews while the GPU trains
RAW_DIR = ROOT / 'artifacts/lewm_m4_canonical/raw/joint'


class PatchState:
    """frames uint8 [n,63,63,3] -> patch states [n,81,D] (float32, cuda)"""
    def __init__(self, spec):
        from d4mj.config import load_recipe
        from d4mj.lewm import LeWMEncoder, LeWMProjector
        # the bundles' execution identity was recorded under PyTorch's default numerics (recontext_stale.Encoder does the same)
        saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        config = load_recipe(ROOT / 'd4mj/recipes/lewm_mamba_raw.json')
        self.enc = LeWMEncoder(config).to(DEV).eval()
        self.proj = None
        if spec.startswith('raw@'):
            from d4mj.checkpoint import read_lewm_bundle
            payload = read_lewm_bundle(RAW_DIR / f'step-{int(spec[4:]):06d}.pt')
            self.enc.load_state_dict(payload['modules']['encoder'])
        elif spec.startswith('spatial-bb:'):                                     # a Spatial encoder's BACKBONE-LN stage (no projector)
            ck = torch.load(spec[11:], map_location=DEV, weights_only=False)
            self.enc.load_state_dict(ck['encoder'])
        elif spec.startswith('spatial:'):
            ck = torch.load(spec[8:], map_location=DEV, weights_only=False)
            self.enc.load_state_dict(ck['encoder'])
            self.proj = LeWMProjector(192, config.encoder).to(DEV).eval()
            self.proj.load_state_dict(ck['patch_proj'])
        else:
            raise ValueError(spec)
        torch.set_float32_matmul_precision(saved[0]); torch.backends.cuda.matmul.allow_tf32 = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]; torch.backends.cudnn.benchmark = saved[3]; torch.backends.cudnn.deterministic = saved[4]
        torch.use_deterministic_algorithms(saved[5])

    @torch.no_grad()
    def __call__(self, frames):
        out = []
        for i in range(0, len(frames), 256):
            x = frames[i:i + 256].to(DEV)[:, None]
            _, _, tok, _, _ = self.enc._hidden(x)                                         # [n,81,192] post backbone LN
            tok = tok.float()
            out.append(F.layer_norm(tok, (192,)) if self.proj is None else self.proj(tok.reshape(-1, 192)).view(tok.shape).float())
        return torch.cat(out)


def ridge(X, Y, lam=1e-2):
    """closed-form ridge on CPU float64; lam relative to the mean eigenvalue of X'X"""
    X = torch.cat([X.cpu().double(), torch.ones(len(X), 1, dtype=torch.float64)], 1); Y = Y.cpu().double()
    A = X.T @ X
    A += lam * torch.trace(A) / len(A) * torch.eye(len(A), dtype=torch.float64)
    return torch.linalg.solve(A, X.T @ Y)


def apply(W, X):
    return (torch.cat([X.cpu().double(), torch.ones(len(X), 1, dtype=torch.float64)], 1) @ W).float()


def auc(score, label):
    s, l = score.cpu(), label.cpu().bool()
    if l.all() or (~l).all():
        return None
    order = s.argsort(); r = torch.empty(len(s)); r[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def assign(X, C):
    return torch.cat([torch.cdist(X[i:i + 20000], C).argmin(-1) for i in range(0, len(X), 20000)])


def kmeans(X, K=1024, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:K]].clone()
    for _ in range(iters):
        a = assign(X, C)
        for k in range(0, K, 256):
            m = (a[:, None] == torch.arange(k, min(k + 256, K), device=X.device)[None]).float()
            n = m.sum(0)
            C[k:k + 256] = torch.where(n[:, None] > 0, (m.T @ X) / n.clamp_min(1)[:, None], C[k:k + 256])
    return C


@torch.no_grad()
def main():
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261070))[:400].tolist()
    split = {'fit': pick[:280], 'test': pick[280:]}
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        enc = PatchState(spec)
        data = {}
        for part, idx in split.items():
            S, TL, MOB, HUD, seq = [], [], [], [], []
            for ri in idx:
                r = rows[ri]
                for f in range(5):
                    alive = int((~r['future_dead'][f].bool()).long().cumprod(0).sum())
                    frames = torch.cat([r['context'][-1:], r['future_frames'][f][:alive]])
                    vis = torch.cat([r['root_visible'][None], r['future_visible'][f][:alive]])
                    s = enc(frames).half().cpu()
                    t, l, sh = OA.layout(vis)
                    S.append(s); TL.append(t); MOB.append(vis[:, 1071:1512].reshape(-1, 63, 7)[..., :2] > 0)
                    HUD.append(sh[:, :4] / 9); seq.append((len(s), vis[:, 1521].float(), vis[:, 1520].float()))
            data[part] = dict(S=S, T=TL, M=MOB, H=HUD, seq=seq)
        fit, test = data['fit'], data['test']
        Sf = torch.cat(fit['S']); St = torch.cat(test['S'])                               # CPU fp16
        Tf, Tt = torch.cat(fit['T']), torch.cat(test['T'])
        Mf, Mt = torch.cat(fit['M']), torch.cat(test['M'])
        Hf, Ht = torch.cat(fit['H']), torch.cat(test['H'])
        res = {'encoder': spec, 'frames_fit': len(Sf), 'frames_test': len(St)}
        # A1
        sub = torch.randperm(len(Sf) * 63, generator=torch.Generator().manual_seed(1))[:400000]
        Xf = Sf[:, :63].reshape(-1, Sf.shape[-1])[sub].float(); yt = Tf.reshape(-1)[sub]
        W = ridge(Xf, F.one_hot(yt, 17).float())
        probes = {'tile': W}
        Xt = St[:, :63].reshape(-1, St.shape[-1])
        pred = torch.cat([apply(W, Xt[i:i + 200000]).argmax(-1) for i in range(0, len(Xt), 200000)])
        res['tile_acc'] = round(float((pred == Tt.reshape(-1)).float().mean()), 4)
        for ch, nm in ((0, 'zombie'), (1, 'cow')):
            Wm = ridge(Xf, Mf[..., ch].reshape(-1)[sub].float()[:, None])
            probes[nm] = Wm
            yf = Mf[..., ch].reshape(-1)[sub].float()                                       # rate-matched threshold on the FIT split
            sf = apply(Wm, Xf)[:, 0]
            probes[f'{nm}_threshold'] = float(sf.sort(descending=True).values[max(int(yf.sum()) - 1, 0)])
            sc = torch.cat([apply(Wm, Xt[i:i + 200000])[:, 0] for i in range(0, len(Xt), 200000)])
            res[f'{nm}_auc'] = round(auc(sc, Mt[..., ch].reshape(-1)), 4)
        Wh = ridge(Sf[:, 63:].reshape(len(Sf), -1).float(), Hf)
        ph = apply(Wh, St[:, 63:].reshape(len(St), -1)); ht = Ht.float()
        r2 = 1 - (ph - ht).pow(2).mean(0) / ht.var(0).clamp_min(1e-9)
        res['hud_r2'] = {k: round(float(v), 4) for k, v in zip(('health', 'food', 'drink', 'energy'), r2)}
        probes['hud'] = Wh                                                                 # saved for Tier B (same readers)
        # A3
        e = Xt[torch.randperm(len(Xt), generator=torch.Generator().manual_seed(2))[:20000]].float().to(DEV)
        sv = torch.linalg.svdvals(e - e.mean(0)); p = sv / sv.sum()
        res['eff_rank'] = round(float(torch.exp(-(p * torch.log(p + 1e-12)).sum())), 2)
        res['mean_dim_std'] = round(float(e.std(0).mean()), 4)
        # A2
        C = kmeans(torch.cat([Sf[:, :63].reshape(-1, Sf.shape[-1])[sub[:200000]], Sf[:, 63:].reshape(-1, Sf.shape[-1])[:50000]]).float().to(DEV))
        drift = {k: [0., 0] for k in ('day_steady', 'day_changing', 'night')}
        flips = {k: [0, 0] for k in ('persist', 'hud_value_changed', 'hud_unchanged')}
        between = []
        pos = 0
        for (n, light, sleep), s, t, l, sh in zip(test['seq'], test['S'], test['T'], [None] * len(test['S']), test['H']):
            s = s.float().to(DEV)
            codes = assign(s.reshape(-1, s.shape[-1]), C).view(n, 81).cpu()
            tiles = t
            # mob layer for persistence labels
            lay = None
            for j in range(n - 1):
                dr, dc = OA.shift_of(tiles[j], tiles[j + 1]); src = OA.sources(dr, dc)
                ok = src >= 0
                same = torch.zeros(63, dtype=torch.bool)
                same[ok] = tiles[j + 1][ok] == tiles[j][src[ok]]
                same[31] = False
                if bool(sleep[j] > 0) or bool(sleep[j + 1] > 0):
                    reg = None
                elif min(float(light[j]), float(light[j + 1])) < 0.5:
                    reg = 'night'
                elif min(float(light[j]), float(light[j + 1])) >= 0.99 and float(light[j]) == float(light[j + 1]):
                    reg = 'day_steady'
                else:
                    reg = 'day_changing'
                idx = torch.nonzero(same)[:, 0]
                if len(idx):
                    d = (s[j + 1, idx.to(DEV)] - s[j, src[idx].to(DEV)]).norm(dim=-1)
                    if reg is not None:
                        drift[reg][0] += float(d.sum()); drift[reg][1] += len(idx)
                    fl = codes[j + 1, idx] != codes[j, src[idx]]
                    flips['persist'][0] += int(fl.sum()); flips['persist'][1] += len(idx)
                hk = 'hud_value_changed' if bool((sh[j + 1] != sh[j]).any()) else 'hud_unchanged'
                fl = codes[j + 1, 63:] != codes[j, 63:]
                flips[hk][0] += int(fl.sum()); flips[hk][1] += 18
                if j % 4 == 0:
                    tj = tiles[j]
                    a_, b_ = torch.randint(63, (64,)), torch.randint(63, (64,))
                    diff = tj[a_] != tj[b_]
                    if bool(diff.any()):
                        between.append((s[j, a_[diff].to(DEV)] - s[j, b_[diff].to(DEV)]).norm(dim=-1).cpu())
        bd = float(torch.cat(between).mean())
        res['between_content_distance'] = round(bd, 4)
        res['drift_ratio'] = {k: round(v[0] / max(v[1], 1) / bd, 4) for k, v in drift.items()}
        res['code_flip'] = {k: round(v[0] / max(v[1], 1), 4) for k, v in flips.items()}
        res['hud_salience'] = round(res['code_flip']['hud_value_changed'] - res['code_flip']['hud_unchanged'], 4)
        print(json.dumps({name: res}, indent=1), flush=True)
        (OUT / f'tierA_{name}.json').write_text(json.dumps(res, indent=1))
        torch.save({'spec': spec, **probes}, OUT / f'probes_{name}.pt')
        del enc; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
