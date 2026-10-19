"""Sleep-stratified, level-separated evaluation (SPATIAL_ENCODER_PREREG.md Amendment 6). Evaluation only; CPU; exact simulator
state labels frames (is_sleeping = visible[1520], light_level = visible[1521], observe.py), never training.

Readers: per stage (backbone_ln, deployed) zombie / cow ridge exactly as spatial_eval_v2.a1 (fit frames of split_v2.json, seeded
400k-cell subsample, lambda 1e-2, fit-only rate-matched threshold, strict >); the deployed readers must equal probes2_<name>.pt.
Population A: Tier A test frames (spatial_eval_v2.gather). Population B: the Tier B test roots' real next frames (4 keys x 17
actions, alive). Level 1 / 2: scores of both stages on both populations. Level 3: one predictor step from each test root's 2-frame
context for the 17 actions (as tierb_eval_v2.py: PatchState float32, expected + one router sample, probes2 readers) and the same
reader on the real next frames, at certain-empty cells (no zombie in any alive key).
Usage: sleep_strat_eval.py <name>=<encoder spec>,<predictor checkpoint>,<probes2 file> ...
   -> artifacts/eda/spatial_encoder_v1/sleep2_units_<name>.pt + sleep2_<name>.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_encoder as SE                                                   # noqa: E402
import spatial_eval as EV                                                      # noqa: E402
import spatial_eval_v2 as EV2                                                  # noqa: E402
import tierb_eval as TB                                                        # noqa: E402

import torch                                                                   # noqa: E402

import objective_audit as OA                                                   # noqa: E402

GROUPS = ('all', 'awake', 'sleeping', 'day', 'night')


def summary(U, th):
    """U: {'z','c' scores [cells], 'zl','cl' labels, 'sleep','night' per frame} -> metrics by frame group"""
    fr = {'all': torch.ones_like(U['sleep']), 'awake': ~U['sleep'], 'sleeping': U['sleep'], 'day': ~U['night'], 'night': U['night']}
    out = {}
    for nm, s, l in (('zombie', U['z'].float(), U['zl']), ('cow', U['c'].float(), U['cl'])):
        fp = (s > th[nm]) & ~l
        row = {'fp_share_in_sleeping': round(float(fp.view(-1, 63)[U['sleep']].sum() / fp.sum().clamp_min(1)), 4)}
        for g in GROUPS:
            m = fr[g].repeat_interleave(63)
            ap, auc = EV2.average_precision(s[m], l[m]), EV2.auc_tie(s[m], l[m])
            row[g] = {'frames': int(fr[g].sum()), 'positives': int(l[m].sum()), 'negatives': int((~l[m]).sum()),
                      'episodes': len(set(U['episode'][fr[g]].tolist())),
                      'ap': None if ap is None else round(ap, 4), 'auc': None if auc is None else round(auc, 4),
                      'fp_rate': round(float(fp[m].sum() / (~l[m]).sum().clamp_min(1)), 5),
                      'recall_at_threshold': round(float(((s > th[nm]) & l)[m].sum() / l[m].sum().clamp_min(1)), 4)}
        out[nm] = row
    return out


@torch.no_grad()
def main():
    rows, _, split = EV2.episode_split()
    assert split == json.loads(EV2.SPLIT.read_text()), 'split differs from split_v2.json'
    test = split['test'][:6] if os.environ.get('SLEEP_SMOKE') else split['test']     # code-path smoke test only
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        st = EV2.Stages(enc_spec)
        ff, _, Lf, _, _, _ = EV2.gather(rows, split['fit'], st)
        ft, _, Lt, _, seqt, ridt = EV2.gather(rows, test, st)
        Lf = torch.cat(Lf)
        # population B: real next frames of the Tier B test roots (alive)
        fb, Lb, sb, nb, eb, ab = {}, [], [], [], [], []
        for ri in test:
            r = rows[ri]
            ok = ~r['onestep_dead'].bool().reshape(-1)
            for k, v in st(r['onestep_frames'].flatten(0, 1)[ok]).items():
                fb.setdefault(k, []).append(v)
            vis = r['onestep_visible'].reshape(68, -1)[ok]
            Lb.append(OA.layout(vis)[1]); sb.append(vis[:, 1520] > 0); nb.append(vis[:, 1521] < .5)
            eb.append(torch.full((int(ok.sum()),), int(r['seed']))); ab.append(torch.arange(17).repeat(4)[ok])
        popA = {'zl': (torch.cat(Lt).reshape(-1) & 1) > 0, 'cl': (torch.cat(Lt).reshape(-1) & 2) > 0,
                'sleep': torch.cat([s[2] > 0 for s in seqt]), 'night': torch.cat([s[1] < .5 for s in seqt]),
                'episode': torch.cat([torch.full((s[0],), int(rows[r]['seed'])) for s, r in zip(seqt, ridt)])}
        Lb = torch.cat(Lb).reshape(-1)
        popB = {'zl': (Lb & 1) > 0, 'cl': (Lb & 2) > 0, 'sleep': torch.cat(sb), 'night': torch.cat(nb),
                'episode': torch.cat(eb), 'action': torch.cat(ab)}
        saved = torch.load(probe_file, weights_only=False)
        U = {'A': {}, 'B': {}, 'thresholds': {}, 'labels': {'A': popA, 'B': popB}}
        res = {'encoder': enc_spec, 'predictor': pred_ck, 'probes': probe_file}
        for stage in ff:
            Sf = torch.cat(ff[stage]); D = Sf.shape[-1]
            sub = torch.randperm(len(Sf) * 63, generator=torch.Generator().manual_seed(1))[:400000]     # = spatial_eval_v2.a1
            Xf = Sf[:, :63].reshape(-1, D)[sub].float()
            th, W = {}, {}
            for bit, nm in ((1, 'zombie'), (2, 'cow')):
                lf = ((Lf.reshape(-1)[sub] & bit) > 0).float()
                W[nm] = EV.ridge(Xf, lf[:, None])
                srt = EV.apply(W[nm], Xf)[:, 0].sort(descending=True).values
                th[nm] = float(srt[min(max(int(round(float(lf.sum()))) - 1, 0), len(srt) - 1)])
                if stage == 'deployed':                                              # must reproduce the Tier B readers
                    assert torch.allclose(W[nm], saved[nm].double(), atol=1e-8) and abs(th[nm] - saved[f'{nm}_threshold']) < 1e-6, (name, nm)
            del Xf
            U['thresholds'][stage] = th
            for pop, feats in (('A', ft[stage]), ('B', fb[stage])):
                X = torch.cat(feats)[:, :63].reshape(-1, D)
                sc = torch.cat([EV.apply(torch.cat([W['zombie'], W['cow']], 1), X[i:i + 200000].float()) for i in range(0, len(X), 200000)])
                U[pop][stage] = {'z': sc[:, 0].clone(), 'c': sc[:, 1].clone()}
                res[f'{pop}_{stage}'] = summary({**U['labels'][pop], **U[pop][stage]}, th)
            print(json.dumps({name: {stage: {p: res[f'{p}_{stage}']['zombie']['sleeping'] for p in 'AB'}}}), flush=True)
        del ff, ft, fb
        # level 3: predictor, deployed stage, Tier B numerics
        enc = EV.PatchState(enc_spec)
        sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
        pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
        g = torch.Generator(device=EV.DEV).manual_seed(20261071)
        zt = saved['zombie_threshold']
        L3 = []
        for ri in test:
            r = rows[ri]
            ctx = enc(r['context'][-2:])
            exp, samp, _, _ = TB.step(pred, ctx[0:1].expand(17, -1, -1), ctx[1:2].expand(17, -1, -1), torch.arange(17), g)
            _, z_exp, _ = TB.read(saved, exp.cpu()); _, z_s, _ = TB.read(saved, samp.cpu())
            _, z_rn, _ = TB.read(saved, enc(r['onestep_frames'].flatten(0, 1)).view(4, 17, 81, -1).cpu())
            dead = r['onestep_dead'].bool()
            _, lay, _ = OA.layout(r['onestep_visible'])
            for a in range(17):
                ok = torch.nonzero(~dead[:, a])[:, 0]
                if len(ok) < 1:
                    continue
                empty = ~(((lay[ok, a] & 1) > 0).any(0))
                if not bool(empty.any()):
                    continue
                L3.append((int(r['seed']), ri, a, float((r['onestep_visible'][ok, a, 1520] > 0).float().mean()), int(empty.sum()),
                           int((z_exp[a][empty] > zt).sum()), int((z_s[a][empty] > zt).sum()),
                           float((z_rn[ok, a][:, empty] > zt).float().mean(0).sum())))
        U['L3'] = L3
        t = torch.tensor([x[3:] for x in L3], dtype=torch.float64)
        for nm_, m in (('sleeping_next', t[:, 0] >= .5), ('awake_next', t[:, 0] < .5)):
            n = t[m, 1].sum().clamp_min(1)
            res[f'L3_{nm_}'] = {'rows': int(m.sum()), 'empty_cells': int(t[m, 1].sum()), 'fp_rate_pred_expected': round(float(t[m, 2].sum() / n), 5),
                                'fp_rate_pred_sample': round(float(t[m, 3].sum() / n), 5), 'fp_rate_real_next': round(float(t[m, 4].sum() / n), 5)}
        torch.save(U, EV.OUT / f'sleep2_units_{name}.pt')
        (EV.OUT / f'sleep2_{name}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({name: {k: v for k, v in res.items() if k.startswith('L3')}}), flush=True)
        del enc, pred, st, U


if __name__ == '__main__':
    main()
