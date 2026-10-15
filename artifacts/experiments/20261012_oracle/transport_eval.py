"""Stage 3 evaluation (TRANSPORT_PREREG.md). Sampling: the 104 bright DEV roots, 64 samples/root, batch 4, generator seed
2026101402 + start (as direct_eval), 32 Gaussian draw pairs then 32 Gumbel draw pairs from that generator (identical in both
arms). Readings: (1) the frozen evaluator direct_coherence.analyze (historical comparability); (2) independent physical
metrics on the 14 strict sources (audit S5 selection) under BOTH frozen readers: valid / absent / multiple / duplicate,
all-source and cow valid movement and stay, distinct valid outcomes, dependence = valid - exact independence baseline;
scored joint occupancy ES; operator diagnostics (swaps per frame, swaps touching a decoded mob, reference-only validity);
(3) paired operator - free intervals: 2,000 root-cluster bootstrap draws, seed 20261015.
Generalized for TRANSPORT_COST_PREREG.md: --train/--eval directories, --arms (free, operator, operator_cost), --free (a
pred_free.pt from another directory, e.g. Stage 3's seed-1 free arm). Defaults reproduce the Stage 3 evaluation exactly.
Usage: transport_eval.py sample|analyze <update> [--train DIR] [--eval DIR] [--arms a,b] [--free PATH]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_s5_direct as A5  # noqa: E402

OUT = A5.BASE / 'transport'
KINDS = ('zombie', 'cow', 'skeleton')


@torch.no_grad()
def sample(update, train_dir=None, eval_dir=None, arms=None):
    import direct_model as DM
    import direct_train as BASE
    import direct_eval as E
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    data, _ = E.cohort('exploratory')
    d = eval_dir or OUT / f'eval_u{update:04d}'; d.mkdir(parents=True, exist_ok=True)
    for arm in arms or TM.ARMS:
        saved = torch.load((train_dir or OUT / 'training') / f'{arm}_u{update:04d}.pt', map_location='cpu', weights_only=False)
        if arm == 'pi':                                                    # TRANSPORT_PI_PREREG (Track B world)
            sys.path.insert(0, str(HERE.parent / '20261015_whiteboard')); import pi_model
            world = pi_model.PIWorld().cuda().eval()
        else:
            world = TM.TransportWorld(arm).cuda().eval()
        world.load_state_dict(saved['world'])
        preds, swaps, refs = [], [], []
        for start in range(0, len(data['roots']), 4):
            ctx = data['context_tokens'][start:start + 4].cuda().float(); ac = data['actions'][start:start + 4].cuda()
            frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
            g = torch.Generator().manual_seed(2026101402 + start)
            noises = torch.cat([DM.draw_noise((len(ctx), 16), g) for _ in range(32)])
            gumbels = torch.cat([TM.draw_gumbel((len(ctx), 16), g) for _ in range(32)])
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
                refs.append(F.layer_norm(world.reference(frames, ac, belief, ha)[:, 14].float(), (192,)).cpu())
                out, sw = [], []
                for k in range(64):
                    o, aux = world.complete(frames, ac, belief, ha, noises[k].cuda(), gumbels[k].cuda())
                    out.append(o[:, 14].float().cpu())
                    if 'partner' in aux:
                        sw.append(aux['partner'][:, 14].cpu())
            preds.append(torch.stack(out, 1))
            if sw:
                swaps.append(torch.stack(sw, 1))
        row = {'arm': arm, 'update': update, 'predictions': torch.cat(preds), 'reference': torch.cat(refs),
               'partner': torch.cat(swaps) if swaps else None}
        torch.save(row, d / f'pred_{arm}.pt')
        print(json.dumps({'stage': 'sampled', 'arm': arm, 'update': update}), flush=True)
        del world; torch.cuda.empty_cache()


def bootstrap(per_source, seed=20261015, draws=2000):
    x = np.asarray(per_source, dtype=float)
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(draws)]
    return [round(float(np.quantile(means, .025)), 4), round(float(np.quantile(means, .975)), 4)]


def analyze(update, eval_dir=None, arms=None, free_path=None):
    import direct_coherence as C
    import direct_eval as E
    import transport_model as TM
    data, _ = E.cohort('exploratory')
    d = eval_dir or OUT / f'eval_u{update:04d}'
    arms = list(arms or TM.ARMS)
    if free_path is not None and 'free' not in arms:
        arms = ['free'] + arms
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    units = json.loads((A5.OUT / 'audit_recompute.json').read_text())['sources_list']
    scroll = A5.load_scroll()
    root_tok = data['root_tokens'].float(); real_tok = data['fan_tokens'].float(); fan_lay = data['fan_layout'].bool()
    real_dec = C.decode(real_tok, bank, head)['mob']
    res = {'update': update, 'arms': {}, 'per_source': {}}
    for arm in arms:
        row = torch.load(free_path if (arm == 'free' and free_path is not None) else d / f'pred_{arm}.pt', map_location='cpu', weights_only=False)
        pred = row['predictions']
        frozen = C.analyze(pred, root_tok, real_tok, data['root_layout'], data['fan_layout'], bank, head, shifts=data['shifts'],
                           actions=data['actions'][:, -1], root_tiles=data['root_tiles'], real_tiles=data['fan_tiles'])
        (d / f'frozen_analysis_{arm}.json').write_text(json.dumps({k: v for k, v in frozen.items() if k != 'raw_per_root'}, default=str))
        dec = C.decode(pred, bank, head)['mob']
        ref_dec = C.decode(row['reference'][:, None], bank, head)['mob']
        reg = torch.stack([(scroll.estimate(root_tok[i:i + 1, None], pred[i:i + 1])[0] == 0) for i in range(len(pred))])
        out = {}
        for reader in ('real', 'bank'):
            per = []
            for root, source, kindname in units:
                kind = KINDS.index(kindname)
                r, c = divmod(source, 9)
                pts = [(r + a) * 9 + c + e for a, e in A5.CROSS]
                f = dec[reader][root][:, pts, kind]; ev = A5.event(f); ok = (ev < 5) & reg[root]
                p = f[reg[root]].double().mean(0)
                ind = sum(float(p[j] * torch.prod(torch.cat([1 - p[:j], 1 - p[j + 1:]]))) for j in range(5)) if bool(reg[root].any()) else 0.
                ref_ev = int(A5.event(ref_dec[reader][root][:, pts, kind])[0])
                per.append({'root': root, 'kind': kindname, 'valid': float(ok.float().mean()), 'absent': float((ev == 5).float().mean()),
                            'multiple': float((ev == 6).float().mean()), 'duplicate': float((f[:, 0] & f[:, 1:].any(-1)).float().mean()),
                            'valid_move': float((ok & (ev > 0)).float().mean()), 'valid_stay': float((ok & (ev == 0)).float().mean()),
                            'distinct_valid': len(torch.unique(ev[ok])), 'dependence': float(ok.float().mean()) - ind,
                            'reference_event': ref_ev, 'real_move_rate': float((A5.event(fan_lay[root][:, pts, kind]) > 0).float().mean())})
            summary = {k: round(float(np.mean([x[k] for x in per])), 4) for k in ('valid', 'absent', 'multiple', 'duplicate', 'valid_move', 'valid_stay', 'distinct_valid', 'dependence', 'real_move_rate')}
            cows = [x for x in per if x['kind'] == 'cow']
            summary['cow_valid_move'] = round(float(np.mean([x['valid_move'] for x in cows])), 4)
            summary['cow_real_move_rate'] = round(float(np.mean([x['real_move_rate'] for x in cows])), 4)
            summary['reference_valid_stay_sources'] = sum(x['reference_event'] == 0 for x in per)
            interior = list(A5.INTERIOR)
            es = A5.energy(dec[reader][:, :, interior, :3].flatten(2).float(), real_dec[reader][:, :, interior, :3].flatten(2).float())
            summary['scored_joint_ES'] = round(float(es.mean()), 6)
            out[reader] = summary
            res['per_source'].setdefault(arm, {})[reader] = per
        static = frozen['static_content']['scored_sampled_corruption_per_root']
        out['scored_static_corruption'] = round(float(np.mean([x for x in static if x is not None])), 6)
        if row['partner'] is not None:
            moved = row['partner'] != torch.arange(81)
            mob = dec['real'][..., :3].any(-1)                       # decoded mob after the operator (63 map cells)
            out['operator'] = {'swaps_per_frame': round(float(moved.sum(-1).float().mean() / 2), 4),
                               'share_of_swaps_touching_a_root_mob': round(float((moved[..., :63] & C.decode(root_tok, bank, head)['mob']['real'][:, None, :, :3].any(-1)).sum() / max(int(moved.sum()), 1)), 4)}
        res['arms'][arm] = out
        print(json.dumps({arm: out}), flush=True)
    pairs = [(x, y) for x, y in (('operator', 'free'), ('operator_cost', 'free'), ('operator_cost', 'operator'), ('ic', 'free'), ('ic', 'operator'), ('icsf', 'free'), ('icsf', 'ic'), ('pi', 'free')) if x in arms and y in arms]
    for x_arm, y_arm in pairs:
        res[f'{x_arm}_minus_{y_arm}'] = compare(res, x_arm, y_arm)
    (d / 'analysis.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if '_minus_' in k}, indent=1), flush=True)


def compare(res, x_arm, y_arm):
    out = {}
    for reader in ('real', 'bank'):
        a, b = res['per_source'][x_arm][reader], res['per_source'][y_arm][reader]
        out[reader] = {k: {'mean': round(float(np.mean([x[k] - y[k] for x, y in zip(a, b)])), 4),
                           'ci95': bootstrap([x[k] - y[k] for x, y in zip(a, b)])}
                       for k in ('valid', 'valid_move', 'duplicate', 'absent', 'multiple', 'dependence', 'distinct_valid')}
        ca = [x for x in a if x['kind'] == 'cow']; cb = [x for x in b if x['kind'] == 'cow']
        out[reader]['cow_valid_move'] = {'mean': round(float(np.mean([x['valid_move'] - y['valid_move'] for x, y in zip(ca, cb)])), 4),
                                         'ci95': bootstrap([x['valid_move'] - y['valid_move'] for x, y in zip(ca, cb)])}
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('sample', 'analyze')); ap.add_argument('update', type=int)
    ap.add_argument('--train', type=Path); ap.add_argument('--eval', type=Path); ap.add_argument('--arms'); ap.add_argument('--free', type=Path)
    a = ap.parse_args()
    arms = a.arms.split(',') if a.arms else None
    if a.mode == 'sample':
        sample(a.update, a.train, a.eval, arms)
    else:
        analyze(a.update, a.eval, arms, a.free)
