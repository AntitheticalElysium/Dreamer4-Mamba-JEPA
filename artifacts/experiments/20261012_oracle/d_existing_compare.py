"""CPU-only paired L1/D audit on immutable saved oracle predictions.

This is descriptive across different parent lineages, not a randomized loss
comparison. Reader coefficients and thresholds are frozen from TRAIN. The
primary includes every root; common alignment is a selected sensitivity.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import snap_eval as S
import scoreboard as SB

OUT = S.OUT
MODELS = ['control_l1', 'oracle_l1', 'control_deltairis', 'oracle_deltairis']
GROUPS = S.GROUPS


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(pred, truth, previous_truth, readers, facts):
    counts, extras = {}, {}
    scores_out = []
    for start in range(0, len(pred), 8):
        x = pred[start:start + 8].float()
        counts_batch = S.counts(x, truth[start:start + 8], previous_truth[start:start + 8], readers)
        for key, vals in counts_batch.items():
            counts.setdefault(key, []).extend(vals)
        scores = readers.mob(x[:, :, S.OE.MAPC].reshape(-1, 192)).view(len(x), 16, 62, 4)
        scores_out.append(scores)
        tiles = readers.tile(x[:, :, S.OE.MAPC].reshape(-1, 192)).view(len(x), 16, 62, 17).argmax(-1)
        f = {k: v[start:start + 8] for k, v in facts.items()}
        correct = tiles == f['tile']
        idx = {c: j for j, c in enumerate(S.OE.MAPC)}
        for name, region in {'all': S.OE.MAPC, 'near': SB.NEAR4, 'interior': SB.INTER, 'edge': SB.EDGE}.items():
            z = correct[:, :, [idx[c] for c in region]]
            extras.setdefault('tile_' + name, []).extend(z.sum(-1).tolist())
        facing = readers.facing(x[:, :, 31].reshape(-1, 192)).argmax(-1).view(len(x), 16)
        extras.setdefault('facing', []).extend((facing == f['facing']).tolist())
        hud = readers.hud(x[:, :, 63:].reshape(-1, 18 * 192)).view(len(x), 16, -1)
        for j, name in enumerate(('health', 'food', 'drink', 'energy')):
            error = hud[:, :, j] - f['hud'][:, :, j]
            for metric, val in [('sse', error.square()), ('abs', error.abs()), ('bias', error)]:
                extras.setdefault(name + '_' + metric, []).extend(val.tolist())
    return {k: np.asarray(v) for k, v in counts.items()}, {k: np.asarray(v) for k, v in extras.items()}, torch.cat(scores_out)


def main():
    torch.set_num_threads(8)
    panel = torch.load(OUT / 'panel.pt', weights_only=False, map_location='cpu')
    visible = torch.load(OUT / 'snap_visible.pt', weights_only=False, map_location='cpu')
    assert panel['roots'] == visible['roots']
    state = torch.load(OUT / 'snap_bank_readers.pt', weights_only=False, map_location='cpu')
    readers = S.Readers(state)
    facts = SB.AllProbes.facts(visible['visible'][:, 16:32])
    truth = torch.stack([panel['layouts'][..., S.OE.LAYOUT_CH[t]] > 0 for t in S.OE.TYPES], -1)
    truth[:, :, 31] = False
    truth, previous_truth = truth[:, 16:32], truth[:, 15:31]
    ids = np.asarray([r['episode_id'] for r in panel['roots']])
    episodes, inverse = np.unique(ids, return_inverse=True)
    n = len(episodes)
    weights = np.random.default_rng(20261012).multinomial(n, np.ones(n) / n, size=2000)
    modes, raw_scores, align = {}, {}, {}
    inputs = [OUT / p for p in ('panel.pt', 'snap_visible.pt', 'snap_bank_readers.pt')]
    for name in ['TRUE'] + MODELS:
        if name == 'TRUE':
            st = {'tf': panel['tokens'][:, 16:32], 'gen': panel['tokens'][:, 16:32],
                  'aligned': torch.ones(200, 16, dtype=torch.bool)}
        else:
            path = OUT / f'pred_{name}.pt'; inputs.append(path)
            st = torch.load(path, weights_only=False, map_location='cpu')
        align[name] = st['aligned'].numpy()
        for mode in ('tf', 'gen'):
            if name == 'TRUE' and mode == 'gen':
                modes[name + '/' + mode] = modes[name + '/tf']
                raw_scores[name + '/' + mode] = raw_scores[name + '/tf']
                continue
            counts, extras, scores = decode(st[mode], truth, previous_truth, readers, facts)
            modes[name + '/' + mode] = {'counts': counts, 'extras': extras}
            raw_scores[name + '/' + mode] = scores
            print(json.dumps({'stage': 'decoded', 'model': name, 'mode': mode}), flush=True)
        del st
    common = np.logical_and.reduce([align[name] for name in MODELS])
    metrics = {'recall': ('true_positive', 'true'), 'precision': ('true_positive', 'predicted'),
               'fp_per_true': ('false_positive', 'true'), 'census': ('predicted', 'true'),
               'arrive_recall': ('arrive_true_positive', 'arrived'), 'residue': ('residue', 'left')}
    contrasts = [(f'{name}/gen', f'{name}/tf') for name in MODELS]
    contrasts += [(f'{arm}_deltairis/{mode}', f'{arm}_l1/{mode}') for arm in ('oracle', 'control') for mode in ('tf', 'gen')]
    contrasts += [(f'oracle_{loss}/{mode}', f'control_{loss}/{mode}') for loss in ('l1', 'deltairis') for mode in ('tf', 'gen')]
    result = {'scope': 'paired post-hoc CPU analysis on reused DEV roots; L1 and D have different parent training lineages',
              'roots': len(ids), 'episodes': n, 'bootstrap': {'draws': 2000, 'seed': 20261012, 'unit': 'episode'},
              'thresholds': state['thresholds'], 'hashes': {p.name: sha(p) for p in inputs},
              'alignment': {k: v.mean(0).tolist() for k, v in align.items()}, 'populations': {}}
    for population, mask in [('all', np.ones((len(ids), 16), bool)), ('common_aligned', common)]:
        result['populations'][population] = {}
        for group, ks in GROUPS.items():
            select = mask[:, ks]
            out = {'selected_root_steps': int(select.sum()), 'models': {}, 'contrasts': {}}
            point, draws = {}, {}
            for key, mode in modes.items():
                cnt = {k: (v[:, ks] * select[:, :, None]).sum(1) for k, v in mode['counts'].items()}
                agg = {}
                for k, v in cnt.items():
                    a = np.zeros((n, 3), np.int64); np.add.at(a, inverse, v); agg[k] = a
                mob_out = {}
                for j, mob in enumerate(S.OE.TYPES):
                    vals = {'counts': {k: int(v[:, j].sum()) for k, v in cnt.items()}}
                    for met, (num, den) in metrics.items():
                        bn, bd = weights @ agg[num][:, j], weights @ agg[den][:, j]
                        z = np.full(2000, np.nan); np.divide(bn, bd, out=z, where=bd > 0)
                        p = vals['counts'][num] / vals['counts'][den] if vals['counts'][den] else None
                        vals[met] = {'estimate': p, 'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
                        point[key, mob, met], draws[key, mob, met] = p, z
                    mob_out[mob] = vals
                semantic = {}
                for metric, arr in mode['extras'].items():
                    rr = (arr[:, ks] * select).sum(1)
                    cluster = np.zeros(n); np.add.at(cluster, inverse, rr)
                    den_root = select.sum(1)
                    den_cluster = np.zeros(n); np.add.at(den_cluster, inverse, den_root)
                    divisor = len({'all': S.OE.MAPC, 'near': SB.NEAR4, 'interior': SB.INTER, 'edge': SB.EDGE}[metric[5:]]) if metric.startswith('tile_') else 1
                    den = den_cluster * divisor
                    z = (weights @ cluster) / (weights @ den)
                    p = rr.sum() / (den_root.sum() * divisor)
                    semantic[metric] = {'estimate': float(p), 'ci95': np.quantile(z, [.025, .975]).tolist()}
                    point[key, 'semantic', metric], draws[key, 'semantic', metric] = float(p), z
                for j, name in enumerate(('health', 'food', 'drink', 'energy')):
                    y = facts['hud'][:, ks, j].numpy()[select]
                    semantic[name + '_r2'] = float(1 - semantic[name + '_sse']['estimate'] / max(y.var(), 1e-9))
                out['models'][key] = {'mobs': mob_out, 'semantic': semantic}
            for a, b in contrasts:
                co = {}
                for typ in list(S.OE.TYPES) + ['semantic']:
                    mets = metrics if typ != 'semantic' else modes[a]['extras']
                    co[typ] = {}
                    for met in mets:
                        pa, pb = point[a, typ, met], point[b, typ, met]
                        z = draws[a, typ, met] - draws[b, typ, met]
                        co[typ][met] = {'difference': pa - pb if pa is not None and pb is not None else None,
                                       'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
                out['contrasts'][a + '_minus_' + b] = co
            result['populations'][population][group] = out
    quantiles = {}
    for key, scores in raw_scores.items():
        quantiles[key] = {}
        for mob, ch in S.OE.PROBE_CH.items():
            j = S.OE.TYPES.index(mob)
            y = truth[:, :, S.OE.MAPC, j]
            sc = scores[..., ch]
            quantiles[key][mob] = {}
            for label, sl in [('truth_present', y), ('truth_absent', ~y)]:
                quantiles[key][mob][label] = {g: torch.quantile(sc[:, ks][sl[:, ks]], torch.tensor([.01, .1, .5, .9, .99])).tolist() for g, ks in GROUPS.items()}
    result['score_quantiles_01_10_50_90_99'] = quantiles
    S.atomic(result, OUT / 'd_existing_compare.json', json_format=True)
    S.atomic({'counts': {k: {ck: v.tolist() for ck, v in m['counts'].items()} for k, m in modes.items()},
              'semantic': {k: {ek: v.tolist() for ek, v in m['extras'].items()} for k, m in modes.items()},
              'roots': panel['roots'], 'alignment': {k: v.tolist() for k, v in align.items()},
              'hashes': result['hashes']}, OUT / 'd_existing_rows.json', json_format=True)
    print(json.dumps({'stage': 'complete', 'out': str(OUT / 'd_existing_compare.json')}), flush=True)


if __name__ == '__main__':
    main()
