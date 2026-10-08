"""Frozen E19 on the exact fresh-arrival TRAIN events already counted by the census.

Rules fixed before execution. The previous detector/estimated-motion census defines
15 sampled ordinary-hit events and25 unchanged events; no new event threshold or
selection. Score C at both Mamba seeds and attention seed7 on (1) every sampled
original window prefix through the current frame and (2) each actually sampled
boundary/context, weighted by actual multiplicity. Real next-state HUD is a
decoder positive control. Force-generating the health token separates candidate
depiction from routing. Future frames/labels never enter the world forward.
This distinguishes own-TRAIN failure from a failure only on new roots. It does
not isolate historical optimization versus scarce unique support. Source/input-
bound atomic case journals; GPU allocator capped16%, batch4, no model training.
No automatic notebook writes. The event condition remains detector-estimated,
not ground-truth simulator adjacency; recorded damage/alive labels are factual.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import e19_diagnose as D
import h16_resume as R

HERE = Path(__file__).resolve().parent
T, E = D.T, D.E


def main():
    torch.set_num_threads(3)
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.16)
    root = HERE / 'evals/resume/e19_fmamba__incoming_exposure'
    oldspec = json.loads((root / 'contract.json').read_text())
    oldstore = R.Store(root, oldspec)
    eventsets = {c: set() for c in ('ordinary_ge2', 'unchanged')}
    for p in root.glob('batch_*.json'):
        part = oldstore.load(p.stem)
        for c in eventsets:
            eventsets[c].update(tuple(e) for e in part['events']['fresh_incoming'][c])
    assert [len(eventsets[c]) for c in eventsets] == [15, 25]
    pp = E.T.POOLS['raw'] / 'pool.pt'
    pool = torch.load(pp, map_location='cpu', mmap=True, weights_only=False)
    trainroot = E.T.OUT / 'state/e19_C_s7_fmamba_from36000'
    training = R.Store(trainroot, json.loads((trainroot / 'contract.json').read_text())).load('train')
    idx, boundary = training['ledger'].long(), training['layout'].long()
    count = torch.bincount((idx * 5 + boundary).flatten(), minlength=len(pool['ids']) * 5).reshape(-1, 5)
    rows = []
    for r, (episode, start) in enumerate(pool['ids']):
        for k in (3, 4):
            event = (episode, int(start) + k)
            label = next((c for c, events in eventsets.items() if event in events), None)
            if label is None or not count[r].sum():
                continue
            rows.append({'pool_row': r, 'target': k, 'lo': 0, 'label': label,
                         'protocol': 'original_context', 'weight': int(count[r].sum())})
            for b in range(5):
                if count[r, b]:
                    cut = 4 - b
                    lo = 0 if k < cut else cut
                    rows.append({'pool_row': r, 'target': k, 'lo': lo, 'label': label,
                                 'protocol': 'sampled_boundary', 'weight': int(count[r, b])})
    assert sum(r['weight'] for r in rows if r['protocol'] == 'original_context' and r['label'] == 'ordinary_ge2') == 111
    meta, fit, seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = T.Probes(cache, meta, fit, seeds)
    health = lambda tokens: probes.hud(tokens[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9
    from d4mj.train import autocast_context
    config = E.config()
    for bb, seed in [('fmamba', 7), ('fmamba', 8), ('full', 7)]:
        name = f'e19_C_s{seed}_{bb}_from36000'
        path = E.T.OUT / (name + '.pt')
        spec = {'scope': __doc__, 'selection': rows,
                'sources': {str(p): R.file_hash(p) for p in
                            [Path(__file__), HERE / 'tworld.py', HERE / 'teval.py', HERE / 'e19.py', HERE / 'e19_diagnose.py']},
                'inputs': {str(path): R.file_hash(path), str(T.META): R.file_hash(T.META),
                           str(root / 'contract.json'): R.file_hash(root / 'contract.json'),
                           str(root / 'result.json'): R.file_hash(root / 'result.json')},
                'pool_sha256': oldspec['inputs'][str(pp)],
                'runtime': {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                            'precision': 'canonical BF16 autocast, FP32 health readout', 'batch': 4}}
        store = R.Store(HERE / 'evals/resume' / (name + '__train_fresh'), spec)
        with store.lock():
            result = store.load('result')
            if result is None:
                world, _ = T.load_world(path, device)
                measurements = []
                for length in range(1, 6):
                    selected = [i for i, r in enumerate(rows) if r['target'] - r['lo'] + 1 == length]
                    for start in range(0, len(selected), 4):
                        key = f'length{length}_batch{start}'
                        part = store.load(key)
                        if part is None:
                            indices = selected[start:start + 4]
                            src = torch.stack([pool['tokens'][rows[i]['pool_row'], rows[i]['lo']:rows[i]['target'] + 1] for i in indices]).float()
                            act = torch.stack([pool['actions'][rows[i]['pool_row'], rows[i]['lo']:rows[i]['target'] + 1] for i in indices])
                            truth = torch.stack([pool['tokens'][rows[i]['pool_row'], rows[i]['target'] + 1] for i in indices]).float()
                            with torch.no_grad(), autocast_context(config):
                                pred, _, gen = world(src.to(device), act.to(device))
                            pred, gen = pred[:, -1].float(), gen[:, -1].float()
                            forced = pred.clone(); forced[:, 63] = F.layer_norm(gen[:, 63], (192,))
                            base = health(src[:, -1])
                            delta = {k: health(v) - base for k, v in [('actual', pred), ('generator63', forced), ('real_successor', truth)]}
                            part = [{'index': i, **{k: float(v[j]) for k, v in delta.items()}} for j, i in enumerate(indices)]
                            store.save(key, part, 1)
                        measurements.extend(part)
                assert len(measurements) == len(rows)
                out = {}
                for protocol in ('original_context', 'sampled_boundary'):
                    out[protocol] = {}
                    for label in eventsets:
                        mm = [m for m in measurements if rows[m['index']]['protocol'] == protocol and rows[m['index']]['label'] == label]
                        denom = sum(rows[m['index']]['weight'] for m in mm)
                        out[protocol][label] = {'unique_window_targets': sum(1 for r in rows if r['protocol'] == 'original_context' and r['label'] == label),
                            'sampled_presentations': denom,
                            'weighted_damage_drawn': {k: sum(rows[m['index']]['weight'] for m in mm if m[k] < -1.5) for k in ('actual', 'generator63', 'real_successor')},
                            'weighted_health_change': {k: sum(rows[m['index']]['weight'] * m[k] for m in mm) / denom for k in ('actual', 'generator63', 'real_successor')}}
                result = {'scope': __doc__, 'name': name, 'contract': spec, 'groups': out}
                store.save('rows', measurements, 1); store.save('result', result, 1)
                del world; torch.cuda.empty_cache()
            R.atomic_json(HERE / 'evals' / (name + '__train_fresh.json'), result)
            print(json.dumps({'name': name, 'groups': result['groups']}), flush=True)


if __name__ == '__main__':
    main()
