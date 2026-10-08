"""Build the declared E20 factual pool and audit expanded fresh-hit support.

All stored targets are observed corpus outcomes. The frozen diagnostic zombie
reader is used only to audit support; no fork state/outcome trains a world or the
health-loss reader. Chunk hashes protect the pool on resume. No notebook writes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e20_data as D
import e19 as E
import h16_resume as R
import teval as V
from scroll import SHIFTS, estimate


def inventory():
    root = D.OUT / 'source_census_corrected'
    store = R.Store(root, json.loads((root / 'contract.json').read_text()))
    return store.load('episodes')


def tables(records):
    old = torch.load(E.T.POOLS['raw'] / 'pool.pt', map_location='cpu', mmap=True, weights_only=False)
    by_id = {e['id']: i for i, e in enumerate(records)}
    sets = [set() for _ in range(4)]
    for r in E.train_rows(old).tolist():
        eid, start = old['ids'][r]
        ep = records[by_id[eid]]
        for k in range(5):
            t = int(start) + k
            if t >= 14:
                cls = int(ep['classes'][t])
                same = int(ep['dh'][t]) == int(old['dh'][r, k])
                terminal_ambiguity = cls == 0 and int(ep['dh'][t]) == -9 and int(old['dh'][r,k]) == 1
                assert same or terminal_ambiguity
                sets[cls].add((by_id[eid], t))
    gen = torch.Generator().manual_seed(20261007)
    for c, cap in ((2, 24576), (3, 4096)):
        q = sorted(sets[c])
        if len(q) > cap:
            q = [q[i] for i in torch.randperm(len(q), generator=gen)[:cap].tolist()]
        sets[c] = set(q)
    broad = {(i, t) for i, e in enumerate(records)
             for t in torch.where(e['classes'][14:] == 1)[0].add(14).tolist()}
    all_events = sorted(set.union(*sets, broad))
    lookup = {v: i for i, v in enumerate(all_events)}
    a = [torch.tensor([lookup[v] for v in sorted(q)], dtype=torch.long) for q in sets]
    b = list(a)
    b[1] = torch.tensor([lookup[v] for v in sorted(broad)], dtype=torch.long)
    assert set(a[1].tolist()) <= set(b[1].tolist())
    assert all(torch.equal(a[c], b[c]) for c in (0, 2, 3))
    return all_events, {'A': a, 'B': b, 'C': b}


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda')
    records = inventory()
    events, cohorts = tables(records)
    episodes, dataset_contract = D.corpus()
    episodes = {e.episode_id: e for e in episodes if e.split == 'train' and e.uniform_eligible}
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = torch.load(D.ENCODER, map_location='cpu', weights_only=False)
    cfg = config_from_dict(payload['config'])
    bundle = ModelBundle.create(cfg)
    bundle.encoder.load_state_dict(payload['modules']['encoder'])
    encoder = bundle.encoder.to(device).freeze()
    spec = {'version': 'e20-endpoint-pool-v1', 'events': events,
            'cohorts': {k: [R.tensor_hash(v) for v in q] for k, q in cohorts.items()},
            'dataset_contract': dataset_contract['sha256'],
            'inputs': {str(p): R.file_hash(p) for p in
                       (D.DATASET, D.ENCODER, E.T.POOLS['raw'] / 'pool.pt',
                        D.OUT / 'source_census_corrected/result.json', V.META,
                        Path(str(V.CACHE).format('raw')))},
            'sources': {str(p): R.file_hash(p) for p in
                        (Path(__file__), HERE / 'e20_data.py', HERE / 'e20_labels.py', HERE / 'e19.py',
                         HERE / 'tworld.py', HERE / 'scroll.py', HERE / 'h16_resume.py',
                         HERE / 'teval.py', HERE / 'E20.md')},
            'forward_sources': {str(p): R.file_hash(p) for p in sorted(D.ROOT.joinpath('d4mj').rglob('*.py'))},
            'runtime': {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                        'precision': 'historical FP32 encoder / FP32 layer norm / fp16 storage',
                        'tf32': torch.backends.cuda.matmul.allow_tf32},
            'shape': [len(events), 16, 81, 192], 'batch': 32}
    store = R.Store(D.OUT / 'pool', spec)
    with store.lock():
        if store.load('complete') is not None:
            print(json.dumps({'stage': 'pool_complete_exists'}), flush=True)
            return
        if store.load('encoder_parity') is None:
            old = torch.load(E.T.POOLS['raw'] / 'pool.pt', map_location='cpu', mmap=True, weights_only=False)
            rows = E.train_rows(old)[torch.linspace(0, len(E.train_rows(old))-1, 32).long()]
            frames = torch.stack([episodes[old['ids'][r][0]].observations[int(old['ids'][r][1]):int(old['ids'][r][1])+6]
                                  for r in rows.tolist()])
            with torch.no_grad():
                tok = encoder._hidden(frames.to(device))[2]
                tok = F.layer_norm(tok.float(), (192,)).half().cpu().reshape(32,6,81,192)
            diff = (tok.float()-old['tokens'][rows].float()).abs()
            parity = {'maximum_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                      'tolerance': .002, 'sentinel_windows': 32}
            if parity['maximum_abs'] > .002:
                raise RuntimeError(f'Historical encoder cache parity failed: {parity}')
            store.save('encoder_parity', parity, 1)
            print(json.dumps({'stage': 'encoder_parity', **parity}), flush=True)
        fc = R.FeatureCache(D.OUT / 'pool/tokens.f16', spec['shape'], batch=32)
        health = []
        for e in records:
            hp = e['health'].long()
            if not bool(((hp >= 0) & (hp <= 9)).all()):
                raise RuntimeError(f"Reward-derived absolute health invalid: {e['id']}")
            health.append(hp)
        for start in range(fc.start, len(events), 32):
            chunk = events[start:start + 32]
            frames = torch.stack([episodes[records[i]['id']].observations[t-14:t+2] for i, t in chunk])
            assert frames.shape[1:] == (16, 63, 63, 3)
            out = []
            flat = frames.flatten(0, 1)
            with torch.no_grad():
                for k in range(0, len(flat), 128):
                    tok = encoder._hidden(flat[k:k+128, None].to(device))[2]
                    out.append(F.layer_norm(tok.float(), (192,)).half().cpu())
            fc.mm[start:start + len(chunk)] = torch.cat(out).reshape(len(chunk), 16, 81, 192).numpy()
            fc.commit(start, start + len(chunk))
            if start % 1024 == 0:
                print(json.dumps({'stage': 'encode', 'events_done': start + len(chunk), 'of': len(events)}), flush=True)
        del encoder, bundle
        torch.cuda.empty_cache()
        labels = {'ids': [(records[i]['id'], t) for i, t in events], 'cohorts': cohorts,
                  'classes': torch.tensor([int(records[i]['classes'][t]) for i, t in events]),
                  'actions': torch.stack([records[i]['actions'][t-14:t+1] for i, t in events]),
                  'dh': torch.stack([records[i]['dh'][t-14:t+1] for i, t in events]),
                  'health': torch.stack([health[i][t-14:t+2] for i, t in events]),
                  'shape': spec['shape'], 'pool_contract': store.contract}
        labels_path = D.OUT / 'pool/labels.pt'
        if not labels_path.exists():
            R.atomic_torch(labels_path, labels)
        else:
            previous = torch.load(labels_path, map_location='cpu', weights_only=False)
            if previous['pool_contract'] != store.contract or previous['ids'] != labels['ids']:
                raise RuntimeError('Pool labels changed')
        cache = torch.load(Path(str(V.CACHE).format('raw')), map_location='cpu', mmap=True, weights_only=False)
        meta, fit, seeds = V.split()
        probes = V.Probes(cache, meta, fit, seeds)
        cells = torch.tensor([r*9+c for r in range(7) for c in range(9) if abs(r-3)+abs(c-4)<=2])
        coords = torch.stack([cells//9, cells%9], 1)
        distance = (coords - torch.tensor([3,4])).abs().sum(1)
        fresh = torch.zeros(len(events), dtype=torch.bool)
        for start in range(0, len(events), 128):
            key = f'support_{start}'
            part = store.load(key)
            if part is None:
                s = torch.from_numpy(np.array(fc.mm[start:start+128, 11:])).float()
                zombie = (probes.zombie(s[:, :4, cells].flatten(0,2)).reshape(len(s), 4, len(cells)) > .3)
                adjacency = zombie[:, :, distance == 1].any(-1).any(1)
                shift = estimate(s[:, 3:4], s[:, 4:5])[:, 0]
                disp = torch.tensor(SHIFTS)[shift]
                afterdist = (coords[None] - torch.tensor([3,4])[None,None] - disp[:,None]).abs().sum(-1)
                incoming = (zombie[:, -1] & (afterdist == 1)).any(-1)
                previous_damage = (labels['dh'][start:start+len(s), 11:14] <= -2).any(1)
                part = incoming & ~adjacency & ~previous_damage
                store.save(key, part, 1)
            fresh[start:start+len(part)] = part
        coverage = {}
        for arm, qs in cohorts.items():
            q = qs[1][fresh[qs[1]]]
            coverage[arm] = {'fresh_damage_events': len(q),
                             'fresh_damage_episodes': len({labels['ids'][i][0] for i in q.tolist()}),
                             'class_counts': [len(x) for x in qs]}
        labels['fresh'] = fresh
        R.atomic_torch(labels_path, labels)
        result = {'coverage': coverage, 'events': len(events), 'contract': spec,
                  'labels_sha256': R.file_hash(labels_path),
                  'fresh_support_pass': coverage['B']['fresh_damage_events'] >= 200 and
                                        coverage['B']['fresh_damage_episodes'] >= 100,
                  'audit_label_scope': 'TRAIN-fit token zombie detector and true-pair motion estimator'}
        store.save('complete', result, 1)
        R.atomic_json(D.OUT / 'pool.json', result, immutable=True)
        print(json.dumps({'stage': 'pool_complete', 'coverage': coverage,
                          'fresh_support_pass': result['fresh_support_pass']}), flush=True)


if __name__ == '__main__':
    main()
