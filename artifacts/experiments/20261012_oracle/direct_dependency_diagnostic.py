"""CPU dependence sensitivity of the declared metric on intact real token marginals.

Independently permute complete 192-D vectors across real branches at each cell.
This preserves every cell's empirical distribution while breaking cross-cell
dependencies. No generated model is retrained, projected, or repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import direct_coherence as C
import direct_eval as E
import direct_model as M
import direct_spread_diagnostic as S
import loss_intervention as L

OUT = L.OUT / 'direct_generator/diagnostics/dependency_sensitivity'
PERMUTATIONS = 16
SEED = 2026101405


def permutation(root, direction, repetition):
    g = torch.Generator().manual_seed(SEED + 10000 * root + 100 * direction + repetition)
    index = torch.stack([torch.randperm(16, generator=g) for _ in range(81)], 1)
    assert torch.equal(index.sort(0).values, torch.arange(16)[:, None].expand(-1, 81))
    return index


def shuffled(x, index):
    return x[index, torch.arange(x.shape[1])[None]]


def source_metrics(flags, units):
    rows = []
    for unit in units:
        local = flags[:, unit['points'], unit['kind']]
        count = local.sum(-1)
        rows.append({'source': unit['source'], 'kind': unit['kind'],
                     'valid': float((count == 1).double().mean()),
                     'absent': float((count == 0).double().mean()),
                     'multiple': float((count > 1).double().mean()),
                     'source_and_destination': float((local[:, 0] & (count > 1)).double().mean())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--stop-after', type=int)
    a = parser.parse_args()
    torch.set_num_threads(a.threads)
    data, inputs = E.cohort('exploratory')
    analysis_path = L.OUT / 'direct_generator/eval_exploratory_u3000/analysis.json'
    analysis = json.loads(analysis_path.read_text())
    bank_path = L.OUT / 'snap_bank_readers.pt'
    real_path = L.OUT / 'learned_innovation/eval_u3000_h01/mob_heads.pt'
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)
    real_head = torch.load(real_path, map_location='cpu', weights_only=False)['heads']['real']
    real_decoded = C.decode(data['fan_tokens'], bank, real_head)['mob']['real']
    cells = [c for c in range(63) if c != C.PLAYER_CELL]
    assert torch.equal(real_decoded[:, :, cells], data['fan_layout'][:, :, cells].bool())
    units = [u for u in analysis['arms']['energy']['source_units']
             if u['primary'] and u['reader_clean']
             and len(torch.unique(C._event(real_decoded[u['root']], u['points'], u['kind']))) > 1]
    assert len(units) == 14 and len({u['root'] for u in units}) == 14
    assert len({r['episode_id'] for r in data['roots']}) == len(data['roots'])
    by_root = {i: [u for u in units if u['root'] == i] for i in range(len(data['roots']))}
    contract = {'format': 'direct_dependency_sensitivity_v1', 'roots': data['roots'], 'inputs': inputs,
                'analysis_sha256': L.sha(analysis_path), 'readers': {str(p.relative_to(L.ROOT)): L.sha(p) for p in (bank_path, real_path)},
                'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p)
                            for p in (__file__, E.__file__, M.__file__, C.__file__, S.__file__)},
                'permutations': PERMUTATIONS, 'directions': [[0, 16, 16, 32], [16, 32, 0, 16]],
                'seed_rule': '2026101405 + 10000*root + 100*direction + repetition; independent per-cell permutations',
                'invariants': ['Each full192D feature vector is retained intact at the same cell',
                               'Every per-cell empirical feature and semantic marginal is identical',
                               'Real-reader exactness and token-local equivariance are validated'],
                'limits': ['Post-hoc diagnostic; no training, scientific-criteria changes, or physical repair',
                          'Real32 full-simulator conditional law differs from H15 belief conditioning',
                          'Permutations break dependence of contextual embeddings as well as physical facts']}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cp = OUT / 'contract.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, 'diagnostic source/input drift'
    else:
        L.atomic(contract, cp, True)
    rows = []
    for i, root in enumerate(data['roots']):
        path = OUT / 'roots' / f'{i:04d}.json'
        if path.exists():
            row = json.loads(path.read_text())
            assert row['contract_id'] == cid and row['root'] == root
        else:
            started = time.perf_counter()
            y = data['fan_tokens'][i].double()
            flags = real_decoded[i]
            row = {'contract_id': cid, 'index': i, 'root': root, 'source_units': by_root[i], 'directions': []}
            for direction, (left, right) in enumerate(((slice(0, 16), slice(16, 32)), (slice(16, 32), slice(0, 16)))):
                x, target, xf, yf = y[left], y[right], flags[left], flags[right]
                coherent = {'latent': S.score(x, target),
                            'decoded_joint': float(C._energy(xf[None, :, M.INTERIOR, :3], yf[None, :, M.INTERIOR, :3])[0]),
                            'sources': source_metrics(xf, by_root[i])}
                repetitions = []
                for repetition in range(PERMUTATIONS):
                    index = permutation(i, direction, repetition)
                    mixed = shuffled(x, index)
                    mf = shuffled(xf, index[:, :63])
                    assert torch.equal(mixed.mean(0), x.mean(0)) or float((mixed.mean(0) - x.mean(0)).abs().max()) < 1e-15
                    assert torch.equal(mf.sum(0), xf.sum(0))
                    # The already exact reader acts token-wise; verify its
                    # equivariance on one permutation per root/direction.
                    if repetition == 0:
                        read = C.decode(mixed.float()[None], bank, real_head)['mob']['real'][0]
                        assert torch.equal(read, mf), 'reader equivariance failed'
                    repetitions.append({'latent': S.score(mixed, target),
                                        'decoded_joint': float(C._energy(mf[None, :, M.INTERIOR, :3], yf[None, :, M.INTERIOR, :3])[0]),
                                        'sources': source_metrics(mf, by_root[i])})
                row['directions'].append({'coherent': coherent, 'shuffled': repetitions})
            row['seconds'] = time.perf_counter() - started
            L.atomic(row, path, True)
        rows.append(row)
        print(json.dumps({'stage': 'dependency_sensitivity', 'completed_roots': i + 1, 'total': len(data['roots'])}), flush=True)
        if a.stop_after and i + 1 >= a.stop_after:
            return
    def average(row, fn):
        return float(np.mean([fn(d) for d in row['directions']]))
    metrics = {}
    for name in ('latent', 'decoded_joint'):
        pick = (lambda reading: reading['latent']['energy']) if name == 'latent' else (lambda reading: reading['decoded_joint'])
        coherent = [average(r, lambda d: pick(d['coherent'])) for r in rows]
        changed = [average(r, lambda d: np.mean([pick(x) for x in d['shuffled']])) for r in rows]
        metrics[name] = {'coherent': S.interval(coherent), 'shuffled': S.interval(changed),
                         'shuffled_minus_coherent': S.interval(np.asarray(changed) - coherent),
                         'fractional_increase_of_coherent_mean': float(np.mean(changed) / np.mean(coherent) - 1)}
    source_rows = [r for r in rows if r['source_units']]
    physical = {}
    for name in ('valid', 'absent', 'multiple', 'source_and_destination'):
        coherent = [average(r, lambda d: np.mean([u[name] for u in d['coherent']['sources']])) for r in source_rows]
        changed = [average(r, lambda d: np.mean([u[name] for x in d['shuffled'] for u in x['sources']])) for r in source_rows]
        physical[name] = {'coherent': S.interval(coherent), 'shuffled': S.interval(changed),
                          'shuffled_minus_coherent': S.interval(np.asarray(changed) - coherent)}
    result = {'format': 'direct_dependency_sensitivity_v1', 'completed': True, 'contract_id': cid,
              'roots': len(rows), 'source_roots': len(source_rows), 'source_units': len(units),
              'permutations_per_direction': PERMUTATIONS, 'directions': 2,
              'metrics': metrics, 'physical': physical,
              'per_cell_feature_and_semantic_marginals_preserved': True, 'reader_equivariance_verified': True,
              'physical_repair_claim': False, 'scope': contract['limits'],
              'total_root_compute_seconds': sum(r['seconds'] for r in rows)}
    L.atomic(result, OUT / 'result.json', True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
