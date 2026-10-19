"""Lineage audit (2026-10-15): did privileged movement information (simulator layouts, the F1b true-reassignment fit) reach the
unprivileged Stage 3 comparison? CPU only, reads saved files.
  1. the Stage 3 training contract's pinned sources/inputs contain no layouts, F1b code or F1b checkpoints
  2. Stage 3's initial worlds (free/operator u0000): every non-completion tensor equals the D parent exactly, and differs from
     the F1b-trained references (fit_assigned / fit_identity, u0500 and u1500)
  3. the D parent's own lineage: its saved args and its oracle-conditioning weights (must be identically zero)
  4. Stage 3 snapshots u0500 / u1500 / u3000 descend from u0000 (same ledger chain prefix is not stored per snapshot; checked:
     every snapshot's contract id equals the training contract id)
Usage: transport_lineage.py -> artifacts/eda/oracle_v1/transport/lineage.json
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/eda/oracle_v1'
T = BASE / 'transport'


def main():
    res = {}
    contract = json.loads((T / 'training/contract.json').read_text())
    pinned = list(contract['sources_inputs'])
    banned = [p for p in pinned if 'layouts' in p or 'transport_stage2' in p or 'fit_' in p or 'fan.pt' in p or 'panel.pt' in p]
    res['contract_pinned_inputs'] = pinned
    res['contract_privileged_inputs'] = banned
    res['contract_declares_no_labels'] = contract.get('training_semantic_or_simulator_labels') is False
    parent = torch.load(BASE / 'worlds/control_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt',
                        map_location='cpu', weights_only=False)
    res['parent_args'] = {k: str(v) for k, v in parent.get('args', {}).items()}
    res['parent_oracle_weights_all_zero'] = all(int(torch.count_nonzero(x)) == 0 for x in parent['oracle'].values())
    res['parent_oracle_tensors'] = len(parent['oracle'])
    pw = parent['world']
    fits = {f'{a}_u{u:04d}': torch.load(T / f'fit_{a}_u{u:04d}.pt', map_location='cpu', weights_only=False)['world']
            for a in ('assigned', 'identity') for u in (500, 1500)}
    for arm in ('free', 'operator'):
        w0 = torch.load(T / f'training/{arm}_u0000.pt', map_location='cpu', weights_only=False)
        world = w0['world']
        shared = [k for k in world if not k.startswith('completion.')]
        res[f'{arm}_u0000'] = {
            'non_completion_tensors': len(shared),
            'all_equal_to_D_parent': all(k in pw and torch.equal(world[k], pw[k]) for k in shared),
            'max_abs_diff_to_F1b': {n: max(float((world[k].float() - f[k].float()).abs().max()) for k in shared if k in f) for n, f in fits.items()}}
        for u in (0, 500, 1500, 3000):
            snap = torch.load(T / f'training/{arm}_u{u:04d}.pt', map_location='cpu', weights_only=False)
            res[f'{arm}_u{u:04d}_contract_id'] = snap['contract_id']
    import hashlib
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    res['training_contract_id'] = cid
    res['all_snapshots_from_this_contract'] = all(v == cid for k, v in res.items() if k.endswith('_contract_id'))
    (T / 'lineage.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'contract_pinned_inputs'}, indent=1))


if __name__ == '__main__':
    main()
