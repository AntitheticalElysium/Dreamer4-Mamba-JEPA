"""Same-population source strata for the frozen dependence diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np

import direct_dependency_diagnostic as D
import direct_spread_diagnostic as S
import loss_intervention as L


def main():
    rows = [json.loads(p.read_text()) for p in sorted((D.OUT / 'roots').glob('*.json'))]
    result = json.loads((D.OUT / 'result.json').read_text())
    assert result['completed'] and len(rows) == result['roots'] == 104
    assert all(r['contract_id'] == result['contract_id'] for r in rows)
    contract = {'format': 'direct_dependency_source_strata_v1', 'base_contract_id': result['contract_id'],
                'base_result_sha256': L.sha(D.OUT / 'result.json'),
                'root_hashes': {p.name: L.sha(p) for p in sorted((D.OUT / 'roots').glob('*.json'))},
                'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p) for p in (__file__, D.__file__, S.__file__)},
                'selection': 'Previously frozen14 valid isolated variable source roots; cow subset kind1, not score selection'}
    summaries = {}
    for name, selected in [('source14', [r for r in rows if r['source_units']]),
                           ('cow8', [r for r in rows if r['source_units'] and r['source_units'][0]['kind'] == 1])]:
        assert len(selected) == (14 if name == 'source14' else 8)
        metrics = {}
        for key in ('latent', 'decoded_joint'):
            def value(reading):
                return reading['latent']['energy'] if key == 'latent' else reading[key]
            coherent = np.array([np.mean([value(d['coherent']) for d in r['directions']]) for r in selected])
            mixed = np.array([np.mean([value(x) for d in r['directions'] for x in d['shuffled']]) for r in selected])
            metrics[key] = {'coherent': S.interval(coherent), 'shuffled': S.interval(mixed),
                            'shuffled_minus_coherent': S.interval(mixed - coherent),
                            'fractional_increase_of_coherent_mean': float(mixed.mean() / coherent.mean() - 1)}
        summaries[name] = metrics
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output = {'completed': True, 'format': contract['format'], 'contract_id': cid,
              'contract': contract, 'summaries': summaries,
              'scope': result['scope'], 'physical_repair_claim': False}
    L.atomic(output, D.OUT / 'source_strata.json', True)
    print(json.dumps({'completed': True, 'contract_id': cid, 'summaries': summaries}), flush=True)


if __name__ == '__main__':
    main()
