"""S0-format new-monitor labels from authoritative collection sidecars. Evaluation only."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    union = json.loads((ROOT / 'artifacts/craftax_union_v1/manifest.json').read_text())
    report = {'monitor_rule': 'sha256(episode_id) % 10000 < 500', 'sources': {}}
    offset = 0
    for source in union['sources']:
        store = ROOT / source['store']
        m = json.loads((store / 'manifest.json').read_text())
        if offset == 0:
            offset += source['shards']; continue
        state, trans, meta = {}, {}, []
        for local, shard in enumerate(m['shards']):
            side = store / shard['audit_file']
            assert hashlib.sha256(side.read_bytes()).hexdigest() == shard['audit_sha256']
            with np.load(side) as z:
                ts = z['ep_T']; so = np.r_[0, np.cumsum(ts + 1)]; to = np.r_[0, np.cumsum(ts)]
                for slot, T in enumerate(ts):
                    eid = f"coverage-v1:{m['contract']['seed']}:{int(z['ep_round'][slot])}:{int(z['ep_slot'][slot])}"
                    if int(hashlib.sha256(eid.encode()).hexdigest(), 16) % 10000 >= 500:
                        continue
                    for key in z.files:
                        if key.startswith('s_'):state.setdefault(key, []).append(z[key][so[slot]:so[slot+1]])
                        elif key.startswith('t_'):trans.setdefault(key, []).append(z[key][to[slot]:to[slot+1]])
                    policy = shard['policy']
                    epsilon = policy['base_epsilon'] if policy['prefix'] is None else float('nan')
                    meta.append((offset + local, slot, int(T), epsilon, eid))
        out = {**{k: np.concatenate(v) for k, v in state.items()},
               **{k: np.concatenate(v) for k, v in trans.items()},
               'ep_shard': np.array([v[0] for v in meta], np.int32),
               'ep_slot': np.array([v[1] for v in meta], np.int32),
               'ep_T': np.array([v[2] for v in meta], np.int32),
               'ep_eps': np.array([v[3] for v in meta], np.float32),
               'ep_key': np.array([v[4] for v in meta])}
        assert len(meta) == m['coverage']['monitor']['episodes']
        assert int(out['ep_T'].sum()) == m['coverage']['monitor']['transitions']
        assert len(out['s_body']) == int((out['ep_T']+1).sum())
        assert len(out['t_action']) == int(out['ep_T'].sum())
        name = 'coverage' if 'coverage' in source['store'] else 'risk'
        path = HERE / f'heldout_{name}.npz'
        np.savez_compressed(path, **out)
        report['sources'][name] = {'file': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((store/'manifest.json').read_bytes()).hexdigest(),
            'episodes': len(meta), 'transitions': int(out['ep_T'].sum()),
            'note': 'Scheduled-policy ep_eps is NaN; actual t_epsilon and t_exogenous are retained.'}
        offset += source['shards']
    (HERE/'heldout.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
