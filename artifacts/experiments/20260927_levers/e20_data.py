"""E20 factual TRAIN consequence inventory; resumable, no notebook writes.

First inventory the complete source corpus, not the old sampled six-frame pool.
Health changes use the previously validated reward decomposition. No fork outcome
is added to training. A target at t uses observations through t, outgoing action t,
and observation t+1; t>=3 supports the gate's four observed frames, t>=14 supports
all proposed context lengths 4..15 without padding or crossing an episode reset.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260921_readout_ladder'))
import h16_resume as R

OUT = ROOT / 'artifacts/eda/levers_e20_v1'
DATASET = ROOT / 'artifacts/lewm_m4_canonical/raw/dataset.json'
ENCODER = ROOT / 'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt'


def corpus():
    from d4mj.config import config_from_dict
    from d4mj.data import load_joint_corpus
    cfg = config_from_dict(torch.load(ENCODER, map_location='cpu', weights_only=False)['config'])
    saved = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(saved['paths'], cfg)
    if contract != saved['contract']:
        raise RuntimeError('Factual corpus contract changed')
    return episodes, contract


def main():
    import e20_labels as L
    torch.set_num_threads(3)
    episodes, contract = corpus()
    eligible = [e for e in episodes if e.split == 'train' and e.uniform_eligible]
    pictures = L.templates()
    spec = {'version': 'e20-source-census-hud-checked-v2', 'scope': __doc__,
            'dataset_contract': contract['sha256'],
            'inputs': {str(p): R.file_hash(p) for p in (DATASET, ENCODER)},
            'sources': {str(p): R.file_hash(p) for p in
                        (Path(__file__), HERE / 'h16_resume.py', ROOT / 'd4mj/data.py',
                         HERE.parent / '20260921_readout_ladder/exposure.py', HERE / 'e20_labels.py')},
            'health_templates': R.tensor_hash(torch.from_numpy(pictures)),
            'runtime': {'torch': str(torch.__version__), 'threads': 3}}
    store = R.Store(OUT / 'source_census_corrected', spec)
    with store.lock():
        result = store.load('result')
        if result is not None:
            print(json.dumps(result['counts']), flush=True)
            return
        chunks = []
        for start in range(0, len(eligible), 128):
            key = f'episodes_{start}'
            part = store.load(key)
            if part is None:
                part = []
                for e in eligible[start:start + 128]:
                    dh, ambiguity = L.health_change(e.rewards, e.terminated)
                    dead = e.terminated.bool()
                    hp = L.decode(e.observations, pictures)
                    if hp[0] != 9 or not torch.equal(hp[1:] - hp[:-1], dh):
                        mismatch = torch.where(hp[1:] - hp[:-1] != dh)[0].tolist()
                        raise RuntimeError(f"HUD/reward health mismatch: {e.episode_id}, {mismatch[:10]}")
                    if dead[:-1].any() or e.truncated[:-1].any():
                        raise RuntimeError('Episode crosses a reset')
                    classes = torch.full((len(e),), 3, dtype=torch.uint8)
                    classes[dh == 0] = 2
                    classes[(dh <= -2) & ~dead] = 1
                    classes[dead] = 0
                    part.append({'id': e.episode_id, 'steps': len(e), 'classes': classes,
                                 'dh': dh.to(torch.int8), 'health': hp.to(torch.int8),
                                 'terminal_ambiguities': int(ambiguity.sum()), 'actions': e.actions_taken,
                                 'split': e.split, 'source': e.episode_id.split(':')[0]})
                store.save(key, part, 1)
            chunks.extend(part)
            print(json.dumps({'stage': 'source_census', 'episodes': len(chunks), 'of': len(eligible)}), flush=True)
        counts = {}
        for minimum in (3, 14):
            hist = torch.zeros(4, dtype=torch.long)
            for e in chunks:
                hist += torch.bincount(e['classes'][minimum:].long(), minlength=4)
            counts[f'target_t_ge_{minimum}'] = dict(zip(('death', 'ordinary_ge2', 'unchanged', 'other'), hist.tolist()))
        result = {'contract': spec, 'counts': counts, 'train_episodes': len(chunks),
                  'class_names': ['death', 'ordinary_ge2', 'unchanged', 'other'],
                  'terminal_reward_ambiguities_corrected': sum(e['terminal_ambiguities'] for e in chunks),
                  'hud_frames_exactly_verified': sum(e['steps']+1 for e in chunks),
                  'label_scope': 'recorded health/termination, independently exact-HUD checked; no simulator hazard annotations'}
        store.save('episodes', chunks, 1)
        store.save('result', result, 1)
        R.atomic_json(OUT / 'source_census_corrected.json', result, immutable=True)
        print(json.dumps({'stage': 'complete', **counts}), flush=True)


if __name__ == '__main__':
    main()
