"""P0: no GPU or fit; same trained head predictions, different fixed policies."""
import importlib
import json
import time
import numpy as np

C = importlib.import_module('artifacts.experiments.20261103_chooser_development.common')


def main():
    start = time.time()
    results = {}; checks = C.checks()
    metas = C.cohorts()
    saved = np.load(C.HERE.parent/'20261101_p1f15/readings/locus/p1f15.npz')
    native = np.load(C.OLD/'native_confirmation_and_noise.npz')
    inputs = {}
    for name, meta in metas.items():
        source = saved if name == 'saved' else native
        cc, ss = (source[k] for k in (('contingency', 'surprise') if name == 'saved' else ('C', 'S')))
        cc, ss = cc[:len(meta['ep'])], ss[:len(meta['ep'])]
        for k in meta:
            assert np.array_equal(meta[k], source[k][:len(meta['ep'])]), (name, k)
        inputs['native_'+name] = (cc, ss)
        for seed in (611601, 611602):
            path = C.OLD/f'predictions_{seed}_{name}.npz'
            z = np.load(path)
            for k in meta:
                assert np.array_equal(meta[k], z[k]), (seed, name, k)
            inputs[str(seed)+'_'+name] = (z['C'], z['S'])
            diff = np.max(np.abs(C.policies(z['C'], z['S'])['existing'] - z['prob']))
            assert diff < 1e-6, diff
            checks[f'existing_probability_max_error_{seed}_{name}'] = float(diff)
    for key, (c, s) in inputs.items():
        name = 'confirmation' if key.endswith('confirmation') else 'saved'
        results[key] = C.summarize(metas[name], c, s)
        np.savez_compressed(C.HERE/f'policy_{key}.npz', **C.policies(c, s), **metas[name])
        print('policy done', key, round(time.time()-start, 2), flush=True)
    C.write(C.HERE/'policy_results.json', results)
    C.write(C.HERE/'policy_checks.json', checks)
    lines = ['# Cached policy ablation', '', 'Two existing heads; frozen checkpoint10. No new training.',
             'Values are exact remote-counter probability mass, not argmax coverage.', '']
    for cohort in ('saved', 'confirmation'):
        lines += [f'## {cohort}', '', '| Policy | Drink | Stone | Coal | Iron |', '|---|---:|---:|---:|---:|']
        for p in C.POLICIES:
            cells = []
            for cl in ('drink', 'stone', 'coal', 'iron'):
                vs = [100*results[f'{seed}_{cohort}'][cl]['policies'][p]['remote_mass']['mean'] for seed in (611601, 611602)]
                cells.append('/'.join(f'{v:.1f}%' for v in vs))
            lines.append('| '+p+' | '+' | '.join(cells)+' |')
        lines += ['', 'Native reference under each matching rule and all contact/union, class, light and',
                  'paired episode-bootstrap intervals are in policy_results.json.', '']
    lines += ['This is conditional allocation on the SAME recorded histories. It does not measure',
              'the policies\' own fixation histories, world learning, or adaptive emergence.',
              'No rule or temperature was selected from these results.', '', f'CPU elapsed: {time.time()-start:.1f}s.']
    (C.HERE/'POLICY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    C.write(C.HERE/'policy_identity.json', dict(seconds=time.time()-start,
        plan_sha256=C.sha(C.HERE/'PLAN.md'), code_sha256={p.name:C.sha(p) for p in C.HERE.glob('*.py')},
        input_sha256={p.name:C.sha(p) for p in C.OLD.glob('predictions_*.npz')}))


if __name__ == '__main__':
    main()
