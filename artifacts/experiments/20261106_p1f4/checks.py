"""P1-F4 preflight (PLAN.md "Preflight"): F2.1's arm checks (../20261104_p1f21/checks.py: run) adapted so that the
bootstrap term is exercised, plus F4's own. No efficacy reading, no launch.
- exact resume across the bootstrap start (small budget, bootstrap from update 2): streams, carry, cache, state, RNG,
  chooser, gaze bytes and every logged term bitwise; model and optimizer within 1e-11 (F2.1's optimizer tolerance);
- the real launcher in a fresh process;
- speed and memory: 64-step and 256-step updates with the bootstrap off (the recipe until update 5,000) and on, and the
  whole run's training time from them;
- update 1 against F3's configuration run beside it (same microbatch split): every term except g1 identical (same init
  except the two new embeddings, same data, gaze and heads; only the denoiser's objective and attention changed);
- the monitor on the 200-update model."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from saccade import data as D, train as TR
F2C = importlib.import_module('artifacts.experiments.20261103_p1f2.checks')
L = importlib.import_module('artifacts.experiments.20261106_p1f4.train_p1f4')
equal, optimizer_error = F2C.equal, F2C.optimizer_error
HERE = Path(__file__).resolve().parent
F3_LOG = HERE.parent / '20261105_p1f3/run/log.jsonl'


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    out = HERE / 'checks'
    if out.exists():
        raise RuntimeError('Preserve previous checks')
    out.mkdir(); torch.set_num_threads(4); start = time.time(); result = {'passed': False}
    cfg = L.check_recipe(L.recipe(TR.load_cfg()))
    small = copy.deepcopy(cfg); small['p1']['frames'] = 20480; small['p1']['monitor']['windows'] = 4
    small['edm']['shortcut']['bootstrap_start'] = 2
    for name, updates in (('straight', (4,)), ('resumed', (2, 4))):
        for u in updates:
            model = L.T.p1(small, 1, out / name, updates=u, log_every=1); del model; torch.cuda.empty_cache()
    a, b = (torch.load(out / n / 'ckpt_02.pt', map_location='cpu', weights_only=False) for n in ('straight', 'resumed'))
    # The model is held to the optimizer's tolerance, not bitwise: the existing cross-process noise (F3's own
    # configuration, two straight 4-update runs in separate processes: optimizer state 2.3e-13 apart, model bitwise
    # equal) reaches the dense blocks' still near-zero adaLN parameters (5e-18 apart; within one process bitwise).
    result['learned_resume'] = {k: bool(equal(a[k], b[k])) for k in ('streams', 'carry', 'cache', 'state', 'rng', 'fixation_controller')}
    result['learned_resume_optimizer_max_abs'] = optimizer_error(a['opt'], b['opt'])
    result['learned_resume_model_max_abs'] = max(float((a['model'][k].float() - b['model'][k].float()).abs().max()) for k in a['model'])
    result['learned_resume_model_bitwise_tensors'] = sum(bool(equal(a['model'][k], b['model'][k])) for k in a['model'])
    result['learned_resume']['gaze_bytes'] = (out / 'straight/gaze.bin').read_bytes() == (out / 'resumed/gaze.bin').read_bytes()
    logs = [[json.loads(x) for x in (out / n / 'log.jsonl').read_text().splitlines()] for n in ('straight', 'resumed')]
    result['learned_resume']['logged_terms'] = [{k: v for k, v in r.items() if k not in ('sec_per_update', 'peak_gib')} for r in logs[0]] == \
        [{k: v for k, v in r.items() if k not in ('sec_per_update', 'peak_gib')} for r in logs[1]]
    assert all(result['learned_resume'].values()) and result['learned_resume_optimizer_max_abs'] < 1e-11 \
        and result['learned_resume_model_max_abs'] < 1e-11, result
    del a, b
    # F3's own configuration (its checkpoint's, the EDM path; the same microbatch split) beside F4's, one update each
    f3cfg = torch.load(L.CONTROL, map_location='cpu', weights_only=False)['cfg']; f3cfg['data'].update(L.MEMORY)
    for name, c in (('update1_f3', f3cfg), ('update1_f4', cfg)):
        c = copy.deepcopy(c); c['p1']['frames'] = 20480; c['p1']['monitor']['windows'] = 4
        model = L.T.p1(c, 1, out / name, updates=1, checkpoints=False, log_every=1); del model; torch.cuda.empty_cache()
    f3, f4 = (json.loads((out / n / 'log.jsonl').read_text().splitlines()[0]) for n in ('update1_f3', 'update1_f4'))
    same = {k: f3[k] == f4[k] for k in f3 if k not in ('g1', 'grad_norm', 'sec_per_update', 'peak_gib')}
    result['update1_vs_f3'] = {'identical_except_g1': same, 'g1': [f3['g1'], f4['g1']], 'grad_norm': [f3['grad_norm'], f4['grad_norm']]}
    assert all(same.values()) and f3['g1'] != f4['g1'], result['update1_vs_f3']
    subprocess.run([sys.executable, '-u', '-m', 'artifacts.experiments.20261106_p1f4.train_p1f4', str(out / 'launcher'),
                    '--updates', '2', '--no-checkpoints'], cwd=TR.ROOT, check=True)
    on = copy.deepcopy(cfg); on['edm']['shortcut']['bootstrap_start'] = 0
    for name, c, u, steps in (('sustained64', cfg, 200, 64), ('sustained64_bootstrap', on, 100, 64),
                              ('phase256', cfg, 3, 256), ('phase256_bootstrap', on, 3, 256)):
        torch.cuda.reset_peak_memory_stats()
        model = L.T.p1(c, 1, out / name, updates=u, steps=steps, checkpoints=False, log_every=25 if steps == 64 else 1)
        rows = [json.loads(x) for x in (out / name / 'log.jsonl').read_text().splitlines()]
        aux = [json.loads(x) for x in (out / name / 'chooser.jsonl').read_text().splitlines()]
        assert len(aux) == u and all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
        result[name] = dict(updates=u, last=rows[-1], head_updates=aux[-1]['head_updates'],
                            mean_final_half_sec_per_update=float(np.mean([r['sec_per_update'] for r in rows[len(rows) // 2:]])),
                            peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
        if name == 'sustained64':
            result['monitor'] = TR.monitor(model, cfg, D.load_split(cfg)['monitor'])
            f3rows = {r['update']: r for r in map(json.loads, F3_LOG.read_text().splitlines()) if 'g1' in r}
            result['g1_vs_f3_same_update'] = {r['update']: [f3rows[r['update']]['g1'], r['g1']] for r in rows if r['update'] in f3rows}
        del model; torch.cuda.empty_cache()
    s = lambda n: result[n]['mean_final_half_sec_per_update']
    n64 = cfg['p1']['frames'] / 2 / (cfg['data']['streams'] * 64); n256 = cfg['p1']['frames'] / 2 / (cfg['data']['streams'] * 256)
    b0 = cfg['edm']['shortcut']['bootstrap_start']
    result['training_hours_estimate'] = (b0 * s('sustained64') + (n64 - b0) * s('sustained64_bootstrap') + n256 * s('phase256_bootstrap')) / 3600
    result['f3_training_hours'] = 11.5
    result.update(passed=True, elapsed_seconds=time.time() - start, peak_gib=max(result[n]['peak_gib'] for n in
                  ('sustained64', 'sustained64_bootstrap', 'phase256', 'phase256_bootstrap')))
    (HERE / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('ALL CHECKS PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
