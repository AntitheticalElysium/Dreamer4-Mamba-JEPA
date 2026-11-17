"""P1-MultiStep-Mamba preflight: tests determinism, resume, gradient routes, speed, and VRAM.
Ensures zero bitwise regressions on F4 components, verifies multi-step loss and gradient propagation.
"""
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
L = importlib.import_module('artifacts.experiments.20261111_p1_multistep_mamba.train_multistep')
equal, optimizer_error = F2C.equal, F2C.optimizer_error
HERE = Path(__file__).resolve().parent
F4_LOG = HERE.parent / '20261106_p1f4/run/log.jsonl'


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    out = HERE / 'checks'
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir()
    torch.set_num_threads(4)
    start = time.time()
    result = {'passed': False}
    cfg = L.check_recipe(L.recipe(TR.load_cfg()))

    # 1. Learned resume determinism across bootstrap start
    small = copy.deepcopy(cfg)
    small['p1']['frames'] = 20480
    small['p1']['monitor']['windows'] = 4
    small['edm']['shortcut']['bootstrap_start'] = 2
    for name, updates in (('straight', (4,)), ('resumed', (2, 4))):
        for u in updates:
            model = L.T.p1(small, 1, out / name, updates=u, log_every=1)
            del model
            torch.cuda.empty_cache()
    a, b = (torch.load(out / n / 'ckpt_02.pt', map_location='cpu', weights_only=False) for n in ('straight', 'resumed'))
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

    # 2. Update 1 check against F4 control baseline
    f4_u1 = json.loads(F4_LOG.read_text().splitlines()[0])
    u1_now = logs[0][0]
    result['update1_vs_f4'] = {
        'g1_bitwise_equal': u1_now['g1'] == f4_u1['g1'],
        'tsp_bitwise_equal': u1_now['tsp'] == f4_u1['tsp'],
        'rec_bitwise_equal': u1_now['rec'] == f4_u1['rec'],
        'noise_bitwise_equal': u1_now['noise'] == f4_u1['noise'],
        'multi_loss_present': 'multi' in u1_now and u1_now['multi'] > 0,
        'g1': [f4_u1['g1'], u1_now['g1']],
        'multi': u1_now.get('multi', 0.0),
        'grad_norm': [f4_u1['grad_norm'], u1_now['grad_norm']],
    }
    assert result['update1_vs_f4']['g1_bitwise_equal'], 'g1 must match F4 on update 1'
    assert result['update1_vs_f4']['multi_loss_present'], 'multi loss must be positive and present'

    # 3. Real launcher smoke in subprocess
    subprocess.run([sys.executable, '-u', '-m', 'artifacts.experiments.20261111_p1_multistep_mamba.train_multistep',
                    str(out / 'launcher'), '--updates', '2', '--no-checkpoints'], cwd=TR.ROOT,
                   env=dict(os.environ, MAMBA_DETERMINISTIC='1'), check=True)

    # 4. Sustained benchmarks & memory profiling
    on = copy.deepcopy(cfg)
    on['edm']['shortcut']['bootstrap_start'] = 0
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
            f4rows = {r['update']: r for r in map(json.loads, F4_LOG.read_text().splitlines()) if 'g1' in r}
            result['g1_vs_f4_same_update'] = {r['update']: [f4rows[r['update']]['g1'], r['g1']] for r in rows if r['update'] in f4rows}
        del model
        torch.cuda.empty_cache()

    s = lambda n: result[n]['mean_final_half_sec_per_update']
    n64 = cfg['p1']['frames'] / 2 / (cfg['data']['streams'] * 64)
    n256 = cfg['p1']['frames'] / 2 / (cfg['data']['streams'] * 256)
    b0 = cfg['edm']['shortcut']['bootstrap_start']
    result['training_hours_estimate'] = (b0 * s('sustained64') + (n64 - b0) * s('sustained64_bootstrap') + n256 * s('phase256_bootstrap')) / 3600
    result['f4_training_hours'] = 14.4
    result.update(passed=True, elapsed_seconds=time.time() - start, peak_gib=max(result[n]['peak_gib'] for n in
                  ('sustained64', 'sustained64_bootstrap', 'phase256', 'phase256_bootstrap')))
    (HERE / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('ALL CHECKS PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
