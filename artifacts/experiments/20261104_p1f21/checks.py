"""Bounded actual-path preflight for an F2 arm (F2.1 here; F2.2 imports `run`); no efficacy reading, no launch.
F2's checks (../20261103_p1f2/checks.py) already qualified the trainer hooks and chooser; these repeat what the arm
changes can break: exact resume (streams over the arm's episode list, chooser, optimizer, emitted gaze), the real
launcher in a fresh process, sustained 64-step and 256-step updates (finite, speed, memory), the monitor."""
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
equal, optimizer_error = F2C.equal, F2C.optimizer_error


def run(L, here, module, prepare=lambda cfg: None, extra=lambda cfg, out: {}):
    """L: the arm's launcher module; module: its import path (the real-launcher check); prepare(cfg): process setup
    before any data access (F2.1: union.use); extra(cfg, out): the arm's own checks (dict, merged)."""
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    out = here / 'checks'
    if out.exists():
        raise RuntimeError('Preserve previous checks')
    out.mkdir(); torch.set_num_threads(4); start = time.time(); result = {'passed': False}
    cfg = L.check_recipe(L.recipe(TR.load_cfg())); prepare(cfg)
    small = copy.deepcopy(cfg); small['p1']['frames'] = 20480; small['p1']['monitor']['windows'] = 4
    for name, updates in (('straight', (4,)), ('resumed', (2, 4))):
        for u in updates:
            model = L.T.p1(small, 1, out / name, updates=u, log_every=1); del model; torch.cuda.empty_cache()
    a, b = (torch.load(out / n / 'ckpt_02.pt', map_location='cpu', weights_only=False) for n in ('straight', 'resumed'))
    result['learned_resume'] = {k: bool(equal(a[k], b[k])) for k in ('model', 'streams', 'carry', 'cache', 'state', 'rng', 'fixation_controller')}
    result['learned_resume_optimizer_max_abs'] = optimizer_error(a['opt'], b['opt'])
    result['learned_resume']['gaze_bytes'] = (out / 'straight/gaze.bin').read_bytes() == (out / 'resumed/gaze.bin').read_bytes()
    assert all(result['learned_resume'].values()) and result['learned_resume_optimizer_max_abs'] < 1e-11, result
    del a, b
    subprocess.run([sys.executable, '-u', '-m', module, str(out / 'launcher'), '--updates', '2', '--no-checkpoints'],
                   cwd=TR.ROOT, check=True)
    for name, u, steps in (('sustained64', 200, 64), ('phase256', 3, 256)):
        model = L.T.p1(cfg, 1, out / name, updates=u, steps=steps, checkpoints=False, log_every=25 if steps == 64 else 1)
        rows = [json.loads(x) for x in (out / name / 'log.jsonl').read_text().splitlines()]
        aux = [json.loads(x) for x in (out / name / 'chooser.jsonl').read_text().splitlines()]
        assert len(aux) == u and all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
        assert all(np.isfinite(r['head_loss']) for r in aux if r['head_loss'] is not None)
        result[name] = dict(updates=u, last=rows[-1], head_updates=aux[-1]['head_updates'], head_loss=aux[-1]['head_loss'],
                            gaze_changed_fraction=float(np.mean([r['changed_fraction'] for r in aux])),
                            mean_final_half_sec_per_update=float(np.mean([r['sec_per_update'] for r in rows[len(rows) // 2:]])))
        if name == 'sustained64':
            result['monitor'] = TR.monitor(model, cfg, D.load_split(cfg)['monitor'])
            result.update(extra(cfg, out / name))
        del model; torch.cuda.empty_cache()
    result.update(passed=True, elapsed_seconds=time.time() - start, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    (here / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('ALL CHECKS PASSED', json.dumps(result), flush=True)


def composition(cfg, out):
    """Share of the sustained run's trained frames (the gaze trace's frame rows) from each source store."""
    C = importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
    rows = np.fromfile(out / 'gaze.bin', dtype=C.TRACE)['target']
    rows = rows[rows >= 0]                                               # episode-first frames carry no target row
    split, m = D.load_split(cfg), D.RP.manifest()
    off = D.frame_offsets(split['train'])
    src = np.array([m['shards'][e.shard]['source'] for e in split['train']])[np.searchsorted(off, rows, side='right') - 1]
    expected = {s: float(sum(e.T + 1 for e in split['train'] if m['shards'][e.shard]['source'] == s) / off[-1]) for s in np.unique(src)}
    return {'trained_frames_by_source': {s: float(np.mean(src == s)) for s in np.unique(src)},
            'union_frame_share_by_source': expected, 'trace_rows': int(len(rows))}


def main():
    L = importlib.import_module('artifacts.experiments.20261104_p1f21.train_p1f21')
    here = Path(__file__).resolve().parent
    if not json.loads((here / 'union_checks.json').read_text()).get('passed'):
        raise RuntimeError('union.py verify has not passed')
    run(L, here, 'artifacts.experiments.20261104_p1f21.train_p1f21', prepare=L.U.use, extra=composition)


if __name__ == '__main__':
    main()
