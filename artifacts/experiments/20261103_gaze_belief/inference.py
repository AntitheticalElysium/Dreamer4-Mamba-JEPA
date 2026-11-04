"""Inference only: full precision E3 errors and physically fixed renderer repeats."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch

X = importlib.import_module('artifacts.experiments.20261103_chooser_development.campaign')
C, Q = X.C, X.Q
from saccade import model as M, train as TR
HERE = Path(__file__).resolve().parent
READ = HERE.parent/'20261102_p1f15_readiness'


@torch.no_grad()
def main():
    start = time.time()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device('cuda')
    z = torch.load(C.OLD/'dev_600_coarse.pt', map_location='cpu', weights_only=False)
    base = Q.Data(z, torch.device('cpu'))
    cohorts = C.cohorts()
    ckpath = C.CKDIR/'ckpt_10.pt'
    assert z['checkpoint_sha256'] == C.sha(ckpath)
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    m = M.M0(ck['cfg']).to(dev)
    m.load_state_dict(ck['model']); m.eval().requires_grad_(False)
    sr = dict(np.load(READ/'static_repeats.npz'))
    frames = torch.tensor(sr['frames'].reshape(-1, 63, 63, 3), device=dev)
    with TR.autocast(dev, ck['cfg']):
        parts = m.perception(frames, torch.ones(len(frames), 2, device=dev, dtype=torch.long))
    current = parts.p.float().cpu().numpy().reshape(8, 4, 81, 16)
    np.savez_compressed(HERE/'static_percepts.npz', p=current, rows=sr['rows'])
    del m, ck, frames, parts
    aux = {
        'coarse': X.sketch(z),
        'rich': torch.load(C.HERE/'dev_preaction_belief.pt', map_location='cpu', weights_only=False)['aux']
    }
    rows = sr['rows']; ids = base.row_ids(rows[:, 0], rows[:, 1])
    models = []
    for seed in X.SEEDS:
        models.extend([(f'e1_{seed}', C.HERE/f'e1_{seed}_final.pt', None),
                       (f'e2_carry_{seed}', C.HERE/f'e2_carry_{seed}_ck10.pt', None)])
        models.extend([(f'e3_{kind}_{seed}', C.HERE/f'e3_{kind}_{seed}.pt', kind)
                       for kind in ('coarse', 'rich')])
    identity = {'checkpoint_sha256': C.sha(ckpath), 'models': {}, 'archive_checks': {}}
    for name, path, kind in models:
        state = torch.load(path, map_location='cpu', weights_only=False)
        h = (X.AuxHead() if kind else Q.Head()).to(dev)
        h.load_state_dict(state['state']); h.eval().requires_grad_(False)
        data = (X.AuxData(z, torch.device('cpu'), aux[kind], state['mean'].cpu(), state['std'].cpu())
                if kind else base)
        def forward(indices):
            inp = tuple(x.to(dev) for x in data.inputs(indices))
            p = h(inp)
            n = h((*inp[:2], torch.zeros_like(inp[2]), *inp[3:]))
            return p.float().cpu().numpy(), n.float().cpu().numpy()
        pp, nn = forward(ids)
        cell_c = ((pp-nn)**2).sum(-1)
        cell_s = ((current-pp[:, None])**2).sum(-1)
        sw = cell_s[:, :, C.WIN].sum(-1)
        cw = np.broadcast_to(cell_c[:, C.WIN].sum(-1)[:, None], sw.shape).copy()
        assert np.array_equal(cw[:, 0], cw[:, -1])
        np.savez_compressed(HERE/(name+'_static.npz'), C=cw, S=sw, rows=rows)
        if kind:
            for cohort, meta in cohorts.items():
                ii = data.row_ids(meta['ep'], meta['t'])
                p, n = zip(*(forward(batch) for batch in ii.split(256)))
                p, n = np.concatenate(p), np.concatenate(n)
                actual = z['p'][ii].float().numpy()
                cell_s = ((p-actual)**2).sum(-1)
                cell_c = ((p-n)**2).sum(-1)
                ar = np.arange(len(ii)); co = np.maximum(meta['counter'], 0)
                old = dict(np.load(C.HERE/f'{name}_endpoint_{cohort}.npz'))
                checks = {
                    'mse': float(np.max(np.abs(cell_s.mean(-1)/16-old['mse']))),
                    'C': float(np.max(np.abs(cell_c[:, C.WIN].sum(-1)-old['C']))),
                    'S': float(np.max(np.abs(cell_s[:, C.WIN].sum(-1)-old['S']))),
                    'counter_mse': float(np.max(np.abs(cell_s[ar, co]/16-old['counter_mse'])))
                }
                assert checks['mse'] < 2e-6 and checks['C'] < 2e-4 and checks['S'] < 2e-4, checks
                assert checks['counter_mse'] < 2e-6, checks
                action = z['prev_action'][ii].numpy()
                assert (cell_c[action == 0] == 0).all()
                np.savez_compressed(HERE/(name+'_'+cohort+'_cells.npz'),
                                    cell_C=cell_c, cell_S=cell_s, **meta)
                identity['archive_checks'][name+'_'+cohort] = checks
        identity['models'][name] = C.sha(path)
        print('inference', name, round(time.time()-start, 1), flush=True)
        del h, data
    identity.update(seconds=time.time()-start, peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    (HERE/'inference_identity.json').write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    main()
