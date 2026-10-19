"""Does the current tworld.py, with every new flag off, train exactly as the code that trained M6 and M16? (2026-10-08 night;
precondition of the from-scratch replication, NOTEBOOK "Could a negative come from continuing".)
Sources are identified by the sha256 recorded in each checkpoint: M6 (corrt_raw_teacher_s7_fmamba_u36000) = tworld.py @60b0c22f,
M16 (corrt_rawlong_teacher_s7_fmamba_L16b40_from36000) = tworld.py @ae0868ec (git show; hashes re-checked here).
On CPU with the Mamba reference backend in every world (identical in old and new; the Triton kernel is the same code in both):
  stage1   TWorld(corrt, fmamba) built as tworld.train builds it (seed 7): every parameter equal; tworld.train's first raw-pool batch
           (held split seed 1, order seed 11; its first 8 windows): teacher loss and every gradient, new vs @60b0c22f
  stage2   TWorld(corrt, fmamba, frames 16) initialized from M6's weights as tworld.train does (--init): parameters equal; the first
           rawlong batch (order seed 11, TERMINAL_SHARE; first 8 windows): teacher loss and every gradient, new vs @ae0868ec
Exit code 1 unless every comparison is exactly equal. -> artifacts/eda/health_chain_v1/tworld_equivalence.json
Usage: tworld_equivalence.py
"""
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
SCRATCH = ROOT / 'artifacts/eda/health_chain_v1/equivalence_sources'
M6 = ROOT / 'artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_u36000.pt'
M16 = ROOT / 'artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt'


def module_at(rev, recorded):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    src = subprocess.run(['git', 'show', f'{rev}:artifacts/experiments/20260927_levers/tworld.py'], cwd=ROOT,
                         capture_output=True, check=True).stdout
    assert hashlib.sha256(src).hexdigest() == recorded, (rev, recorded)
    path = SCRATCH / f'tworld_{rev}.py'
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(f'tworld_{rev}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(world):
    for layer in getattr(world, 'layers', []):
        layer.mix.settings = dataclasses.replace(layer.mix.settings, backend='reference')
    return world


def build(mod, frames):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        args = ('corrt', None, 'fmamba', 'all', 0, False) + ((frames,) if frames is not None else ())   # as each source's train()
        return reference(mod.TWorld(*args))


def compare(a, b, s, act, mod_a, mod_b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    out = {'same_parameter_names': sorted(pa) == sorted(pb),
           'max_parameter_difference': max(float((pa[k] - pb[k]).abs().max()) for k in pa)}
    la = mod_a.rollout_losses(a, s, act, 'teacher'); la.backward()
    lb = mod_b.rollout_losses(b, s, act, 'teacher'); lb.backward()
    out['loss'] = [float(la), float(lb)]
    none_a = sorted(k for k in pa if pa[k].grad is None); none_b = sorted(k for k in pb if pb[k].grad is None)
    out['parameters_without_gradient'] = none_a
    out['same_gradient_support'] = none_a == none_b
    out['max_gradient_difference'] = max(float((pa[k].grad - pb[k].grad).abs().max()) for k in pa if pa[k].grad is not None)
    out['exact'] = out['same_parameter_names'] and out['max_parameter_difference'] == 0 and float(la) == float(lb) \
        and out['same_gradient_support'] and out['max_gradient_difference'] == 0
    return out


def main():
    import tworld as NEW
    import spatial as S
    old6 = module_at('60b0c22f', torch.load(M6, map_location='cpu', weights_only=False)['script_sha256'])
    old16 = module_at('ae0868ec', torch.load(M16, map_location='cpu', weights_only=False)['script_sha256'])
    res = {'current_sha256': hashlib.sha256((HERE / 'tworld.py').read_bytes()).hexdigest()}
    # stage 1: raw pool, tworld.train's first batch
    pool = torch.load(NEW.POOLS['raw'] / 'pool.pt', weights_only=False, mmap=True)
    main_rows = torch.where(~pool['terminal'])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool['terminal'])[0]])
    order = torch.Generator().manual_seed(11)
    idx = rows[torch.randint(len(rows), (NEW.BATCH,), generator=order)][:8]
    b = S.batch_of(pool, idx, 'tokens', torch.device('cpu'))
    res['stage1_vs_60b0c22f'] = compare(build(NEW, None), build(old6, None), b['s'].float(), b['actions'], NEW, old6)
    print(json.dumps({'stage1': res['stage1_vs_60b0c22f']}), flush=True)
    # stage 2: rawlong pool, init from M6 as tworld.train does
    w0 = torch.load(M6, map_location='cpu', weights_only=False)['world']
    worlds = []
    for mod in (NEW, old16):
        w = build(mod, 16)
        init = dict(w0)
        init['time'] = init['time'][torch.arange(w.time.shape[0]) % init['time'].shape[0]]
        w.load_state_dict(init)
        worlds.append(w)
    lab = torch.load(NEW.POOLS['rawlong'] / 'labels.pt', weights_only=False, mmap=True)
    tokens = np.memmap(NEW.POOLS['rawlong'] / 'tokens.f16', dtype=np.float16, mode='r', shape=(len(lab['terminal']), 64, 81, 192))
    main_rows, term_rows = torch.where(~lab['terminal'])[0], torch.where(lab['terminal'])[0]
    order = torch.Generator().manual_seed(11)
    term = torch.rand(40, generator=order) < NEW.TERMINAL_SHARE
    r = torch.where(term, term_rows[torch.randint(len(term_rows), (40,), generator=order)],
                    main_rows[torch.randint(len(main_rows), (40,), generator=order)])
    t0 = torch.where(term, 64 - 16, torch.randint(0, 49, (40,), generator=order))
    s = torch.stack([torch.from_numpy(np.array(tokens[int(i), int(j):int(j) + 16])) for i, j in zip(r[:8], t0[:8])]).float()
    a = torch.stack([lab['actions'][int(i), int(j):int(j) + 15] for i, j in zip(r[:8], t0[:8])])
    res['stage2_vs_ae0868ec'] = compare(worlds[0], worlds[1], s, a, NEW, old16)
    print(json.dumps({'stage2': res['stage2_vs_ae0868ec']}), flush=True)
    (ROOT / 'artifacts/eda/health_chain_v1/tworld_equivalence.json').write_text(json.dumps(res, indent=1))
    return 0 if res['stage1_vs_60b0c22f']['exact'] and res['stage2_vs_ae0868ec']['exact'] else 1


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    torch.set_num_threads(8)
    raise SystemExit(main())
