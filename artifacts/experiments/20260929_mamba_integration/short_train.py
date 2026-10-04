"""Source-bound state/RNG-resume 18k six-frame fmamba+corrg run; recipe copied from 20260927_levers/tworld.py.

All models use the same window IDs, heldout rule, batch RNG, optimizer, loss and schedule. The
only treatment is the frozen encoder whose patch tokens populate the pool. Source and input hashes
are bound at start; a resume fails on drift. Saves atomic optimizer/RNG state every 500 updates.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE = Path(__file__).resolve().parent
LEVERS = ROOT / 'artifacts/experiments/20260927_levers'
SPATIAL = ROOT / 'artifacts/experiments/20260921_readout_ladder'
sys.path[:0] = [str(ROOT), str(LEVERS), str(SPATIAL)]
import tworld as T  # noqa: E402
import spatial as S  # noqa: E402
from d4mj.config import config_from_dict  # noqa: E402
from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer  # noqa: E402

OUT = ROOT / 'artifacts/eda/levers_mamba_integration_v1'
SOURCES = [HERE / 'short_train.py', LEVERS / 'tworld.py', SPATIAL / 'spatial.py', ROOT / 'd4mj/train.py',
           ROOT / 'd4mj/mamba_recurrence.py']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def state(world, opt, order, history, update, contract):
    return {'world': world.state_dict(), 'optimizer': opt.state_dict(), 'order_rng': order.get_state(),
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'history': history, 'update': update, 'contract': contract}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool', required=True, choices=('raw', 'ldad1', 'ldad10'))
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--total', type=int, default=18000)
    p.add_argument('--until', type=int, default=None, help='For exact-resume smoke; full schedule remains --total')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    assert 0 < (a.until or a.total) <= a.total
    assert S.W == 6 and S.ANCHOR == 3 and T.BATCH == 40
    torch.backends.cuda.matmul.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    name = f'int_corrg_{a.pool}_suffix_s{a.seed}_fmamba_u{a.total}'
    resume = a.out / f'{name}.resume.pt'
    final = a.out / f'{name}.pt'
    if final.exists():
        raise SystemExit(f'final exists: {final}; refusing overwrite')
    source_hashes = {str(x.relative_to(ROOT)): digest(x) for x in SOURCES}
    pool_path = T.POOLS[a.pool]
    pool_json = json.loads((pool_path / 'pool.json').read_text())
    pool_sha = digest(pool_path / 'pool.pt')
    assert pool_sha == pool_json['pool_sha256'], 'pool content differs from its declared hash'
    cfg_sha = digest(S.CHECKPOINT)
    contract = {'name': name, 'pool': a.pool, 'pool_sha256': pool_sha,
                'encoder_sha256': pool_json.get('encoder_sha256', pool_json['checkpoint_sha256']),
                'config_checkpoint_sha256': cfg_sha, 'source_sha256': source_hashes,
                'total_updates': a.total, 'batch': T.BATCH, 'loss': 'suffix', 'head': 'corrg',
                'backbone': 'fmamba', 'seed': a.seed, 'sampler_seed': 11, 'heldout_seed': 1}
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    pool = torch.load(pool_path / 'pool.pt', weights_only=False, mmap=True)
    assert pool['tokens'].shape[1:] == (6, 81, 192)
    main_rows = torch.where(~pool['terminal'])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool['terminal'])[0]])
    assert len(rows) == len(pool['terminal']) - 2048
    device = torch.device('cuda')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        world = T.TWorld('corrg', backbone='fmamba').to(device)
    opt = phase_optimizer([world], config)
    params = [q for g in opt.param_groups for q in g['params']]
    order = torch.Generator().manual_seed(11)
    history, start = [], 0
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['contract'] == contract, 'source/input/recipe drift on resume'
        world.load_state_dict(saved['world'])
        opt.load_state_dict(saved['optimizer'])
        order.set_state(saved['order_rng'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, start = saved['history'], saved['update']
        print(json.dumps({'stage': 'resume', 'name': name, 'update': start}), flush=True)
    else:
        print(json.dumps({'stage': 'init', 'contract': contract,
                          'parameters': sum(q.numel() for q in world.parameters())}), flush=True)
    stop = a.until or a.total
    began = time.time()
    for update in range(start, stop):
        b = S.batch_of(pool, rows[torch.randint(len(rows), (T.BATCH,), generator=order)], 'tokens', device)
        with autocast_context(config):
            objective = T.rollout_losses(world, b['s'], b['actions'], 'suffix')
        norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
        u = update + 1
        if u % 500 == 0 or u == stop:
            row = {'update': u, 'objective': float(objective.detach()), 'gradient_norm': float(norm),
                   'seconds_since_resume': round(time.time() - began, 1),
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
            history.append(row)
            print(json.dumps({'stage': 'train', 'name': name, **row}), flush=True)
            atomic_save(state(world, opt, order, history, u, contract), resume)
        if u in (6000, 12000, a.total):
            tag = name if u == a.total else f'{name}_at{u}'
            eval_payload = {'name': tag, 'args': {'head': 'corrg', 'pool': a.pool, 'loss': 'suffix',
                            'seed': str(a.seed), 'backbone': 'fmamba', 'regions': 'all'},
                            'world': world.state_dict(), 'history': history, 'contract': contract}
            path = final if u == a.total else a.out / f'{tag}.pt'
            assert not path.exists(), f'checkpoint exists: {path}'
            atomic_save(eval_payload, path)
            print(json.dumps({'stage': 'snapshot', 'name': tag, 'sha256': digest(path)}), flush=True)
    print(json.dumps({'stage': 'complete' if stop == a.total else 'paused', 'name': name,
                      'update': stop, 'seconds_since_resume': round(time.time() - began, 1)}), flush=True)


if __name__ == '__main__':
    main()
