"""E19: matched continuation / target-preserving boundaries / boundaries + health dose.

All arms see the same 40 original six-frame windows per update and all five factual
transitions. B/C split EVERY window (terminal and nonterminal) at a uniform boundary
start=4-r, r in 0..4. A predicts the original window. Prefix and suffix each start
with an empty recurrent state/time row zero; shared boundary frames have no duplicate
target. Loss normalization remains B*5*81*192. C adds lambda times the mask-normalized
L1 of health token63 where the stored TRAIN-only next-frame zombie context is true.
This is a boundary/context intervention, not an isolated positional-embedding test.
The target-derived mask is training-only, never an inference input.

Automatic resume binds code/data/parent/recipe/runtime and saves full optimizer,
sampler, boundary RNG, CPU/CUDA RNG, sampled-row/boundary ledger and update atomically.
Outputs load through the existing teval.load_world; no fork labels train these worlds.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tworld as T
import h16_resume as R

MASK = T.ROOT / 'artifacts/eda/hpctx_labels_v1.pt'


def segments(r):
    start = 4 - int(r)
    return ([(0, start)] if start else []) + [(start, 5)]


def objective(world, s, a, mask, offsets, arm, dose=1.0):
    """Return differentiable scalar and detached components with exact target coverage."""
    if s.shape[1:] != (6, 81, 192) or a.shape != (len(s), 5):
        raise ValueError('E19 requires original six-frame/five-action windows')
    if mask.shape != a.shape or offsets.shape != (len(s),):
        raise ValueError('Mask/boundary shape mismatch')
    denominator = s[:, 1:].numel()
    if arm == 'A':
        # Literal historical expression/order, including its final unused output.
        value = T.rollout_losses(world, s, a, 'teacher')
        return value, {'teacher': float(value.detach()), 'health': 0.0,
                       'selected_health_targets': int(mask.sum())}
    total = s.new_zeros((), dtype=torch.float32)
    hp = s.new_zeros((), dtype=torch.float32)
    for r in range(5):
        rows = torch.where(offsets == r)[0]
        if not len(rows):
            continue
        for lo, hi in segments(r):
            src = s[rows, lo:hi + 1]
            pred = world(src, F.pad(a[rows, lo:hi], (0, 1)))[0][:, :hi - lo]
            err = (pred - s[rows, lo + 1:hi + 1]).abs()
            total = total + err.sum()
            if arm == 'C':
                hp = hp + (err[:, :, 63] * mask[rows, lo:hi, None]).sum()
    teacher = total / denominator
    health = hp / (mask.sum().clamp_min(1) * s.shape[-1])
    value = teacher + (dose * health if arm == 'C' else 0)
    return value, {'teacher': float(teacher.detach()), 'health': float(health.detach()),
                   'selected_health_targets': int(mask.sum())}


def train_rows(pool):
    main = torch.where(~pool['terminal'])[0]
    held = main[torch.randperm(len(main), generator=torch.Generator().manual_seed(1))[:2048]]
    return torch.cat([main[~torch.isin(main, held)], torch.where(pool['terminal'])[0]])


def config():
    from d4mj.config import config_from_dict
    return config_from_dict(torch.load(T.S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])


def make_world(parent, backbone, device):
    st = torch.load(parent, map_location='cpu', weights_only=False)
    if st['args']['head'] != 'corrt' or st['args'].get('backbone', 'full') != backbone:
        raise ValueError('Parent must be the matched corrt backbone')
    w = T.TWorld('corrt', backbone=backbone).to(device)
    w.load_state_dict(st['world'])
    return w


def numeric_sources():
    files = set(T.ROOT.joinpath('d4mj').rglob('*.py'))
    files.update(HERE / (name + '.py') for name in ('e19', 'tworld', 'scroll', 'h16_resume'))
    files.add(Path(T.S.__file__).resolve())
    # Bind the actual local Mamba and custom convolution implementation as well.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and filename.endswith('.py'):
            p = Path(filename).resolve()
            if p.is_file() and p.is_relative_to(T.ROOT) and '.venv' not in p.relative_to(T.ROOT).parts:
                files.add(p)
    return {str(p): R.file_hash(p) for p in sorted(files)}


def contract(args, parent, rows):
    return {'version': 'e19-target-preserving-v1', 'arm': args.arm, 'seed': args.seed,
            'backbone': args.backbone, 'updates': args.updates, 'batch': T.BATCH,
            'dose': args.dose, 'sampler_seed': 11, 'boundary_seed': 19,
            'boundary_on_all_windows': True, 'targets_per_window': 5, 'hp_token': 63,
            'source': numeric_sources(), 'rows': R.tensor_hash(rows),
            'inputs': {str(p): R.file_hash(p) for p in
                       (parent, T.POOLS['raw'] / 'pool.pt', MASK, T.S.CHECKPOINT)},
            'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0), 'tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'triton_f32': os.environ.get('TRITON_F32_DEFAULT'),
                        'config': str(config())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--backbone', choices=['fmamba', 'full'], default='fmamba')
    p.add_argument('--seed', type=int, choices=[7, 8], default=7)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--dose', type=float, default=1.0)
    p.add_argument('--state-every', type=int, default=500)
    args = p.parse_args()
    if args.updates <= 0 or args.state_every <= 0 or args.dose != 1.0:
        p.error('Positive budget/save interval and predeclared dose1 required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    from d4mj.train import phase_optimizer, optimizer_step, autocast_context, _phase_lr
    parent_name = f'corrt_raw_teacher_s{args.seed}' + ('_fmamba' if args.backbone == 'fmamba' else '') + '_u36000'
    parent = T.OUT / (parent_name + '.pt')
    name = f'e19_{args.arm}_s{args.seed}_{args.backbone}_from36000'
    final = T.OUT / (name + '.pt')
    pool = torch.load(T.POOLS['raw'] / 'pool.pt', mmap=True, weights_only=False)
    mask = torch.load(MASK, weights_only=False)['beside'].bool()
    rows = train_rows(pool)
    if mask.shape != (len(pool['terminal']), 5):
        raise ValueError('Health labels do not align with original pool')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        world = make_world(parent, args.backbone, device)
    cfg = config()
    opt = phase_optimizer([world], cfg)
    params = [q for group in opt.param_groups for q in group['params']]
    order = torch.Generator().manual_seed(11)
    boundaries = torch.Generator().manual_seed(19)
    spec = contract(args, parent, rows)
    store = R.Store(T.OUT / 'state' / name, spec)
    started = time.monotonic()
    def log(**kw):
        print(json.dumps({'name': name, 'seconds_session': round(time.monotonic() - started, 2), **kw}), flush=True)
    with store.lock():
        complete = store.load('complete')
        if complete is not None:
            if not final.exists() or R.file_hash(final) != complete['sha256']:
                raise RuntimeError('Final checkpoint missing/changed after completion')
            log(stage='complete_exists', update=args.updates)
            return
        if final.exists():
            # A crash between final write and completion journal is safe to recover.
            st = torch.load(final, map_location='cpu', weights_only=False)
            if st.get('contract') != store.contract or st.get('update') != args.updates:
                raise RuntimeError('Unbound or incomplete final checkpoint')
            store.save('complete', {'sha256': R.file_hash(final)}, 1)
            log(stage='complete_recovered', update=args.updates)
            return
        ledger = torch.empty(args.updates, T.BATCH, dtype=torch.int32)
        layout = torch.empty(args.updates, T.BATCH, dtype=torch.int8)
        history, start, coverage = [], 0, torch.zeros(5, dtype=torch.long)
        death_rows = torch.zeros(5, dtype=torch.long)
        checkpoint = store.load('train')
        if checkpoint is not None:
            world.load_state_dict(checkpoint['world'])
            opt.load_state_dict(checkpoint['optimizer'])
            order.set_state(checkpoint['order'])
            boundaries.set_state(checkpoint['boundaries'])
            R.restore_rng(checkpoint['rng'], device)
            start, history = checkpoint['update'], checkpoint['history']
            ledger[:start], layout[:start] = checkpoint['ledger'], checkpoint['layout']
            coverage, death_rows = checkpoint['coverage'], checkpoint['death_rows']
            log(stage='resume', update=start)
        else:
            log(stage='init', parent=str(parent), contract=store.contract,
                parameters=sum(q.numel() for q in world.parameters()), train_windows=len(rows))
        world.train()
        for u in range(start, args.updates):
            idx = rows[torch.randint(len(rows), (T.BATCH,), generator=order)]
            offsets = torch.randint(5, (T.BATCH,), generator=boundaries)
            ledger[u], layout[u] = idx, offsets
            coverage += torch.bincount(offsets, minlength=5)
            # Actual death labels, not selected-terminal flag (some main rows also end in death).
            dead = pool['alive'][idx, :-1] & ~pool['alive'][idx, 1:]
            # All current deaths are final; refuse silent changes to this assumption.
            if dead[:, :4].any():
                raise RuntimeError('Pool terminal layout changed')
            dr = offsets if args.arm != 'A' else torch.full_like(offsets, 4)
            death_rows += torch.bincount(dr[dead[:, 4]], minlength=5)
            b = T.S.batch_of(pool, idx, 'tokens', device)
            with autocast_context(cfg):
                loss, parts = objective(world, b['s'], b['actions'], mask[idx].to(device),
                                        offsets.to(device), args.arm, args.dose)
            norm = optimizer_step(opt, loss, params, learning_rate=_phase_lr(cfg, u),
                                  grad_clip=cfg.agent.grad_clip, strict=True, zero_grad=True)
            if (u + 1) % 100 == 0 or u == start:
                row = {'update': u + 1, 'objective': float(loss.detach()), 'gradient_norm': float(norm),
                       'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3), **parts}
                history.append(row)
                log(stage='train', **row)
            if (u + 1) % args.state_every == 0 or u + 1 == args.updates:
                store.save('train', {'update': u + 1, 'world': world.state_dict(), 'optimizer': opt.state_dict(),
                                    'order': order.get_state(), 'boundaries': boundaries.get_state(),
                                    'rng': R.rng_state(device), 'history': history, 'ledger': ledger[:u + 1].clone(),
                                    'layout': layout[:u + 1].clone(), 'coverage': coverage, 'death_rows': death_rows}, u + 1)
                log(stage='checkpoint', update=u + 1, boundary_counts=coverage.tolist(), death_rows=death_rows.tolist())
        R.atomic_torch(final, {'name': name, 'args': {'head': 'corrt', 'pool': 'raw', 'loss': 'teacher',
                           'seed': str(args.seed), 'backbone': args.backbone, 'regions': 'all', 'skip': 'False',
                           'frames': 'None', 'e19_arm': args.arm, 'updates': str(args.updates)},
                           'world': world.state_dict(), 'history': history, 'update': args.updates,
                           'contract': store.contract, 'script_sha256': spec['source'][str(Path(__file__).resolve())],
                           'parent': str(parent)})
        store.save('complete', {'sha256': R.file_hash(final)}, 1)
        log(stage='saved', update=args.updates, path=str(final))


if __name__ == '__main__':
    main()
