"""Two matched direct-generator objectives; exact serialized-state resume.

Training reads cached TRAIN observations/actions and an RGB-first-frame ledger.
It never reads evaluation fans, simulator labels, semantic readers or codebooks.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

import numpy as np
import torch
from torch.nn import functional as F

import direct_data as D
import direct_model as M
import loss_intervention as L
from d4mj.config import config_from_dict
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'direct_generator'
PREREG = Path(__file__).with_name('DIRECT_PREREG.md')
PARENT = L.OUT / 'worlds/control_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt'


def settings():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert os.environ['TRITON_F32_DEFAULT'] == 'ieee'
    return config_from_dict(torch.load(L.TW.S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])


def initialized_worlds(device):
    torch.manual_seed(7)
    template = M.DirectWorld()
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert parent['args']['arm'] == 'control'
    assert all(torch.count_nonzero(x) == 0 for x in parent['oracle'].values())
    missing = template.load_state_dict(parent['world'], strict=False)
    assert not missing.unexpected_keys and all(x.startswith('completion.') for x in missing.missing_keys)
    assert len(missing.missing_keys) == len(template.completion.state_dict())
    return {arm: copy.deepcopy(template).to(device).train() for arm in M.ARMS}


def load_data():
    manifest, windows = D.read_manifest()
    pool_manifest = json.loads((D.POOL / 'manifest.json').read_text())
    pool = torch.load(D.POOL / 'labels.pt', map_location='cpu', weights_only=False, mmap=True)
    path = D.POOL / 'tokens.f16'
    verified = json.loads((L.OUT / 'generated_prefix/token_verification.json').read_text())
    assert verified == {'sha256': pool_manifest['tokens_sha256'], 'size': path.stat().st_size,
                        'mtime_ns': path.stat().st_mtime_ns}
    tokens = np.memmap(path, dtype=np.float16, mode='r', shape=(len(pool['terminal']), 64, 81, 192))
    return manifest, windows, pool['actions'], tokens


def batch(tokens, actions, windows, indices, device='cuda'):
    choices = windows[indices]
    frames = torch.stack([torch.from_numpy(np.array(tokens[int(r), int(s):int(s) + 16]))
                          for r, s in choices]).to(device).float()
    ac = torch.stack([actions[int(r), int(s):int(s) + 15] for r, s in choices]).to(device)
    return frames, F.pad(ac, (0, 1)), choices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--directory', type=Path, default=OUT / 'training')
    a = p.parse_args()
    assert a.updates > 0 and a.batch > 0 and PREREG.exists()
    config = settings()
    manifest, windows, actions, tokens = load_data()
    criteria_path = OUT / 'criteria.json'
    criteria = json.loads(criteria_path.read_text())
    assert criteria['prepared_from_real_only'] and criteria['primary_checkpoint'] == a.updates
    assert all(L.sha(L.ROOT / name) == digest for name, digest in criteria['sources'].items())
    source = [Path(__file__), Path(M.__file__), Path(D.__file__), Path(L.__file__), Path(L.TW.__file__),
              Path(L.TW.S.__file__), L.ROOT / 'd4mj/train.py', L.ROOT / 'd4mj/mamba_recurrence.py',
              PREREG, criteria_path, PARENT, L.TW.S.CHECKPOINT, D.OUT / 'manifest.json', D.OUT / 'windows.pt']
    contract = {'format': 'direct_two_arm_v1', 'sources_inputs': {str(x.relative_to(L.ROOT)): L.sha(x) for x in source},
                'data_contract_id': manifest['contract_id'], 'criteria_id': criteria['criteria_id'],
                'updates': a.updates, 'batch': a.batch,
                'arms': list(M.ARMS), 'input': 'frozen Raw 81x192 cached TRAIN observations and recorded actions',
                'backbone': 'same trainable corrt/fmamba parent; one live causal backbone call per arm/update',
                'generator': 'one spatial attention block plus FF; local4 and shared4 Gaussian noise; route and generation',
                'targets_per_window': 15, 'scored_tokens': list(M.INTERIOR), 'patches': [list(x) for x in M.PATCHES],
                'loss': 'point=.5d(X1,Y)+.5d(X2,Y); energy=point-.5d(X1,X2); exponent1; fixed .5global+.5local',
                'sampler': 'uniform eligible RGB-first-view windows; no terminal oversampling',
                'seeds': {'initialization': 7, 'windows': 20261014, 'noise': 2026101401},
                'precision': 'BF16/IEEE with FP32 output distances; TF32 disabled',
                'training_semantic_or_simulator_labels': False, 'scope': 'one-step controlled-subset completion, one seed'}
    contract = json.loads(json.dumps(contract))
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = a.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'contract.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract, 'source/input/preregistration drift'
    else:
        L.atomic(contract, path, True)
    worlds = initialized_worlds('cuda')
    optimizers = {arm: phase_optimizer([world], config) for arm, world in worlds.items()}
    order = torch.Generator().manual_seed(20261014)
    noise_rng = torch.Generator().manual_seed(2026101401)
    start, chain, history = 0, 'initial', []
    totals = {arm: {'steps': 0, 'targets': 0, 'backbone_seconds': 0., 'head_seconds': 0.,
                    'optimizer_seconds': 0., 'gradient_clipped_steps': 0} for arm in M.ARMS}
    state = directory / 'state.pt'
    if state.exists():
        saved = torch.load(state, map_location='cpu', weights_only=False)
        assert saved['contract_id'] == cid
        for arm in M.ARMS:
            worlds[arm].load_state_dict(saved['worlds'][arm])
            optimizers[arm].load_state_dict(saved['optimizers'][arm])
            assert L.restored_equal(L.cpu_state(worlds[arm]), saved['worlds'][arm])
            assert L.restored_equal(optimizers[arm].state_dict(), saved['optimizers'][arm])
        order.set_state(saved['order'])
        noise_rng.set_state(saved['noise_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, chain, history, totals = saved['update'], saved['ledger_chain'], saved['history'], saved['totals']
        print(json.dumps({'stage': 'resume', 'update': start, 'serialized_state_exact': True}), flush=True)
    if a.stop_after and start >= a.stop_after:
        return

    def save(update, snapshot=False):
        model_states = {arm: L.cpu_state(world) for arm, world in worlds.items()}
        L.atomic({'contract_id': cid, 'update': update, 'worlds': model_states,
                  'optimizers': {arm: copy.deepcopy(opt.state_dict()) for arm, opt in optimizers.items()},
                  'order': order.get_state(), 'noise_rng': noise_rng.get_state(),
                  'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                  'ledger_chain': chain, 'history': history, 'totals': totals}, state)
        if snapshot:
            for arm in M.ARMS:
                L.atomic({'contract_id': cid, 'contract': contract, 'update': update, 'arm': arm,
                          'world': model_states[arm], 'ledger_chain': chain}, directory / f'{arm}_u{update:04d}.pt')

    if start == 0:
        save(0, snapshot=True)
    started = time.perf_counter()
    for update in range(start, a.updates):
        indices = torch.randint(len(windows), (a.batch,), generator=order)
        frames, ac, choices = batch(tokens, actions, windows, indices)
        noise = M.draw_noise(frames.shape[:2], noise_rng).to('cuda')
        raw = choices.numpy().tobytes() + noise.cpu().numpy().tobytes()
        chain = hashlib.sha256((b'' if chain == 'initial' else bytes.fromhex(chain)) + raw).hexdigest()
        shared_cpu_rng, shared_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        records = {}
        for arm in M.ARMS:
            torch.set_rng_state(shared_cpu_rng)
            torch.cuda.set_rng_state_all(shared_cuda_rng)
            world = worlds[arm]
            torch.cuda.synchronize()
            began = time.perf_counter()
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
            torch.cuda.synchronize()
            backbone_seconds = time.perf_counter() - began
            began = time.perf_counter()
            with autocast_context(config):
                outputs = torch.stack([world.complete(frames, ac, belief, ha, n)[0] for n in noise])[:, :, :15]
            values = M.losses(outputs, frames[:, 1:])
            torch.cuda.synchronize()
            head_seconds = time.perf_counter() - began
            began = time.perf_counter()
            norm = float(optimizer_step(optimizers[arm], values[arm], list(world.parameters()),
                                       learning_rate=_phase_lr(config, update), grad_clip=config.agent.grad_clip,
                                       strict=True, zero_grad=True))
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - began
            total = totals[arm]
            total['steps'] += 1
            total['targets'] += a.batch * 15
            total['backbone_seconds'] += backbone_seconds
            total['head_seconds'] += head_seconds
            total['optimizer_seconds'] += optimizer_seconds
            total['gradient_clipped_steps'] += int(norm > config.agent.grad_clip)
            records[arm] = {name: float(value.detach()) for name, value in values.items()}
            records[arm].update(gradient_norm=norm, backbone_seconds=backbone_seconds,
                                two_head_seconds=head_seconds, optimizer_seconds=optimizer_seconds)
        if (update + 1) % 100 == 0 or update == start:
            row = {'update': update + 1, 'arms': records, 'ledger_chain': chain,
                   'active_seconds': time.perf_counter() - started,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
            history.append(row)
            print(json.dumps(row), flush=True)
        snapshot = update + 1 in (500, 1500, a.updates)
        if (update + 1) % 100 == 0 or snapshot or (update + 1) % 25 == 0:
            save(update + 1, snapshot=snapshot)
        if a.stop_after and update + 1 >= a.stop_after:
            save(update + 1)
            return
    L.atomic({'contract_id': cid, 'update': a.updates, 'history': history, 'totals': totals,
              'ledger_chain': chain, 'completed': True}, directory / 'training.json', True)
    print(json.dumps({'stage': 'done', 'update': a.updates, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
