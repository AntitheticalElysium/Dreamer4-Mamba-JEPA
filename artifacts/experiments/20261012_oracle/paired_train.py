"""Matched direct-energy and paired-event training with serialized-state resume.

Only cached TRAIN observations/actions enter optimization. Both arms share the
same factual windows and two independent Gaussian draws at every update.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

import torch

import direct_data as D
import direct_model as DIRECT
import direct_train as BASE
import paired_model as M
import loss_intervention as L
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'paired_event'
PREREG = Path(__file__).with_name('PAIRED_PREREG.md')
PARENT = BASE.PARENT
settings = BASE.settings
load_data = BASE.load_data
batch = BASE.batch
initialized_worlds = M.initialized_worlds


def make_contract(manifest, criteria, updates, batch_size, parameter_counts):
    """Pin the complete study before any optimization or evidence reuse."""
    source = [Path(__file__), Path(M.__file__), Path(DIRECT.__file__), Path(BASE.__file__),
              Path(D.__file__), Path(L.__file__), Path(L.TW.__file__), Path(L.TW.S.__file__),
              L.ROOT / 'd4mj/train.py', L.ROOT / 'd4mj/mamba_recurrence.py', PREREG,
              OUT / 'criteria.json', PARENT, L.TW.S.CHECKPOINT,
              D.OUT / 'manifest.json', D.OUT / 'windows.pt']
    # These paths are scientific execution sources, not optional provenance.
    source += [Path(__file__).with_name(name) for name in
               ('paired_checks.py', 'paired_model_test.py', 'paired_eval.py', 'paired_analysis.py',
                'paired_diagnostics.py', 'run_paired.sh', 'direct_eval.py', 'direct_analysis.py',
                'direct_coherence.py', 'direct_fresh.py', 'direct_joint_shuffle.py')]
    source += [L.ROOT / 'artifacts/experiments/20260927_levers/scroll.py',
               D.OUT / 'new_seed_confirmation_reservation.json']
    assert all(path.is_file() for path in source), 'complete and pin the harness before training'
    contract = {
        'format': 'paired_event_two_arm_v1',
        'sources_inputs': {str(path.relative_to(L.ROOT)): L.sha(path) for path in source},
        'data_contract_id': manifest['contract_id'], 'criteria_id': criteria['criteria_id'],
        'updates': updates, 'batch': batch_size, 'arms': list(M.ARMS),
        'parameter_counts': parameter_counts,
        'compute_reporting': {
            'per_arm_timing': 'CUDA synchronized backbone, two completion draws plus energy distances, optimizer; cumulative seconds per arm',
            'memory': 'per-arm allocator high-water and increase over arm start with both models resident; whole-process maxima retained; actual B16 preflight enforces <6GiB allocated',
            'timing_limit': 'Compilation/warmup included in observed wall times; distinguish steady-state measurements during analysis',
        },
        'input': 'frozen Raw 81x192 cached TRAIN observations and recorded actions',
        'backbone': 'same trainable corrt/fmamba original D parent; one causal call per arm/update',
        'contrast': 'unchanged direct energy head versus source-associated paired contextual event editor',
        'noise': 'two independent draws; local4 and shared4 Gaussian coordinates; unchanged across arms',
        'targets_per_window': 15, 'scored_tokens': list(DIRECT.INTERIOR),
        'patches': [list(x) for x in DIRECT.PATCHES],
        'loss': 'both arms: .5d(X1,Y)+.5d(X2,Y)-.5d(X1,X2); unchanged exponent1 .5global+.5local',
        'sampler': 'uniform eligible RGB-first-view windows; no terminal oversampling',
        'seeds': {'initialization': 7, 'windows': 20261014, 'noise': 2026101401},
        'precision': 'BF16/IEEE with FP32 output distances; TF32 disabled',
        'training_semantic_or_simulator_labels': False,
        'scope': 'one-step controlled-subset completion, one seed; no H16/actor/four-way separation',
    }
    return json.loads(json.dumps(contract))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--initialize-only', action='store_true',
                        help='seal the main contract/state/milestone0 without any optimization')
    parser.add_argument('--directory', type=Path, default=OUT / 'training')
    args = parser.parse_args()
    assert args.updates == 3000 and args.batch == 16 and PREREG.is_file(), 'preregistered budget and batch'
    assert args.stop_after is None or args.stop_after > 0, 'use --initialize-only for zero updates'
    config = settings()
    manifest, windows, actions, tokens = load_data()
    criteria = json.loads((OUT / 'criteria.json').read_text())
    assert criteria['prepared_from_real_only'] and criteria['primary_checkpoint'] == args.updates
    assert all(L.sha(L.ROOT / name) == digest for name, digest in criteria['sources'].items())
    worlds = initialized_worlds('cuda')
    assert tuple(worlds) == tuple(M.ARMS)
    contract = make_contract(manifest, criteria, args.updates, args.batch, M.parameter_counts(worlds))
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'contract.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract, 'source/input/preregistration drift'
    else:
        L.atomic(contract, path, True)
    optimizers = {arm: phase_optimizer([world], config) for arm, world in worlds.items()}
    order = torch.Generator().manual_seed(20261014)
    noise_rng = torch.Generator().manual_seed(2026101401)
    start, chain, history = 0, 'initial', []
    totals = {arm: {'steps': 0, 'targets': 0, 'backbone_seconds': 0., 'head_seconds': 0.,
                    'optimizer_seconds': 0., 'gradient_clipped_steps': 0,
                    'peak_allocated_gib': 0., 'peak_reserved_gib': 0.,
                    'peak_extra_allocated_gib': 0.} for arm in M.ARMS}
    state = directory / 'state.pt'
    state_loaded = state.exists()
    if state_loaded:
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
    if args.stop_after and start >= args.stop_after:
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
                          'world': model_states[arm], 'ledger_chain': chain},
                         directory / f'{arm}_u{update:04d}.pt')

    if start == 0 and not state_loaded:
        save(0, snapshot=True)
    if args.initialize_only:
        assert start == 0, 'cannot label a trained checkpoint initialization-only'
        print(json.dumps({'stage': 'initialized', 'update': 0, 'contract_id': cid,
                          'optimization_steps': 0, 'existing_state_reused': state_loaded}), flush=True)
        return
    started = time.perf_counter()
    for update in range(start, args.updates):
        indices = torch.randint(len(windows), (args.batch,), generator=order)
        frames, ac, choices = batch(tokens, actions, windows, indices)
        noise = DIRECT.draw_noise(frames.shape[:2], noise_rng).to('cuda')
        raw = choices.numpy().tobytes() + noise.cpu().numpy().tobytes()
        chain = hashlib.sha256((b'' if chain == 'initial' else bytes.fromhex(chain)) + raw).hexdigest()
        shared_cpu_rng, shared_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        records = {}
        for arm in M.ARMS:
            torch.set_rng_state(shared_cpu_rng)
            torch.cuda.set_rng_state_all(shared_cuda_rng)
            world = worlds[arm]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_start = torch.cuda.memory_allocated() / 2**30
            began = time.perf_counter()
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
            torch.cuda.synchronize()
            backbone_seconds = time.perf_counter() - began
            began = time.perf_counter()
            with autocast_context(config):
                outputs = torch.stack([world.complete(frames, ac, belief, ha, n)[0] for n in noise])[:, :, :15]
            values = DIRECT.losses(outputs, frames[:, 1:])
            assert all(bool(torch.isfinite(value)) for value in values.values())
            torch.cuda.synchronize()
            head_seconds = time.perf_counter() - began
            began = time.perf_counter()
            norm = float(optimizer_step(optimizers[arm], values['energy'], list(world.parameters()),
                                       learning_rate=_phase_lr(config, update), grad_clip=config.agent.grad_clip,
                                       strict=True, zero_grad=True))
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - began
            allocated_peak = torch.cuda.max_memory_allocated() / 2**30
            reserved_peak = torch.cuda.max_memory_reserved() / 2**30
            total = totals[arm]
            total['steps'] += 1
            total['targets'] += args.batch * 15
            total['backbone_seconds'] += backbone_seconds
            total['head_seconds'] += head_seconds
            total['optimizer_seconds'] += optimizer_seconds
            total['gradient_clipped_steps'] += int(norm > config.agent.grad_clip)
            total['peak_allocated_gib'] = max(total['peak_allocated_gib'], allocated_peak)
            total['peak_reserved_gib'] = max(total['peak_reserved_gib'], reserved_peak)
            total['peak_extra_allocated_gib'] = max(total['peak_extra_allocated_gib'], allocated_peak - allocated_start)
            records[arm] = {name: float(value.detach()) for name, value in values.items()}
            records[arm].update(gradient_norm=norm, backbone_seconds=backbone_seconds,
                                two_head_seconds=head_seconds, optimizer_seconds=optimizer_seconds,
                                allocated_at_arm_start_gib=allocated_start,
                                peak_allocated_gib=allocated_peak, peak_reserved_gib=reserved_peak,
                                peak_extra_allocated_gib=allocated_peak - allocated_start)
        if (update + 1) % 100 == 0 or update == start:
            row = {'update': update + 1, 'arms': records, 'ledger_chain': chain,
                   'active_seconds': time.perf_counter() - started,
                   'peak_allocated_gib': max(value['peak_allocated_gib'] for value in totals.values()),
                   'peak_reserved_gib': max(value['peak_reserved_gib'] for value in totals.values())}
            history.append(row)
            print(json.dumps(row), flush=True)
        snapshot = update + 1 in (500, 1500, args.updates)
        if (update + 1) % 25 == 0 or snapshot:
            save(update + 1, snapshot=snapshot)
        if args.stop_after and update + 1 >= args.stop_after:
            save(update + 1)
            return
    L.atomic({'contract_id': cid, 'update': args.updates, 'history': history, 'totals': totals,
              'ledger_chain': chain, 'completed': True}, directory / 'training.json', True)
    print(json.dumps({'stage': 'done', 'update': args.updates, 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
