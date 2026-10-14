"""Atomic one-step direct-generator sampling; no inverse/future-conditioned path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

import torch

import direct_model as M
import direct_train as T
import loss_intervention as L
from d4mj.train import autocast_context


def cohort(name):
    if name == 'fresh':
        path = T.OUT / 'fresh/fresh.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data, {'fresh_sha256': L.sha(path)}
    base = L.OUT
    selection = json.loads((T.OUT / 'data/existing_eval_selection.json').read_text())
    indices = selection['selected_indices']
    panel = torch.load(base / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    fan = torch.load(base / 'learned_innovation/real_futures/fan.pt', map_location='cpu', weights_only=False, mmap=True)
    meta = torch.load(T.OUT / 'eval_reference/metadata.pt', map_location='cpu', weights_only=False)
    assert meta['selected_indices'] == indices
    data = {'roots': [panel['roots'][i] for i in indices],
            'context_tokens': panel['tokens'][indices, 1:16], 'actions': panel['actions'][indices, 1:16],
            'root_tokens': panel['tokens'][indices, 15], 'fan_tokens': fan['tokens'][indices],
            'root_layout': panel['layouts'][indices, 15], 'fan_layout': fan['layouts'][indices],
            'root_tiles': meta['root_tiles'], 'fan_tiles': meta['real_tiles'], 'shifts': meta['shifts']}
    return data, {'panel_sha256': L.sha(base / 'panel.pt'),
                  'fan_sha256': L.sha(base / 'learned_innovation/real_futures/fan.pt'),
                  'metadata_sha256': L.sha(T.OUT / 'eval_reference/metadata.pt'),
                  'selection_sha256': L.sha(T.OUT / 'data/existing_eval_selection.json')}


@torch.no_grad()
def sample(world, context, actions, noises, config):
    # The unused last input preserves the training tensor shape and is not a
    # true successor. Causal checks prove it cannot influence index14.
    frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
    ac = torch.nn.functional.pad(actions, (0, 1))
    torch.cuda.synchronize()
    then = time.perf_counter()
    with autocast_context(config):
        belief, ha = world.backbone_full(frames, ac)
    torch.cuda.synchronize()
    backbone_seconds = time.perf_counter() - then
    output = []
    elapsed = []
    for noise in noises:
        then = time.perf_counter()
        with autocast_context(config):
            prediction = world.complete(frames, ac, belief, ha, noise)[0][:, 14]
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - then)
        output.append(prediction.cpu())
    return torch.stack(output, 1), backbone_seconds, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('exploratory', 'fresh'), default='exploratory')
    parser.add_argument('--updates', type=int, nargs='+', default=[0, 500, 1500, 3000])
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--limit-batches', type=int)
    a = parser.parse_args()
    assert a.samples >= 2
    config = T.settings()
    data, inputs = cohort(a.cohort)
    for update in a.updates:
        directory = T.OUT / f'eval_{a.cohort}_u{update:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        for arm in M.ARMS:
            checkpoint = T.OUT / 'training' / f'{arm}_u{update:04d}.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['update'] == update and saved['arm'] == arm
            for name, digest in saved['contract']['sources_inputs'].items():
                assert L.sha(L.ROOT / name) == digest, f'training/evaluation source or input drift: {name}'
            world = M.DirectWorld().cuda().eval()
            world.load_state_dict(saved['world'])
            contract = {'format': 'direct_samples_v1', 'training_contract_id': saved['contract_id'],
                        'checkpoint_sha256': L.sha(checkpoint), 'cohort': a.cohort, 'update': update, 'arm': arm,
                        'samples': a.samples, 'batch': a.batch, 'inputs': inputs,
                        'seed_rule': '2026101402 + 100000*cohort_id + start; identical across arms/milestones',
                        'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p)
                                    for p in (__file__, M.__file__, T.__file__, L.TW.__file__)}}
            cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            cp = directory / f'{arm}_contract.json'
            if cp.exists():
                assert json.loads(cp.read_text()) == contract
            else:
                L.atomic(contract, cp, True)
            rows = []
            for index, start in enumerate(range(0, len(data['roots']), a.batch)):
                path = directory / 'batches' / f'{arm}_{start:04d}.pt'
                if path.exists():
                    row = torch.load(path, map_location='cpu', weights_only=False)
                    assert row['contract_id'] == cid and row['start'] == start
                else:
                    context = data['context_tokens'][start:start + a.batch].cuda().float()
                    actions = data['actions'][start:start + a.batch].cuda()
                    generator = torch.Generator().manual_seed(2026101402 + (100000 if a.cohort == 'fresh' else 0) + start)
                    noises = torch.cat([M.draw_noise((len(context), 16), generator)
                                        for _ in range((a.samples + 1) // 2)])[:a.samples].cuda()
                    pred, backbone_seconds, head_seconds = sample(world, context, actions, noises, config)
                    # Exact same input/noise/shape verifies repeated sampling;
                    # repeating outputs through a different batch was misleading before.
                    repeat, _, _ = sample(world, context, actions, noises[:1], config)
                    replay_maxabs = float((repeat[:, 0] - pred[:, 0]).abs().max())
                    assert replay_maxabs == 0, replay_maxabs
                    row = {'contract_id': cid, 'start': start, 'predictions': pred,
                           'noise_replay_maxabs': replay_maxabs, 'backbone_seconds': backbone_seconds,
                           'head_seconds': head_seconds, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
                    L.atomic(row, path)
                rows.append(row)
                print(json.dumps({'stage': 'sample', 'cohort': a.cohort, 'update': update, 'arm': arm,
                                  'completed_roots': start + len(row['predictions']), 'total': len(data['roots'])}), flush=True)
                if a.limit_batches and index + 1 >= a.limit_batches:
                    del world
                    return
            L.atomic({'contract_id': cid, 'contract': contract, 'arm': arm, 'update': update,
                      'roots': data['roots'],
                      'predictions': torch.cat([row['predictions'] for row in rows]),
                      'costs': [{'roots': len(row['predictions']), 'backbone_seconds': row['backbone_seconds'],
                                'head_seconds': row['head_seconds'],
                                'cumulative_head_seconds': {str(n): sum(row['head_seconds'][:n])
                                                            for n in (1, 2, 16, 64) if n <= len(row['head_seconds'])},
                                'peak_allocated_gib': row['peak_allocated_gib']}
                               for row in rows], 'noise_replay_maxabs': max(row['noise_replay_maxabs'] for row in rows)},
                     directory / f'pred_{arm}.pt')
            del world
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
