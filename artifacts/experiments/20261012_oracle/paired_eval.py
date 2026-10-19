"""Atomic one-step sampling for the matched direct/paired event experiment.

The shared real cohorts and historical readers are reused without fitting.
Reference predictions are exported separately, never substituted into scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

import torch
from torch.nn import functional as F

import direct_eval as E
import direct_model as M
import direct_train as T
import loss_intervention as L
import paired_model as P
import paired_train as PT
from d4mj.train import autocast_context

ARMS = ('direct', 'paired')
OUT = L.OUT / 'paired_event'


def cohort(name):
    if name == 'fresh':
        path = OUT / 'fresh/fresh.pt'
        return torch.load(path, map_location='cpu', weights_only=False), {'fresh_sha256': L.sha(path)}
    return E.cohort(name)


def model(arm):
    assert arm in ARMS
    return M.DirectWorld() if arm == 'direct' else P.PairedWorld()


@torch.no_grad()
def reference(world, frames, actions, h, ha):
    if hasattr(world, 'reference'):
        value = world.reference(frames, actions, h, ha)
    else:
        grid = frames.float().view(*frames.shape[:2], 9, 9, 192)
        padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
        parts = [grid] + [padded[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc]
                          for dr, dc in L.TW.NEIGHBOURS]
        parts = torch.stack([x.reshape(frames.shape) for x in parts] + [world.proj(h).float()], -2)
        logits = world.choose(h).float()
        moved = world.frame(ha).float()[:, :, None, :]
        target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
        htarget = h.gather(2, target[:, :, None, None].expand(-1, -1, 1, h.shape[-1]))[:, :, 0]
        moved = moved + (((actions >= 1) & (actions <= 4)).float()[:, :, None, None]
                         * world.target_gate(htarget).float()[:, :, None, :])
        logits = logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])
        value = (logits.softmax(-1)[..., None] * parts).sum(-2)
    return F.layer_norm(value.float(), (192,))


@torch.no_grad()
def sample(world, context, actions, noises, config):
    frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
    ac = F.pad(actions, (0, 1))
    torch.cuda.synchronize()
    then = time.perf_counter()
    with autocast_context(config):
        h, ha = world.backbone_full(frames, ac)
        ref = reference(world, frames, ac, h, ha)[:, 14]
    torch.cuda.synchronize()
    backbone_seconds = time.perf_counter() - then
    predictions, elapsed = [], []
    for noise in noises:
        torch.cuda.synchronize()
        then = time.perf_counter()
        with autocast_context(config):
            pred = world.complete(frames, ac, h, ha, noise)[0][:, 14]
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - then)
        predictions.append(pred.cpu())
    return torch.stack(predictions, 1), ref.cpu(), backbone_seconds, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', choices=('exploratory', 'fresh'), default='exploratory')
    p.add_argument('--updates', type=int, nargs='+', default=[0, 500, 1500, 3000])
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--limit-batches', type=int)
    a = p.parse_args()
    assert a.samples >= 2
    config = T.settings()
    data, inputs = cohort(a.cohort)
    for update in a.updates:
        directory = OUT / f'eval_{a.cohort}_u{update:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            checkpoint = OUT / 'training' / f'{arm}_u{update:04d}.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['arm'] == arm and saved['update'] == update
            for name, digest in saved['contract']['sources_inputs'].items():
                assert L.sha(L.ROOT / name) == digest, f'training/evaluation drift: {name}'
            world = model(arm).cuda().eval()
            world.load_state_dict(saved['world'])
            contract = {'format': 'paired_samples_v1', 'training_contract_id': saved['contract_id'],
                        'checkpoint_sha256': L.sha(checkpoint), 'cohort': a.cohort, 'update': update,
                        'arm': arm, 'samples': a.samples, 'batch': a.batch, 'inputs': inputs,
                        'seed_rule': '2026101402 + 100000*cohort_id + start; identical across arms/milestones',
                        'sources': {str(Path(s).relative_to(L.ROOT)): L.sha(s)
                                    for s in (__file__, P.__file__, M.__file__, E.__file__, T.__file__, PT.__file__, L.TW.__file__)}}
            cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            path = directory / f'{arm}_contract.json'
            if path.exists():
                assert json.loads(path.read_text()) == contract
            else:
                L.atomic(contract, path, True)
            rows = []
            for index, start in enumerate(range(0, len(data['roots']), a.batch)):
                path = directory / 'batches' / f'{arm}_{start:04d}.pt'
                if path.exists():
                    row = torch.load(path, map_location='cpu', weights_only=False)
                    assert row['contract_id'] == cid and row['start'] == start
                else:
                    context = data['context_tokens'][start:start+a.batch].cuda().float()
                    actions = data['actions'][start:start+a.batch].cuda()
                    generator = torch.Generator().manual_seed(2026101402 + (100000 if a.cohort == 'fresh' else 0) + start)
                    noises = torch.cat([M.draw_noise((len(context), 16), generator)
                                        for _ in range((a.samples+1)//2)])[:a.samples].cuda()
                    torch.cuda.reset_peak_memory_stats()
                    pred, ref, bs, hs = sample(world, context, actions, noises, config)
                    repeat, ref_repeat, _, _ = sample(world, context, actions, noises[:1], config)
                    error = float((repeat[:, 0] - pred[:, 0]).abs().max())
                    assert error == 0 and torch.equal(ref, ref_repeat), 'sample replay mismatch'
                    row = {'contract_id': cid, 'start': start, 'predictions': pred, 'reference': ref,
                           'noise_replay_maxabs': error, 'backbone_seconds': bs, 'head_seconds': hs,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
                    L.atomic(row, path)
                rows.append(row)
                print(json.dumps({'stage': 'paired_sample', 'cohort': a.cohort, 'update': update,
                                  'arm': arm, 'completed_roots': start+len(row['predictions']),
                                  'total': len(data['roots'])}), flush=True)
                if a.limit_batches and index+1 >= a.limit_batches:
                    return
            L.atomic({'contract_id': cid, 'contract': contract, 'arm': arm, 'update': update,
                      'roots': data['roots'], 'predictions': torch.cat([r['predictions'] for r in rows]),
                      'reference': torch.cat([r['reference'] for r in rows]),
                      'noise_replay_maxabs': max(r['noise_replay_maxabs'] for r in rows),
                      'costs': [{'roots': len(r['predictions']), 'backbone_seconds': r['backbone_seconds'],
                                 'head_seconds': r['head_seconds'],
                                 'cumulative_head_seconds': {str(n): sum(r['head_seconds'][:n])
                                                            for n in (1, 2, 16, 64) if n <= len(r['head_seconds'])},
                                 'peak_allocated_gib': r['peak_allocated_gib']} for r in rows]},
                     directory/f'pred_{arm}.pt')
            del world
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
