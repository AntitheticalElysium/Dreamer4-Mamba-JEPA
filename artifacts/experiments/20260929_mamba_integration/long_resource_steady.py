"""Warm full-optimizer length-64 resource check before fixing Subrun-2 batch size.

Synthetic tokens/actions exercise the exact train graph; no treatment checkpoint or outcome is opened.
Each candidate is warmed once, then timed for three complete train steps with peak GPU allocation.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'artifacts/experiments/20260927_levers'),
                str(ROOT / 'artifacts/experiments/20260921_readout_ladder')]
from long_world import LongTWorld, rollout_loss
import spatial as S
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    out = []
    for batch in (8, 16, 24, 32):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        world = LongTWorld().cuda().train()
        opt = phase_optimizer([world], config)
        params = [p for g in opt.param_groups for p in g['params']]
        s = F.layer_norm(torch.randn(batch, 64, 81, 192, device='cuda'), (192,)).half()
        a = torch.randint(17, (batch, 63), device='cuda')
        durations = []
        error = None
        try:
            for j in range(4):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with autocast_context(config):
                    objective = rollout_loss(world, s, a)
                norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, 18000 + j),
                                      grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
                torch.cuda.synchronize()
                if j:
                    durations.append(time.perf_counter() - start)
            row = {'batch': batch, 'ok': True, 'warmed_seconds': durations,
                   'median_warmed_seconds': sorted(durations)[1],
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                   'last_objective': float(objective.detach()), 'last_grad_norm': float(norm)}
        except torch.OutOfMemoryError as exc:
            error = str(exc).splitlines()[0]
            row = {'batch': batch, 'ok': False, 'error': error,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
        print(json.dumps(row), flush=True)
        out.append(row)
        del world, opt, params, s, a
        if 'objective' in locals():
            del objective
        if 'norm' in locals():
            del norm
        torch.cuda.empty_cache()
        if error:
            break
    report = {'source_sha256': {str(p.relative_to(ROOT)): digest(p)
               for p in (HERE / 'long_resource_steady.py', HERE / 'long_world.py',
                         ROOT / 'artifacts/experiments/20260927_levers/tworld.py',
                         ROOT / 'artifacts/experiments/20260921_readout_ladder/spatial.py')},
              'hardware': {'name': torch.cuda.get_device_name(0),
                           'total_bytes': torch.cuda.get_device_properties(0).total_memory},
              'results': out}
    (HERE / 'long_resource_steady.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
