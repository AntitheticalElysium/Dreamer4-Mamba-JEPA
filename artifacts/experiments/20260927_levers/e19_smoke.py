"""Actual batch40 CUDA E19 update/memory measurement (exclusive device required)."""
import json
import time
import torch
import e19 as E
from d4mj.train import phase_optimizer, optimizer_step, autocast_context, _phase_lr

T, R = E.T, E.R
torch.set_num_threads(4)
device, cfg = torch.device('cuda'), E.config()
pool = torch.load(T.POOLS['raw'] / 'pool.pt', mmap=True, weights_only=False)
masks = torch.load(E.MASK, weights_only=False)['beside']
rows = E.train_rows(pool)
idx = rows[torch.randint(len(rows), (40,), generator=torch.Generator().manual_seed(11))]
off = torch.arange(40, device=device) % 5
b = T.S.batch_of(pool, idx, 'tokens', device)
out = {}
for backbone in ('fmamba', 'full'):
    parent = T.OUT / ('corrt_raw_teacher_s7' + ('_fmamba' if backbone == 'fmamba' else '') + '_u36000.pt')
    for arm in ('A', 'B', 'C'):
        w = E.make_world(parent, backbone, device)
        opt = phase_optimizer([w], cfg)
        params = [p for g in opt.param_groups for p in g['params']]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        with autocast_context(cfg):
            loss, parts = E.objective(w, b['s'], b['actions'], masks[idx].to(device), off, arm)
        norm = optimizer_step(opt, loss, params, learning_rate=_phase_lr(cfg, 0),
                              grad_clip=cfg.agent.grad_clip, strict=True, zero_grad=True)
        torch.cuda.synchronize()
        out[backbone + '_' + arm] = {'batch': 40, 'seconds_one_update': time.monotonic() - start,
                'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                'loss': float(loss.detach()), 'gradient': float(norm), **parts}
        print(json.dumps({backbone + '_' + arm: out[backbone + '_' + arm]}), flush=True)
        del w, opt, params, loss
        torch.cuda.empty_cache()
R.atomic_json(T.HERE / 'e19_batch40_smoke.json', out)
