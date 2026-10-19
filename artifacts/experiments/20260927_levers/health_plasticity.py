"""Could a negative E21 result come from continuing a trained world instead of training from scratch? (2026-10-08, night.)
Two diagnostics; the decisive test (an arm trained from scratch through the canonical recipe) is a separate lane.

Stage `metrics <world.pt|init> ...` -- loss-of-plasticity signatures (literature: Lyle et al. 2022, arXiv 2204.09560; Sokar et al.
2023, arXiv 2302.12902), on a fixed sample of 1,024 health_chain subset events (seed 0), window = the 5 frames 10..14 for every
world (M6-family worlds have a 6-row time table), plus window 15 for L16 worlds:
  dormant      per layer, the MLP's GELU units: score s_i = E|a_i| / mean_k E|a_k| over every position (Sokar eq. 1); fraction with
               s_i <= tau for tau = 0, 0.025, 0.1 (Sokar's analysis / ReDo-figure / ReDo-Atari thresholds)
  rank         feature rank (Lyle Def. 2): #singular values of phi(X) / sqrt(n) above eps = 0.01 (Lyle's cutoff), phi = backbone output
               h at slot 63 and at all 81 slots (n = 1,024 events at the last position; all slots: 1,024 x 81 rows)
  fit          target-fitting capacity (Lyle Def. 1, their random-target protocol shortened): a copy of the world (all parameters,
               AdamW lr 1e-4, no warm-up) regresses the backbone output of a freshly initialized TWorld (seed 1234) on the same inputs,
               MSE, 1,000 steps of 8 windows; MSE on 64 fixed held windows at steps 0, 100, 250, 500, 1,000 (Lyle: 10,000 steps)
  'init' = a fresh TWorld(corrt, fmamba, frames 16) built exactly as tworld.train builds it (manual seed 7).
  -> artifacts/eda/health_chain_v1/plasticity_metrics.json

Stage `oracle <world.pt> [--updates N] [--lam L]` -- positive control: can the trained backbone still acquire the missing ingredients
when the objective asks for them directly? The world continues with its own recipe (tworld.train's rawlong sampler: batch order seed 11,
40 windows of 16 frames, TERMINAL_SHARE end-aligned deaths; phase AdamW, lr warm-up, clip, bf16) on the teacher L1 PLUS lam x oracle
losses read from h at slot 63 (the corrt head's only input for token 63), output positions 0..14:
  since    CE over 16 classes: transitions since the last health drop before position t (0 = none in the window); health from teval's
           Probes.hud reader on the true frames, drop = next < current - 0.5
  adj      BCE: a zombie (Probes.zombie > 0.3) in true frame t beside the player's post-move cell (scroll.estimate between frames t, t+1)
Labels come from the true frames (TRAIN ledger) and are diagnostic only, never a proposed fix. lam default: set on the first
fine-tuning batch so that lam x the oracle loss = the teacher loss (logged); the teacher term is asserted equal to
tworld.rollout_losses' on the first batch.
LP-FT (Kumar et al. 2022, arXiv 2202.10054; added 21:50 before the first run): the two oracle heads are first fitted alone on the
frozen world's h63 (60 batches of order seed 13, AdamW lr 1e-3, 3,000 steps), so the continuation's gradient into the backbone comes
from fitted heads, not random ones; the LP losses are logged. Snapshots in
tworld's checkpoint format at 500 / 1,000 / 2,000 updates -> levers_tworlds_v1/<name>[_atN].pt, read by health_chain.py / health_evidence.
`--gen-loss` (23:05, sufficiency test): + tworld's generator loss (the generate candidate's own L1 on every token, as E21 G), so the
continuation satisfies B1 (oracle) and B2 (generator) together; name gets `_gl`.
Usage: health_plasticity.py metrics <world.pt|init> ... | health_plasticity.py oracle <world.pt> [--updates N] [--lam L (default: auto)] [--gen-loss]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
WORLDS = ROOT / 'artifacts/eda/levers_tworlds_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def fresh_world(seed, device):
    import tworld as TW
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        return TW.TWorld('corrt', None, 'fmamba', 'all', 0, False, 16).to(device)


def metrics(paths):
    import teval as T
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    idx = torch.randperm(len(sub['classes']), generator=torch.Generator().manual_seed(0))[:1088].sort().values
    x = torch.from_numpy(np.array(mm[idx.numpy()])).float()                       # [1088, 16, 81, 192]
    acts = sub['actions'][idx]
    target_net = fresh_world(1234, dev).eval()
    res = {}
    for path in paths:
        if path == 'init':
            w, name, L = fresh_world(7, dev), 'init_seed7', 16
        else:
            w, st = T.load_world(Path(path), dev)
            name, L = st['name'], w.time.shape[0]
        w.eval()
        r = {}
        for W in ((5, 15) if L >= 16 else (5,)):
            gelu = {}
            hooks = [layer.mlp[1].register_forward_hook(lambda m, i, o, k=k: gelu.__setitem__(k, gelu.get(k, 0) + o.float().abs().sum((0, 1, 2)))) for k, layer in enumerate(w.layers)]
            hs = []
            with torch.no_grad():
                for i in range(0, 1024, 16):
                    with autocast_context(config):
                        h = w.backbone(x[i:i + 16, 15 - W:15].to(dev), acts[i:i + 16, 15 - W:15].to(dev))
                    hs.append(h[:, -1].float().cpu())
            for hk in hooks:
                hk.remove()
            dormant = {}
            for k, tot in gelu.items():
                s = tot / tot.mean()
                dormant[f'layer{k}'] = {str(tau): round(float((s <= tau).float().mean()), 4) for tau in (0.0, 0.025, 0.1)}
            h = torch.cat(hs)                                                          # [1024, 81, 256]
            rank = lambda m: int((torch.linalg.svdvals(m / m.shape[0] ** 0.5) > 0.01).sum())
            r[f'w{W}'] = {'dormant': dormant, 'feature_rank_slot63': rank(h[:, 63]), 'feature_rank_all_slots': rank(h.flatten(0, 1)),
                          'slot63_singular_values_top5': [round(float(v), 4) for v in torch.linalg.svdvals(h[:, 63] / 32)[:5]]}
        # target-fitting capacity on 5-frame windows (frames 10..14), all slots' backbone output
        import copy
        net = copy.deepcopy(w).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        g = torch.Generator().manual_seed(1)
        held = torch.arange(1024, 1088)
        def target(xb, ab):
            with torch.no_grad(), autocast_context(config):
                return target_net.backbone(xb, ab).float()
        def held_mse():
            with torch.no_grad():
                tot = 0.0
                for i in range(0, 64, 16):
                    xb, ab = x[held[i:i + 16], 10:15].to(dev), acts[held[i:i + 16], 10:15].to(dev)
                    with autocast_context(config):
                        tot += float((net.backbone(xb, ab).float() - target(xb, ab)).pow(2).mean()) / 4
            return round(tot, 5)
        curve = {0: held_mse()}
        for step in range(1, 1001):
            b = torch.randint(1024, (8,), generator=g)
            xb, ab = x[b, 10:15].to(dev), acts[b, 10:15].to(dev)
            with autocast_context(config):
                loss = (net.backbone(xb, ab).float() - target(xb, ab)).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step in (100, 250, 500, 1000):
                curve[step] = held_mse()
        r['random_target_fit_mse'] = curve
        res[name] = r
        print(json.dumps({name: r}), flush=True)
        del net, opt
    path = OUT / 'plasticity_metrics.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))


def oracle_labels(s, P, estimate):
    """s [B,16,81,192] true frames (CPU float) -> since [B,15] long (0..15), adj [B,15] float."""
    B, L = s.shape[:2]
    health = (P.hud(s[:, :, 63:81].flatten(-2).flatten(0, 1))[:, 0] * 9).float().view(B, L)
    drop = health[:, 1:] < health[:, :-1] - 0.5                                    # [B,15] transition j drops
    idx = torch.arange(L - 1)
    since = torch.zeros(B, L - 1, dtype=torch.long)
    for t in range(L - 1):                                                         # last drop among transitions 0..t-1
        prev = drop[:, :t]
        last = torch.where(prev, idx[:t], torch.tensor(-1)).amax(1) if t else torch.full((B,), -1)
        since[:, t] = torch.where(last >= 0, t - last, torch.tensor(0))
    coords = torch.tensor([[c // 9, c % 9] for c in range(63)])
    z = (P.zombie(s[:, :L - 1, :63].flatten(0, 2))[:, 0] > 0.3).view(B, L - 1, 63)
    sc = estimate(s[:, :-1].flatten(0, 1), s[:, 1:].flatten(0, 1)).view(B, L - 1)
    post = torch.tensor([3, 4]) + torch.tensor(SHIFTS)[sc]                          # [B,15,2]
    d = (coords[None, None] - post[:, :, None]).abs().sum(-1)                      # [B,15,63]
    adj = (z & (d == 1)).any(-1).float()
    return since, adj


def oracle(path, updates, lam, gen_loss=False):
    import teval as T
    import tworld as TW
    from scroll import estimate
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    import spatial as S
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    world, st = T.load_world(Path(path), dev)
    world.train()
    L = world.time.shape[0]
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    aux = nn.ModuleDict({'since': nn.Linear(S.D, 16), 'adj': nn.Linear(S.D, 1)}).to(dev)
    pool = torch.load(TW.POOLS['rawlong'] / 'labels.pt', weights_only=False, mmap=True)
    tokens = np.memmap(TW.POOLS['rawlong'] / 'tokens.f16', dtype=np.float16, mode='r', shape=(len(pool['terminal']), 64, 81, 192))
    main_rows, term_rows = torch.where(~pool['terminal'])[0], torch.where(pool['terminal'])[0]

    def batch(order):
        term = torch.rand(TW.BATCH, generator=order) < TW.TERMINAL_SHARE
        r = torch.where(term, term_rows[torch.randint(len(term_rows), (TW.BATCH,), generator=order)],
                        main_rows[torch.randint(len(main_rows), (TW.BATCH,), generator=order)])
        t0 = torch.where(term, 64 - L, torch.randint(0, 65 - L, (TW.BATCH,), generator=order))
        s_cpu = torch.stack([torch.from_numpy(np.array(tokens[int(i), int(j):int(j) + L])) for i, j in zip(r, t0)]).float()
        return s_cpu, torch.stack([pool['actions'][int(i), int(j):int(j) + L - 1] for i, j in zip(r, t0)])
    lp_order, lp_cache = torch.Generator().manual_seed(13), []
    with torch.no_grad():                                                          # LP: frozen h63 and labels, 60 batches
        for _ in range(60):
            s_cpu, a = batch(lp_order)
            since, adj = oracle_labels(s_cpu, P, estimate)
            with autocast_context(config):
                h = world(s_cpu.to(dev), F.pad(a, (0, 1)).to(dev))[1]
            lp_cache.append((h[:, :L - 1, 63].float(), since.to(dev), adj.to(dev)))
    lp_opt = torch.optim.AdamW(aux.parameters(), lr=1e-3)
    lp_log = []
    for step in range(3000):
        h63, since, adj = lp_cache[step % len(lp_cache)]
        ce = F.cross_entropy(aux['since'](h63).flatten(0, 1), since.flatten())
        bce = F.binary_cross_entropy_with_logits(aux['adj'](h63)[..., 0], adj)
        lp_opt.zero_grad(); (ce + bce).backward(); lp_opt.step()
        if (step + 1) % 500 == 0:
            lp_log.append({'lp_step': step + 1, 'since_ce': float(ce), 'adj_bce': float(bce)})
            print(json.dumps(lp_log[-1]), flush=True)
    del lp_cache
    opt = phase_optimizer([world, aux], config)
    params = [p for g in opt.param_groups for p in g['params']]
    order = torch.Generator().manual_seed(11)
    name = st['name'].replace('corrt_rawlong_teacher_s7_fmamba_', 'corrt_rawlong_teacher_s7_fmamba_oracle_' + ('gl_' if gen_loss else ''))
    args = dict(st['args']); args.update(init=str(path), updates=str(updates), oracle_lam=str(lam), gen_loss=str(gen_loss))
    save = lambda tag, h: torch.save({'name': tag, 'args': args, 'world': world.state_dict(), 'history': h}, WORLDS / f'{tag}.pt')
    history, started, checked = list(lp_log), time.time(), False
    for update in range(updates):
        s_cpu, a = batch(order)
        a = a.to(dev)
        since, adj = oracle_labels(s_cpu, P, estimate)
        s = s_cpu.to(dev)
        with autocast_context(config):
            pred, h, gen = world(s, F.pad(a, (0, 1)))
            teacher = (pred[:, :L - 1] - s[:, 1:]).abs().mean()
            if gen_loss:                                                        # tworld.rollout_losses' gen_loss term
                teacher = teacher + (F.layer_norm(gen[:, :L - 1], (S.WIDTH,)) - s[:, 1:]).abs().mean()
            if not checked:
                with torch.no_grad():
                    ref = TW.rollout_losses(world, s, a, 'teacher', gen_loss)
                assert abs(float(ref) - float(teacher)) < 1e-4, (float(ref), float(teacher))
                checked = True
            h63 = h[:, :L - 1, 63].float()
            ce = F.cross_entropy(aux['since'](h63).flatten(0, 1), since.to(dev).flatten())
            bce = F.binary_cross_entropy_with_logits(aux['adj'](h63)[..., 0], adj.to(dev))
            if lam is None:                                                      # oracle term = teacher term at the start
                lam = float(teacher) / float(ce + bce)
                args['oracle_lam'] = str(lam)
                history.append({'oracle_lam': lam})
                print(json.dumps({'oracle_lam': lam}), flush=True)
            objective = teacher + lam * (ce + bce)
        norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
        if (update + 1) % 100 == 0:
            row = {'update': update + 1, 'objective': float(objective), 'teacher': float(teacher), 'since_ce': float(ce),
                   'adj_bce': float(bce), 'gradient_norm': float(norm), 'seconds': round(time.time() - started, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)
        if update + 1 in (500, 1000) and update + 1 < updates:
            save(f'{name}_at{update + 1}', history)
    save(name, history)
    print(json.dumps({'saved': name}), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    a = sys.argv[1:]
    if a[0] == 'metrics':
        metrics(a[1:])
    else:
        oracle(a[1], int(a[a.index('--updates') + 1]) if '--updates' in a else 2000, float(a[a.index('--lam') + 1]) if '--lam' in a else None,
               '--gen-loss' in a)
