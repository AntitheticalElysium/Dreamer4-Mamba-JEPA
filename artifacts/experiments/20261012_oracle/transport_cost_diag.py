"""Pre-launch diagnostics for a transition cost (2026-10-15, user mandate 2). Nothing here trains the experiment's arms.
  crossterrain  (CPU, DEV, exact tiles: evaluation-side) the contextual-token problem a cost cannot solve: share of real
                in-view mob moves whose source and destination terrain differ; terrain decoded (frozen bank tile head) at both
                endpoints after swapping the TRUE afterstate tokens (the representation's own ceiling) and in the seed-1
                operator's 3k samples, same- vs cross-terrain.
  benefit <u>   (GPU, TRAIN windows, label-free tile classes) what the training objective pays per unit of move probability.
                Seed-1 operator checkpoint u; 48 training-ledger windows whose camera never moves (parameter-free scroll reader
                on real tokens), 8 noise pairs each, all 15 steps. The straight-through relaxation is linear in the soft
                proposals q, so B[j,d] = -dES/dq[j,d] is the marginal value of probability mass on moving tile j's content in
                direction d (two-draw energy score against the real successor, as in training). Classes from tokens only:
                  static   the tile's real token never changes by more than RMS 0.1 over the window (the static floor p90 is
                           0.087), not the player
                  dynamic  some step changes it by more than RMS 0.3; at such a step the realized direction is the neighbour
                           whose next token is closest to this tile's current token, if closer than half its own change
                Reported: static tiles' mean benefit per direction with its window-level standard error; dynamic tiles' benefit
                along realized directions at moving steps, along all directions at staying steps, and the realized moving
                frequency; plus, for interpretation only, the same split by exact simulator mob presence (layouts).
  probe         (GPU) the declared go/no-go check of the chosen cost (TRANSPORT_COST_PREREG.md): from the seed-1 operator u3000,
                two matched 500-update continuations (same windows, noise; with and without the cost), then P(move) on the
                benefit windows by class.
Usage: transport_cost_diag.py crossterrain | benefit <update> | probe <lambda> -> artifacts/eda/oracle_v1/transport/cost_diag_<mode>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_s5_direct as A5  # noqa: E402
import transport_stage2 as S2  # noqa: E402

OUT = A5.BASE / 'transport'
STATIC, DYNAMIC = .1, .3


def crossterrain():
    import direct_eval as E
    import direct_coherence as C
    data, _ = E.cohort('exploratory')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    root_tok, real_tok = data['root_tokens'].float(), data['fan_tokens'].float()
    real_tiles = data['fan_tiles'].long()
    n, K = real_tok.shape[:2]
    rows = {'same': [0, 0, 0], 'cross': [0, 0, 0]}                    # moves, endpoint tile errors after true swap, after real
    op = torch.load(OUT / 'eval_u3000/pred_operator.pt', map_location='cpu', weights_only=False)
    oprow = {'same': [0, 0], 'cross': [0, 0]}
    for i in range(n):
        dr, dc = (int(x) for x in data['shifts'][i, 0])
        after = S2.afterstate_layout(data['root_layout'][i], dr, dc)
        tile = -torch.ones(63, dtype=torch.long)
        aligned = root_tok[i].clone()
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    tile[r * 9 + c] = int(data['root_tiles'][i, (r + dr) * 9 + c + dc])
                    aligned[r * 9 + c] = root_tok[i, (r + dr) * 9 + c + dc]
        swapped, cls = [], []
        for k in range(K):
            pr = [(s, d, t) for s, d, t in S2.swaps(after, data['fan_layout'][i, k].bool()) if t < 3 and 31 not in (s, d) and tile[s] >= 0 and tile[d] >= 0]
            if pr:
                swapped.append((k, pr, S2.apply(aligned, pr)))
        if swapped:
            dec = C.decode(torch.stack([x[2] for x in swapped]), bank, head)['tile']
            dreal = C.decode(real_tok[i, [x[0] for x in swapped]], bank, head)['tile']
            for m, (k, pr, _) in enumerate(swapped):
                for s, d, _t in pr:
                    key = 'same' if tile[s] == tile[d] else 'cross'
                    rows[key][0] += 1
                    rows[key][1] += int(dec[m, s] != real_tiles[i, k, s]) + int(dec[m, d] != real_tiles[i, k, d])
                    rows[key][2] += int(dreal[m, s] != real_tiles[i, k, s]) + int(dreal[m, d] != real_tiles[i, k, d])
        # the operator's own mob moves (its afterstate tiles are the root's; terrain does not move with a mob)
        mob = (after[:, :3] == 1).any(-1)
        dop = C.decode(op['predictions'][i], bank, head)['tile']
        for k in range(op['partner'].shape[1]):
            p = op['partner'][i, k]
            for j in range(63):
                q = int(p[j])
                if q <= j or q >= 63 or tile[j] < 0 or tile[q] < 0 or bool(mob[j]) == bool(mob[q]):
                    continue
                key = 'same' if tile[j] == tile[q] else 'cross'
                oprow[key][0] += 1
                oprow[key][1] += int(dop[k, j] != tile[j]) + int(dop[k, q] != tile[q])
    res = {'real_moves': {k: {'moves': v[0], 'endpoint_tile_error_rate_true_swap': round(v[1] / max(2 * v[0], 1), 4),
                              'endpoint_tile_error_rate_real_successor_reader': round(v[2] / max(2 * v[0], 1), 4)} for k, v in rows.items()},
           'cross_terrain_share_of_real_moves': round(rows['cross'][0] / max(rows['cross'][0] + rows['same'][0], 1), 4),
           'operator_seed1_u3000_mob_moves': {k: {'moves': v[0], 'endpoint_tile_error_rate': round(v[1] / max(2 * v[0], 1), 4)} for k, v in oprow.items()},
           'operator_cross_terrain_share': round(oprow['cross'][0] / max(oprow['cross'][0] + oprow['same'][0], 1), 4)}
    return res


def stationary_windows(count=48, seed=20261020):
    import direct_train as BASE
    _, windows, actions, tokens = BASE.load_data()
    scroll = A5.load_scroll()
    g = torch.Generator().manual_seed(seed)
    picked = []
    for idx in torch.randperm(len(windows), generator=g).tolist():
        row, start = windows[idx].tolist()
        tok = torch.from_numpy(np.array(tokens[row, start:start + 16])).float()
        if bool((scroll.estimate(tok[:-1], tok[1:]) == 0).all()):
            picked.append(idx)
            if len(picked) == count:
                break
    return windows[picked], actions, tokens


def classes(frames):
    """frames [b,16,81,192] real tokens, stationary windows -> static [b,63], dynamic [b,63], realized dir [b,15,63] (-1 none)"""
    ch = (frames[:, 1:, :63] - frames[:, :-1, :63]).square().mean(-1).sqrt()          # [b,15,63]
    static = (ch.amax(1) < STATIC); static[:, 31] = False
    dynamic = (ch.amax(1) > DYNAMIC); dynamic[:, 31] = False
    target = torch.tensor([[j if d == 0 else (((j // 9) + dr) * 9 + (j % 9) + dc if 0 <= (j // 9) + dr < 7 and 0 <= (j % 9) + dc < 9 else -1)
                            for d, (dr, dc) in enumerate(A5.CROSS)] for j in range(63)])                     # [63,5]
    real_dir = -torch.ones(ch.shape, dtype=torch.long)
    for d in range(1, 5):
        ok = target[:, d] >= 0
        k = target[:, d].clamp(min=0)
        dist = (frames[:, 1:, k] - frames[:, :-1, :63]).square().mean(-1).sqrt()       # next token at neighbour vs my current
        dist[..., ~ok] = float('inf')
        if d == 1:
            best, bestd = dist, torch.full(ch.shape, 1, dtype=torch.long)
        else:
            better = dist < best
            best, bestd = torch.where(better, dist, best), torch.where(better, torch.full_like(bestd, d), bestd)
    moving = (ch > DYNAMIC) & (best < .5 * ch)
    real_dir = torch.where(moving, bestd, real_dir)
    return static, dynamic, real_dir


def benefit(update, seed_dir='training'):
    import direct_model as DM
    import direct_train as BASE
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    wins, actions, tokens = stationary_windows()
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(OUT / f'{seed_dir}/operator_u{update:04d}.pt', map_location='cpu', weights_only=False)['world'])
    for prm in world.parameters():
        prm.requires_grad_(False)
    comp = world.completion
    g = torch.Generator().manual_seed(20261021)
    acc = {k: [] for k in ('static', 'dyn_move_realized', 'dyn_move_other', 'dyn_stay', 'mob_label', 'nomob_label')}
    per_window_static, moving_steps, dyn_steps, pmove = [], 0, 0, {'static': [], 'dynamic': [], 'mob_label': [], 'nomob_label': []}
    for b0 in range(0, len(wins), 4):
        frames, ac, choices = BASE.batch(tokens, actions, wins, torch.arange(b0, min(b0 + 4, len(wins))))
        static, dynamic, rdir = classes(frames.float().cpu())
        labels = torch.stack([torch.from_numpy(np.array(lay[int(r), int(s):int(s) + 16]) > 0) for r, s in choices.tolist()])  # [b,16,63,4]
        mob_lab = labels[:, :15, :, :3].any(-1)                                                                            # [b,15,63]
        with torch.no_grad(), autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            ref = world.reference(frames, ac, belief, ha).float()
        grads = torch.zeros(len(frames), 15, 81, 5, device='cuda')
        probs = torch.zeros(len(frames), 15, 81, 5, device='cuda')
        for _ in range(8):
            noise = DM.draw_noise(frames.shape[:2], g).cuda(); gum = TM.draw_gumbel(frames.shape[:2], g).cuda()
            outs, qs = [], []
            for k in range(2):
                with torch.no_grad(), autocast_context(config):
                    hidden = comp.trunk(belief, noise[k])
                    logits = comp.propose(hidden).float().masked_fill(~comp.valid, -torch.inf)
                probs += logits.softmax(-1)[:, :15] / 16
                pert = logits + gum[k]
                choice = pert.argmax(-1)
                score = pert.gather(-1, choice[..., None])[..., 0] - pert[..., 0]
                partner = comp.matching(score, choice)
                q = pert.softmax(-1).detach().requires_grad_(True)
                hard = ref.gather(2, partner[..., None].expand(-1, -1, -1, 192))
                relax = torch.zeros_like(ref)
                for d in range(1, 5):
                    kk = comp.target[:, d]; other = ref[:, :, kk]; w = q[..., d:d + 1] * comp.valid[:, d].float()[:, None]
                    relax = relax + w * (other - ref)
                    relax = relax.index_add(2, kk, w * (ref - other))
                outs.append(F.layer_norm(hard + relax - relax.detach(), (192,))[:, :15]); qs.append(q)
            es = DM.losses(torch.stack(outs), frames[:, 1:])
            # per-frame sum so each frame's gradient is its own (losses averages over frames: undo the mean)
            total = es['energy'] * (frames.shape[0] * 15)
            gq = torch.autograd.grad(total, qs)
            grads += -(gq[0] + gq[1])[:, :15] / 16
        grads, probs = grads.cpu(), probs.cpu()
        for b in range(len(frames)):
            st = static[b]
            if st.any():
                v = grads[b, :, :63][:, st][..., 1:]                    # [15, n_static, 4] (invalid dirs carry 0 benefit and 0 prob)
                val = comp.valid[:63][st][:, 1:].cpu()[None].expand_as(v)
                acc['static'].append(v[val]); per_window_static.append(float(v[val].mean()))
                pmove['static'].append(float((1 - probs[b, :, :63][:, st][..., 0]).mean()))
            for j in torch.where(dynamic[b])[0].tolist():
                dyn_steps += 15
                pmove['dynamic'].append(float((1 - probs[b, :, j, 0]).mean()))
                for t in range(15):
                    d = int(rdir[b, t, j])
                    if d > 0:
                        moving_steps += 1
                        acc['dyn_move_realized'].append(grads[b, t, j, d:d + 1])
                        others = [x for x in range(1, 5) if x != d and bool(comp.valid[j, x])]
                        if others:
                            acc['dyn_move_other'].append(grads[b, t, j, others])
                    else:
                        valid = [x for x in range(1, 5) if bool(comp.valid[j, x])]
                        acc['dyn_stay'].append(grads[b, t, j, valid])
            for t in range(15):
                for j in range(63):
                    if j == 31:
                        continue
                    valid = [x for x in range(1, 5) if bool(comp.valid[j, x])]
                    (acc['mob_label'] if bool(mob_lab[b, t, j]) else acc['nomob_label']).append(grads[b, t, j, valid])
                    (pmove['mob_label'] if bool(mob_lab[b, t, j]) else pmove['nomob_label']).append(float(1 - probs[b, t, j, 0]))
    cat = {k: torch.cat(v) if v else torch.zeros(0) for k, v in acc.items()}
    se = float(np.std(per_window_static) / np.sqrt(len(per_window_static))) if per_window_static else None
    res = {'update': update, 'windows': len(wins), 'noise_pairs': 8,
           'benefit_per_unit_move_probability': {k: {'n': int(len(v)), 'mean': float(v.mean()) if len(v) else None,
                                                     'p10_p50_p90': [float(x) for x in torch.quantile(v.float(), torch.tensor([.1, .5, .9]))] if len(v) else None}
                                                 for k, v in cat.items()},
           'static_window_mean_se': se, 'static_window_means_mean': float(np.mean(per_window_static)) if per_window_static else None,
           'dynamic_moving_frequency_per_step': round(moving_steps / max(dyn_steps, 1), 4),
           'p_move_mean': {k: round(float(np.mean(v)), 4) if v else None for k, v in pmove.items()}}
    return res


@torch.no_grad()
def p_move_by_class(world, wins, actions, tokens, config):
    import direct_model as DM
    import direct_train as BASE
    import transport_model as TM
    from d4mj.train import autocast_context
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    g = torch.Generator().manual_seed(20261022)
    acc = {k: [] for k in ('static', 'cow', 'zombie', 'skeleton', 'no_mob')}
    for b0 in range(0, len(wins), 4):
        frames, ac, choices = BASE.batch(tokens, actions, wins, torch.arange(b0, min(b0 + 4, len(wins))))
        static, _, _ = classes(frames.float().cpu())
        labels = torch.stack([torch.from_numpy(np.array(lay[int(r), int(s):int(s) + 16]) > 0) for r, s in choices.tolist()])[:, :15]
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            pm = torch.zeros(len(frames), 15, 81)
            for _ in range(16):
                noise = DM.draw_noise(frames.shape[:2], g)[0].cuda()
                logits = world.completion.propose(world.completion.trunk(belief, noise)).float().masked_fill(~world.completion.valid, -torch.inf)
                pm += (1 - logits.softmax(-1)[..., 0])[:, :15].cpu() / 16
        pm = pm[..., :63]
        acc['static'].append(pm[static[:, None].expand(-1, 15, -1)])
        for k, name in enumerate(('zombie', 'cow', 'skeleton')):
            m = labels[..., k].clone(); m[..., 31] = False
            acc[name].append(pm[m])
        nm = ~labels[..., :3].any(-1); nm[..., 31] = False
        acc['no_mob'].append(pm[nm])
    return {k: round(float(torch.cat(v).mean()), 5) for k, v in acc.items()}


def probe(lam, updates=1000):
    import direct_model as DM
    import direct_train as BASE
    import loss_intervention as L
    import transport_model as TM
    from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    wins, _, _ = stationary_windows()
    st = torch.load(OUT / 'training/state.pt', map_location='cpu', weights_only=False)
    assert st['update'] == 3000
    worlds, opts = {}, {}
    for arm in ('no_cost', 'cost'):
        w = TM.TransportWorld('operator').cuda().train(); w.load_state_dict(st['worlds']['operator'])
        o = phase_optimizer([w], config); o.load_state_dict(st['optimizers']['operator'])
        worlds[arm], opts[arm] = w, o
    res = {'lambda': lam, 'updates': updates, 'start_u3000': p_move_by_class(worlds['cost'].eval(), wins, actions, tokens, config)}
    worlds['cost'].train()
    order = torch.Generator().manual_seed(20261030); noise_rng = torch.Generator().manual_seed(2026103001)
    log = []
    for u in range(updates):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda(); gumbel = TM.draw_gumbel(frames.shape[:2], noise_rng).cuda()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        row = {'update': 3000 + u + 1}
        for arm in ('no_cost', 'cost'):
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            w = worlds[arm]
            with autocast_context(config):
                belief, ha = w.backbone_full(frames, ac)
                outs, lgs = [], []
                for k in range(2):
                    o, aux = w.complete(frames, ac, belief, ha, noise[k], gumbel[k])
                    outs.append(o[:, :15]); lgs.append(aux['logits'][:, :15])
            values = DM.losses(torch.stack(outs), frames[:, 1:])
            changes = torch.stack([TM.expected_changes(x) for x in lgs]).mean()
            objective = values['energy'] + (lam * changes if arm == 'cost' else 0)
            optimizer_step(opts[arm], objective, list(w.parameters()), learning_rate=_phase_lr(config, 3000 + u),
                           grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
            row[arm] = {'energy': round(float(values['energy'].detach()), 6), 'expected_changes_per_frame': round(float(changes.detach()), 4)}
        if (u + 1) % 100 == 0:
            log.append(row); print(json.dumps(row), flush=True)
    for arm in ('no_cost', 'cost'):
        res[f'end_{arm}'] = p_move_by_class(worlds[arm].eval(), wins, actions, tokens, config)
    a, b = res['end_cost'], res['end_no_cost']
    res['static_reduction_vs_twin'] = round(1 - a['static'] / b['static'], 4)
    res['cow_reduction_vs_twin'] = round(1 - a['cow'] / b['cow'], 4)
    res['pass'] = res['static_reduction_vs_twin'] >= .30 and res['cow_reduction_vs_twin'] <= .15
    res['log'] = log
    return res


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'crossterrain':
        res, name = crossterrain(), 'crossterrain'
    elif mode == 'benefit':
        res, name = benefit(int(sys.argv[2])), f'benefit_u{int(sys.argv[2]):04d}'
    elif mode == 'probe':
        res, name = probe(float(sys.argv[2])), 'probe'
    else:
        raise SystemExit(mode)
    (OUT / f'cost_diag_{name}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
