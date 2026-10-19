"""Inferred-chance design checks (2026-10-15), before any IC training. Simulator layouts are diagnostic only.

  calib     (CPU) does the label convention keep the sampler calibrated? The operator's sampler (independent Gumbel-max
            proposals per tile + the parallel greedy matching) is run on synthetic boards where every isolated adjacent
            pair has a target swap probability p. Convention BOTH: each endpoint proposes the swap with probability p (the
            preregistered "label at both endpoints"); convention CANONICAL: only the up/left endpoint of each unordered pair
            proposes it (down/right direction) with probability p. Reported: accepted swap frequency / p.
  posterior (GPU inference) the label-free posterior on TRAINING windows (seeded sample of the training ledger, layout rows
            only), with the initial reference (the D parent) and a trained operator's reference. For each interior adjacent
            pair: gain = |a_j - z_j| + |a_k - z_k| - |a_k - z_j| - |a_j - z_k| (first-power, the afterstate a = LN(b) vs the
            observed successor z). Rules:
              P0  gain > 0 (as preregistered)
              P1  Laplace-likelihood MAP with a measured scale and an estimated prior: under p(z|a,T) ~ exp(-sum_j |z_j -
                  (Ta)_j| / s) the swap log-likelihood ratio is gain / s; s = the frame's mean residual norm |z_j - a_j| over
                  interior cells; prior per pair pi estimated by EM on this sample (start from P0's rate, 3 iterations);
                  accept if gain / s > log((1 - pi) / pi). No label enters; nothing is tuned on labels.
            Scored against T* (simulator, diagnostic): recall of true interior mob moves, precision, false swaps per frame
            (same / different decoded terrain, touching a mob), split day / night by the frozen bank light reader.
  misses    (GPU inference) why true moves are not inferred: every true interior mob move on the same training windows is
            classed by what the afterstate a already holds (frozen real mob reader on a): PREDICTED (a shows the mob at the
            destination and not at the source: the deterministic part already moved it, nothing to sample), GHOST (a shows
            it at both or neither: blurred/faded), STAY (a shows it at the source only: a genuine chance event), and for
            STAY moves whether P0 / P1 captured the swap. By mob type and camera still / moving.
  milestones  (GPU inference) TRANSPORT_IC diagnostic: the v3 posterior (transport_ic_train.infer under each checkpoint's own
            chance-head prior) on the same 400 training windows, for every IC milestone of both seeds: share of STAY-class
            (genuine chance) moves captured, accepted swaps per frame, false accepted swaps per frame by class (scored vs T*).
  pi        (CPU inference, 2026-10-15 night, NOTEBOOK R23: PI's zombies under-chase) the same 400 training windows and the
            same true STAY-class moves (layouts, diagnostic only), under each PI u3000 checkpoint's TRAINING-TIME inference
            (pi_train.losses: copy under the label-free scroll estimate, provisional afterstate with the gate's own decisions,
            the IC v3 posterior with the chance head's prior, then the edit posterior q > 0.5). Each true move is classed:
            SWAP (the posterior's swap captures it), EDIT (no swap; the edit posterior regenerates its source or destination),
            MISSED (neither). By mob type. If EDIT is large, the gate absorbs moves the chance head should learn.
Usage: ic_posterior.py calib | posterior | misses | milestones | pi -> artifacts/eda/oracle_v1/transport/ic_<mode>.json
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
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8)]


def calib():
    import transport_model as TM
    comp = TM.TransportCompletion('operator')
    g = torch.Generator().manual_seed(1)
    # isolated pairs: tile j = (r, c) with partner (r, c+1); rows 1,3,5 cols 1,4 -> no shared cells, no other proposals
    pairs = [(r * 9 + c, r * 9 + c + 1) for r in (1, 3, 5) for c in (1, 4) if 31 not in (r * 9 + c, r * 9 + c + 1)]   # the player never moves
    res = {}
    for p in (.02, .05, .1, .2, .4):
        for conv in ('both', 'canonical'):
            logp = torch.full((4000, 81, 5), -30.0); logp[..., 0] = 0.0
            for j, k in pairs:
                # direction 4 = right (j -> k), direction 3 = left (k -> j)
                logp[:, j, 4] = np.log(p) - np.log(1 - p)
                if conv == 'both':
                    logp[:, k, 3] = np.log(p) - np.log(1 - p)
            logits = logp.masked_fill(~comp.valid, -torch.inf)
            u = torch.rand(4000, 81, 5, generator=g).clamp(1e-10, 1 - 1e-10)
            pert = logits - torch.log(-torch.log(u))
            choice = pert.argmax(-1)
            score = pert.gather(-1, choice[..., None])[..., 0] - pert[..., 0]
            partner = comp.matching(score[:, None], choice[:, None])[:, 0]
            freq = float(np.mean([(partner[:, j] == k).float().mean() for j, k in pairs]))
            res[f'p{p}_{conv}'] = {'accepted_freq': round(freq, 4), 'ratio_to_target': round(freq / p, 3)}
    return res


@torch.no_grad()
def references(names, n_windows=400):
    import direct_train as BASE
    import paired_model as PM
    import transport_objective as TO
    from d4mj.train import autocast_context
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261016))[:n_windows]
    frames_all, ac_all, choices_all = [], [], []
    for b0 in range(0, n_windows, 8):
        frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
        frames_all.append(frames.half()); ac_all.append(ac); choices_all.append(choices)
    frames = torch.cat(frames_all); ac = torch.cat(ac_all); choices = torch.cat(choices_all)
    base_state = torch.load(TO.REFERENCES['D_parent'], map_location='cpu', weights_only=False)['world']
    refs = {}
    for name in names:
        saved = torch.load(TO.REFERENCES[name], map_location='cpu', weights_only=False)['world']
        state = dict(base_state); state.update({k: v for k, v in saved.items() if not k.startswith('completion.')})
        world = PM.PairedWorld().cuda().eval(); world.load_state_dict(state)
        out = []
        for b0 in range(0, n_windows, 8):
            fr = frames[b0:b0 + 8].cuda().float(); a = ac[b0:b0 + 8].cuda()
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, a)
                out.append(F.layer_norm(world.reference(fr, a, belief, ha)[:, :15].float(), (192,)).half().cpu())
        refs[name] = torch.cat(out)                                         # [n,15,81,192] afterstates for steps 1..15
        del world; torch.cuda.empty_cache()
    return frames, choices, refs


def gains(a, z, pairs):
    """a, z [m,81,192] -> gain [m,P], keep residual norms [m,81]"""
    res = (a - z).norm(dim=-1)                                              # [m,81]
    j = torch.tensor([p[0] for p in pairs]); k = torch.tensor([p[1] for p in pairs])
    cross = (a[:, k] - z[:, j]).norm(dim=-1) + (a[:, j] - z[:, k]).norm(dim=-1)
    return res[:, j] + res[:, k] - cross, res


def accept(gain, pairs, bar):
    """greedy one-to-one acceptance per frame by descending gain over gain > bar; returns list of sets"""
    out = []
    for m in range(len(gain)):
        g = gain[m]; b = bar[m] if bar.ndim else bar
        order = torch.argsort(g, descending=True).tolist()
        used, acc = set(), set()
        for q in order:
            if g[q] <= b:
                break
            j, k = pairs[q]
            if j in used or k in used:
                continue
            used |= {j, k}; acc.add(frozenset((j, k)))
        out.append(acc)
    return out


def posterior():
    import direct_coherence as C
    import transport_objective as TO
    names = ('D_parent', 'stage3_seed1_operator')
    frames, choices, refs = references(names)
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    pos = {x: i for i, x in enumerate(INTERIOR)}
    pairs = [(x, x + d) for x in INTERIOR for d in (1, 9) if x + d in pos and 31 not in (x, x + d)]
    n = len(frames)
    z = frames[:, 1:].float()                                               # [n,15,81,192] observed successors
    # truth and strata (diagnostic)
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())   # [n,15]
    truth, mobcells = [], []
    for i, (row, start) in enumerate(choices.tolist()):
        L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
        for t in range(15):
            prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
            truth.append({frozenset((s_, d_)) for s_, d_, ty in S2.swaps(prev, L[t + 1]) if ty < 3 and s_ in pos and d_ in pos and 31 not in (s_, d_)})
            mobcells.append(set(torch.where(L[t + 1][:, :3].any(-1) | (prev[:, :3] == 1).any(-1))[0].tolist()))
    lt = frames[:, :-1, :63].float().mean(2)                                # [n,15,192] mean map token per frame
    hl = bank['heads']['light']
    zl = ((lt - hl['mu']) / hl['sd']).double()
    light = (torch.cat([zl, torch.ones(*zl.shape[:-1], 1, dtype=torch.float64)], -1) @ hl['w'].double())[..., 0].float()
    night = (light < .5).flatten()
    res = {'windows': n, 'frames': n * 15, 'night_share': round(float(night.float().mean()), 4), 'pairs_per_frame': len(pairs), 'rules': {}}
    for name, ref in refs.items():
        a = ref.float().view(-1, 81, 192); zz = z.reshape(-1, 81, 192)
        tiles = C.decode(a.view(n, 15, 81, 192), bank, head)['tile'].view(-1, 63)
        gain, resid = gains(a, zz, pairs)
        s = resid[:, INTERIOR].mean(-1)                                     # per-frame Laplace scale estimate
        rules = {'P0': torch.zeros(len(gain))}
        pi = max(np.mean([len(x) for x in accept(gain, pairs, torch.zeros(len(gain)))]) / len(pairs), 1e-4)
        for _ in range(3):
            bar = s * float(np.log((1 - pi) / pi))
            acc = accept(gain, pairs, bar)
            pi = max(np.mean([len(x) for x in acc]) / len(pairs), 1e-4)
        rules['P1'] = s * float(np.log((1 - pi) / pi))
        for rule, bar in rules.items():
            acc = accept(gain, pairs, bar)
            for part, mask in (('day', ~night), ('night', night)):
                tp = fn = fp_same = fp_diff = fp_mob = 0
                frames_n = int(mask.sum())
                for m in torch.where(mask)[0].tolist():
                    t, got = truth[m], acc[m]
                    tp += len(t & got); fn += len(t - got)
                    for pr in got - t:
                        j, k = tuple(pr)
                        if j in mobcells[m] or k in mobcells[m]:
                            fp_mob += 1
                        elif tiles[m, j] == tiles[m, k]:
                            fp_same += 1
                        else:
                            fp_diff += 1
                fp = fp_same + fp_diff + fp_mob
                res['rules'].setdefault(name, {}).setdefault(rule, {})[part] = {
                    'frames': frames_n, 'true_moves': tp + fn, 'recall': round(tp / max(tp + fn, 1), 4), 'precision': round(tp / max(tp + fp, 1), 4),
                    'false_per_frame': {'same_terrain': round(fp_same / max(frames_n, 1), 4), 'different_terrain': round(fp_diff / max(frames_n, 1), 4),
                                        'touching_mob': round(fp_mob / max(frames_n, 1), 4)}}
            if rule == 'P1':
                res['rules'][name]['P1']['prior_pi'] = round(pi, 5)
                res['rules'][name]['P1']['scale_s_day_night_median'] = [round(float(s[~night].median()), 4), round(float(s[night].median()), 4) if night.any() else None]
        print(json.dumps({name: res['rules'][name]}), flush=True)
    return res


def misses():
    import direct_coherence as C
    names = ('D_parent', 'stage3_seed1_operator')
    frames, choices, refs = references(names)
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    pos = {x: i for i, x in enumerate(INTERIOR)}
    pairs = [(x, x + d) for x in INTERIOR for d in (1, 9) if x + d in pos and 31 not in (x, x + d)]
    n = len(frames)
    z = frames[:, 1:].float()
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
    moves = []                                                              # (frame m, src, dst, type, camera moving)
    for i, (row, start) in enumerate(choices.tolist()):
        L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
        for t in range(15):
            prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
            for s_, d_, ty in S2.swaps(prev, L[t + 1]):
                if ty < 3 and s_ in pos and d_ in pos and 31 not in (s_, d_):
                    moves.append((i * 15 + t, s_, d_, ty, int(est[i, t]) != 0))
    res = {'true_moves': len(moves)}
    for name, ref in refs.items():
        a = ref.float().view(-1, 81, 192); zz = z.reshape(-1, 81, 192)
        dec = C.decode(a, bank, head)['mob']['real'][..., :3]                    # [m,63,3] layout order
        gain, resid = gains(a, zz, pairs)
        s = resid[:, INTERIOR].mean(-1)
        acc0 = accept(gain, pairs, torch.zeros(len(gain)))
        pi = 0.00155 if name == 'D_parent' else 0.00262                       # the EM priors measured by `posterior`
        acc1 = accept(gain, pairs, s * float(np.log((1 - pi) / pi)))
        tab = {}
        for m, s_, d_, ty, cam in moves:
            at_src, at_dst = bool(dec[m, s_, ty]), bool(dec[m, d_, ty])
            cls = 'PREDICTED' if (at_dst and not at_src) else ('STAY' if (at_src and not at_dst) else 'GHOST')
            key = f"{('zombie', 'cow', 'skeleton')[ty]}|{'moving' if cam else 'still'}"
            row = tab.setdefault(key, {'n': 0, 'PREDICTED': 0, 'GHOST': 0, 'STAY': 0, 'STAY_captured_P0': 0, 'STAY_captured_P1': 0})
            row['n'] += 1; row[cls] += 1
            if cls == 'STAY':
                pr = frozenset((s_, d_))
                row['STAY_captured_P0'] += int(pr in acc0[m]); row['STAY_captured_P1'] += int(pr in acc1[m])
        res[name] = tab
        tot = {k: sum(r[k] for r in tab.values()) for k in ('n', 'PREDICTED', 'GHOST', 'STAY', 'STAY_captured_P0', 'STAY_captured_P1')}
        res[name + '_total'] = tot
        print(json.dumps({name: tot}), flush=True)
    return res


@torch.no_grad()
def milestones():
    import direct_coherence as C
    import direct_model as DM
    import direct_train as BASE
    import transport_model as TM
    import transport_ic_train as IC
    from d4mj.train import autocast_context
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261016))[:400]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    pos = {x: i for i, x in enumerate(INTERIOR)}
    res = {}
    for seed in (1, 2):
        for u in (0, 500, 1500, 3000):
            path = OUT / f'ic_seed{seed}/ic_u{u:04d}.pt'
            if not path.exists():
                continue
            world = TM.TransportWorld('operator').cuda().eval(); world.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['world'])
            g = torch.Generator().manual_seed(1)
            tally = {'n_stay': 0, 'stay_captured': 0, 'accepted': 0, 'false_terrain': 0, 'false_mob': 0, 'frames': 0}
            for b0 in range(0, 400, 8):
                frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
                noise = DM.draw_noise(frames.shape[:2], g)[0].cuda()
                with autocast_context(config):
                    belief, ha = world.backbone_full(frames, ac)
                    b = world.reference(frames, ac, belief, ha).float()[:, :15]
                    logits = world.completion.propose(world.completion.trunk(belief, noise)).float()[:, :15]
                logits = logits.masked_fill(~world.completion.valid, -torch.inf)
                a = F.layer_norm(b, (192,)); z = frames[:, 1:].float()
                partner, _, nacc, _, _, _ = IC.infer(a, z, IC.pair_prior_logit(logits))
                dec = C.decode(a.cpu().reshape(-1, 81, 192), bank, head)['mob']['real'][..., :3]
                est = scroll.estimate(frames[:, :-1].float().cpu(), frames[:, 1:].float().cpu())
                partner = partner.cpu()
                for i, (row, start) in enumerate(choices.tolist()):
                    L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
                    for t in range(15):
                        m = i * 15 + t
                        prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
                        truth = {frozenset((s_, d_)): ty for s_, d_, ty in S2.swaps(prev, L[t + 1]) if ty < 3 and s_ in pos and d_ in pos and 31 not in (s_, d_)}
                        got = {frozenset((x, int(partner[i, t, x]))) for x in INTERIOR if int(partner[i, t, x]) != x}
                        tally['frames'] += 1; tally['accepted'] += len(got)
                        for pr, ty in truth.items():
                            s_, d_ = tuple(pr)
                            src, dst = (s_, d_) if bool(prev[s_, ty] == 1) else (d_, s_)
                            if bool(dec[m, src, ty]) and not bool(dec[m, dst, ty]):
                                tally['n_stay'] += 1; tally['stay_captured'] += int(pr in got)
                        mobs = set(torch.where(L[t + 1][:, :3].any(-1) | (prev[:, :3] == 1).any(-1))[0].tolist())
                        for pr in got - set(truth):
                            x, y = tuple(pr)
                            tally['false_mob' if (x in mobs or y in mobs) else 'false_terrain'] += 1
            f = tally['frames']
            res[f'seed{seed}_u{u}'] = {'stay_capture': round(tally['stay_captured'] / max(tally['n_stay'], 1), 4), 'stay_events': tally['n_stay'],
                                       'accepted_per_frame': round(tally['accepted'] / f, 4), 'false_terrain_per_frame': round(tally['false_terrain'] / f, 4),
                                       'false_mob_per_frame': round(tally['false_mob'] / f, 4)}
            print(json.dumps({f'seed{seed}_u{u}': res[f'seed{seed}_u{u}']}), flush=True)
            del world; torch.cuda.empty_cache()
    return res


def pi_capture():
    import dataclasses
    import direct_coherence as C
    import direct_model as DM
    import direct_train as BASE
    import transport_ic_train as IC
    sys.path.insert(0, str(HERE.parent / '20261015_whiteboard')); import pi_model as PI
    from d4mj.mamba_recurrence import FunctionalMamba2
    torch.set_num_threads(8)
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261016))[:400]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    pos = {x: i for i, x in enumerate(INTERIOR)}
    res = {}
    for seed in (1, 2):
        world = PI.PIWorld().eval(); world.load_state_dict(torch.load(OUT / f'pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        for m_ in world.modules():
            if isinstance(m_, FunctionalMamba2):
                m_.settings = dataclasses.replace(m_.settings, backend='reference')
        g = torch.Generator().manual_seed(1)
        tally = {t: {'n': 0, 'swap': 0, 'edit': 0, 'missed': 0} for t in ('zombie', 'cow', 'skeleton')}
        frames_n = 0
        with torch.no_grad():
            for b0 in range(0, 400, 8):
                frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
                noise = DM.draw_noise(frames.shape[:2], g)[0]
                est = scroll.estimate(frames[:, :-1], frames[:, 1:])
                belief, ha = world.backbone_full(frames, ac)
                copy, gen, glogit, enter, _ = world.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
                logits = world.completion.propose(world.completion.trunk(belief, noise)).float()
                copy, gen, glogit, enter, logits = (x[:, :15] for x in (copy, gen, glogit, enter, logits))
                logits = logits.masked_fill(~world.completion.valid, -torch.inf)
                z = frames[:, 1:]
                a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
                partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
                z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
                res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
                gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
                s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
                edit = (torch.sigmoid(gain / s + glogit) > .5) | enter
                dec = C.decode(F.layer_norm(a0, (192,)).reshape(-1, 81, 192), bank, head)['mob']['real'][..., :3]
                for i, (row, start) in enumerate(choices.tolist()):
                    L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
                    for t in range(15):
                        m = i * 15 + t
                        frames_n += 1
                        prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
                        truth = {frozenset((s_, d_)): ty for s_, d_, ty in S2.swaps(prev, L[t + 1]) if ty < 3 and s_ in pos and d_ in pos and 31 not in (s_, d_)}
                        for pr, ty in truth.items():
                            s_, d_ = tuple(pr)
                            src, dst = (s_, d_) if bool(prev[s_, ty] == 1) else (d_, s_)
                            if not (bool(dec[m, src, ty]) and not bool(dec[m, dst, ty])):
                                continue                                    # STAY class only, as milestones
                            k = ('zombie', 'cow', 'skeleton')[ty]
                            tally[k]['n'] += 1
                            if int(partner[i, t, src]) == dst:
                                tally[k]['swap'] += 1
                            elif bool(edit[i, t, src]) or bool(edit[i, t, dst]):
                                tally[k]['edit'] += 1
                            else:
                                tally[k]['missed'] += 1
                if b0 % 80 == 0:
                    print(json.dumps({'seed': seed, 'windows_done': b0 + 8}), flush=True)
        res[f'pi_seed{seed}'] = {k: {**v, **{f'{c}_share': round(v[c] / max(v['n'], 1), 4) for c in ('swap', 'edit', 'missed')}} for k, v in tally.items()}
        res[f'pi_seed{seed}']['frames'] = frames_n
        print(json.dumps({f'pi_seed{seed}': res[f'pi_seed{seed}']}), flush=True)
    return res


if __name__ == '__main__':
    mode = sys.argv[1]
    res = {'calib': calib, 'posterior': posterior, 'misses': misses, 'milestones': milestones, 'pi': pi_capture}[mode]()
    (OUT / f'ic_{mode}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])
