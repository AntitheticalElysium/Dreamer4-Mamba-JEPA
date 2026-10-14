"""Stage 3 frozen diagnostics (declared in TRANSPORT_PREREG.md; not gates). Operator checkpoint at a milestone, the 104 bright
DEV roots, the evaluation noise rule (64 draws, seed 2026101402 + start), predicted step 14:
  proposals  the Gumbel-max sampling probabilities softmax(P(e)) (e depends on the noise through the trunk), averaged over
             draws, grouped by the afterstate content of the tile (root layout aligned by the exact camera offset: zombie /
             cow / skeleton / no mob); P(move) per group against the real one-step move rate of that content in the fan
             (adjacent same-type swaps / mob cells); at the 14 strict sources, the direction distribution of ACCEPTED swaps
             vs the real event distribution; non-mob swaps per frame.
  payload    provenance: at strict sources and draws where the source was swapped, replace the source's reference token by a
             non-mob interior token of the same root before the (fixed) permutation; the destination's decoded mob must vanish.
  margin     (CPU, model-free) does the training objective prefer the TRUE movement law over always-stay, and by how much?
             Per root: keys 0:16 are samples X of the true law; X_stay = the same successors with their adjacent swaps undone
             (T* is an involution: every mob back at its source, everything else unchanged); keys 16:32 are the targets Y.
             ES = mean d(X,Y) - .5 mean_{i!=j} d(X_i,X_j) with the exact training distance direct_model.distance (35
             scored tokens). Reported for the 14 strict roots and all roots with any in-view move; plus the attraction-only
             comparison (the per-sample term a straight-through estimator sees first).
  gradsplit  (GPU, frozen operator checkpoint) what the training signal asks of the source tile's proposal: on each of the
             14 strict roots, 32 draw pairs (evaluation noise rule), the straight-through gradient of the energy objective
             w.r.t. the source tile's move logits, split into attraction (mean distance to all 32 real successors) and
             diversity (-.5 d(X1, X2)). Reported: the push on P(move) = -(sum over move logits of dL/dlogit) for each term,
             averaged over pairs (positive = the term asks the source to move more), and the share of pairs whose two draws
             differ at the source cross. The same push is also reported at the root's interior tiles that hold no mob
             (exact layout, aligned; evaluation only) and are not adjacent to one (terrain-only proposals).
  swaps      (CPU, saved evaluation samples) what the accepted swaps are: exact root tiles and layouts (evaluation only)
             aligned by the camera offset; each swap classed mob<->passable (grass/path/sand), mob<->solid (Craftax
             SOLID_BLOCKS or lava: an impossible move), mob<->mob, terrain<->same class, terrain<->different class (static
             damage), or touching an entering cell; per frame. Scored static corruption (frozen-evaluator definition: decoded
             tile != exact root tile on scored cells whose exact facts are stable across all 32 keys) for the operator's
             samples vs its own deterministic reference alone (T = I), and the same for the free arm's samples.
Usage: transport_diag.py <update> | margin | gradsplit <update> | swaps <update> -> transport/diag_u<update>.json | diag_margin.json | diag_gradsplit_u<update>.json
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


@torch.no_grad()
def main(update):
    import direct_model as DM
    import direct_train as BASE
    import direct_eval as E
    import direct_coherence as C
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    data, _ = E.cohort('exploratory')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(OUT / f'training/operator_u{update:04d}.pt', map_location='cpu', weights_only=False)['world'])
    units = json.loads((A5.OUT / 'audit_recompute.json').read_text())['sources_list']
    strict = {root: (source, ('zombie', 'cow', 'skeleton').index(kind)) for root, source, kind in units}
    n = len(data['roots'])
    groups = {g: [] for g in ('zombie', 'cow', 'skeleton', 'no_mob')}
    real_rate = {g: [0, 0] for g in ('zombie', 'cow', 'skeleton')}
    strict_dirs, real_dirs, nonmob_swaps, frames_seen, payload = {}, {}, 0, 0, {'checked': 0, 'destination_mob_vanished': 0}
    for start in range(0, n, 4):
        ctx = data['context_tokens'][start:start + 4].cuda().float(); ac = data['actions'][start:start + 4].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
        g = torch.Generator().manual_seed(2026101402 + start)
        noises = torch.cat([DM.draw_noise((len(ctx), 16), g) for _ in range(32)])
        gumbels = torch.cat([TM.draw_gumbel((len(ctx), 16), g) for _ in range(32)])
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            ref = world.reference(frames, ac, belief, ha).float()
            probs, partners = [], []
            for k in range(64):
                hidden = world.completion.trunk(belief, noises[k].cuda())
                logits = world.completion.propose(hidden).float().masked_fill(~world.completion.valid, -torch.inf)
                probs.append(logits[:, 14].softmax(-1).cpu())
                _, aux = world.completion(ref, belief, noises[k].cuda(), gumbels[k].cuda())
                partners.append(aux['partner'][:, 14].cpu())
        probs = torch.stack(probs, 1); partners = torch.stack(partners, 1)      # [b,64,81,5], [b,64,81]
        ref14 = ref[:, 14].cpu()
        for b in range(len(ctx)):
            i = start + b
            dr, dc = (int(x) for x in data['shifts'][i, 0])
            after = S2.afterstate_layout(data['root_layout'][i], dr, dc)        # [63,4], -1 entering
            pm = 1 - probs[b, :, :63, 0].mean(0)                                 # P(move) per map tile
            for x in range(63):
                if x == 31 or after[x, 0] < 0:
                    continue
                kinds = [k for k in range(3) if after[x, k] == 1]
                groups[('zombie', 'cow', 'skeleton')[kinds[0]] if kinds else 'no_mob'].append(float(pm[x]))
            for k in range(32):
                for s, dst, kind in S2.swaps(after, data['fan_layout'][i, k].bool()):
                    if kind < 3:
                        real_rate[('zombie', 'cow', 'skeleton')[kind]][0] += 1
            for kind in range(3):
                real_rate[('zombie', 'cow', 'skeleton')[kind]][1] += 32 * int((after[:, kind] == 1).sum())
            moved = partners[b] != torch.arange(81)
            mob = (after[:, :3] == 1).any(-1)
            nonmob_swaps += int((moved[:, :63] & ~mob).sum())             # swapped map tiles holding no mob, over 64 draws
            frames_seen += 64
            if i in strict:
                source, kind = strict[i]
                r, c = divmod(source, 9)
                pts = [(r + a) * 9 + c + e for a, e in A5.CROSS]
                p = partners[b][:, source]
                d = [pts.index(int(q)) if int(q) in pts else -1 for q in p]
                strict_dirs[i] = np.bincount(np.array(d) + 1, minlength=6)[1:].tolist()
                real_dirs[i] = np.bincount(A5.event(data['fan_layout'][i][:, pts, kind].bool()).numpy(), minlength=7)[:5].tolist()
                # payload substitution: replace the source content by a non-mob interior token, keep the permutation fixed
                empties = [x for x in A5.INTERIOR if not bool(mob[x]) and x != 31 and abs(x // 9 - r) + abs(x % 9 - c) > 2]
                for k in range(64):
                    dst = int(p[k])
                    if dst == source or not empties:
                        continue
                    content = ref14[b].clone(); content[source] = ref14[b, empties[0]]
                    out = F.layer_norm(content.gather(0, partners[b, k][:, None].expand(-1, 192)), (192,))
                    before = F.layer_norm(ref14[b].gather(0, partners[b, k][:, None].expand(-1, 192)), (192,))
                    m_after = C.decode(out[None], bank, head)['mob']['real'][0, dst, kind]
                    m_before = C.decode(before[None], bank, head)['mob']['real'][0, dst, kind]
                    if bool(m_before):
                        payload['checked'] += 1; payload['destination_mob_vanished'] += int(not bool(m_after))
    res = {'update': update,
           'p_move_by_afterstate_content': {g: {'tiles': len(v), 'mean': round(float(np.mean(v)), 4) if v else None} for g, v in groups.items()},
           'real_one_step_move_rate': {g: round(v[0] / max(v[1], 1), 4) for g, v in real_rate.items()},
           'swapped_nonmob_map_tiles_per_frame': round(nonmob_swaps / max(frames_seen, 1), 4),
           'strict_sources_accepted_direction_counts[stay,up,down,left,right]': {str(k): v for k, v in strict_dirs.items()},
           'strict_sources_real_event_counts[stay,up,down,left,right]': {str(k): v for k, v in real_dirs.items()},
           'payload_substitution': payload}
    (OUT / f'diag_u{update:04d}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def margin():
    import direct_model as DM
    import direct_eval as E
    data, _ = E.cohort('exploratory')
    units = json.loads((A5.OUT / 'audit_recompute.json').read_text())['sources_list']
    strict_roots = {u[0] for u in units}
    rows = []
    for i in range(len(data['roots'])):
        dr, dc = (int(x) for x in data['shifts'][i, 0])
        after = S2.afterstate_layout(data['root_layout'][i], dr, dc)
        Y = data['fan_tokens'][i].float()
        pairs = [S2.swaps(after, data['fan_layout'][i, k].bool()) for k in range(32)]
        if not any(pairs[k] for k in range(32)):
            continue
        X = Y[:16]
        Xs = torch.stack([S2.apply(Y[k], pairs[k]) for k in range(16)])
        T = Y[16:]
        def es(x):
            att = DM.distance(x[:, None], T[None]).mean()
            dif = DM.distance(x[:, None], x[None])
            div = dif.sum() / (len(x) * (len(x) - 1))
            return float(att), float(att - .5 * div)
        (a_true, e_true), (a_stay, e_stay) = es(X), es(Xs)
        rows.append({'root': i, 'strict': i in strict_roots, 'moves_in_samples': sum(len(pairs[k]) for k in range(16)),
                     'ES_true': e_true, 'ES_stay': e_stay, 'att_true': a_true, 'att_stay': a_stay})
    def summary(sel):
        r = [x for x in rows if sel(x)]
        f = lambda k: float(np.mean([x[k] for x in r]))
        return {'roots': len(r), 'ES_true': round(f('ES_true'), 6), 'ES_stay': round(f('ES_stay'), 6),
                'ES_margin_stay_minus_true': round(f('ES_stay') - f('ES_true'), 6),
                'ES_margin_relative': round((f('ES_stay') - f('ES_true')) / f('ES_true'), 4),
                'attraction_true': round(f('att_true'), 6), 'attraction_stay': round(f('att_stay'), 6),
                'roots_where_ES_prefers_true': sum(x['ES_stay'] > x['ES_true'] for x in r),
                'roots_where_attraction_prefers_true': sum(x['att_stay'] > x['att_true'] for x in r)}
    res = {'strict_roots': summary(lambda x: x['strict']), 'all_roots_with_moves': summary(lambda x: True), 'rows': rows}
    (OUT / 'diag_margin.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


def gradsplit(update):
    import direct_model as DM
    import direct_train as BASE
    import direct_eval as E
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    data, _ = E.cohort('exploratory')
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(OUT / f'training/operator_u{update:04d}.pt', map_location='cpu', weights_only=False)['world'])
    for prm in world.parameters():
        prm.requires_grad_(False)
    store = {}
    hook = world.completion.propose.register_forward_hook(lambda m, i, o: (o.requires_grad_(True), o.retain_grad(), store.__setitem__('logits', o))[-1])
    units = json.loads((A5.OUT / 'audit_recompute.json').read_text())['sources_list']
    rows = []
    for root, source, kind in units:
        start = root - root % 4; b = root % 4
        ctx = data['context_tokens'][start:start + 4].cuda().float(); ac = data['actions'][start:start + 4].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
        g = torch.Generator().manual_seed(2026101402 + start)
        noises = torch.cat([DM.draw_noise((len(ctx), 16), g) for _ in range(32)])
        gumbels = torch.cat([TM.draw_gumbel((len(ctx), 16), g) for _ in range(32)])
        with torch.no_grad(), autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
        r, c = divmod(source, 9)
        pts = [(r + a) * 9 + c + e for a, e in A5.CROSS]
        Y = data['fan_tokens'][root].float().cuda()
        att_push, div_push, differ = [], [], 0
        lay = S2.afterstate_layout(data['root_layout'][root], *(int(x) for x in data['shifts'][root, 0]))
        mob = (lay[:, :3] == 1).any(-1)
        nearmob = mob.clone()
        for x in torch.where(mob)[0].tolist():
            for a, e in A5.CROSS:
                rr, cc = divmod(x, 9); rr += a; cc += e
                if 0 <= rr < 7 and 0 <= cc < 9:
                    nearmob[rr * 9 + cc] = True
        terrain = [x for x in A5.INTERIOR if x != 31 and not bool(nearmob[x]) and lay[x, 0] >= 0]
        att_t, div_t = [], []
        for k in range(0, 64, 2):
            outs, logits = [], []
            for j in (k, k + 1):
                with autocast_context(config):
                    o, aux = world.complete(frames, ac, belief, ha, noises[j].cuda(), gumbels[j].cuda())
                outs.append(o[b, 14]); logits.append(store['logits'])
            x1, x2 = outs
            att = .5 * (DM.distance(x1[None], Y).mean() + DM.distance(x2[None], Y).mean())
            div = -.5 * DM.distance(x1[None], x2[None]).mean()
            ga = torch.autograd.grad(att, logits, retain_graph=True)
            gd = torch.autograd.grad(div, logits)
            att_push.append(-float(sum(x[b, 14, source, 1:].float().sum() for x in ga)))
            div_push.append(-float(sum(x[b, 14, source, 1:].float().sum() for x in gd)))
            att_t.append(-float(sum(x[b, 14, terrain, 1:].float().sum(-1).mean() for x in ga)))
            div_t.append(-float(sum(x[b, 14, terrain, 1:].float().sum(-1).mean() for x in gd)))
            differ += int(bool((x1[pts] - x2[pts]).abs().max() > 1e-4))
        rows.append({'root': root, 'kind': kind, 'attraction_push_on_move_mean': float(np.mean(att_push)),
                     'diversity_push_on_move_mean': float(np.mean(div_push)), 'pairs_differing_at_source_cross': differ / 32,
                     'terrain_tiles': len(terrain), 'terrain_attraction_push_mean': float(np.mean(att_t)), 'terrain_diversity_push_mean': float(np.mean(div_t))})
        print(json.dumps(rows[-1]), flush=True)
    hook.remove()
    res = {'update': update, 'rows': rows,
           'mean_attraction_push': float(np.mean([x['attraction_push_on_move_mean'] for x in rows])),
           'mean_diversity_push': float(np.mean([x['diversity_push_on_move_mean'] for x in rows])),
           'roots_where_total_push_favours_moving': sum(x['attraction_push_on_move_mean'] + x['diversity_push_on_move_mean'] > 0 for x in rows),
           'mean_share_pairs_differing_at_source_cross': float(np.mean([x['pairs_differing_at_source_cross'] for x in rows])),
           'terrain_mean_attraction_push': float(np.mean([x['terrain_attraction_push_mean'] for x in rows])),
           'terrain_mean_diversity_push': float(np.mean([x['terrain_diversity_push_mean'] for x in rows])),
           'roots_where_terrain_total_push_favours_moving': sum(x['terrain_attraction_push_mean'] + x['terrain_diversity_push_mean'] > 0 for x in rows)}
    (OUT / f'diag_gradsplit_u{update:04d}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


def swaps_mode(update):
    import direct_eval as E
    import direct_coherence as C
    data, _ = E.cohort('exploratory')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    d = OUT / f'eval_u{update:04d}'
    op = torch.load(d / 'pred_operator.pt', map_location='cpu', weights_only=False)
    fr = torch.load(d / 'pred_free.pt', map_location='cpu', weights_only=False)
    solid = {3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 14}
    passable = {2, 7, 13}
    counts = dict.fromkeys(('mob_passable', 'mob_solid', 'mob_mob', 'terrain_same', 'terrain_different', 'entering_or_unknown'), 0)
    n, K = op['partner'].shape[:2]
    for i in range(n):
        dr, dc = (int(x) for x in data['shifts'][i, 0])
        lay = S2.afterstate_layout(data['root_layout'][i], dr, dc)
        tile = -torch.ones(63, dtype=torch.long)
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    tile[r * 9 + c] = int(data['root_tiles'][i, (r + dr) * 9 + c + dc])
        mob = (lay[:, :3] == 1).any(-1)
        for k in range(K):
            p = op['partner'][i, k]
            for j in range(63):
                q = int(p[j])
                if q <= j:
                    continue
                if q >= 63 or lay[j, 0] < 0 or lay[q, 0] < 0:
                    counts['entering_or_unknown'] += 1
                elif mob[j] and mob[q]:
                    counts['mob_mob'] += 1
                elif mob[j] or mob[q]:
                    other = q if mob[j] else j
                    counts['mob_passable' if int(tile[other]) in passable else 'mob_solid'] += 1
                else:
                    counts['terrain_same' if int(tile[j]) == int(tile[q]) else 'terrain_different'] += 1
    frames = n * K
    res = {'update': update, 'swaps_per_frame_by_class': {k: round(v / frames, 4) for k, v in counts.items()},
           'share_of_mob_moves_into_solid': round(counts['mob_solid'] / max(counts['mob_solid'] + counts['mob_passable'], 1), 4)}
    # static corruption: frozen definition, exact tiles
    root_tiles, real_tiles = data['root_tiles'].long(), data['fan_tiles'].long()
    stable = (real_tiles == root_tiles[:, None]).all(1) & (data['fan_layout'].bool() == data['root_layout'][:, None].bool()).all((1, 3))
    stable[:, 31] = False
    scored = torch.zeros(63, dtype=torch.bool); scored[list(A5.INTERIOR)] = True
    stable &= scored
    def corruption(tokens):
        t = C.decode(tokens, bank, head)['tile']
        bad = (t != root_tiles[:, None]) & stable[:, None]
        return round(float(bad.sum() / (stable.sum() * tokens.shape[1])), 6)
    res['scored_static_corruption'] = {'operator_samples': corruption(op['predictions']),
                                       'operator_reference_only': corruption(op['reference'][:, None].contiguous()),
                                       'free_samples': corruption(fr['predictions']),
                                       'free_reference_only': corruption(fr['reference'][:, None].contiguous())}
    (OUT / f'diag_swaps_u{update:04d}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'swaps':
        swaps_mode(int(sys.argv[2]))
    elif sys.argv[1] == 'margin':
        margin()
    elif sys.argv[1] == 'gradsplit':
        gradsplit(int(sys.argv[2]))
    else:
        main(int(sys.argv[1]))
