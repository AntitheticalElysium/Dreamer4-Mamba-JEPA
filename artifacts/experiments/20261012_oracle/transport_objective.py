"""Model-free investigation of the objective's locality (2026-10-15, user mandate 3). No model is trained here.
  afterstate  Is a clean pre-chance afterstate learned WITHOUT privileged assignments? For each reference (D parent, the F1b
              oracle-assigned and identity fits, every unprivileged Stage 3 / seed-1 / seed-2 arm at 3k), the TRUE
              reassignment T* of every real key (simulator layouts: diagnostic readout only) is applied to that reference's
              own deterministic afterstate, zhat = LN(T* b): strict-source events (both frozen readers), interior panel moves
              with both endpoints correct, and the frozen evaluator's scored static corruption. The same for the true
              afterstate tokens (aligned real root tokens): the representation-only ceiling of whole-token transport.
  rules       (CPU, model-free, exact) the scoring rules on real futures. Fan roots (200 DEV roots x 32 keys): keys 0-15 are the
              model's outcomes, keys 16-31 the targets (and the reverse, averaged). For a model distribution p over outcome
              tokens and uniform targets, ES(p) = p.A - 0.5 p'Dp exactly (A = mean distance to targets, D = outcome distances),
              so every derivative below is exact (no sampling noise). The first-power formulation is kept; each rule is a
              distance d (Euclidean norms over fixed token sets, normalised per token), and ES uses it unchanged.
              Rules on the 35 scored interior tokens: E0 = the existing training distance (.5 global + .5 mean of nine fixed
              3x3 patches); G = global; TOK = mean per-token norm (marginal-only control); DOM = mean over all adjacent
              interior pairs; CROSS = mean over the 5-cell cross of every interior cell (truncated to the interior); hybrids
              H(a, L) = a G + (1-a) L for a in {.1, .25, .5, .75} and L in {DOM, CROSS}. Neighbourhoods are fixed by grid
              geometry, never chosen from realized futures or labels.
              Outcome sets per root, all from the 16 model keys (simulator layouts only to build the controls):
                true      the real keys (the true law)
                stay      the same keys with every adjacent same-type mob move undone (T* is an involution)
                spurious  the true outcomes with ONE swap of two adjacent interior cells whose tokens persist (RMS < 0.1 from
                          the aligned root token in all 32 keys); pairs classed same / different decoded terrain (frozen
                          bank tile head); up to 4 per class per root
                shuffled  every interior cell's token permuted independently across the 16 keys (marginals exact)
                decoupled the same, but only inside the cross of each moving source (localized dependence break)
              Readings per rule: movement margin ES(stay) - ES(true); value of genuine movement per unit event probability
              -dES/dq / f along p(q) = (1-q) stay + q true at q = .5 (f = share of keys with a move); spurious penalty per unit
              probability dES/de along p(e) = (1-e) true + e spurious at e = 0, .02, .1; dependence ES(shuffled) - ES(true)
              and ES(decoupled) - ES(true), as fractions of the movement margin.
  headgrad <ckpt>  (GPU) the learning signal each rule gives the stochastic head: the frozen operator checkpoint (path relative
              to transport/), the 48 stationary training windows of transport_cost_diag (label-free static / dynamic tile
              classes), 8 noise pairs. The training estimator per frame, ES = .5 d(X1,Y) + .5 d(X2,Y) - .5 d(X1,X2), with d each
              rule's distance (vector norms: zero-safe gradients), differentiated through the straight-through relaxation:
              B[j,dir] = -dES/dq = the value of move probability. Reported per rule: static mean (+ window SE), dynamic at
              realized moves, dynamic at staying steps, labelled mob / no-mob (interpretation), and the ratio of the expected
              mob value to the static penalty.
  biasnoise   (GPU) why the score rewards spread at static cells: the two-sample first-power ES with locally identical draws
              never penalizes a one-draw perturbation unless the prediction is unbiased (triangle inequality). Where does
              the bias come from? At interior cells with no mob in the root or any of the 32 futures (fan layouts:
              evaluation-side), on all 200 DEV fan roots: nuisance = RMS over keys of the real successor token around its
              key-mean (unpredictable target variation), bias = RMS of each reference's LN(b) minus that key-mean (its error
              against the predictable part). References: D parent, F1b oracle, unprivileged operators, a free arm.
              Also: the TRUE afterstate (aligned root tokens) and HARD CorrT afterstates (each cell takes its highest-weight
              candidate instead of the weighted blend; computed here, the models are unchanged).
  headgrad <ckpt> hard   the same head-gradient test with the hard CorrT afterstate in place of the soft one.
  headgrad <ckpt> oraclestatic   mechanism test (privileged, diagnostic only): at the window's label-free static tiles the
              afterstate is replaced by the REAL next token (zero prediction error where content persists); everything else
              unchanged. If spurious static swaps are a hedge against afterstate error, their value turns negative here.
  posterior   (CPU+GPU) can the chance outcome be INFERRED from the observed successor without labels? Parameter-free local
              search: for every adjacent interior pair (j, k) of the afterstate tokens a = LN(b), gain = (|a_j - z_j| + |a_k - z_k|)
              - (|a_k - z_j| + |a_j - z_k|) against the observed successor z; swaps with gain > 0 are accepted greedily by
              descending gain, each cell at most once (no tuned threshold). Scored against the simulator's T* (diagnostic):
              recall of true mob moves, precision, false swaps per frame split by decoded same / different terrain, on the
              104 + 96 DEV fan roots x 32 keys, for the true afterstate tokens and the D parent / unprivileged operator
              references. Also the gain distribution of true moves vs false swaps (is there a margin at 0?).
Usage: transport_objective.py afterstate | rules | headgrad <ckpt> [hard|oraclestatic] | biasnoise | posterior -> artifacts/eda/oracle_v1/transport/objective_<mode>.json
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

T = A5.BASE / 'transport'
REFERENCES = {
    'D_parent': A5.BASE / 'paired_event/training/paired_u0000.pt',
    'F1b_oracle_assigned_u1500': T / 'fit_assigned_u1500.pt',
    'F1b_identity_u1500': T / 'fit_identity_u1500.pt',
    'stage3_seed1_free': T / 'training/free_u3000.pt',
    'stage3_seed1_operator': T / 'training/operator_u3000.pt',
    'seed1_operator_replicate': T / 'seed1_cost/operator_u3000.pt',
    'seed1_operator_cost': T / 'seed1_cost/operator_cost_u3000.pt',
    'seed2_free': T / 'seed2/free_u3000.pt',
    'seed2_operator': T / 'seed2/operator_u3000.pt',
    'seed2_operator_cost': T / 'seed2/operator_cost_u3000.pt',
}


@torch.no_grad()
def afterstate():
    import paired_model as PM
    import direct_train as BASE
    import direct_eval as E
    import direct_coherence as C
    from d4mj.train import autocast_context
    config = BASE.settings()
    data, _ = E.cohort('exploratory')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    base_state = torch.load(REFERENCES['D_parent'], map_location='cpu', weights_only=False)['world']
    n, K = data['fan_tokens'].shape[:2]
    root_tok, real_tok = data['root_tokens'].float(), data['fan_tokens'].float()
    pairs, entering, aligned = {}, torch.zeros(n, 63, dtype=torch.bool), root_tok.clone()
    for i in range(n):
        dr, dc = (int(x) for x in data['shifts'][i, 0])
        prev = S2.afterstate_layout(data['root_layout'][i], dr, dc)
        entering[i] = prev[:, 0] < 0
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    aligned[i, r * 9 + c] = root_tok[i, (r + dr) * 9 + c + dc]
        for k in range(K):
            pairs[(i, k)] = S2.swaps(prev, data['fan_layout'][i, k].bool())
    units = json.loads((A5.OUT / 'audit_recompute.json').read_text())['sources_list']

    def readout(tokens):                                                    # tokens [n,K,81,192]
        mob = C.decode(tokens, bank, head)['mob']
        out = {}
        for reader in ('real', 'bank'):
            mv = st = mvc = stc = 0
            for root, source, kind in units:
                k = ('zombie', 'cow', 'skeleton').index(kind)
                r, c = divmod(source, 9)
                pts = [(r + a) * 9 + c + e for a, e in A5.CROSS]
                truth = A5.event(data['fan_layout'][root][:, pts, k].bool())
                ev = A5.event(mob[reader][root][:, pts, k])
                m = truth > 0
                mv += int(m.sum()); mvc += int((ev == truth)[m].sum()); st += int((~m).sum()); stc += int((ev == truth)[~m].sum())
            both = tot = 0
            for (i, k), pr in pairs.items():
                for s, d, kind in pr:
                    if kind > 2 or 31 in (s, d) or entering[i, s] or entering[i, d]:
                        continue
                    tot += 1; both += int(bool(mob[reader][i, k, d, kind]) and not bool(mob[reader][i, k, s, kind]))
            out[reader] = {'strict_move_correct': round(mvc / mv, 4), 'strict_stay_correct': round(stc / st, 4), 'panel_moves_both_correct': round(both / tot, 4)}
        frozen = C.analyze(tokens, root_tok, real_tok, data['root_layout'], data['fan_layout'], bank, head, shifts=data['shifts'],
                           actions=data['actions'][:, -1], root_tiles=data['root_tiles'], real_tiles=data['fan_tiles'])
        static = [x for x in frozen['static_content']['scored_sampled_corruption_per_root'] if x is not None]
        out['scored_static_corruption'] = round(float(np.mean(static)), 6)
        return out
    res = {'definition': 'zhat = LN(T* b) with the TRUE reassignment of each real key (diagnostic); static corruption by the frozen evaluator'}
    res['true_afterstate_tokens'] = readout(torch.stack([torch.stack([S2.apply(aligned[i], pairs[(i, k)]) for k in range(K)]) for i in range(n)]))
    res['real_successors'] = readout(real_tok)
    print(json.dumps({'true_afterstate_tokens': res['true_afterstate_tokens'], 'real_successors': res['real_successors']}), flush=True)
    for name, path in REFERENCES.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)['world']
        state = dict(base_state); state.update({k: v for k, v in saved.items() if not k.startswith('completion.')})
        world = PM.PairedWorld().cuda().eval(); world.load_state_dict(state)
        refs = []
        for s0 in range(0, n, 4):
            ctx = data['context_tokens'][s0:s0 + 4].cuda().float(); ac = data['actions'][s0:s0 + 4].cuda()
            frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
                refs.append(world.reference(frames, ac, belief, ha)[:, 14].float().cpu())
        b = torch.cat(refs)
        swapped = torch.stack([torch.stack([F.layer_norm(S2.apply(b[i], pairs[(i, k)]), (192,)) for k in range(K)]) for i in range(n)])
        res[name] = readout(swapped)
        print(json.dumps({name: res[name]}), flush=True)
        del world; torch.cuda.empty_cache()
    return res


GRID = [(r, c) for r in range(1, 6) for c in range(1, 8)]                # the 35 scored cells, interior order
POS = {rc: i for i, rc in enumerate(GRID)}


def neighbourhoods():
    dom = [(POS[(r, c)], POS[(r + dr, c + dc)]) for (r, c) in GRID for dr, dc in ((0, 1), (1, 0)) if (r + dr, c + dc) in POS]
    cross = [[POS[(r + dr, c + dc)] for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)) if (r + dr, c + dc) in POS] for (r, c) in GRID]
    import direct_model as DM
    patches = [[POS[divmod(x, 9)] for x in p] for p in DM.PATCHES]
    return dom, cross, patches


def rule_distances(sq):
    """sq [..., 35] per-token squared distances -> {rule: [...]} first-power normalised distances"""
    dom, cross, patches = NB
    g = sq.sum(-1).sqrt() / np.sqrt(35 * 192)
    tok = sq.sqrt().mean(-1) / np.sqrt(192)
    d_dom = torch.stack([(sq[..., i] + sq[..., j]).sqrt() for i, j in dom], -1).mean(-1) / np.sqrt(2 * 192)
    d_cross = torch.stack([sq[..., n].sum(-1).sqrt() / np.sqrt(len(n) * 192) for n in cross], -1).mean(-1)
    d_patch = torch.stack([sq[..., p].sum(-1).sqrt() for p in patches], -1).mean(-1) / np.sqrt(9 * 192)
    out = {'E0': .5 * g + .5 * d_patch, 'G': g, 'TOK': tok, 'DOM': d_dom, 'CROSS': d_cross}
    for a in (.1, .25, .5, .75):
        out[f'H{a}_DOM'] = a * g + (1 - a) * d_dom
        out[f'H{a}_CROSS'] = a * g + (1 - a) * d_cross
    return out


NB = None


def rules():
    global NB
    NB = neighbourhoods()
    import direct_coherence as C
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    fan = torch.load(A5.BASE / 'learned_innovation/real_futures/fan.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll()
    interior = [r * 9 + c for r, c in GRID]
    g = torch.Generator().manual_seed(20261031)
    rows = []
    for i in range(len(panel['roots'])):
        root = panel['tokens'][i, 15].float(); keys = fan['tokens'][i].float(); lay = fan['layouts'][i].bool()
        est = scroll.estimate(root[None].expand(32, -1, -1), keys)
        shift = scroll.SHIFTS[int(est.mode().values)]
        dr, dc = shift
        prev = S2.afterstate_layout(panel['layouts'][i, 15], dr, dc)
        aligned = root.clone()
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    aligned[r * 9 + c] = root[(r + dr) * 9 + c + dc]
        pr = [S2.swaps(prev, lay[k]) for k in range(32)]
        moved_keys = [k for k in range(32) if any(t < 3 and 31 not in (s_, d_) and s_ in interior and d_ in interior for s_, d_, t in pr[k])]
        persist = ((keys[:, interior] - aligned[interior]).square().mean(-1).sqrt() < .1).all(0)   # [35]
        tile = C.decode(aligned[None], bank, head)['tile'][0]
        static_pairs = [(interior[a], interior[b]) for a, b in NB[0] if persist[a] and persist[b] and 31 not in (interior[a], interior[b])]
        same = [p for p in static_pairs if tile[p[0]] == tile[p[1]]]; diff = [p for p in static_pairs if tile[p[0]] != tile[p[1]]]
        pick = lambda xs: [xs[j] for j in torch.randperm(len(xs), generator=g)[:4].tolist()]
        spur = [('same', p) for p in pick(same)] + [('different', p) for p in pick(diff)]
        cross_src = set()
        for k in moved_keys:
            for s_, d_, t in pr[k]:
                if t < 3 and s_ in interior:
                    r_, c_ = divmod(s_, 9)
                    cross_src |= {(r_ + a) * 9 + c_ + e for a, e in A5.CROSS if (r_ + a) * 9 + c_ + e in interior}
        for half, (mk, tk) in enumerate(((range(16), range(16, 32)), (range(16, 32), range(16)))):
            mk, tk = list(mk), list(tk)
            O_true = keys[mk]; Y = keys[tk]
            O_stay = torch.stack([S2.apply(keys[k], pr[k]) for k in mk])
            sets = [O_true, O_stay]
            for _, (a, b) in spur:
                o = O_true.clone(); o[:, a], o[:, b] = O_true[:, b], O_true[:, a]; sets.append(o)
            perm = torch.stack([torch.randperm(16, generator=g) for _ in range(81)], 1)
            O_shuf = O_true[perm, torch.arange(81)[None]]
            O_dec = O_true.clone()
            if cross_src:
                cs = sorted(cross_src); O_dec[:, cs] = O_shuf[:, cs]
            sets += [O_shuf, O_dec]
            O = torch.cat(sets)[:, interior].double(); Yi = Y[:, interior].double()
            sqA = torch.stack([torch.cdist(O[:, j], Yi[:, j]).square() for j in range(35)], -1)   # [nO,16,35]
            sqD = torch.stack([torch.cdist(O[:, j], O[:, j]).square() for j in range(35)], -1)    # [nO,nO,35]
            dA, dD = rule_distances(sqA), rule_distances(sqD)
            n = 16
            idx = {'true': range(0, n), 'stay': range(n, 2 * n)}
            for m, _ in enumerate(spur):
                idx[f'spur{m}'] = range((2 + m) * n, (3 + m) * n)
            idx['shuf'] = range((2 + len(spur)) * n, (3 + len(spur)) * n); idx['dec'] = range((3 + len(spur)) * n, (4 + len(spur)) * n)
            nO = O.shape[0]

            def vec(name):
                v = torch.zeros(nO, dtype=torch.float64); v[list(idx[name])] = 1 / n; return v
            f = len([k for k in mk if k in moved_keys]) / 16
            for rule in dA:
                A = dA[rule].mean(1); D = dD[rule]
                es = lambda p: float(p @ A - .5 * p @ D @ p)
                grad = lambda p, dirn: float((A - D @ p) @ dirn)
                pt, ps = vec('true'), vec('stay')
                row = {'root': i, 'half': half, 'selected': i in selected, 'rule': rule, 'f_moved': f,
                       'ES_true': es(pt), 'ES_stay': es(ps), 'ES_shuf': es(vec('shuf')), 'ES_dec': es(vec('dec')) if cross_src else None}
                if f > 0:
                    row['move_value_per_prob'] = -grad(.5 * pt + .5 * ps, pt - ps) / f
                for m, (cls, _) in enumerate(spur):
                    psp = vec(f'spur{m}')
                    for e in (0., .02, .1):
                        row.setdefault(f'spur_{cls}_{e}', []).append(grad((1 - e) * pt + e * psp, psp - pt))
                rows.append(row)
        if i % 20 == 0:
            print(json.dumps({'root': i, 'static_pairs_same_diff': [len(same), len(diff)], 'moved_keys': len(moved_keys)}), flush=True)
    return summarize(rows)


def rule_distances_diff(err):
    """err [..., 35, 192] -> {rule: [...]} with zero-safe vector norms (differentiable at identical tokens)"""
    dom, cross, patches = NB
    vn = torch.linalg.vector_norm
    g = vn(err.flatten(-2), dim=-1) / np.sqrt(35 * 192)
    tok = vn(err, dim=-1).mean(-1) / np.sqrt(192)
    d_dom = torch.stack([vn(err[..., [i, j], :].flatten(-2), dim=-1) for i, j in dom], -1).mean(-1) / np.sqrt(2 * 192)
    d_cross = torch.stack([vn(err[..., n, :].flatten(-2), dim=-1) / np.sqrt(len(n) * 192) for n in cross], -1).mean(-1)
    d_patch = torch.stack([vn(err[..., p, :].flatten(-2), dim=-1) for p in patches], -1).mean(-1) / np.sqrt(9 * 192)
    out = {'E0': .5 * g + .5 * d_patch, 'G': g, 'TOK': tok, 'DOM': d_dom, 'CROSS': d_cross}
    for a in (.1, .25, .5, .75):
        out[f'H{a}_CROSS'] = a * g + (1 - a) * d_cross
    out['H0.5_DOM'] = .5 * g + .5 * d_dom
    return out


def headgrad(ckpt, hard=False, oraclestatic=False):
    global NB
    use_hard, use_oracle = hard, oraclestatic          # 'hard' is reused below for the permuted tokens
    NB = neighbourhoods()
    import direct_model as DM
    import direct_train as BASE
    import transport_model as TM
    import transport_cost_diag as CD
    from d4mj.train import autocast_context
    config = BASE.settings()
    wins, actions, tokens = CD.stationary_windows()
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(T / ckpt, map_location='cpu', weights_only=False)['world'])
    for prm in world.parameters():
        prm.requires_grad_(False)
    comp = world.completion
    interior = [r * 9 + c for r, c in GRID]
    g = torch.Generator().manual_seed(20261021)
    acc = {}
    def add(rule, key, x):
        acc.setdefault(rule, {}).setdefault(key, []).append(x)
    per_window_static = {}
    for b0 in range(0, len(wins), 4):
        frames, ac, choices = BASE.batch(tokens, actions, wins, torch.arange(b0, min(b0 + 4, len(wins))))
        static, dynamic, rdir = CD.classes(frames.float().cpu())
        labels = torch.stack([torch.from_numpy(np.array(lay[int(r), int(s):int(s) + 16]) > 0) for r, s in choices.tolist()])
        mob_lab = labels[:, :15, :, :3].any(-1)
        with torch.no_grad(), autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            ref = (hard_reference(world, frames, ac, belief, ha) if use_hard else world.reference(frames, ac, belief, ha)).float()
        if use_oracle:
            st = static.cuda()[:, None, :].expand(-1, 15, -1)                       # [b,15,63]
            nxt = frames[:, 1:, :63].float()
            ref = ref.clone()
            ref[:, :15, :63] = torch.where(st[..., None], nxt, ref[:, :15, :63])
        grads = {}
        for _ in range(8):
            noise = DM.draw_noise(frames.shape[:2], g).cuda(); gum = TM.draw_gumbel(frames.shape[:2], g).cuda()
            outs, qs = [], []
            for k in range(2):
                with torch.no_grad(), autocast_context(config):
                    hidden = comp.trunk(belief, noise[k])
                    logits = comp.propose(hidden).float().masked_fill(~comp.valid, -torch.inf)
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
                outs.append(F.layer_norm(hard + relax - relax.detach(), (192,))[:, :15, interior]); qs.append(q)
            y = frames[:, 1:, interior].float()
            d1, d2, d12 = (rule_distances_diff(e) for e in (outs[0] - y, outs[1] - y, outs[0] - outs[1]))
            for rule in d1:
                es = (.5 * d1[rule] + .5 * d2[rule] - .5 * d12[rule]).sum()
                gq = torch.autograd.grad(es, qs, retain_graph=True)
                grads[rule] = grads.get(rule, 0) + (-(gq[0] + gq[1])[:, :15] / 16).cpu()
        for rule, gr in grads.items():
            for b in range(len(frames)):
                st = static[b]
                if st.any():
                    v = gr[b, :, :63][:, st][..., 1:]
                    val = comp.valid[:63][st][:, 1:].cpu()[None].expand_as(v)
                    add(rule, 'static', v[val]); per_window_static.setdefault(rule, []).append(float(v[val].mean()))
                for j in torch.where(dynamic[b])[0].tolist():
                    for t in range(15):
                        d = int(rdir[b, t, j])
                        valid = [x for x in range(1, 5) if bool(comp.valid[j, x])]
                        if d > 0:
                            add(rule, 'dyn_move_realized', gr[b, t, j, d:d + 1])
                        else:
                            add(rule, 'dyn_stay', gr[b, t, j, valid])
                for t in range(15):
                    for j in range(63):
                        if j == 31:
                            continue
                        valid = [x for x in range(1, 5) if bool(comp.valid[j, x])]
                        add(rule, 'mob_label' if bool(mob_lab[b, t, j]) else 'nomob_label', gr[b, t, j, valid])
        print(json.dumps({'windows_done': b0 + 4}), flush=True)
    res = {'checkpoint': ckpt, 'hard_reference': use_hard, 'oracle_static': use_oracle}
    for rule, a in acc.items():
        cat = {k: torch.cat(v) for k, v in a.items()}
        st = float(cat['static'].mean()); se = float(np.std(per_window_static[rule]) / np.sqrt(len(per_window_static[rule])))
        mob = float(cat['mob_label'].mean()); nomob = float(cat['nomob_label'].mean())
        res[rule] = {'static_mean': st, 'static_window_se': se, 'dyn_move_realized': float(cat['dyn_move_realized'].mean()),
                     'dyn_stay': float(cat['dyn_stay'].mean()), 'mob_label_mean': mob, 'nomob_label_mean': nomob,
                     'static_t': st / se if se else None,
                     'mob_over_minus_nomob': mob / -nomob if nomob < 0 else None,
                     'realized_over_abs_static': float(cat['dyn_move_realized'].mean()) / abs(st) if st else None}
    return res


def hard_reference(world, frames, actions, belief, action_belief):
    """PairedWorld.reference with each cell's highest-weight CorrT candidate instead of the weighted blend (exact copy)"""
    import paired_model as PM
    gen = world.proj(belief).float()
    grid = frames.float().reshape(*frames.shape[:2], 9, 9, 192)
    padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
    candidates = [grid] + [padded[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in PM.NEIGHBOURS]
    candidates = torch.stack([x.reshape(frames.shape) for x in candidates] + [gen], -2)
    logits = world.choose(belief).float()
    moved = world.frame(action_belief).float()[:, :, None, :]
    target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
    htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
    moved = moved + (((actions >= 1) & (actions <= 4)).float()[:, :, None, None] * world.target_gate(htarget).float()[:, :, None, :])
    logits = logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])
    pick = logits.argmax(-1)
    return candidates.gather(-2, pick[..., None, None].expand(*pick.shape, 1, 192))[..., 0, :]


@torch.no_grad()
def biasnoise():
    import paired_model as PM
    import direct_train as BASE
    from d4mj.train import autocast_context
    config = BASE.settings()
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    fan = torch.load(A5.BASE / 'learned_innovation/real_futures/fan.pt', map_location='cpu', weights_only=False, mmap=True)
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    interior = [r * 9 + c for r, c in GRID]
    n = len(panel['roots'])
    keys = fan['tokens'].float()                                            # [200,32,81,192]
    mean = keys.mean(1)
    nuis = (keys - mean[:, None]).square().mean(-1).mean(1).sqrt()          # [200,81] RMS over keys
    lay = fan['layouts'].bool()[..., :3].any(-1).any(1) | panel['layouts'][:, 15].bool()[..., :3].any(-1)   # [200,63]
    static = torch.zeros(n, 81, dtype=torch.bool)
    static[:, interior] = ~lay[:, interior]; static[:, 31] = False
    base_state = torch.load(REFERENCES['D_parent'], map_location='cpu', weights_only=False)['world']
    res = {'cells': int(static.sum())}
    for part, rows in (('eval104', [i for i in range(n) if i in selected]), ('heldout96', [i for i in range(n) if i not in selected])):
        m = static[rows]
        res.setdefault(part, {})['nuisance_rms_mean'] = round(float(nuis[rows][m].mean()), 4)
    scroll = A5.load_scroll()
    aligned = panel['tokens'][:, 15].float().clone()
    for i in range(n):
        root = panel['tokens'][i, 15].float()
        dr, dc = scroll.SHIFTS[int(scroll.estimate(root[None].expand(32, -1, -1), keys[i]).mode().values)]
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    aligned[i, r * 9 + c] = root[(r + dr) * 9 + c + dc]
    bias = (aligned - mean).square().mean(-1).sqrt()
    for part, rows in (('eval104', [i for i in range(n) if i in selected]), ('heldout96', [i for i in range(n) if i not in selected])):
        m = static[rows]
        res[part]['true_afterstate'] = {'bias_rms_mean': round(float(bias[rows][m].mean()), 4),
                                        'bias_over_nuisance': round(float(bias[rows][m].mean() / nuis[rows][m].mean()), 2)}
    print(json.dumps({'true_afterstate': {p: res[p]['true_afterstate'] for p in ('eval104', 'heldout96')}}), flush=True)
    for name, hard in (('D_parent', False), ('F1b_oracle_assigned_u1500', False), ('stage3_seed1_operator', False), ('seed2_operator', False),
                       ('stage3_seed1_free', False), ('D_parent', True), ('stage3_seed1_operator', True), ('seed2_operator', True)):
        saved = torch.load(REFERENCES[name], map_location='cpu', weights_only=False)['world']
        state = dict(base_state); state.update({k: v for k, v in saved.items() if not k.startswith('completion.')})
        world = PM.PairedWorld().cuda().eval(); world.load_state_dict(state)
        refs = []
        for s0 in range(0, n, 4):
            ctx = panel['tokens'][s0:s0 + 4, 1:16].cuda().float(); ac = panel['actions'][s0:s0 + 4, 1:16].cuda()
            frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
                rf = hard_reference(world, frames, ac, belief, ha) if hard else world.reference(frames, ac, belief, ha)
                refs.append(F.layer_norm(rf[:, 14].float(), (192,)).cpu())
        b = torch.cat(refs)
        bias = (b - mean).square().mean(-1).sqrt()                          # [200,81]
        name = name + ('_HARD' if hard else '')
        for part, rows in (('eval104', [i for i in range(n) if i in selected]), ('heldout96', [i for i in range(n) if i not in selected])):
            m = static[rows]
            res[part][name] = {'bias_rms_mean': round(float(bias[rows][m].mean()), 4),
                               'bias_over_nuisance': round(float(bias[rows][m].mean() / nuis[rows][m].mean()), 2),
                               'share_cells_bias_gt_nuisance': round(float((bias[rows][m] > nuis[rows][m]).float().mean()), 3)}
        print(json.dumps({name: {p: res[p][name] for p in ('eval104', 'heldout96')}}), flush=True)
        del world; torch.cuda.empty_cache()
    return res


def infer_swaps(a, z, interior_pairs):
    """a, z [81,192] -> accepted (j, k, gain) list: greedy by gain > 0, each cell once"""
    rows = []
    for j, k in interior_pairs:
        keep = (a[j] - z[j]).norm() + (a[k] - z[k]).norm()
        swap = (a[k] - z[j]).norm() + (a[j] - z[k]).norm()
        rows.append((float(keep - swap), j, k))
    rows.sort(reverse=True)
    used, out = set(), []
    for g, j, k in rows:
        if g <= 0:
            break
        if j in used or k in used:
            continue
        used |= {j, k}; out.append((j, k, g))
    return out, rows


@torch.no_grad()
def posterior():
    import paired_model as PM
    import direct_train as BASE
    import direct_coherence as C
    from d4mj.train import autocast_context
    config = BASE.settings()
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    fan = torch.load(A5.BASE / 'learned_innovation/real_futures/fan.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll()
    n = len(panel['roots'])
    interior = [r * 9 + c for r, c in GRID]
    pairs = [(interior[a], interior[b]) for a, b in neighbourhoods()[0] if 31 not in (interior[a], interior[b])]
    keys = fan['tokens'].float()
    aligned = panel['tokens'][:, 15].float().clone(); truth = {}
    for i in range(n):
        root = panel['tokens'][i, 15].float()
        dr, dc = scroll.SHIFTS[int(scroll.estimate(root[None].expand(32, -1, -1), keys[i]).mode().values)]
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    aligned[i, r * 9 + c] = root[(r + dr) * 9 + c + dc]
        prev = S2.afterstate_layout(panel['layouts'][i, 15], dr, dc)
        for k in range(32):
            truth[(i, k)] = {frozenset((s_, d_)) for s_, d_, t in S2.swaps(prev, fan['layouts'][i, k].bool()) if t < 3 and s_ in interior and d_ in interior and 31 not in (s_, d_)}
    tiles = C.decode(aligned, bank, head)['tile']                          # decoded terrain of the afterstate (observation)
    refs = {'true_afterstate': aligned}
    base_state = torch.load(REFERENCES['D_parent'], map_location='cpu', weights_only=False)['world']
    for name in ('D_parent', 'stage3_seed1_operator', 'seed2_operator'):
        saved = torch.load(REFERENCES[name], map_location='cpu', weights_only=False)['world']
        state = dict(base_state); state.update({k: v for k, v in saved.items() if not k.startswith('completion.')})
        world = PM.PairedWorld().cuda().eval(); world.load_state_dict(state)
        out = []
        for s0 in range(0, n, 4):
            ctx = panel['tokens'][s0:s0 + 4, 1:16].cuda().float(); ac = panel['actions'][s0:s0 + 4, 1:16].cuda()
            frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(frames, ac)
                out.append(F.layer_norm(world.reference(frames, ac, belief, ha)[:, 14].float(), (192,)).cpu())
        refs[name] = torch.cat(out)
        del world; torch.cuda.empty_cache()
    res = {}
    for name, a in refs.items():
        for part, rows in (('eval104', [i for i in range(n) if i in selected]), ('heldout96', [i for i in range(n) if i not in selected])):
            tp = fn = fp_same = fp_diff = fp_mob = 0; frames_n = 0; g_true, g_false = [], []
            for i in rows:
                for k in range(32):
                    acc, allrows = infer_swaps(a[i], keys[i, k], pairs)
                    t = truth[(i, k)]
                    got = {frozenset((j, q)) for j, q, _ in acc}
                    tp += len(t & got); fn += len(t - got); frames_n += 1
                    for j, q, g_ in acc:
                        if frozenset((j, q)) in t:
                            g_true.append(g_)
                        else:
                            g_false.append(g_)
                            lay = fan['layouts'][i, k].bool()[..., :3].any(-1) | panel['layouts'][i, 15].bool()[..., :3].any(-1)
                            if bool(lay[j]) or bool(lay[q]):
                                fp_mob += 1
                            elif tiles[i, j] == tiles[i, q]:
                                fp_same += 1
                            else:
                                fp_diff += 1
            res.setdefault(name, {})[part] = {
                'true_moves': tp + fn, 'recall': round(tp / max(tp + fn, 1), 4),
                'precision': round(tp / max(tp + fp_same + fp_diff + fp_mob, 1), 4),
                'false_swaps_per_frame': {'same_terrain': round(fp_same / frames_n, 4), 'different_terrain': round(fp_diff / frames_n, 4),
                                          'touching_a_mob': round(fp_mob / frames_n, 4)},
                'gain_true_p10_p50': [round(float(np.quantile(g_true, q)), 3) for q in (.1, .5)] if g_true else None,
                'gain_false_p50_p90': [round(float(np.quantile(g_false, q)), 3) for q in (.5, .9)] if g_false else None}
        print(json.dumps({name: res[name]}), flush=True)
    return res


def summarize(rows):
    out = {}
    rule_names = sorted({r['rule'] for r in rows})
    for part, sel in (('heldout96', lambda r: not r['selected']), ('eval104', lambda r: r['selected']), ('all200', lambda r: True)):
        out[part] = {}
        for rule in rule_names:
            rr = [r for r in rows if r['rule'] == rule and sel(r)]
            mv = [r for r in rr if r['f_moved'] > 0]
            margin = np.mean([r['ES_stay'] - r['ES_true'] for r in mv]); val = np.mean([r['move_value_per_prob'] for r in mv])
            item = {'roots_with_moves': len({r['root'] for r in mv}), 'ES_true_mean': float(np.mean([r['ES_true'] for r in rr])),
                    'movement_margin': float(margin), 'movement_margin_rel': float(margin / np.mean([r['ES_true'] for r in mv])),
                    'share_roots_margin_positive': float(np.mean([r['ES_stay'] > r['ES_true'] for r in mv])),
                    'move_value_per_prob': float(val),
                    'dependence_shuffle_frac_of_margin': float(np.mean([r['ES_shuf'] - r['ES_true'] for r in mv]) / margin),
                    'dependence_local_decouple_frac_of_margin': float(np.mean([r['ES_dec'] - r['ES_true'] for r in mv if r['ES_dec'] is not None]) / margin)}
            for cls in ('same', 'different'):
                for e in (0., .02, .1):
                    v = [x for r in rr for x in r.get(f'spur_{cls}_{e}', [])]
                    item[f'spur_{cls}_penalty_e{e}'] = float(np.mean(v)) if v else None
                    item[f'spur_{cls}_n'] = len(v)
                    if v and e == .02:
                        item[f'spur_{cls}_penalty_over_move_value_e0.02'] = float(np.mean(v) / val)
                        item[f'spur_{cls}_penalty_over_move_value_e0.1'] = None
                    if v and e == .1:
                        item[f'spur_{cls}_penalty_over_move_value_e0.1'] = float(np.mean(v) / val)
            out[part][rule] = item
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'headgrad':
        hard, oracle = sys.argv[3:] == ['hard'], sys.argv[3:] == ['oraclestatic']
        res = headgrad(sys.argv[2], hard, oracle)
        mode = 'headgrad_' + sys.argv[2].replace('/', '_').replace('.pt', '') + ('_hard' if hard else '') + ('_oraclestatic' if oracle else '')
    else:
        res = {'afterstate': afterstate, 'rules': rules, 'biasnoise': biasnoise, 'posterior': posterior}[mode]()
    (T / f'objective_{mode}.json').write_text(json.dumps(res, indent=1))
