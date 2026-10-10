"""10-10 PROGRAM Part II, B4 (prediction P-C): the anatomy of a world's training gradient by token category. GPU.
Held windows from the diagnosis futures with the simulator state of every frame: [root, factual future 1..L-1] (L = the world's
frames: 6 for the stage-1 worlds, 16 for L16), the factual actions, roots alive through the window (sample 0). Teacher-forced as in
training (tworld.rollout_losses' teacher term), autocast as training. Every target token is put in one of budget.CATS (root -> key-0
successor categories applied to every transition of the window: entering, own state changed, player sprite left, neighbour changed,
sleep changed, no state change, HUD changed / same, player token).
The objective L (`l1`: mean |e|; `deltairis`: mean e^2 + 0.1 mean |e| + 0.01 mean over frames of the frame's worst element e^2, the
worst element's term assigned to its token) is split EXACTLY into per-category terms L_c (sum_c L_c = L). Per category and batch, the
gradient g_c of L_c on the BACKBONE (every parameter except the corrt head's proj / choose / frame / target_gate) is taken.
Reported per category (over the batches): loss share; gradient norm ratio |g_c| / |g| (mean); projection share <g_c, g> / |g|^2
(mean; additive: sums to 1 over categories); consistency |sum_b g_c| / sum_b |g_c| (1 = the same direction in every batch,
~1 / sqrt(batches) = noise); and the mean cosine between categories within a batch (conflict when negative).
Both objectives are scored on every world (`--loss l1,deltairis`), so what the world IS and what the LOSS allocates are separated.
Check: on the first batch, sum_c L_c must equal tworld.rollout_losses(world, s, a, 'teacher', recon=...) (printed, relative error).
--device cpu: the fmamba layers on the reference backend (tests).
--facts (added 11:1x, the loss-currency question on the gradient side): the categories are instead the world's OWN fact errors in each
batch -- every predicted map cell decoded by scoreboard.AllProbes (fitted on true TRAIN tokens; as currency.py) is map_fact_wrong when
its tile class or any of the four mob flags differs from the simulator's, else map_fact_correct; the 18 HUD tokens are hud_value_wrong
when any decoded health / food / drink / energy (rounded) differs, else hud_value_correct; player_token. TEST roots only.
--seen (added 17:4x, Codex's point that `entering` mixes seen and never-seen terrain): entering cells whose world position was in
view in an earlier frame of the window go to entering_seen (true-tile-map scroll offsets, as budget.py / entering_memory.py).
--adam (added 17:4x, Codex's point that a raw gradient is not Adam's update): the run's saved AdamW state
(levers_tworlds_v1/state/<world>.state.pt; endpoints only, its weights checked equal to the loaded world's) gives the
preconditioner P = 1 / (sqrt(v / (1 - b2^t)) + eps). Each category's update u_c = P g_c is reported as: projection share
<u_c, u> / |u|^2, consistency, and the share of the MEAN update over batches <sum_b u_c, sum_b u> / |sum_b u|^2 (momentum
averages batches; noise cancels there). Also its cosine with the run's own last update direction m_hat P. Every run also reports
the raw mean-update share.
Usage: gradanat.py <world.pt> ... [--loss l1,deltairis] [--batches N] [--device cpu] [--facts] [--seen] [--adam]
       -> artifacts/eda/health_chain_v1/gradanat_<world>[_facts][_seen][_adam].json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/health_chain_v1'
HEAD = ('proj.', 'choose.', 'frame.', 'target_gate.', 'event_head.')


def main():
    import teval as T
    import budget as BG
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    args = sys.argv[1:]
    losses = (args[args.index('--loss') + 1] if '--loss' in args else 'l1,deltairis').split(',')
    nb_max = int(args[args.index('--batches') + 1]) if '--batches' in args else 10 ** 9
    paths = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] not in ('--loss', '--batches', '--device'))]
    dev = torch.device(args[args.index('--device') + 1] if '--device' in args else 'cuda')
    facts, seen, adam = '--facts' in args, '--seen' in args, '--adam' in args
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    names = ('map_fact_wrong', 'map_fact_correct', 'hud_value_wrong', 'hud_value_correct', 'player_token') if facts else BG.CATS
    if facts:
        import scoreboard as SB
        probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
        thr = torch.tensor(probes.thr)
    R = len(meta['seed'])
    root, fut, fa = cache['ctx'][:, -1], cache['fut'], cache['fut_a']
    vis = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1)                # [R,17,1534]
    toks = torch.cat([root[:, None], fut], 1)                                                         # [R,17,81,192]
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    ncat = len(names)
    for path in paths:
        world, st = T.load_world(Path(path), dev)
        world.eval()
        if dev.type == 'cpu':
            import dataclasses
            for layer in getattr(world, 'layers', []):
                layer.mix.settings = dataclasses.replace(layer.mix.settings, backend='reference')
        L = world.time.shape[0]
        ok = alive[:, L - 2]                                                                          # alive through future frame L-1
        idx = torch.where(ok & ~train_roots if facts else ok)[0]
        # categories of every transition in the window (frame t -> t+1, t = 0..L-2)
        cats = torch.empty(len(idx), L - 1, 81, dtype=torch.long)
        if seen:                                                                                      # window offsets (true tile maps)
            import encoder_diag as ED
            from scroll import estimate
            tl = [ED.frame_state(vis[idx, t])[0] for t in range(L)]
            dv = []
            for t in range(L - 1):
                frac = []
                for dr, dc in BG.SHIFTS:
                    src = ED.neighbour(tl[t], dr, dc, -1); ov = src >= 0
                    frac.append(((src != tl[t + 1]) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
                est = estimate(toks[idx, t].float(), toks[idx, t + 1].float())
                dv.append(torch.tensor(BG.SHIFTS)[(torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)])
            off = torch.cat([torch.zeros(len(idx), 1, 2, dtype=torch.long), torch.stack(dv, 1).cumsum(1)], 1)    # [n,L,2]
        for t in range(L - 1):
            c, _ = BG.categories(vis[idx, t], vis[idx, t + 1], toks[idx, t].float(), toks[idx, t + 1].float(),
                                 (off[:, :t + 1] - off[:, t:t + 1]) if seen else None)
            cats[:, t] = c
        params = [p for n, p in world.named_parameters() if not n.startswith(HEAD) and p.requires_grad]
        bs = 16 if L <= 6 else 4
        res = {'world': st['name'], 'frames': L, 'windows': int(len(idx)), 'batch': bs, 'backbone_params': sum(p.numel() for p in params)}
        if adam:
            from d4mj.train import phase_optimizer
            sp = Path(path).parent / 'state' / f"{st['name']}.state.pt"
            sd = torch.load(sp, map_location='cpu', weights_only=False)
            same = all(k in sd['world'] and torch.equal(v.cpu(), sd['world'][k]) for k, v in world.state_dict().items())
            opt = phase_optimizer([world], config)
            opt.load_state_dict(sd['optimizer'])
            b1, b2 = opt.param_groups[0]['betas']; eps = opt.param_groups[0]['eps']
            pre, mom = [], []
            for p in params:
                s_ = opt.state.get(p, {})
                if 'exp_avg_sq' in s_:
                    t_ = float(s_['step'])
                    den = (s_['exp_avg_sq'].to(dev).float() / (1 - b2 ** t_)).sqrt() + eps
                    pre.append((1 / den).flatten()); mom.append((s_['exp_avg'].to(dev).float() / (1 - b1 ** t_) / den).flatten())
                else:
                    pre.append(torch.zeros(p.numel(), device=dev)); mom.append(torch.zeros(p.numel(), device=dev))
            Pre, Mupd = torch.cat(pre), torch.cat(mom)
            res['adam_state'] = {'file': sp.name, 'update': int(sd['update']), 'weights_equal_loaded_world': same,
                                 'params_with_state': sum(int('exp_avg_sq' in opt.state.get(p, {})) for p in params)}
            del opt, sd
        for loss in losses:
            acc = {c: {'loss': 0.0, 'norm_ratio': 0.0, 'proj': 0.0, 'gsum': None, 'nsum': 0.0, 'aproj': 0.0, 'usum': None, 'unsum': 0.0} for c in names}
            utot_sum, utot_n = None, 0.0
            cos = torch.zeros(ncat, ncat, dtype=torch.float64)
            gtot_sum, gtot_n, nb = None, 0.0, 0
            for j in range(0, len(idx), bs):
                if nb >= nb_max:
                    break
                ii = idx[j:j + bs]
                s = toks[ii, :L].float().to(dev)
                a = torch.nn.functional.pad(fa[ii, :L - 1], (0, 1)).to(dev)
                with autocast_context(config):
                    pred = world(s, a)[0]
                e = pred[:, :L - 1].float() - s[:, 1:]                                                # [b,L-1,81,192]
                n_el = e.numel()
                ct = cats[j:j + bs].to(dev)                                                           # [b,L-1,81]
                if facts:
                    with torch.no_grad():
                        P = pred[:, :L - 1].detach().float().cpu()
                        cells = P[:, :, SB.MAPC].flatten(0, 2)
                        f_ = SB.AllProbes.facts(vis[ii, 1:L])
                        wrong = (probes.tile(cells).view(*P.shape[:2], 62, 17).argmax(-1) != f_['tile']) | \
                            ((probes.mob(cells) > thr).view(*P.shape[:2], 62, 4) != f_['mobs']).any(-1)
                        hw = (probes.hud(P[:, :, 63:81].flatten(0, 1).flatten(1))[:, :4].round().view(*P.shape[:2], 4)
                              != f_['hud'][..., :4].round()).any(-1)
                        ct = torch.full(P.shape[:3], 4, dtype=torch.long)
                        ct[:, :, SB.MAPC] = torch.where(wrong, 0, 1)
                        ct[:, :, 63:] = torch.where(hw, 2, 3)[..., None]
                        ct = ct.to(dev)
                if loss == 'l1':
                    per_tok = e.abs().sum(-1) / n_el                                                  # [b,L-1,81]
                else:
                    e2 = e ** 2
                    per_tok = (e2.sum(-1) + 0.1 * e.abs().sum(-1)) / n_el
                    worst = e2.flatten(2).max(-1)                                                     # per frame [b,L-1]
                    wtok = worst.indices // 192
                    add = torch.zeros_like(per_tok)
                    add.scatter_(2, wtok[..., None], (0.01 * worst.values / (e.shape[0] * e.shape[1]))[..., None])
                    per_tok = per_tok + add
                total = per_tok.sum()
                if nb == 0:
                    import tworld as TW
                    with torch.no_grad(), autocast_context(config):
                        ref = float(TW.rollout_losses(world, s, a[:, :L - 1], 'teacher', recon='l1' if loss == 'l1' else 'deltairis'))
                    res.setdefault('objective_check', {})[loss] = {'sum_of_terms': float(total), 'rollout_losses': ref,
                                                                   'rel_err': abs(float(total) - ref) / ref}
                g_tot = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
                g_tot = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g_tot, params)]).float()
                gn = float(g_tot.norm()) + 1e-12
                if adam:
                    u_tot = g_tot * Pre; un = float(u_tot.norm()) + 1e-12
                    utot_sum = u_tot.clone() if utot_sum is None else utot_sum + u_tot; utot_n += un
                gs = []
                for ci, c in enumerate(names):
                    m = ct == ci
                    lc = (per_tok * m).sum()
                    if m.any():
                        g = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)
                        g = torch.cat([(gg if gg is not None else torch.zeros_like(p)).flatten() for gg, p in zip(g, params)]).float()
                    else:
                        g = torch.zeros_like(g_tot)
                    gs.append(g)
                    A = acc[c]
                    A['loss'] += float(lc); A['norm_ratio'] += float(g.norm()) / gn; A['proj'] += float(g @ g_tot) / gn ** 2
                    A['gsum'] = g.clone() if A['gsum'] is None else A['gsum'] + g; A['nsum'] += float(g.norm())
                    if adam:
                        u = g * Pre
                        A['aproj'] += float(u @ u_tot) / un ** 2
                        A['usum'] = u.clone() if A['usum'] is None else A['usum'] + u; A['unsum'] += float(u.norm())
                G = torch.stack(gs)                                                                   # [ncat, P]
                Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
                cos += (Gn @ Gn.T).double().cpu()
                gtot_sum = g_tot.clone() if gtot_sum is None else gtot_sum + g_tot; gtot_n += gn
                nb += 1
                del G, Gn, gs, g_tot, pred, e
            tot_loss = sum(A['loss'] for A in acc.values())
            out = {}
            for c, A in acc.items():
                out[c] = {'loss_share': round(A['loss'] / tot_loss, 4), 'grad_norm_ratio': round(A['norm_ratio'] / nb, 4),
                          'projection_share': round(A['proj'] / nb, 4),
                          'consistency': round(float(A['gsum'].norm()) / A['nsum'], 4) if A['nsum'] > 0 else None,
                          'mean_update_share': round(float(A['gsum'] @ gtot_sum) / float(gtot_sum @ gtot_sum), 4)}
                if adam:
                    out[c].update({'adam_projection_share': round(A['aproj'] / nb, 4),
                                   'adam_consistency': round(float(A['usum'].norm()) / A['unsum'], 4) if A['unsum'] > 0 else None,
                                   'adam_mean_update_share': round(float(A['usum'] @ utot_sum) / float(utot_sum @ utot_sum), 4),
                                   'adam_cos_with_saved_update': round(float(A['usum'] @ Mupd) / (float(A['usum'].norm()) * float(Mupd.norm()) + 1e-12), 4)})
            res[loss] = {'batches': nb, 'noise_consistency': round(nb ** -0.5, 4),
                         'total_consistency': round(float(gtot_sum.norm()) / gtot_n, 4), 'categories': out,
                         'cosine_matrix': {names[i]: {names[k]: round(float(cos[i, k] / nb), 3) for k in range(ncat)} for i in range(ncat)}}
            if adam:
                res[loss]['adam_total_consistency'] = round(float(utot_sum.norm()) / utot_n, 4)
                res[loss]['adam_total_cos_with_saved_update'] = round(float(utot_sum @ Mupd) / (float(utot_sum.norm()) * float(Mupd.norm()) + 1e-12), 4)
            print(json.dumps({st['name']: {loss: {c: [v['loss_share'], v['projection_share'], v['consistency'], v['mean_update_share']]
                                                  + ([v['adam_projection_share'], v['adam_consistency'], v['adam_mean_update_share']] if adam else [])
                                                  for c, v in out.items()}}}), flush=True)
            for A in acc.values():
                A['gsum'] = A['usum'] = None
        (OUT / f"gradanat_{st['name']}{'_facts' if facts else ''}{'_seen' if seen else ''}{'_adam' if adam else ''}.json").write_text(json.dumps(res, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
