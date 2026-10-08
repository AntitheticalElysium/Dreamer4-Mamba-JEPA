"""What happens to zombies in a world's predictions, one step and in imagination? (2026-10-09 09:00; D's imagination gap: an imagined
zombie is beside the player at 8.9% of real damage steps vs 73.6% in truth.) Held diagnosis futures (teval's Raw cache, sample 0,
window min(4 + k, 15)); true zombie cells from the simulator's visible state (mob channel 0 of visible[1071:1512], 7 x 9 view);
predicted zombie scores = teval Probes.zombie on the predicted frame's 63 map tokens.
Craftax-Classic rule (game_logic.py update_mobs, read 2026-10-09): a zombie within Manhattan 10 of the player moves toward the player
with p = 0.75 (on a max-|difference| axis, chosen at random when tied), else one uniformly random step; at distance 1 with cooldown
<= 0 it attacks and stays.
1. one step, true history (steps with exactly one zombie in view at t and t+1, relative distance 1..6 to the post-move player):
   the true move classified as `toward` (reduces a max-|difference| component by 1), `stay` (no world move: attack or blocked) or
   `other`; then on the predicted frame: zombie score at the true next cell, argmax cell == true cell, cells with score > 0.3,
   total positive score mass; also the score at the true cell on the TRUE next frame (probe ceiling).
2. imagination (self-fed), roots with exactly one zombie in view at the root: per depth 1, 2, 4, 8, 16: peak zombie score and
   positive mass on imagined frames vs on true frames, and share of roots whose imagined frame has any cell > 0.3.
3. calibration of the one-step prediction (added 09:05): steps with one zombie at relative distance 2..6 (no attack possible),
   its 5 candidate next cells in the next view (stay, and the 4 world moves; view cell = v0 - player displacement + move), grouped
   as stay / toward (moves that reduce a max-|difference| component) / other; the world's positive predicted score share per group
   (normalized over the 5 cells) vs the empirical frequency of the true next cell's group. A mean-seeking world matches the
   frequencies; a mode-seeking one puts its share on the most frequent group.
Player displacement per step = scroll.estimate between consecutive true frames (SHIFTS), as health_strata.py.
E22 pre-flight (P2, 09:50): a dworld stage-A file (has 'post') is read with the POSTERIOR's Delta conditioning every window frame
(its code for the transition leaving that frame, the last one from the TRUE next frame), so parts 1 and 3 measure what the channel
carries; part 2 (imagination) is then reported with Delta zeroed (no prior yet).
Usage: health_zombie.py <name=world.pt> ...  -> artifacts/eda/health_chain_v1/zombie_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import check_decision_step as CD  # noqa: E402
SD, T = CD.SD, CD.T
H = CD.H
OUT = Path('artifacts/eda/health_chain_v1')
SHIFTS = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
CENTER = torch.tensor([3, 4])


def cell_rc(mask):
    """[n, 7, 9] bool with exactly one True per row -> [n, 2] (row, col)."""
    idx = mask.flatten(1).float().argmax(1)
    return torch.stack([idx // 9, idx % 9], 1)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device('cuda')
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache('raw', device)
    R = len(meta['seed'])
    ctx, ca, fa = cache['ctx'], cache['ctx_a'], cache['fut_a']
    root = ctx[:, -1]
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()                                             # teval's: sample 0, [R, 16]
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()      # [R, 17, 1520]
    zt = allv[..., 1071:1512].reshape(R, 17, 7, 9, 7)[..., 0] > 0                                  # true zombies [R, 17, 7, 9]
    frames_true = torch.cat([root[:, None], fut5[:, 0]], 1).float().cpu()                           # [R, 17, 81, 192]
    sc = torch.cat([estimate(frames_true[i:i + 16, :-1].flatten(0, 1), frames_true[i:i + 16, 1:].flatten(0, 1)).view(-1, 16)
                    for i in range(0, R, 16)])                                                       # [R, 16] scroll class
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    zscore = lambda tok: probes.zombie(tok[:, :63].float().flatten(0, 1))[:, 0].view(len(tok), 7, 9)
    # one-step cases
    n1 = zt.flatten(2).sum(-1)                                                                       # zombies per frame
    one = (n1[:, :-1] == 1) & (n1[:, 1:] == 1) & alive                                              # [R, 16]
    res_all = {}
    import dworld as DW
    for name, path in (a.split('=', 1) for a in sys.argv[1:]):
        post = None
        if 'post' in torch.load(path, map_location='cpu', weights_only=False):
            world, post, st = DW.load_a(Path(path), device)
        else:
            world, st = T.load_world(Path(path), device)
        pred_tf = torch.empty(R, H, 7, 9)
        gen_peak, gen_mass = torch.zeros(R, H), torch.zeros(R, H)
        for i in range(0, R, 16):
            b = min(16, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            hist = [a3[:, j] for j in range(3)]
            tfr, gfr = [c4[:, j] for j in range(4)], [c4[:, j] for j in range(4)]
            for k in range(H):
                w = min(4 + k, 15)
                acts = torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                if post is not None:                                                        # posterior Delta incl. the true next frame
                    s_ext = torch.stack(tfr[-w:] + [fut5[i:i + b, 0, k].float()], 1).to(device)
                    world.delta = post(s_ext, acts.to(device))[0][:, :w]
                t = T.step(world, torch.stack(tfr[-w:], 1), acts, device, config)
                world.delta = None
                g = T.step(world, torch.stack(gfr[-w:], 1), acts, device, config)
                pred_tf[i:i + b, k] = zscore(t)
                zg = zscore(g)
                gen_peak[i:i + b, k] = zg.flatten(1).max(1).values
                gen_mass[i:i + b, k] = zg.clamp(min=0).flatten(1).sum(1)
                tfr.append(fut5[i:i + b, 0, k].float()); gfr.append(g.to(c4.device)); hist.append(fk[:, k])
        true_score = torch.stack([zscore(frames_true[:, k + 1]) for k in range(H)], 1)                # [R, H, 7, 9] probe on truth
        # 1. one-step by move type
        r_i, k_i = torch.where(one)
        v0, v1 = cell_rc(zt[r_i, k_i]), cell_rc(zt[r_i, k_i + 1])
        disp = SHIFTS[sc[r_i, k_i]]
        rel0 = v0 - (CENTER + disp)                                                                  # vs the post-move player
        rel1 = v1 - CENTER
        move = rel1 - rel0
        d0 = rel0.abs().sum(1)
        keep = (d0 >= 1) & (d0 <= 6)
        amax = rel0.abs().max(1, keepdim=True).values
        on_max_axis = (rel0.abs() == amax)
        toward = ((move.abs().sum(1) == 1) & ((move * -rel0.sign()) == 1).any(1)
                  & (((move != 0) & on_max_axis).any(1)))
        stay = move.abs().sum(1) == 0
        kinds = {'toward': toward & keep, 'stay': stay & keep, 'other': ~toward & ~stay & keep}
        p_cells = pred_tf[r_i, k_i]                                                                  # [n, 7, 9]
        t_cells = true_score[r_i, k_i]
        at_true = p_cells[torch.arange(len(r_i)), v1[:, 0], v1[:, 1]]
        ceiling = t_cells[torch.arange(len(r_i)), v1[:, 0], v1[:, 1]]
        hitcell = p_cells.flatten(1).argmax(1) == (v1[:, 0] * 9 + v1[:, 1])
        spread = (p_cells > 0.3).flatten(1).sum(1).float()
        mass = p_cells.clamp(min=0).flatten(1).sum(1)
        one_step = {}
        for kname, m in kinds.items():
            one_step[kname] = {'n': int(m.sum()), 'score_at_true_cell': round(float(at_true[m].mean()), 3),
                               'probe_ceiling_on_true_frame': round(float(ceiling[m].mean()), 3),
                               'argmax_is_true_cell': round(float(hitcell[m].float().mean()), 3),
                               'cells_above_0.3': round(float(spread[m].mean()), 2), 'positive_mass': round(float(mass[m].mean()), 3)}
        # 3. calibration over the 5 candidate cells, distance 2..6
        dirs = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
        far = keep & (d0 >= 2)
        base = v0 - disp                                                                             # zombie's old cell in the new view
        cand = base[:, None] + dirs[None]                                                            # [n, 5, 2]
        inside = ((cand[..., 0] >= 0) & (cand[..., 0] < 7) & (cand[..., 1] >= 0) & (cand[..., 1] < 9)).all(1)
        far = far & inside & (move.abs().sum(1) <= 1)                                               # one legal step (scroll errors excluded)
        cr, cc = cand[..., 0].clamp(0, 6), cand[..., 1].clamp(0, 8)
        pscore = p_cells[torch.arange(len(r_i))[:, None], cr, cc].clamp(min=0)                        # [n, 5]
        share = pscore / pscore.sum(1, keepdim=True).clamp(min=1e-6)
        tow = torch.zeros(len(r_i), 5, dtype=torch.bool)
        for j in range(1, 5):
            mv = dirs[j][None].expand(len(r_i), 2)
            tow[:, j] = ((mv * -rel0.sign()) == 1).any(1) & ((mv != 0) & on_max_axis).any(1)
        group = torch.where(tow, 1, 2); group[:, 0] = 0                                              # 0 stay, 1 toward, 2 other
        true_j = ((move[:, None] == dirs[None]).all(-1)).float().argmax(1)
        true_g = group[torch.arange(len(r_i)), true_j]
        calib = {}
        for g_i, g_name in enumerate(('stay', 'toward', 'other')):
            calib[g_name] = {'empirical_frequency': round(float((true_g[far] == g_i).float().mean()), 3),
                             'predicted_share': round(float((share[far] * (group[far] == g_i)).sum(1).mean()), 3)}
        calib['n'] = int(far.sum())
        calib['predicted_mass_over_5_cells'] = round(float(pscore[far].sum(1).mean()), 3)
        calib['true_frame_mass_over_5_cells'] = round(float(t_cells[torch.arange(len(r_i))[:, None], cr, cc].clamp(min=0)[far].sum(1).mean()), 3)
        # 2. imagination, roots with exactly one zombie at the root
        roots1 = n1[:, 0] == 1
        true_peak = true_score.flatten(2).max(2).values
        true_mass = true_score.clamp(min=0).flatten(2).sum(2)
        imag = {}
        for d in (1, 2, 4, 8, 16):
            m = roots1 & alive[:, d - 1] & (n1[:, d] >= 1)
            imag[str(d)] = {'roots': int(m.sum()),
                            'imagined_peak': round(float(gen_peak[m, d - 1].mean()), 3), 'true_peak': round(float(true_peak[m, d - 1].mean()), 3),
                            'imagined_mass': round(float(gen_mass[m, d - 1].mean()), 3), 'true_mass': round(float(true_mass[m, d - 1].mean()), 3),
                            'imagined_any_above_0.3': round(float((gen_peak[m, d - 1] > 0.3).float().mean()), 3),
                            'true_any_above_0.3': round(float((true_peak[m, d - 1] > 0.3).float().mean()), 3)}
        res_all[name] = {'world': st['name'], 'posterior_delta_in_one_step': post is not None, 'one_step_by_true_move': one_step,
                         'imagination_one_zombie_roots': imag,
                         'one_step_calibration_distance_2_to_6': calib,
                         'move_share': {k: round(float(v.sum() / keep.sum()), 3) for k, v in kinds.items()}}
        print(json.dumps({name: res_all[name]}), flush=True)
        del world; torch.cuda.empty_cache()
    for name, r in res_all.items():
        (OUT / f'zombie_{name}.json').write_text(json.dumps(r, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
