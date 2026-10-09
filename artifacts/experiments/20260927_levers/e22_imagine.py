"""E22 readings 1-2 (NOTEBOOK "E22", predeclared 09:18; E22F 10:37): imagination with Delta SAMPLED by the stage-B prior.
Data, masks and reader exactly as health_imagination.py (damage) and health_zombie.py (zombie localization): held diagnosis futures,
sample 0, window min(4 + k, world frames - 1) (15 for L16 worlds, as both), health = teval Probes.hud x 9, zombie score = teval
Probes.zombie on the 63 map tokens. The rollout is check_e16's self-fed rollout: context transitions carry the stage-A posterior's
codes (it sees the true context frames), every imagined transition's 4 codes are sampled by the prior from the imagined history
(last <= 21 blocks), and each window frame is conditioned on the code of the transition leaving it (check_e16.step).
Rollouts per root: `zeroed` (Delta None: for a frozen-decoder file this IS the decoder's deterministic rollout, a built-in check
against health_imagination / health_zombie on that decoder) and S prior samples (generator seed 20261009).
  1. localization (health_zombie part 2): roots with exactly one zombie at the root, alive (sample 0) and >= 1 zombie at depth d:
     imagined peak zombie score and share of roots with a cell > 0.3, d = 1, 2, 4, 8, 16 (truth: health_zombie's true_*)
  2. damage (health_imagination `gen`): a drop is drawn when the predicted health < the imagined current health - 0.5; caught on
     true damage transitions (all, and view-aligned: imagined scroll offsets equal the true ones at every earlier step), false on
     true unchanged transitions (health_imagination's alive: no sample dead)
Sampled readings are the mean over the S samples (each sample's rate, then the mean; the per-sample values are kept).
In-distribution readings (added 14:03; E22Z sufficiency: reading 1 passed, reading 2 failed with 6% false): a sampled world is judged
against reality's RATES, not against the one realized trajectory: how often an imagined zombie is beside the player (probe > 0.3 on
the 4 neighbour cells of the current imagined frame) vs a true one (simulator, same cells), and the drawn-hit rate given imagined
adjacency vs the true hit rate given true adjacency; plus caught / false split by imagined adjacency.
Zombie census by depth (added 14:15; sampled E22Z puts a zombie beside the player 2.5x as often as reality): per depth 1, 2, 4, 8, 16
over alive roots, imagined vs true: cells with zombie score > 0.3 (a count of drawn zombies), and among roots with >= 1 such cell the
Manhattan distance of the nearest one to the player (3, 4); truth = the probe on the true frames (as true_peak).
Usage: e22_imagine.py <stage-B prior .pt> [--samples S]  -> artifacts/eda/health_chain_v1/e22_imagine_<prior name>.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import check_e16 as CE  # noqa: E402
import dworld as DW  # noqa: E402
DA, SD, T, H = CE.DA, CE.SD, CE.T, CE.H
OUT = Path('artifacts/eda/health_chain_v1')
NEAR = [22, 30, 32, 40]


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    args = sys.argv[1:]
    S = int(args[args.index('--samples') + 1]) if '--samples' in args else 4
    prior_path = Path(args[0])
    device = torch.device('cuda')
    st_b = torch.load(prior_path, map_location='cpu', weights_only=False)
    world, post, st_a = DW.load_a(Path(st_b['stage_a']), device)
    prior = DW.Prior().to(device); prior.load_state_dict(st_b['prior']); prior.eval()
    cap = world.time.shape[0] - 1
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache('raw', device)
    R = len(meta['seed'])
    ctx, ca, fa = cache['ctx'], cache['ctx_a'], cache['fut_a']
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9
    zscore = lambda tok: probes.zombie(tok[:, :63].float().cpu().flatten(0, 1))[:, 0].view(len(tok), 7, 9)
    zombie_near = lambda tok: (probes.zombie(tok[:, NEAR].float().cpu().flatten(0, 1))[:, 0].view(len(tok), 4) > 0.3).any(1)
    # truth, health_imagination's masks
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    hp = allv[:, :, 1512] * 9
    alive_any = ~meta['future_dead'].cumsum(2).bool().any(1)
    drop = (hp[:, 1:] < hp[:, :-1] - 0.5) & alive_any
    same = (hp[:, 1:] == hp[:, :-1]) & alive_any
    root = ctx[:, -1].cpu()
    prev = torch.cat([root[:, None], fut5[:, 0, :-1].cpu()], 1)
    true_off = DA.offsets(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, 0].float().cpu()) for i in range(0, R, 16)])).long()
    # truth, health_zombie's
    alive0 = ~meta['future_dead'][:, 0].cumsum(1).bool()
    zt = allv[..., 1071:1512].reshape(R, 17, 7, 9, 7)[..., 0] > 0
    n1 = zt.flatten(2).sum(-1)
    frames_true = torch.cat([root[:, None], fut5[:, 0].cpu()], 1).float()
    true_score = torch.stack([zscore(frames_true[:, k + 1]) for k in range(H)], 1)
    true_peak = true_score.flatten(2).max(2).values
    true_beside = zt[:, :-1].flatten(2)[:, :, NEAR] .any(-1) & alive_any                        # [R, H] true zombie beside, current frame
    true_frames = torch.cat([ctx.cpu(), fut5[:, 0].cpu()], 1)                                   # [R, 4 + 16, 81, 192]
    acts_all = torch.cat([ca.cpu(), fa.cpu()], 1)                                                # [R, 3 + 16]
    names = ['zeroed'] + [f'sample{j}' for j in range(S)]
    g = torch.Generator(device=device).manual_seed(20261009)
    res = {'prior': st_b.get('name', prior_path.stem), 'stage_a': st_a['name'], 'samples': S, 'window_cap': cap,
           'damage_transitions': int(drop.sum()), 'unchanged_transitions': int(same.sum())}
    per = {}
    for name in names:
        pred_h, cur_h, znear = torch.empty(R, H), torch.empty(R, H), torch.empty(R, H, dtype=torch.bool)
        peak, gen = torch.empty(R, H), torch.empty(R, H, 81, 192, dtype=torch.float16)
        zmap = torch.empty(R, H, 7, 9)
        for i in range(0, R, 32):
            b = min(32, R - i)
            tf = true_frames[i:i + b].float().to(device); ac = acts_all[i:i + b].to(device)
            pc = post.quantizer(post.encode(tf[:, :3].flatten(0, 1), ac[:, :3].flatten(), tf[:, 1:4].flatten(0, 1)))[1].view(b, 3, DW.K)
            imagined, icodes = [tf[:, j] for j in range(4)], [pc[:, j] for j in range(3)]
            for k in range(H):
                c = 3 + k
                iw = min(len(imagined), cap)
                win, wa = torch.stack(imagined[-iw:], 1), ac[:, c + 1 - iw:c + 1]
                if name == 'zeroed':
                    world.delta = None
                    nxt = T.step(world, win, wa, device, config).to(device)
                    scodes = torch.zeros(b, DW.K, dtype=torch.long, device=device)              # never read
                else:
                    isn = torch.stack(imagined[-DW.BLOCKS:], 1)
                    ian = ac[:, c + 1 - isn.shape[1]:c + 1]
                    scodes, _ = prior.sample(isn, ian, torch.stack(icodes[-(isn.shape[1] - 1):], 1), generator=g)
                    cw = torch.stack(icodes[len(icodes) - (iw - 1):] + [scodes], 1)
                    nxt = CE.step(world, post, win, wa, cw, device, config).to(device)
                cur = imagined[-1]
                pred_h[i:i + b, k] = health(nxt); cur_h[i:i + b, k] = health(cur); znear[i:i + b, k] = zombie_near(cur)
                zs_ = zscore(nxt)
                peak[i:i + b, k] = zs_.flatten(1).max(1).values
                zmap[i:i + b, k] = zs_
                gen[i:i + b, k] = nxt.half().cpu()
                imagined.append(nxt); icodes.append(scodes)
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])).long()
        aligned = torch.cat([torch.ones(R, 1, dtype=torch.bool), (off == true_off).all(-1)[:, :-1]], 1)
        pdrop = pred_h < cur_h - 0.5
        r = {'caught_all': round(float(pdrop[drop].float().mean()), 4),
             'caught_aligned': [round(float(pdrop[drop & aligned].float().mean()), 4), int((drop & aligned).sum())],
             'false_all': round(float(pdrop[same].float().mean()), 4),
             'zombie_beside_on_damage': round(float(znear[drop].float().mean()), 4),
             'caught_where_zombie_beside': round(float(pdrop[drop & znear].float().mean()), 4),
             'false_where_zombie_beside': [round(float(pdrop[same & znear].float().mean()), 4), int((same & znear).sum())],
             'false_where_no_zombie_beside': [round(float(pdrop[same & ~znear].float().mean()), 4), int((same & ~znear).sum())],
             'beside_rate': [round(float(znear[alive_any].float().mean()), 4), round(float(true_beside[alive_any].float().mean()), 4)],
             'hit_rate_given_beside': [round(float(pdrop[znear & alive_any].float().mean()), 4),
                                       round(float(drop[true_beside].float().mean()), 4)]}
        roots1 = n1[:, 0] == 1
        dist = (torch.arange(7)[:, None] - 3).abs() + (torch.arange(9)[None, :] - 4).abs()            # [7, 9] to the player
        census = lambda zm: ((zm > 0.3).flatten(-2).sum(-1).float(),
                             torch.where(zm > 0.3, dist.float(), torch.tensor(99.)).flatten(-2).min(-1).values)
        cnt_i, near_i = census(zmap); cnt_t, near_t = census(true_score)
        r['census'] = {}
        for d in (1, 2, 4, 8, 16):
            m = alive0[:, d - 1]
            mi, mt = m & (near_i[:, d - 1] < 99), m & (near_t[:, d - 1] < 99)
            r['census'][str(d)] = {'cells_above_0.3': [round(float(cnt_i[m, d - 1].mean()), 3), round(float(cnt_t[m, d - 1].mean()), 3)],
                                   'nearest_distance': [round(float(near_i[mi, d - 1].mean()), 2), round(float(near_t[mt, d - 1].mean()), 2)],
                                   'nearest_within_1': [round(float((near_i[m, d - 1] <= 1).float().mean()), 4),
                                                        round(float((near_t[m, d - 1] <= 1).float().mean()), 4)]}
        r['localization'] = {}
        for d in (1, 2, 4, 8, 16):
            m = roots1 & alive0[:, d - 1] & (n1[:, d] >= 1)
            r['localization'][str(d)] = {'roots': int(m.sum()), 'imagined_peak': round(float(peak[m, d - 1].mean()), 3),
                                         'imagined_any_above_0.3': round(float((peak[m, d - 1] > 0.3).float().mean()), 3),
                                         'true_peak': round(float(true_peak[m, d - 1].mean()), 3),
                                         'true_any_above_0.3': round(float((true_peak[m, d - 1] > 0.3).float().mean()), 3)}
        per[name] = r
        print(json.dumps({name: r}), flush=True)
    res['per_rollout'] = per
    smp = [per[n] for n in names[1:]]
    mean = lambda f: round(sum(f(x) for x in smp) / len(smp), 4)
    res['sampled_mean'] = {'caught_aligned': mean(lambda x: x['caught_aligned'][0]), 'false_all': mean(lambda x: x['false_all']),
                           'caught_all': mean(lambda x: x['caught_all']),
                           'depth8_any_above_0.3': mean(lambda x: x['localization']['8']['imagined_any_above_0.3']),
                           'depth8_peak': mean(lambda x: x['localization']['8']['imagined_peak'])}
    res['readings'] = {'e22_localized': res['sampled_mean']['depth8_any_above_0.3'] >= 0.5 and res['sampled_mean']['depth8_peak'] >= 0.4,
                       'e22_damage': res['sampled_mean']['caught_aligned'] >= 0.3 and res['sampled_mean']['false_all'] <= 0.02}
    print(json.dumps({'sampled_mean': res['sampled_mean'], 'readings': res['readings']}), flush=True)
    (OUT / f"e22_imagine_{res['prior']}.json").write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
