"""Why does D draw 73% of real damage from true history but 6% in its own imagination? (2026-10-09 03:20; diagnosis by substitution.)
check_damage.py's data, reader and masks exactly (held diagnosis futures, sample 0; health = Probes.hud x 9; damage = true health
drop > 0.5, unchanged = equal; 0.5 cut; window min(4 + k, 15)). Rollouts where the frame fed back at each step is
  gen          the world's own prediction (= check_damage's self-fed)
  gen_trueHUD  its prediction with the 18 HUD tokens (63..80: health and the other intrinsics, i.e. the cooldown-phase history)
               replaced by the true next frame's
  gen_trueMAP  its prediction with the 63 map tokens (zombies, terrain, view) replaced by the true next frame's
  teacher      the true next frame (= check_damage's teacher-forced)
At each step, a drop is drawn when the predicted health < the CURRENT health of the fed-back frame - 0.5 (imagined for gen /
gen_trueMAP, true for gen_trueHUD / teacher). Reported: caught on true damage transitions (all, and view-aligned as check_damage:
imagined scroll offsets equal the true ones at every earlier step), false drops on true unchanged transitions, and on damage steps
whether a zombie is drawn beside the player in the CURRENT fed-back frame (teval Probes.zombie > 0.3 on the 4 neighbour cells).
Usage: health_imagination.py <world.pt> ...  -> artifacts/eda/health_chain_v1/imagination_<world>.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import check_decision_step as CD  # noqa: E402
DA, SD, T = CD.DA, CD.SD, CD.T
H = CD.H
OUT = Path('artifacts/eda/health_chain_v1')
NEAR = [22, 30, 32, 40]


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
    alive = ~meta['future_dead'].cumsum(2).bool().any(1)
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    hp = allv[:, :, 1512] * 9
    prev = torch.cat([root[:, None], fut5[:, 0, :-1]], 1)
    true_off = DA.offsets(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, 0].float()) for i in range(0, R, 16)])).long()
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().flatten(-2))[..., 0] * 9
    zombie_near = lambda tok: (probes.zombie(tok[:, NEAR].float().flatten(0, 1))[:, 0].view(len(tok), 4) > 0.3).any(1)
    drop = (hp[:, 1:] < hp[:, :-1] - 0.5) & alive
    same = (hp[:, 1:] == hp[:, :-1]) & alive
    variants = ('gen', 'gen_trueHUD', 'gen_trueMAP', 'teacher')
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        pred_h = {v: torch.empty(R, H) for v in variants}
        cur_h = {v: torch.empty(R, H) for v in variants}
        znear = {v: torch.empty(R, H, dtype=torch.bool) for v in variants}
        fed = {v: torch.empty(R, H, 81, 192, dtype=torch.float16) for v in ('gen', 'gen_trueHUD')}   # for the alignment rule
        for i in range(0, R, 16):
            b = min(16, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            hist = [a3[:, j] for j in range(3)]
            frames = {v: [c4[:, j] for j in range(4)] for v in variants}
            for k in range(H):
                w = min(4 + k, 15)
                acts = torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                true_next = fut5[i:i + b, 0, k].float().cpu()
                for v in variants:
                    cur = frames[v][-1]
                    pred = T.step(world, torch.stack(frames[v][-w:], 1), acts, device, config)
                    pred_h[v][i:i + b, k] = health(pred); cur_h[v][i:i + b, k] = health(cur.cpu())
                    znear[v][i:i + b, k] = zombie_near(cur.cpu())
                    if v == 'gen':
                        nxt = pred
                    elif v == 'gen_trueHUD':
                        nxt = torch.cat([pred[:, :63], true_next[:, 63:]], 1)
                    elif v == 'gen_trueMAP':
                        nxt = torch.cat([true_next[:, :63], pred[:, 63:]], 1)
                    else:
                        nxt = true_next
                    if v in fed:
                        fed[v][i:i + b, k] = nxt.half()
                    frames[v].append(nxt.to(c4.device) if nxt.device != c4.device else nxt)
                hist.append(fk[:, k])
        res = {'world': st['name'], 'damage_transitions': int(drop.sum()), 'unchanged_transitions': int(same.sum())}
        for v in variants:
            pdrop = pred_h[v] < cur_h[v] - 0.5
            if v in fed:
                g = fed[v]
                gprev = torch.cat([root[:, None].cpu(), g[:, :-1]], 1)
                off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].float(), g[j:j + 32].float()) for j in range(0, R, 32)])).long()
                aligned = torch.cat([torch.ones(R, 1, dtype=torch.bool), (off == true_off.cpu()).all(-1)[:, :-1]], 1)
            else:
                aligned = torch.ones(R, H, dtype=torch.bool)
            res[v] = {'caught_all': round(float(pdrop[drop].float().mean()), 4),
                      'caught_aligned': [round(float(pdrop[drop & aligned].float().mean()), 4), int((drop & aligned).sum())],
                      'false_all': round(float(pdrop[same].float().mean()), 4),
                      'zombie_beside_on_damage': round(float(znear[v][drop].float().mean()), 4),
                      'caught_where_zombie_beside': round(float(pdrop[drop & znear[v]].float().mean()), 4)}
        print(json.dumps(res), flush=True)
        (OUT / f"imagination_{st['name']}.json").write_text(json.dumps(res, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
