"""Does a camera scroll alone change how persisting content is rendered? (2026-10-16, CPU, pixels + exact state; diagnostic)

diagnosis_futures_v1, 240 seeded roots x 5 futures, every transition. Persisting cells: exact content unchanged after the exact
camera shift (aligned). Per transition: share of persisting cells whose 7x7x3 pixels change (max |diff| > 2). Compared for camera
STILL vs SCROLL transitions: share of transitions with zero changed cells, and the mean changed share; for scroll transitions,
which cells change by distance from the player (Manhattan) and by view border.
Usage: scroll_render.py -> artifacts/eda/whiteboard/scroll_render.json
"""
import json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rep_layers as RL

def main():
    import rollouts as Rr
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261028))[:240].tolist()
    stats = {'still': [], 'scroll': []}
    by_dist = {}
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            frames = torch.cat([r['context'][-1:], r['future_frames'][k]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k]])
            tiles, mobs, light = RL.facts(vis)
            px = frames.reshape(17, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(17, 81, 147).int()
            for t in range(16):
                dr, dc = RL.camera(tiles[t], tiles[t + 1])
                ch, tot = 0, 0
                for x in range(63):
                    rr, cc = divmod(x, 9)
                    if x == 31 or not (0 <= rr + dr < 7 and 0 <= cc + dc < 9):
                        continue
                    src = (rr + dr) * 9 + cc + dc
                    if src == 31 or int(tiles[t + 1, x]) != int(tiles[t, src]) or bool((mobs[t + 1, x] != mobs[t, src]).any()):
                        continue
                    c = bool((px[t + 1, x] - px[t, src]).abs().amax() > 2)
                    tot += 1; ch += c
                    if (dr, dc) != (0, 0):
                        dkey = abs(rr - 3) + abs(cc - 4)
                        a = by_dist.setdefault(dkey, [0, 0]); a[0] += 1; a[1] += c
                if tot:
                    stats['still' if (dr, dc) == (0, 0) else 'scroll'].append(ch / tot)
    res = {}
    for k, v in stats.items():
        v = torch.tensor(v)
        res[k] = {'transitions': len(v), 'share_with_zero_changed': round(float((v == 0).float().mean()), 4),
                  'mean_changed_share': round(float(v.mean()), 4), 'median_changed_share': round(float(v.median()), 4)}
    res['scroll_changed_share_by_player_distance'] = {d: round(a[1] / a[0], 4) for d, a in sorted(by_dist.items())}
    print(json.dumps(res, indent=1))
    (HERE.parents[1] / 'eda/whiteboard/scroll_render.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
