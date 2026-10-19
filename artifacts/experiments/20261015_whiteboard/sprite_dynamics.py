"""Do mob sprites change appearance while the mob stays in place? (2026-10-16, CPU, pixels + exact state; diagnostic)

diagnosis_futures_v1, 240 seeded roots x 5 futures. For consecutive frames with NO camera scroll, every cell whose exact content
(tile class and mob channels) is unchanged is classed: zombie / cow / skeleton (the mob stayed in this cell) / terrain. Per class:
share of cells whose 7x7x3 pixels change (max |diff| > 2), split by whether the frame's terrain cells changed at all (global
rendering event) -- 'quiet' transitions have zero changed terrain cells, so any change there is the cell's own appearance.
For zombies also: whether the change coincides with the zombie being adjacent to the player (attack phase visuals?).
Usage: sprite_dynamics.py -> artifacts/eda/whiteboard/sprite_dynamics.json
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
    tally = {}
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            frames = torch.cat([r['context'][-1:], r['future_frames'][k]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k]])
            tiles, mobs, light = RL.facts(vis)
            px = frames.reshape(17, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(17, 81, 147).int()
            for t in range(16):
                if bool(r['future_dead'][k][max(t - 1, 0)]) or RL.camera(tiles[t], tiles[t + 1]) != (0, 0):
                    continue
                same = (tiles[t + 1, :63] == tiles[t, :63]) & (mobs[t + 1, :63] == mobs[t, :63]).all(-1)
                changed = (px[t + 1, :63] - px[t, :63]).abs().amax(-1) > 2
                terrain = same & ~mobs[t, :63].any(-1); terrain[31] = False
                quiet = not bool(changed[terrain].any())
                adj = torch.zeros(63, dtype=torch.bool); adj[[22, 30, 32, 40]] = True
                for name, m in (('zombie', same & mobs[t, :63, 0]), ('cow', same & mobs[t, :63, 1]), ('skeleton', same & mobs[t, :63, 2]), ('terrain', terrain)):
                    for x in torch.where(m)[0].tolist():
                        key = f"{name}_{'quiet' if quiet else 'global'}" + ('_adjacent' if name == 'zombie' and bool(adj[x]) else '')
                        a = tally.setdefault(key, [0, 0]); a[0] += 1; a[1] += int(changed[x])
    res = {k: {'cells': v[0], 'pixels_changed_share': round(v[1] / max(v[0], 1), 4)} for k, v in sorted(tally.items())}
    print(json.dumps(res, indent=1))
    (HERE.parents[1] / 'eda/whiteboard/sprite_dynamics.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
