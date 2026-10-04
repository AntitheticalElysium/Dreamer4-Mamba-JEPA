"""Item 9 follow-up (2026-10-02): do the worlds that erase stone ignore the pickaxe condition? craftax_classic game_logic (read):
mining a tree always turns it into grass (can_mine_tree = True); stone becomes path only with inventory.wood_pickaxe (Inventory
field 6 = visible[1522 + 6]). DO steps facing stone on the diagnosis futures (sample 0); the world's teacher-forced prediction
of the faced cell (probe class != stone = drawn mined), split by pickaxe.
Result (2026-10-02): data: no pickaxe -> stays 962 / 962; pickaxe -> mined 344 / 344 (the rule holds). Drawn mined without /
with a pickaxe: s7 36k 0.005 / 0.904; s8 18k 0.601 / 0.724; s8 36k 0.199 / 0.954 -- the stone-erasing worlds ignore (s8 18k) or
partly ignore (s8 36k) the inventory condition; s7 36k learned it.
Usage: check_stone_pickaxe.py <world.pt> ...
"""
import sys, json, torch
sys.path[:0] = ["artifacts/experiments/20260927_levers", "artifacts/experiments/20260926_diagnosis", "artifacts/experiments/20260921_readout_ladder"]
import check_dohalluc as CH
T, SD, H = CH.T, CH.SD, CH.H
from d4mj.config import config_from_dict
import spatial as Sp
device = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
meta, tr, ts = T.split()
fut5, _ = SD.token_cache(device); cache = T.build_cache("raw", device)
R = len(meta["seed"]); ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]; fut0 = fut5[:, 0]
alive = ~meta["future_dead"].cumsum(2).bool().any(1)
allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()
tiles = allv[:, :, :1071].reshape(R, 17, 63, 17).argmax(-1); facing = allv[:, :, 1516:1520].argmax(-1)
pick = allv[:, :, 1522 + 6] > 0                                      # wood_pickaxe (craftax Inventory field 6)
probes = T.Probes(cache, meta, tr, ts)
sel = []
for r in range(R):
    for k in range(H):
        if int(fa[r, k]) == 5 and bool(alive[r, k]):
            cell = CH.FACED[int(facing[r, k])]
            if int(tiles[r, k, cell]) == 4:                                # DO facing stone
                sel.append((r, k, cell, bool(pick[r, k]), int(tiles[r, k + 1, cell]) != 4))
data = {}
for p, ch in ((p, c) for _, _, _, p, c in sel):
    data[(p, ch)] = data.get((p, ch), 0) + 1
print(json.dumps({"data: (has_pickaxe, stone_changed) -> count": {str(k): v for k, v in data.items()}}))
for path in sys.argv[1:]:
    world, st = T.load_world(path, device)
    cnt = {}
    for j in range(0, len(sel), 128):
        chunk = sel[j:j + 128]
        for w in (4, 5):
            idx = [q for q, (r, k, *_ ) in enumerate(chunk) if (4 if k == 0 else 5) == w]
            if not idx: continue
            wins = torch.stack([torch.cat([ctx[chunk[q][0]].float(), fut0[chunk[q][0], :chunk[q][1]].float()], 0)[-w:] for q in idx])
            acts = torch.stack([torch.cat([ca[chunk[q][0]], fa[chunk[q][0], :chunk[q][1] + 1]], 0)[-w:] for q in idx])
            pt = T.step(world, wins, acts, device, config)
            cls = probes.tile(torch.stack([pt[n, chunk[q][2]] for n, q in enumerate(idx)])).argmax(-1)
            for n, q in enumerate(idx):
                _, _, _, p, ch = chunk[q]
                key = ("pickaxe" if p else "no_pickaxe") + ("|changed" if ch else "|stays")
                c = cnt.setdefault(key, [0, 0]); c[0] += 1; c[1] += int(int(cls[n]) != 4)
    print(json.dumps({st["name"]: {k: {"n": v[0], "drawn_removed": round(v[1] / v[0], 4)} for k, v in sorted(cnt.items())}}))
