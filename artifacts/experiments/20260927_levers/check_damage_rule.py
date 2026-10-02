"""Decision check (2026-10-03): is the damage a world must draw RANDOM, or a rule on what the world sees?
The 2026-10-02 note "damage is mostly aleatoric (mean P 0.35-0.51 across 5 futures)" compared the 5 futures at the same STEP,
where their states have already diverged (zombies move at random). From ONE state the drop repeats: one-step successors (17
actions x 4 keys from the root) drop in 4/4 keys for 459 of 473 dropping (root, action) pairs, 2/4 for 14; futures step 1 (shared
root) 5/5 for 20 of 21 roots.
Craftax-Classic source (game_logic.py, read 2026-10-03): craftax_step = action, move_player, update_mobs, ..., intrinsics. A zombie
hits (2; 7 if asleep) iff its PRE-move position is at Manhattan distance 1 from the player's POST-move position and its hidden
attack_cooldown <= 0 (reset to 5 on a hit, -1 every other step); arrows hit for 2; a hidden recover counter costs 1 health about
every 16 steps without food / drink / energy; lava kills.
Diagnosis futures (all 5 samples, 16 steps; frame 0 = root): per transition t -> t+1, the player's displacement from the exact map
tiles (best of 5 shifts, overlap agreement >= 0.9, else ambiguous), the zombies of frame t (visible mob channel 0) with their hidden
cooldown (hidden grid channel 0 x 5), the rule's predicted zombie hit, and the true health change (visible HUD x 9).
  rule_hit           some zombie adjacent to the post-move player with cooldown <= 0
  adjacent           some zombie adjacent to the post-move player (VISIBLE: positions, action and the move's outcome)
  hit_in_window      a drop of >= 2 in the 3 transitions inside the world's 4-frame input (t-3 -> t), visible; k >= 3 only
Readings, declared before running:
  rule_exact      the rule (with the hidden cooldown) predicts >= 0.95 of the drops of size >= 2, with <= 0.05 false hits
  damage_visible  P(hit | adjacent, no hit in window) >= 0.8: a deterministic 4-frame world could draw the hit from its input
Usage: check_damage_rule.py   (CPU, data only)
"""
import json
import sys

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import teval as T  # noqa: E402

SHIFTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]          # player displacement (row, col)


def main():
    meta, _, _ = T.split()
    R = len(meta["seed"])
    vis = torch.cat([meta["root_visible"][:, None, None].expand(R, 5, 1, 1534),
                     meta["future_visible"]], 2).float()                          # [R,5,17,1534]
    hid = torch.cat([meta["root_hidden"][:, None, None].expand(R, 5, 1, 319),
                     meta["future_hidden"]], 2).float()
    dead = meta["future_dead"].bool()                                             # [R,5,16] death at transition k
    alive_before = ~torch.cat([torch.zeros(R, 5, 1, dtype=torch.bool), dead.cumsum(2).bool()[:, :, :-1]], 2)   # alive at t
    tiles = vis[..., :1071].reshape(R, 5, 17, 7, 9, 17).argmax(-1)
    zomb = vis[..., 1071:1512].reshape(R, 5, 17, 7, 9, 7)[..., 0] > 0
    cd = hid[..., :315].reshape(R, 5, 17, 7, 9, 5)[..., 0] * 5
    hp = (vis[..., 1512] * 9).round()
    asleep = vis[..., 1520] > 0.5
    dh = hp[:, :, 1:] - hp[:, :, :-1]                                             # [R,5,16]
    a, b = tiles[:, :, :-1], tiles[:, :, 1:]
    agree = []
    for dr, dc in SHIFTS:                                  # tiles_{t+1}[r, c] == tiles_t[r + dr, c + dc] on the overlap
        r0, r1, c0, c1 = max(0, -dr), 7 - max(0, dr), max(0, -dc), 9 - max(0, dc)
        eq = (b[..., r0:r1, c0:c1] == a[..., r0 + dr:r1 + dr, c0 + dc:c1 + dc]).float().mean((-1, -2))
        agree.append(eq)
    agree = torch.stack(agree, -1)                                                # [R,5,16,5]
    best = agree.argmax(-1)
    ok = agree.max(-1).values >= 0.9
    rr = torch.arange(7)[:, None].expand(7, 9); cc = torch.arange(9)[None].expand(7, 9)
    disp = torch.tensor(SHIFTS)[best]                                             # [R,5,16,2]
    pr, pc = 3 + disp[..., 0], 4 + disp[..., 1]
    dist = (rr - pr[..., None, None]).abs() + (cc - pc[..., None, None]).abs()    # [R,5,16,7,9]
    adj_cells = (dist == 1) & zomb[:, :, :-1]
    adjacent = adj_cells.any((-1, -2))
    rule_hit = (adj_cells & (cd[:, :, :-1] <= 0)).any((-1, -2))
    valid = alive_before & ok & ~dead                                             # alive after, displacement identified
    drop2 = dh <= -2
    out = {"transitions": int(valid.sum()), "ambiguous_displacement": int((alive_before & ~ok & ~dead).sum()),
           "health_change_counts": {str(int(v)): int(n) for v, n in zip(*torch.unique(dh[valid], return_counts=True))}}
    v = valid
    out["rule"] = {"drops_ge2": int((drop2 & v).sum()),
                   "rule_hit_on_drops_ge2": float(rule_hit[drop2 & v].float().mean()),
                   "rule_hit_without_drop": float((~drop2[rule_hit & v]).float().mean()),
                   "hit_size_7_when_asleep": int((dh[v & rule_hit & asleep[:, :, :-1]] <= -7).sum()),
                   "asleep_rule_hits": int((v & rule_hit & asleep[:, :, :-1]).sum())}
    # visible predictability: transitions with k >= 3 so the world's 4-frame window is inside the stored trajectory
    k = torch.arange(16)
    win = torch.zeros_like(drop2)
    for j in (1, 2, 3):
        win[:, :, j:] |= drop2[:, :, :-j]
    w3 = v & (k >= 3)
    sel = w3 & adjacent
    out["visible"] = {"adjacent_transitions": int(sel.sum()),
                      "p_hit_given_adjacent": float(drop2[sel].float().mean()),
                      "p_hit_given_adjacent_no_hit_in_window": float(drop2[sel & ~win].float().mean()),
                      "n_adjacent_no_hit_in_window": int((sel & ~win).sum()),
                      "p_hit_given_adjacent_hit_in_window": float(drop2[sel & win].float().mean()),
                      "n_adjacent_hit_in_window": int((sel & win).sum()),
                      "p_hit_given_not_adjacent": float(drop2[w3 & ~adjacent].float().mean()),
                      "drops_ge2_not_adjacent": int((drop2 & w3 & ~adjacent).sum())}
    # same-state repeat (the corrected aleatoric claim): one-step successors from the root, 17 actions x 4 keys
    hp0 = (meta["root_visible"][:, 1512].float() * 9).round()
    hp1 = (meta["onestep_visible"][..., 1512].float() * 9).round()
    drop1 = (hp1 <= hp0[:, None, None] - 2) | meta["onestep_dead"].bool()
    n = drop1.sum(1)
    out["same_state_onestep"] = {"pairs_with_drop": int((n > 0).sum()), "keys_dropping_hist_1_to_4": [int((n == q).sum()) for q in range(1, 5)]}
    out["readings"] = {"rule_exact": out["rule"]["rule_hit_on_drops_ge2"] >= 0.95 and out["rule"]["rule_hit_without_drop"] <= 0.05,
                       "damage_visible": out["visible"]["p_hit_given_adjacent_no_hit_in_window"] >= 0.8}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
