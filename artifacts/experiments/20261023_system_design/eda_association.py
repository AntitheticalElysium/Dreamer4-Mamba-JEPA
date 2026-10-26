"""A2 (2026-10-25): how ambiguous is entity identity between consecutive frames in Craftax-Classic, and who attacked?
Facts about the environment and our data for the record-update design (user question: belief over where an entity is
now vs where it moves next; Codex: one image cannot tell which previous record a detection matches). Reads the existing
A1 rollouts (eda_signal_incentives.py: 256 expert episodes + 1,000 episodes per epsilon); nothing is trained.

CLAIM 1: matching each visible entity to the nearest same-kind entity of the previous frame, in world coordinates,
  recovers its identity in nearly all steps (mobs move at most one cell; Franconeri et al. 2012: proximity alone links views).
COMPETING EXPLANATIONS: (a) same-kind entities near each other swap apparent identities (sprites are identical within a
  kind); (b) an entity entering the view where another left creates a false match; (c) without self-motion compensation
  (screen coordinates) every player move shifts all entities and creates ambiguity.
CONTROLS: world vs screen coordinates (gate Manhattan 1 vs 2); ambiguity counted before resolution; min-total-distance
  assignment within kind as the resolver, ties counted as unresolved; confusers = entries landing in the gate of a
  record that left the view or died in the same step.
OUTCOMES: unresolved rate well below 1% means association needs only the previous positions (no multi-hypothesis
  identity in the belief); a rate of several % means the belief must carry identity uncertainty.
CLAIM 2: a zombie attack (cooldown reset to 5) usually has a single adjacent zombie, so the event identifies the attacker.
COMPETING: zombies converge on the player at night, so several can be adjacent. CONTROL: count adjacent zombies at the
  attack. OUTCOME: a substantial multi-adjacent share means attack events must stay latent explanations.
CLAIM 3 (descriptive): attackers have been visible for several steps before attacking (adjacency is inside the view),
  so the out-of-view belief matters for anticipating entries more than for the attack itself.
Usage (scratch venv with numpy): eda_association.py A1_NPZ OUT_JSON
"""
import itertools, json, sys
import numpy as np

KINDS = {"zombies": 3, "cows": 3, "skeletons": 2}


def visible(rel):
    return (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)


def assign(cost):
    """cost [R, D] with np.inf outside the gate -> list of optimal partial assignments (tuples of (r, d)), max matches
    first, then min total cost; every optimum is returned so ties can be detected."""
    R, D = cost.shape
    best, sols = None, []
    for k in range(min(R, D), -1, -1):
        for rs in itertools.combinations(range(R), k):
            for ds in itertools.permutations(range(D), k):
                c = sum(cost[r, d] for r, d in zip(rs, ds))
                if not np.isfinite(c):
                    continue
                key = (-k, c)
                if best is None or key < best:
                    best, sols = key, [tuple(zip(rs, ds))]
                elif key == best:
                    sols.append(tuple(zip(rs, ds)))
        if sols:
            return sols
    return [()]


def main():
    D = np.load(sys.argv[1])
    ep = D["episode"]; pos = D["pos"].astype(int)
    nxt = np.nonzero(ep[:-1] == ep[1:])[0]                           # rows t with t+1 in the same episode
    out = {"n_transitions": int(len(nxt))}
    for kind, n in KINDS.items():
        rel = D[kind + "_rel"].astype(int); msk = D[kind + "_mask"]
        world = rel + pos[:, None, :]
        vis = msk & visible(rel)
        r0, r1 = nxt, nxt + 1
        cont = msk[r0] & msk[r1] & (np.abs(world[r1] - world[r0]).sum(-1) <= 1)  # same entity (slot not reused)
        res = {}
        for frame, gate in (("world", 1), ("screen", 2)):
            P0 = world[r0] if frame == "world" else rel[r0]
            P1 = world[r1] if frame == "world" else rel[r1]
            dist = np.abs(P1[:, None, :, :] - P0[:, :, None, :]).sum(-1)          # [N, rec i, det j]
            ok = vis[r0][:, :, None] & vis[r1][:, None, :] & (dist <= gate)
            target = cont & vis[r0] & vis[r1]                                    # continuing visible entities
            n_rec = ok.sum(1)                                                    # records in the gate of det j
            n_det = ok.sum(2)                                                    # dets in the gate of record i
            uniq = target & (n_rec == 1) & (n_det == 1)                         # a det's true record has slot j
            amb_rows = np.nonzero((target & ~uniq).any(1))[0]
            resolved = wrong = tie = 0
            for row in amb_rows:
                c = np.where(ok[row], dist[row], np.inf).astype(float)
                sols = assign(c)
                for j in np.nonzero(target[row] & ~uniq[row])[0]:
                    got = {dict((d, r) for r, d in s).get(j) for s in sols}
                    if len(got) > 1:
                        tie += 1
                    elif got == {j}:
                        resolved += 1
                    else:
                        wrong += 1
            # confusers: an entity entering view (or spawning) inside the gate of a visible record that is gone at t+1
            entering = vis[r1] & ~(cont & vis[r0])
            gone = vis[r0] & ~(cont & vis[r1])
            conf = (entering[:, None, :] & gone[:, :, None] & (dist <= gate)).any(1) & entering
            res[frame] = {"continuing_visible": int(target.sum()), "unique": int(uniq.sum()),
                          "ambiguous_resolved": resolved, "ambiguous_wrong": wrong, "ambiguous_tie": tie,
                          "unresolved_rate": (wrong + tie) / max(1, int(target.sum())),
                          "entering": int(entering.sum()), "entering_confusable": int(conf.sum())}
            print(kind, frame, res[frame], flush=True)
        out[kind] = res
    # attacks: zombie cooldown reset to 5 during t -> t+1
    rel = D["zombies_rel"].astype(int); msk = D["zombies_mask"]; cd = D["zombies_cd"].astype(int)
    r0, r1 = nxt, nxt + 1
    att = msk[r0] & msk[r1] & (cd[r0] <= 0) & (cd[r1] == 5)                 # game_logic: reset to 5 on attack
    adj1 = msk[r1] & (np.abs(rel[r1]).sum(-1) == 1)
    rows, slots = np.nonzero(att)
    n_adj = adj1[rows].sum(1)
    # visible run length of the attacker up to t+1
    vis_z = msk & visible(rel)
    runs = []
    for row, j in zip(r1[rows], slots):
        k = 0
        while row - k >= 0 and ep[row - k] == ep[row] and vis_z[row - k, j]:
            k += 1
        runs.append(k)
    runs = np.array(runs)
    sleep = D["sleep"][r0[rows]]
    out["attacks"] = {"n": int(len(rows)), "adjacent_zombies_hist": {int(k): int((n_adj == k).sum()) for k in range(4)},
                      "share_multi_adjacent": float((n_adj >= 2).mean()),
                      "multi_attacks_same_step": int((att.sum(1) >= 2).sum()),
                      "visible_run_median": float(np.median(runs)), "visible_run_p10": float(np.percentile(runs, 10)),
                      "share_visible_le_3": float((runs <= 3).mean()), "share_asleep": float(sleep.mean())}
    print("attacks", out["attacks"], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
