"""A13 (2026-10-26): how many entities are uncertain AT ONCE, and does ambiguous identity carry different hidden state?
Records pass v2 (scene-particle belief, as in Structured World Belief): the number of particles needed grows with the
number of entities whose state is uncertain simultaneously (out of view but tracked), and coherent association
matters only when the candidates for an identity differ in hidden state. Reads the A1 rollouts (simulator state);
nothing is trained.

(1) CLAIM: at most one or two entities are out of view and tracked at once, so a joint (scene-level) particle belief
    needs few particles. COMPETING: skeleton areas keep several entities flickering in and out. CONTROL: count alive
    entities not visible now but seen within W = 8 / 16 / 32 steps. OUTCOME: the distribution of that count.
(2) CLAIM (Codex): swapping identical entities is harmless unless their hidden states differ. CONTROL: at zombie /
    skeleton re-entries with another reachable same-kind out-of-view candidate (A12 definition), compare attack
    cooldown "ready" (cd <= 0) between the true entity and each candidate. OUTCOME: the share of ambiguous re-entries
    where a swap would change cooldown readiness (health is not recorded in A1).
Usage: eda_joint_uncertainty.py A1_NPZ OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_record_needs import segments, KINDS


def main():
    D = np.load(sys.argv[1]); ep = D["episode"]; pos = D["pos"].astype(np.int32)
    same = np.r_[False, ep[1:] == ep[:-1]]; N = len(ep); rows = np.arange(N)
    E = {k: segments(D, k, same, pos) for k in KINDS}
    out = {}
    for W in (8, 16, 32):
        cnt = np.zeros(N, int)
        for k in KINDS:
            e = E[k]
            cnt += (e["msk"] & ~e["vis"] & (e["last"] >= 0) & ((rows[:, None] - e["last"]) <= W)).sum(1)
        out[f"out_of_view_tracked_W{W}"] = {str(c): float((cnt == c).mean()) for c in range(0, 6)}
        out[f"out_of_view_tracked_W{W}"]["ge6"] = float((cnt >= 6).mean())
        out[f"out_of_view_tracked_W{W}"]["p99"] = float(np.percentile(cnt, 99))
        print("joint", W, out[f"out_of_view_tracked_W{W}"], flush=True)
    for k in ("zombies", "skeletons"):
        e = E[k]; S = e["msk"].shape[1]; cd = D[k + "_cd"].astype(int)
        prev_vis = np.r_[e["vis"][:1] & False, e["vis"][:-1]]
        prev_last = np.r_[np.full((1, S), -1), e["last"][:-1]]
        reent = e["vis"] & ~prev_vis & e["cont"] & (prev_last >= 0)
        r, j = np.nonzero(reent)
        amb = differ = 0
        for t, s in zip(r, j):
            entry = e["world"][t, s]; cands = []
            for o in range(S):
                if o == s or not e["msk"][t - 1, o] or prev_vis[t, o] or prev_last[t, o] < 0:
                    continue
                if np.abs(entry - e["world"][prev_last[t, o], o]).sum() <= t - prev_last[t, o]:
                    cands.append(o)
            if cands:
                amb += 1
                differ += any((cd[t, o] <= 0) != (cd[t, s] <= 0) for o in cands)
        out[f"identity_{k}"] = {"ambiguous_reentries": amb, "share_cooldown_readiness_differs": differ / max(amb, 1)}
        print("identity", k, out[f"identity_{k}"], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
