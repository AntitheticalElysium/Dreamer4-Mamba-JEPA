"""A12 (2026-10-26): what must the record machinery hold, how fast, and how ambiguous is identity at re-entry?
Data facts for the records pass; reads the A1 rollouts (simulator state, true positions and identities: a perfect
observer, so these are environment properties, not tracker performance). Nothing is trained.

(1) LOAD. CLAIM: the number of entities a perfect tracker must hold (visible + alive out of view and seen within the
    last W steps) rarely exceeds a small cap. COMPETING: skeleton areas and nights pile up entities. CONTROL: W = 16 / 64;
    share of steps above K = 4, 8, 11. OUTCOME: tail sizes for the working cap (K stays a working number).
(2) URGENCY. CLAIM (Codex): new entities can matter on first appearance, so records must be creatable immediately
    (tentative), not only after persistence. COMPETING: entities enter at the view edge, several steps from the player.
    CONTROL: first sightings vs re-entries; steps until the entity is within Manhattan 1 of the player (zombies, arrows)
    and until a zombie attack. OUTCOME: the share of first sightings that become adjacent within 0 / 1 / 2 / 4 steps.
(3) RE-ENTRY IDENTITY. CLAIM: when an entity re-enters view, the nearest same-kind out-of-view entity (by last-seen
    position) is the right identity almost always, so a single identity hypothesis per record suffices. COMPETING: other
    same-kind entities out of view nearby make re-entry ambiguous. CONTROL: candidates = same-kind entities alive and out
    of view whose last-seen position can reach the entry cell (Manhattan <= steps since last seen); nearest-candidate
    correctness, ties counted as failures; by gap length. OUTCOME: the ambiguous share, i.e. how often identity must be
    carried as a distribution across records.
Usage: eda_record_needs.py A1_NPZ OUT_JSON
"""
import json, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")


def segments(D, kind, same, pos):
    rel = D[kind + "_rel"].astype(np.int32); msk = D[kind + "_mask"]
    world = rel + pos[:, None, :]
    prev_w = np.r_[world[:1], world[:-1]]; prev_m = np.r_[msk[:1] & False, msk[:-1]]
    cont = same[:, None] & prev_m & msk & (np.abs(world - prev_w).sum(-1) <= 1)
    new = msk & ~cont
    seg = np.where(msk, np.cumsum(new, 0), -1)                       # segment id per slot (unique within slot)
    vis = msk & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
    N, S = msk.shape
    idx = np.where(vis, np.arange(N)[:, None], -1)
    last = np.maximum.accumulate(idx, 0)                             # last visible row (any segment)
    ok = (last >= 0) & (seg >= 0)
    last_seg = np.where(last >= 0, seg[np.clip(last, 0, None), np.arange(S)[None]], -2)
    last = np.where(ok & (last_seg == seg), last, -1)                # last visible row within the same segment
    return dict(rel=rel, msk=msk, world=world, cont=cont, new=new, seg=seg, vis=vis, last=last)


def main():
    D = np.load(sys.argv[1]); ep = D["episode"]; pos = D["pos"].astype(np.int32)
    same = np.r_[False, ep[1:] == ep[:-1]]; N = len(ep); rows = np.arange(N)
    E = {k: segments(D, k, same, pos) for k in KINDS}
    out = {}
    # (1) load
    for W in (16, 64):
        held = np.zeros(N, int)
        for k in KINDS:
            e = E[k]
            recent = e["msk"] & (e["last"] >= 0) & ((rows[:, None] - e["last"]) <= W)
            held += (e["vis"] | recent).sum(1)
        out[f"load_W{W}"] = {"p50": float(np.percentile(held, 50)), "p90": float(np.percentile(held, 90)),
                             "p99": float(np.percentile(held, 99)), "max": int(held.max()),
                             "share_gt4": float((held > 4).mean()), "share_gt8": float((held > 8).mean()),
                             "share_gt11": float((held > 11).mean())}
        print("load", W, out[f"load_W{W}"], flush=True)
    # (2) urgency at first sighting vs re-entry
    zcd = D["zombies_cd"].astype(int)
    for k in ("zombies", "arrows", "skeletons", "cows"):
        e = E[k]; S = e["msk"].shape[1]
        prev_vis = np.r_[e["vis"][:1] & False, e["vis"][:-1]]
        prev_last = np.r_[np.full((1, S), -1), e["last"][:-1]]
        first = e["vis"] & ~(prev_vis & e["cont"]) & ((prev_last < 0) | e["new"])
        reent = e["vis"] & ~prev_vis & e["cont"] & (prev_last >= 0)
        dist = np.abs(e["rel"]).sum(-1)
        res = {}
        for name, ev in (("first", first), ("reentry", reent)):
            r, j = np.nonzero(ev)
            steps = np.full(len(r), 99); attack = np.zeros(len(r), bool)
            for h in range(0, 17):
                rr = np.clip(r + h, 0, N - 1)
                alive = (e["seg"][rr, j] == e["seg"][r, j]) & (ep[rr] == ep[r])
                hit = alive & (dist[rr, j] <= 1) & (steps == 99)
                steps[hit] = h
                if k == "zombies" and h >= 1:
                    attack |= alive & (zcd[rr, j] == 5) & (zcd[np.clip(rr - 1, 0, None), j] <= 0)
            res[name] = {"n": int(len(r)), "dist_median": float(np.median(dist[r, j])) if len(r) else None,
                         **{f"adjacent_within_{h}": float((steps <= h).mean()) for h in (0, 1, 2, 4)}}
            if k == "zombies":
                res[name]["attack_within_16"] = float(attack.mean())
        out[f"urgency_{k}"] = res
        print("urgency", k, res, flush=True)
    # (3) re-entry identity
    for k in ("zombies", "cows", "skeletons"):
        e = E[k]; S = e["msk"].shape[1]
        prev_vis = np.r_[e["vis"][:1] & False, e["vis"][:-1]]
        prev_last = np.r_[np.full((1, S), -1), e["last"][:-1]]
        reent = e["vis"] & ~prev_vis & e["cont"] & (prev_last >= 0)
        r, j = np.nonzero(reent)
        gaps, ncand, nn_ok = [], [], []
        for t, s in zip(r, j):
            entry = e["world"][t, s]
            g_true = t - prev_last[t, s]
            d_true = np.abs(entry - e["world"][prev_last[t, s], s]).sum()
            c = 0; best_other = 1 << 30
            for o in range(S):
                if o == s or not e["msk"][t - 1, o] or prev_vis[t, o] or prev_last[t, o] < 0:
                    continue
                if ep[prev_last[t, o]] != ep[t]:
                    continue
                g_o = t - prev_last[t, o]; d_o = np.abs(entry - e["world"][prev_last[t, o], o]).sum()
                if d_o <= g_o:
                    c += 1; best_other = min(best_other, d_o)
            gaps.append(g_true); ncand.append(c); nn_ok.append(d_true < best_other)
        gaps = np.array(gaps); ncand = np.array(ncand); nn_ok = np.array(nn_ok)
        res = {"n_reentries": int(len(gaps)), "share_with_other_candidate": float((ncand > 0).mean()),
               "nearest_last_seen_correct": float(nn_ok.mean())}
        for lo, hi in ((1, 2), (3, 8), (9, 32), (33, 10 ** 6)):
            m = (gaps >= lo) & (gaps <= hi)
            if m.any():
                res[f"gap_{lo}_{hi}"] = {"n": int(m.sum()), "with_other": float((ncand[m] > 0).mean()),
                                         "nearest_correct": float(nn_ok[m].mean())}
        out[f"reentry_{k}"] = res
        print("reentry", k, res, flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
