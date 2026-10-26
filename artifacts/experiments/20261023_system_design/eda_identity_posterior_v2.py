"""A14b (2026-10-26): A14 rerun with two defects fixed (Codex): (1) when testing an alternative identity, every
entity whose identity is in question (the returning one and all candidates) is removed from the blocker set, so the true
returner can no longer block an alternative from the entry cell; (2) zombies that are adjacent with cooldown <= 0
attack and stay put (game_logic), tracked with a per-particle cooldown started from the candidate's cooldown at its
last sighting. Health is recorded, so "consequential" covers cooldown readiness OR health.

A14 protocol (unchanged otherwise): how often is identity at re-entry GENUINELY ambiguous under the true motion model, and consequential?
User question: does a belief of several coherent scene hypotheses (particles) differ from one coherent hypothesis with
a distribution per object? It differs only where association is uncertain. A12 / A13 used a maximum-speed envelope;
here each candidate identity's probability of producing the observed re-entry is computed with the game's own mob rules
(A3 machinery: Monte-Carlo propagation from the candidate's last sighting along the real player path, real map and real
other mobs, conditioned on not being seen in between). Simulator state is used for evaluation only; nothing is trained.

CLAIM: with a correct motion model, the identity posterior at re-entry is nearly always concentrated (P(true) > 0.9),
  so a single hypothesis (one hard association per step) is rarely wrong and alternatives rarely matter.
COMPETING EXPLANATION: same-kind entities wander in the same area (cows graze, zombies converge), so posteriors are
  often split and a single hypothesis is often wrong.
CONTROLS: equal prior over candidates; candidates = same-kind entities alive and out of view with reachable last
  sightings (A12); gaps <= 32 steps; births of never-seen entities are NOT included as alternatives (so ambiguity is a
  lower bound). Consequential = an alternative with posterior >= 0.1 differs from the true zombie in cooldown readiness.
OUTCOMES: the MAP-association error rate, the share of re-entries with P(true) < 0.9, and consequential ambiguous
  events per 1,000 steps. Rare -> one hypothesis per step (plus revision) is enough; frequent -> keep alternatives.
Usage (project venv): eda_identity_posterior.py OUT_JSON N_EPISODES N_PARTICLES
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_outofview_belief as A3
RP = A3.RP
MAXGAP = 32


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "map": s.map.astype(jnp.int8), "zombies_cd": s.zombies.attack_cooldown,
                     "zombies_hp": s.zombies.health, "cows_hp": s.cows.health,
                     **{f"{m}_p": getattr(s, m).position for m in A3.KINDS}, **{f"{m}_m": getattr(s, m).mask for m in A3.KINDS}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


def last_visible(vis, cont):
    """Per slot, the last visible row within the same continuous segment (else -1)."""
    T, S = vis.shape
    last = np.full((T, S), -1)
    cur = np.full(S, -1)
    for t in range(T):
        cur = np.where(cont[t], cur, -1)
        cur = np.where(vis[t], t, cur)
        last[t] = cur
    return last


def weight(R, kind, h, t0, t1, entry, N, rng, excluded):
    """P(entity h, last seen at t0, unseen until t1 - 1, appears at `entry` at t1) by Monte Carlo. Entities in
    `excluded` (all identities in question) are removed from the blocker set."""
    R2 = dict(R); m = R[f"{kind}_m"].copy(); m[:, list(excluded)] = False; R2[f"{kind}_m"] = m; R2["_j"] = h
    x = np.repeat(R[f"{kind}_p"][t0, h][None].astype(int), N, 0); w = 1.0
    cd = np.full(N, int(R["zombies_cd"][t0, h])) if kind == "zombies" else None
    for s in range(t0, t1):
        x_new, desp = A3.step_particles(x, kind, R2, s, rng)
        if kind == "zombies":
            attack = (np.abs(x - R["pos"][s + 1][None].astype(int)).sum(1) == 1) & (cd <= 0)
            x_new = np.where(attack[:, None], x, x_new); cd = np.where(attack, 5, cd - 1)
        x = x_new
        keep = ~desp
        if s + 1 < t1:
            keep &= ~A3.visible(x - R["pos"][s + 1][None].astype(int))
        f = keep.mean(); w *= f
        if f == 0:
            return 0.0
        if s + 1 < t1:
            idx = rng.choice(np.nonzero(keep)[0], N); x = x[idx]
            if cd is not None:
                cd = cd[idx]
        else:
            return w * float(((x == entry).all(1) & keep).sum() / max(keep.sum(), 1))
    return 0.0


def main():
    out_path, n_eps, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261027)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps_ = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps_) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn(); rows = []; total_steps = 0
    for e, (sh, slot) in enumerate(picks):
        acts = RP.episode_fields(sh, slot)["actions_taken"].numpy().astype(np.int32); T = len(acts); total_steps += T
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        R = {k: v[: T + 1] for k, v in R.items()}
        pos = R["pos"].astype(int)
        for kind in ("zombies", "cows"):
            P, M = R[f"{kind}_p"].astype(int), R[f"{kind}_m"]
            vis = M & A3.visible(P - pos[:, None, :])
            cont = np.zeros_like(M); cont[1:] = M[1:] & M[:-1] & (np.abs(P[1:] - P[:-1]).sum(-1) <= 1)
            last = last_visible(vis, cont)
            for t in range(1, T + 1):
                for s in range(P.shape[1]):
                    if not (vis[t, s] and not vis[t - 1, s] and cont[t, s] and last[t - 1, s] >= 0):
                        continue
                    g = t - last[t - 1, s]
                    if g > MAXGAP:
                        continue
                    entry = P[t, s]
                    cands = [o for o in range(P.shape[1]) if o != s and M[t - 1, o] and not vis[t - 1, o]
                             and last[t - 1, o] >= 0 and t - last[t - 1, o] <= MAXGAP
                             and np.abs(entry - P[last[t - 1, o], o]).sum() <= t - last[t - 1, o]]
                    if not cands:
                        rows.append({"kind": kind, "gap": int(g), "ambiguous_geom": False}); continue
                    hs = [s] + cands
                    ws = np.array([weight(R, kind, h, last[t - 1, h], t, entry, N, rng, hs) for h in hs])
                    if ws.sum() == 0:
                        rows.append({"kind": kind, "gap": int(g), "ambiguous_geom": True, "degenerate": True}); continue
                    post = ws / ws.sum()
                    row = {"kind": kind, "gap": int(g), "ambiguous_geom": True, "p_true": float(post[0]),
                           "map_correct": bool(post[0] > post[1:].max())}
                    hp = R[f"{kind}_hp"][t]
                    row["consequential_health"] = bool(any(post[i + 1] >= 0.1 and hp[o] != hp[s] for i, o in enumerate(cands)))
                    if kind == "zombies":
                        cd = R["zombies_cd"][t]
                        row["consequential_cooldown"] = bool(any(post[i + 1] >= 0.1 and (cd[o] <= 0) != (cd[s] <= 0)
                                                                 for i, o in enumerate(cands)))
                    row["consequential"] = row["consequential_health"] or row.get("consequential_cooldown", False)
                    rows.append(row)
        print(f"episode {e + 1}/{len(picks)} T={T} rows={len(rows)}", flush=True)
        json.dump({"total_steps": total_steps, "rows": rows}, open(out_path, "w"))
    res = {"total_steps": total_steps}
    for kind in ("zombies", "cows"):
        r = [x for x in rows if x["kind"] == kind]
        amb = [x for x in r if x.get("ambiguous_geom") and "p_true" in x]
        pt = np.array([x["p_true"] for x in amb]) if amb else np.zeros(0)
        res[kind] = {"reentries_le32": len(r), "geom_ambiguous": len(amb),
                     "degenerate": sum(1 for x in r if x.get("degenerate")),
                     "map_error_rate_all_reentries": sum(1 for x in amb if not x["map_correct"]) / max(len(r), 1),
                     "share_p_true_lt_0.9_all_reentries": float((pt < 0.9).sum() / max(len(r), 1)),
                     "p_true_quantiles_ambiguous": np.percentile(pt, [10, 25, 50]).tolist() if len(pt) else None,
                     "per_1000_steps_p_true_lt_0.9": float((pt < 0.9).sum() / total_steps * 1000)}
        cons = sum(1 for x in amb if x.get("consequential"))
        res[kind]["consequential_events"] = cons
        res[kind]["consequential_health"] = sum(1 for x in amb if x.get("consequential_health"))
        res[kind]["consequential_cooldown"] = sum(1 for x in amb if x.get("consequential_cooldown"))
        res[kind]["consequential_per_1000_steps"] = cons / total_steps * 1000
        print(kind, res[kind], flush=True)
    json.dump({"summary": res, "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
