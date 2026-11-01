"""A8 (2026-10-25): which recorded actions were the collector's uniform-random branch, and how long are the runs?
Sampled futures can be scored against recorded ones only where the actions are exogenous (Ortega et al. 2021; the
expert reacted to outcomes and saw symbolic observations). The collector (collect_support.rollout_batch) is a
deterministic scan, so its per-step `take` flags (uniform(ku) < epsilon) and random actions (randint(kn)) can be
regenerated from the saved seed tree (Codex's check: 22/22 matches on two prefixes). Data fact, nothing trained.

CLAIM: the regenerated flags reproduce every stored random action; consecutive random runs long enough for multi-step
  scoring exist mostly in epsilon = 1.0 episodes. CONTROL: on regenerated random steps the stored action must equal the
  regenerated random action (a mismatch means the reconstruction is wrong). OUTCOME: the number of start positions with
  k consecutive exogenous actions (k = 1, 2, 4, 8, 16) by epsilon, i.e. how much data supports k-step prior-sampled
  scoring.
Usage (project venv): eda_exogenous_steps.py OUT_JSON
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

N_ACTIONS = 17
KS = (1, 2, 4, 8, 16)


@jax.jit
def flags(step_keys, eps):
    def one(k):
        ka, kn, ku, sk = jax.random.split(k, 4)
        return jax.random.uniform(ku, (RP.NUM_ENVS,)) < eps, jax.random.randint(kn, (RP.NUM_ENVS,), 0, N_ACTIONS)
    return jax.vmap(one)(step_keys)


def runs(mask):
    """Lengths of consecutive True runs."""
    out, c = [], 0
    for m in mask:
        if m:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def main():
    man = RP.manifest()
    stats = {}
    checked = matched = 0
    for r in range(len(man["shards"])):
        eps_list = RP._shard(r)["episodes"]
        _, step_keys = RP.round_keys(r)
        cache = {}
        for slot, e in enumerate(eps_list):
            if e["split"] != "train":
                continue
            eps = float(e["epsilon"])
            if eps not in cache:
                t_, n_ = flags(step_keys, eps)
                cache[eps] = (np.asarray(t_), np.asarray(n_))
            take, noise = cache[eps]
            acts = e["actions_taken"].numpy(); T = len(acts)
            tk = take[:T, slot]; nz = noise[:T, slot]
            checked += int(tk.sum()); matched += int((acts[tk] == nz[tk]).sum())
            s = stats.setdefault(eps, {"episodes": 0, "steps": 0, "random_steps": 0, "starts_k": {k: 0 for k in KS}, "runs": []})
            s["episodes"] += 1; s["steps"] += T; s["random_steps"] += int(tk.sum())
            rl = runs(tk)
            s["runs"].extend(rl)
            for k in KS:
                s["starts_k"][k] += int(sum(max(0, L - k + 1) for L in rl))
        print(f"round {r + 1}/{len(man['shards'])} checked {checked} matched {matched}", flush=True)
    out = {"random_steps_checked": checked, "random_action_matches": matched}
    for eps, s in sorted(stats.items()):
        rl = np.array(s.pop("runs"))
        s["run_length_median"] = float(np.median(rl)) if len(rl) else 0.0
        s["run_length_p90"] = float(np.percentile(rl, 90)) if len(rl) else 0.0
        out[str(eps)] = s
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
