"""M1 (2026-10-23): does competent behaviour in Craftax-Classic use memory, and how much? Causal intervention on the
memory of the policy that generated our data (the PPO-RNN expert, 512-unit GRU, symbolic observation with the same
7 x 9 field of view as the pixels). Simulator only; nothing is trained.

CLAIM: the expert's behaviour depends on information older than L observations (L = 16, 64).
COMPETING EXPLANATION: its behaviour is determined by the current observation plus a short history; long gaps in the data
  are incidental.
SEPARATING CONTROL: from the SAME simulator state, with the SAME future random keys (common random numbers for the
  environment and for action sampling), run four branches that differ only in the expert's recurrent state at t0:
  A = intact; M64 / M16 = the state rebuilt from only the last 64 / 16 observations (GRU re-run from the episode-start
  state over them); M0 = the episode-start state (no memory). Any difference between branches is caused by the removed
  memory. A within-state control for "general disorientation": water-seeking when water was NEVER seen before t0
  (memory cannot help) vs when it was seen before t0 at a known age.
OUTCOMES AND WHAT THEY RULE OUT:
  - M64 ~ A on survival, achievements and water-seeking => information older than 64 steps is not used by competent
    behaviour here; the long memory scale is not a behavioural requirement (for this expert).
  - M64 worse than A, specifically for water last seen > 64 steps before t0, and not for never-seen water => remembered
    out-of-view resources guide competent behaviour; long memory is a real requirement.
  - M0 / M16 worse than A in both seen and never-seen cases => the state matters, but not as place memory (e.g. phase /
    counter estimates); needs further decomposition.
Limits: one competent policy (a GRU's learned memory use), not optimal behaviour; t0 at three episode times; water is the
  only static resource tested.
Usage: eda_expert_amnesia_v2.py OUT_JSON [N_ENVS] [K]

v2 (2026-10-23, after Codex's review): v1 rebuilt the memory over a buffer that ENDED with o_t0 and then processed
o_t0 again when acting, while the intact state had only processed o_0..o_t0-1. v2 rebuilds over exactly the L
observations o_t0-L .. o_t0-1 (the same ones the intact state saw last), so every branch processes o_t0 once, when
acting. An alignment check is added: rebuilding over the FULL history from o_0 must reproduce the intact state.
Pre-declared equivalence margins (95% paired bootstrap interval of branch - intact entirely inside the margin):
survival over 300 steps +-0.02; achievements gained over 300 steps +-0.15; steps to drink +-10. An interval crossing a
margin is reported as "not established either way".
"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import functools
import jax, jax.numpy as jnp, numpy as np
ROOT = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"
sys.path.insert(0, ROOT + "/d4_mamba_jepa")
from expert.ppo_expert import ScannedRNN, load_expert
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType

PARAMS = ROOT + "/d4_mamba_jepa/artifacts/expert/ppo_expert_v2.msgpack"
LAYER, WATER = 512, BlockType.WATER.value


def main():
    out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 512; K = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False); P = env.default_params
    obs_dim = int(env.observation_space(P).shape[0]); n_act = int(env.action_space(P).n)
    net, params, _ = load_expert(PARAMS, obs_dim=obs_dim, action_dim=n_act, layer_size=LAYER)
    reset_v = jax.vmap(env.reset, in_axes=(0, None)); step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def dead(s):
        return (s.map[s.player_position[0], s.player_position[1]] == BlockType.LAVA.value) | (s.player_health <= 0)

    def water_in_view(s):
        pad = jnp.pad(s.map, 4, constant_values=1)
        v = jax.lax.dynamic_slice(pad, (s.player_position[0] - 3 + 4, s.player_position[1] - 4 + 4), (7, 9))
        return (v == WATER).any()

    dead_v, wiv_v = jax.vmap(dead), jax.vmap(water_in_view)

    def policy(h, obs, key):
        h, logits, _ = net.apply(params, h, (obs[None], jnp.zeros((1, obs.shape[0]), bool)))
        return h, jax.random.categorical(key, logits[0])

    @functools.partial(jax.jit, static_argnums=(1,))
    def phase1(key, t0):
        kr, ks = jax.random.split(key)
        obs, st = reset_v(jax.random.split(kr, N), P)
        h = ScannedRNN.initialize_carry(N, LAYER)
        ring = jnp.zeros((65, N, obs_dim)); last_w = jnp.full((N,), -10 ** 6); alive = jnp.ones(N, bool)

        def body(c, xs):
            h, st, obs, ring, last_w, alive = c
            t, k = xs
            ka, ke = jax.random.split(k)
            ring = jnp.concatenate([ring[1:], obs[None]], 0)          # ring holds the observations the policy consumed
            h, a = policy(h, obs, ka)
            obs2, st2, _, _, _ = step_v(jax.random.split(ke, N), st, a, P)
            alive = alive & ~dead_v(st2)
            last_w = jnp.where(wiv_v(st2), t + 1, last_w)
            return (h, st2, obs2, ring, last_w, alive), None

        (h, st, obs, ring, last_w, alive), _ = jax.lax.scan(body, (h, st, obs, ring, last_w, alive),
                                                            (jnp.arange(t0), jax.random.split(ks, t0)))
        return h, st, obs, ring, last_w, alive

    @jax.jit
    def rewarm(ring_L):                                     # GRU from the episode-start state over the last L observations
        h = ScannedRNN.initialize_carry(N, LAYER)
        h, _, _ = net.apply(params, h, (ring_L, jnp.zeros(ring_L.shape[:2], bool)))
        return h

    @functools.partial(jax.jit, static_argnums=(4,))
    def phase2(h, st, obs, key, K):
        def body(c, k):
            h, st, obs, alive = c
            ka, ke = jax.random.split(k)
            h2, a = policy(h, obs, ka)
            obs2, st2, r, _, _ = step_v(jax.random.split(ke, N), st, a, P)
            alive2 = alive & ~dead_v(st2)
            rec = {"a": a, "alive": alive2, "drink": st2.player_drink, "food": st2.player_food, "health": st2.player_health,
                   "ach": st2.achievements.sum(-1), "r": r * alive, "wiv": wiv_v(st2)}
            return (h2, st2, obs2, alive2), rec
        _, recs = jax.lax.scan(body, (h, st, obs, jnp.ones(N, bool)), jax.random.split(key, K))
        return recs

    res = {"N": N, "K": K, "batches": {}}
    t_start = time.time()
    # alignment check: at t0 = 64 the buffer holds the whole history, so rebuilding must reproduce the intact state
    h64, _, _, ring64, _, _ = phase1(jax.random.PRNGKey(5), 64)
    err = float(jnp.abs(rewarm(ring64[-64:]) - h64).max())
    res["alignment_check_max_abs_diff"] = err
    print(json.dumps({"alignment_check_max_abs_diff": err}), flush=True)
    assert err < 1e-3, "rebuilt state does not reproduce the intact state"
    for b, t0 in enumerate((200, 500, 1000)):
        key = jax.random.PRNGKey(20261031 + b)
        h, st, obs, ring, last_w, alive0 = phase1(key, t0)
        st0 = jax.tree.map(lambda x: x, st)
        # ring[-L:] = o_{t0-L} .. o_{t0-1}: exactly the last L observations the intact state consumed
        branches = {"A": h, "M64": rewarm(ring[-64:]), "M16": rewarm(ring[-16:]), "M0": ScannedRNN.initialize_carry(N, LAYER)}
        k2 = jax.random.PRNGKey(777 + b)
        R = {name: jax.tree.map(np.asarray, phase2(hb, st0, obs, k2, K)) for name, hb in branches.items()}
        alive0 = np.asarray(alive0); last_w = np.asarray(last_w)
        drink0 = np.asarray(st0.player_drink); wiv0 = np.asarray(wiv_v(st0))
        age = t0 - last_w
        thirsty = alive0 & (drink0 <= 4) & ~wiv0
        split = {"seen_le64": thirsty & (age <= 64), "seen_64_256": thirsty & (age > 64) & (age <= 256),
                 "seen_gt256": thirsty & (age > 256) & (age < 10 ** 5), "never_seen": thirsty & (age >= 10 ** 5)}
        out_b = {"t0": t0, "alive_at_t0": int(alive0.sum()), "branches": {}}
        for name, r in R.items():
            alive_end = r["alive"][-1] & alive0
            drank = (np.diff(np.concatenate([drink0[None], r["drink"]], 0), axis=0) > 0)   # [K, N]
            first = np.where(drank.any(0), drank.argmax(0), K)
            agree = (r["a"] == R["A"]["a"]).mean(1)                                       # per step, all envs
            surv_d = (alive_end.astype(float) - (R["A"]["alive"][-1] & alive0).astype(float))[alive0]
            ach_d = (r["ach"][-1] - R["A"]["ach"][-1]).astype(float)[alive0]
            rng_b = np.random.default_rng(1)
            def ci(d):
                bs = [d[rng_b.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            ob = {"survive_K": float(alive_end[alive0].mean()), "survive_diff_vs_A": ci(surv_d), "ach_diff_vs_A": ci(ach_d),
                  "achievements_gained": float((r["ach"][-1] - np.asarray(st0.achievements.sum(-1)))[alive0].mean()),
                  "return_K": float(r["r"].sum(0)[alive0].mean()),
                  "action_agreement_with_A": {f"steps_{a}-{b2}": float(agree[a:b2].mean()) for a, b2 in ((0, 1), (0, 10), (10, 50), (50, K))},
                  "water": {}}
            for sname, m in split.items():
                if m.sum() == 0:
                    ob["water"][sname] = None; continue
                ob["water"][sname] = {"n": int(m.sum()), "mean_steps_to_drink": float(first[m].mean()),
                                      "drank_within_50": float((first[m] < 50).mean()),
                                      "paired_diff_vs_A_steps": None}
            out_b["branches"][name] = ob
            out_b.setdefault("_first", {})[name] = first
        firstA = out_b["_first"]["A"]
        for name in R:
            for sname, m in split.items():
                if m.sum() == 0 or out_b["branches"][name]["water"][sname] is None:
                    continue
                d = (out_b["_first"][name] - firstA)[m].astype(float)
                rng = np.random.default_rng(0)
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                out_b["branches"][name]["water"][sname]["paired_diff_vs_A_steps"] = [float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        del out_b["_first"]
        res["batches"][str(t0)] = out_b
        print(json.dumps({"t0": t0, "alive": out_b["alive_at_t0"], "sec": round(time.time() - t_start)}), flush=True)
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
