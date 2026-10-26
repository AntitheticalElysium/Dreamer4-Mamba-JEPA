"""A1 (2026-10-24): which grounded learning signals pay a representation to keep which content? Decision A (the learning
signal) needs to know, for each candidate signal Y available offline without simulator labels, how much a predictor of Y
gains from each kind of content X that the world must keep (zombies by distance, cows, skeletons, body, faced cell,
terrain, inventory, light), beyond everything else in the state. Simulator rollouts of the data-generating collector;
simulator state is used to DEFINE content, never as a training signal; nothing is trained except the estimators.

QUANTITY: the incentive of Y for X = CE(Y | C) - CE(Y | C, X) in bits per step, where C is the rest of the state (and,
  per variant, the action history or the future action sequence). Under a proper scoring loss this is the conditional
  mutual information I(Y; X | C): the loss reduction an ideal predictor of Y obtains by keeping X. A learning signal can
  only make dropping X costly through this term.
CLAIM UNDER TEST: some dense grounded signal (expert next action, multi-horizon reward / termination / HUD change, inverse
  dynamics) carries a non-negligible incentive for the content the periphery must keep (zombies 2-4 and 5+ cells away,
  cows, skeletons), not only for content near the player.
COMPETING EXPLANATIONS AND CONTROLS:
  - estimator artefact (overfitting, capacity): train / early-stop / test split by episode; permutation control (X
    shuffled across steps within the training data) gives the estimator's floor; episode-clustered bootstrap intervals.
  - confounding (zombies come with night, low health ...): incentives are conditional on all other groups, so light,
    body and terrain are in C.
  - copycat (Wen et al. 2020): with action history in C, the expert-action incentive is measured with and without it.
OUTCOMES AND WHAT THEY RULE OUT (necessary condition only; Codex's caution, agreed): an incentive near the permutation
  floor rules the signal OUT as a protector of that content (a loss whose optimum does not depend on X cannot make
  dropping X costly). A clear incentive does NOT show joint training will keep X (optimisation, capacity, competing
  self-prediction pressure); it only leaves the signal in. Content with no incentive under any signal has no grounded
  teacher among these signals: it is either decision-irrelevant for them (value equivalence) or needs another source.
Limits: one expert (PPO-RNN on symbolic input), the collector's epsilon mixture, Craftax-Classic only; a nonparametric
  estimator (LightGBM) bounds CE from above, so incentives can be underestimated for content it cannot exploit.

Usage (project venv, CPU JAX): eda_signal_incentives.py OUT_NPZ N_EXPERT N_PER_EPS
  Expert-only episodes (eps 0, limit 2500, the expert archive's recipe) and the support-v2 recipe (eps 0.1 / 0.25 / 0.5 /
  1.0, limit 1500): action = uniform with prob eps, else a sample from the expert's logits; the GRU sees every
  observation (collect_support.rollout_batch). Fresh seeds; the stored corpus is not touched.
"""
import functools, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
ROOT = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"
sys.path.insert(0, ROOT + "/d4_mamba_jepa")
from expert.ppo_expert import ScannedRNN, load_expert
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType

PARAMS = ROOT + "/d4_mamba_jepa/artifacts/expert/ppo_expert_v2.msgpack"
LAYER, CHUNK = 512, 250
MOBS = ("zombies", "cows", "skeletons", "arrows")


def recorder(env, P, net, params, n, eps, limit, seed):
    reset_v = jax.vmap(env.reset, in_axes=(0, None)); step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    n_act = int(env.action_space(P).n)

    def snap(s):
        pad = jnp.pad(s.map, 4, constant_values=BlockType.OUT_OF_BOUNDS.value)
        view = jax.lax.dynamic_slice(pad, (s.player_position[0] + 1, s.player_position[1]), (7, 9)).astype(jnp.int8)
        inv = jnp.stack([getattr(s.inventory, f) for f in s.inventory.__dataclass_fields__]).astype(jnp.int8)
        rec = {"view": view, "inv": inv, "pos": s.player_position.astype(jnp.int16), "dir": s.player_direction.astype(jnp.int8),
               "body": jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]).astype(jnp.int8),
               "sleep": s.is_sleeping, "light": s.light_level.astype(jnp.float16), "t": s.timestep.astype(jnp.int16),
               "ach": s.achievements.sum().astype(jnp.int8)}
        for m in MOBS:
            o = getattr(s, m)
            rec[m + "_rel"] = (o.position - s.player_position).astype(jnp.int16)
            rec[m + "_mask"] = o.mask
            rec[m + "_cd"] = o.attack_cooldown.astype(jnp.int8)
        return rec

    snap_v = jax.vmap(snap)

    def dead(s):
        return (s.map[s.player_position[0], s.player_position[1]] == BlockType.LAVA.value) | (s.player_health <= 0)

    @jax.jit
    def chunk(carry, keys):
        def body(c, k):
            h, st, obs, done_prev, finished = c
            ka, kn, ku, sk = jax.random.split(k, 4)
            h, logits, _ = net.apply(params, h, (obs[None], done_prev[None]))
            sample = jax.random.categorical(ka, logits[0])
            noise = jax.random.randint(kn, (n,), 0, n_act)
            take = jax.random.uniform(ku, (n,)) < eps
            a = jnp.where(take, noise, sample)
            obs2, st2, r, done, _ = step_v(jax.random.split(sk, n), st, a, P)
            rec = snap_v(st) | {"logits": logits[0].astype(jnp.float16), "a": a.astype(jnp.int8), "noise": take,
                                "r": r.astype(jnp.float16), "done": done, "dead": jax.vmap(dead)(st2), "live": ~finished}
            return (h, st2, obs2, done, finished | done), rec
        return jax.lax.scan(body, carry, keys)

    key = jax.random.PRNGKey(seed)
    kr, ks = jax.random.split(key)
    obs, st = reset_v(jax.random.split(kr, n), P)
    carry = (ScannedRNN.initialize_carry(n, LAYER), st, obs, jnp.zeros(n, bool), jnp.zeros(n, bool))
    keys = jax.random.split(ks, limit)
    out = []
    for i in range(0, limit, CHUNK):
        carry, rec = chunk(carry, keys[i:i + CHUNK])
        out.append(jax.tree.map(np.asarray, rec))
        if bool(carry[4].all()):
            break
    return {k: np.concatenate([o[k] for o in out], 0) for k in out[0]}     # [T, n, ...]


def main():
    out, n_exp, n_eps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False); P = env.default_params
    net, params, _ = load_expert(PARAMS, obs_dim=int(env.observation_space(P).shape[0]),
                                 action_dim=int(env.action_space(P).n), layer_size=LAYER)
    groups = {}
    for gi, (eps, n, limit) in enumerate([(0.0, n_exp, 2500)] + [(e, n_eps, 1500) for e in (0.1, 0.25, 0.5, 1.0)]):
        t0 = time.time()
        R = recorder(env, P, net, params, n, eps, limit, seed=20261024 + gi)
        groups[eps] = R
        live = R["live"]
        print(f"eps {eps}: {n} envs, scanned {live.shape[0]} steps, live steps {int(live.sum())}, "
              f"deaths {int((R['dead'] & R['live']).sum())}, {time.time() - t0:.0f}s", flush=True)
    flat = {}
    for eps, R in groups.items():
        T, n = R["live"].shape
        ep_id = np.broadcast_to(np.arange(n)[None], (T, n))
        sel = R["live"].T.reshape(-1)                                      # env-major order: episode contiguous
        for k, v in R.items():
            v = np.moveaxis(v, 0, 1).reshape((T * n,) + v.shape[2:])[sel]
            flat.setdefault(k, []).append(v)
        flat.setdefault("episode", []).append(ep_id.T.reshape(-1)[sel] + int(1000 * (eps * 100 + 1)))
        flat.setdefault("eps", []).append(np.full(int(sel.sum()), eps, np.float16))
    np.savez_compressed(out, **{k: np.concatenate(v, 0) for k, v in flat.items()})
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
