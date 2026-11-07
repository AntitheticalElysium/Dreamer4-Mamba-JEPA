"""Exact hidden counters (PLAN.md 11): every episode of an S0 labels file replayed through the simulator's own step
(artifacts/eda/replay.py's reset / step keys and stored actions, as advance_to), the state after each action recorded:
hunger, thirst, fatigue, recover (game_logic.py 1256-1325), with health, food, drink, energy and sleep checked equal to
the labels at every state (any mismatch stops the run). CPU (JAX).
Usage: python -m artifacts.experiments.20261107_f4_followups.counters SPLIT   (dev | train)"""
import functools, os, sys, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eda"))
import replay as RP                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = HERE.parents[1] / "eda/s0"
FIELDS = ("hunger", "thirst", "fatigue", "recover")
BATCH = 32


@functools.cache
def scan():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: jnp.stack([s.player_hunger, s.player_thirst, s.player_fatigue, s.player_recover, s.player_health,
                               s.player_food, s.player_drink, s.player_energy, s.is_sleeping]).astype(jnp.float32)

    def run(reset_key, env_keys, actions):
        _, state = env.reset(reset_key, params)

        def body(s, xs):
            _, nxt, _, _, _ = env.step(xs[0], s, xs[1], params)
            return nxt, rec(nxt)
        _, ys = jax.lax.scan(body, state, (env_keys, actions))
        return jnp.concatenate([rec(state)[None], ys])                  # [bucket + 1, 9]: states 0..bucket
    return jax.jit(jax.vmap(run))


def main(split):
    L = np.load(LABELS / f"s0_{split}_labels.npz")
    T, s_off = L["ep_T"], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = np.zeros((s_off[-1], 4), np.float32)
    buckets = np.array([next(b for b in RP.BUCKETS if b >= max(int(t), 1)) for t in T])
    t0, done = time.time(), 0
    for b in sorted(set(buckets.tolist())):
        idx = np.nonzero(buckets == b)[0]
        for a in range(0, len(idx), BATCH):
            es = idx[a:a + BATCH]
            keys = [RP._slot_keys(int(L["ep_shard"][e]), int(L["ep_slot"][e])) for e in es]
            acts = np.zeros((len(es), b), np.int32)
            for i, e in enumerate(es):
                x = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["actions_taken"].numpy()
                assert len(x) == T[e]
                acts[i, :T[e]] = x
            ys = np.asarray(scan()(jnp.stack([k[0] for k in keys]), jnp.stack([k[1][:b] for k in keys]), jnp.asarray(acts)))
            for i, e in enumerate(es):
                y, lo, hi = ys[i, :T[e] + 1], s_off[e], s_off[e + 1]
                assert np.array_equal(y[:, 4:8].astype(int), L["s_body"][lo:hi]) and np.array_equal(y[:, 8] > 0.5, L["s_sleep"][lo:hi]), \
                    f"replay differs from the labels: episode {e}"
                out[lo:hi] = y[:, :4]
            done += len(es)
            print(f"{split} bucket {b}: {done}/{len(T)} episodes, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(HERE / f"readings/counters_{split}.npz", **{k: out[:, i] for i, k in enumerate(FIELDS)})
    print(split, "all states equal the labels; saved", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
