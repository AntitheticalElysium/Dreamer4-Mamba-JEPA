"""S35's design, applied to nonterminal damage instead of death.

S35 held (state, action) fixed and varied only the simulator RNG, and found death
varies in 0.94% of pairs. Mob damage may not be that deterministic, so the same
question is asked of the damage label the forward classifier is trained on.

For every hazard root, all 17 actions are executed under the same 64 RNG draws --
matched across actions, so within-state comparisons are paired. The last statistic
is the one that bounds the forward task: the within-state AUC obtainable by a
predictor that knows the exact simulator state, and therefore the exact
p(damage | s, a), but not which RNG draw will be realised.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay

DRAWS = 64
BASE = jax.random.PRNGKey(90210)


def main() -> None:
    records = []
    for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
        records += torch.load(path, weights_only=False)
    usable = []
    for record in records:
        health, dead = record["health"].numpy(), record["dead"].numpy()
        positive = (health <= -1) | dead
        if positive.any() and ((health >= 0) & ~dead).any():
            usable.append(record)
    print(f"{len(usable)} hazard roots of {len(records)} collected", flush=True)

    env, params, _, _, _ = replay.env_and_render()
    keys = jax.vmap(lambda j: jax.random.fold_in(BASE, j))(jnp.arange(DRAWS))
    actions = jnp.repeat(jnp.arange(17), DRAWS)
    all_keys = jnp.tile(keys, (17, 1))

    def step_many(state, keys_, actions_):
        def one(key, action):
            _, nxt, _, _, _ = env.step(key, state, action, params)
            from craftax.craftax_classic.constants import BlockType

            lava = nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value
            dead = lava | (nxt.player_health <= 0)
            return nxt.player_health, dead

        return jax.vmap(one)(keys_, actions_)

    step_many = jax.jit(step_many)

    rows = []
    started = time.time()
    for n, record in enumerate(usable):
        state = replay.advance_to(record["shard"], record["slot"], record["t"])
        base_health = float(state.player_health)
        health, dead = step_many(state, all_keys, actions)
        health = np.asarray(health).reshape(17, DRAWS)
        dead = np.asarray(dead).reshape(17, DRAWS)
        damaged = (health - base_health <= -1) | dead
        rows.append(dict(
            shard=record["shard"], slot=record["slot"], t=record["t"],
            band=record["band"], epsilon=record["epsilon"],
            p_damage=damaged.mean(1), first=damaged[:, 0],
            reference=((record["health"].numpy() <= -1) | record["dead"].numpy()),
            base_health=base_health,
        ))
        if (n + 1) % 100 == 0:
            rate = (n + 1) / (time.time() - started)
            print(f"  {n+1}/{len(usable)} [{time.time()-started:.0f}s, "
                  f"{(len(usable)-n-1)/rate:.0f}s left]", flush=True)

    p = np.stack([r["p_damage"] for r in rows])          # (roots, 17)
    first = np.stack([r["first"] for r in rows])
    reference = np.stack([r["reference"] for r in rows])
    varies = (p > 0) & (p < 1)
    print()
    print("=" * 110)
    print(f"S35-FOR-DAMAGE  --  {len(rows)} hazard roots x 17 actions x {DRAWS} RNG draws")
    print("=" * 110)
    print(f"  fixed (state, action) pairs:                    {p.size:,}")
    print(f"  damage outcome varies with RNG:                 {varies.mean():.2%} "
          f"({int(varies.sum()):,} pairs)")
    print(f"  deterministically safe   (p = 0):               {(p == 0).mean():.2%}")
    print(f"  deterministically damaging (p = 1):             {(p == 1).mean():.2%}")
    print(f"  near-deterministic (p < 0.05 or p > 0.95):      {((p < 0.05) | (p > 0.95)).mean():.2%}")
    print()
    print("  distribution of p(damage | s, a) over the pairs that vary at all")
    inside = p[varies]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"    p{int(q*100):<3} {np.quantile(inside, q):.3f}")
    print(f"    mean {inside.mean():.3f}")
    print()
    agree = (reference == first).mean()
    print(f"  the collection's single reference draw agrees with draw 0 here: {agree:.2%}")
    print(f"  reference-damaging pairs are damaging on {p[reference].mean():.1%} of draws")
    print(f"  reference-safe pairs are damaging on      {p[~reference].mean():.1%} of draws")

    def within_auc(scores, labels):
        out = []
        for i in range(len(scores)):
            y, s = labels[i].astype(float), scores[i]
            if not (y > 0).any() or not (y <= 0).any():
                continue
            order = np.argsort(s)
            ranks = np.empty(len(s), float)
            ranks[order] = np.arange(1, len(s) + 1)
            unique, inverse = np.unique(s, return_inverse=True)
            for v in range(len(unique)):
                tie = inverse == v
                if tie.sum() > 1:
                    ranks[tie] = ranks[tie].mean()
            pos = y > 0
            out.append((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                       / (pos.sum() * (~pos).sum()))
        return np.array(out)

    # oracle score from draws 1..63; label from the held-out draw 0
    held = np.stack([(np.asarray(r["p_damage"]) * DRAWS - r["first"]) / (DRAWS - 1)
                     for r in rows])
    ceiling = within_auc(held, first)
    boot = np.array([ceiling[np.random.default_rng(s).integers(0, len(ceiling),
                     len(ceiling))].mean() for s in range(2000)])
    print()
    print("  CEILING for any predictor that knows the exact simulator state but not the RNG")
    print(f"    within-state AUC of p(damage|s,a) against a held-out draw: "
          f"{ceiling.mean():.4f} [{np.quantile(boot,0.025):.4f}, {np.quantile(boot,0.975):.4f}]"
          f"  over {len(ceiling)} roots")
    same = within_auc(held, reference)
    print(f"    against the collection's own reference labels:             {same.mean():.4f}")
    np.savez(HERE / "s35_damage.npz", p=p, first=first, reference=reference,
             band=np.array([r["band"] for r in rows]),
             epsilon=np.array([r["epsilon"] for r in rows]))
    print("\nwrote s35_damage.npz")


if __name__ == "__main__":
    main()
