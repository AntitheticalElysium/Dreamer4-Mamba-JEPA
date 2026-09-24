"""Repeated-key replay pilot: given the FULL simulator state, how random is within-root death?

Every fork root was stepped once, under one key shared by all 17 actions
(`collect_broad_forks.py:176`), so each stored outcome is one draw. Craftax draws zombie movement
and mob spawns from that key. A probe scored against a single draw cannot tell a model that failed
to learn a predictable consequence from one facing an outcome no function of (state, action) can
predict. This pilot measures that directly.

It re-walks a sample of the judgement seeds with the collector's own frozen BC policy, loop and
keys, so each trajectory is the one that produced the stored rows, and PROVES it before measuring:
at every stored root the replayed 32-frame history must match the stored frames, and the original
key must reproduce the stored one-step deaths and stored NOOP-second-step deaths exactly. Any
mismatch aborts the run.

Then, at every stored root, it steps all 17 actions under K independent keys, each followed by the
same NOOP second step under an independent key, giving P(death1 | s, a) and P(death2 | s, a) with
the full state known -- hidden mob cooldowns included. That is the simulator-randomness ceiling for
any predictor. It is not the pixel ceiling: a predictor that sees only frames also lacks the hidden
state, which this cannot measure.

Reported on the realized (original-key) outcomes of roots offering both a fatal and a safe action:
the full-state oracle's choice argmin_a P(death | s, a) against the FIT-root action prior and
always-RIGHT, paired and seed-clustered; its expected safe rate; how many (root, action) outcomes
are genuinely random (0 < P < 1); and NOOP / RIGHT / SLEEP probabilities for the delayed hazard.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
EDA = ROOT / "artifacts/eda"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EDA))

from d4mj.config import Config
from d4mj.data import _sha256, patchify
from d4mj.env import _env, reset, step as env_step
from d4mj.lewm_diagnostics import FORK_STORE
from d4mj.transition import observe

from collect_broad_forks import HISTORY, N_ACTIONS, NOOP, frames_of, load_policy, make_fork  # noqa: E402
from confirm import seeds_for  # noqa: E402
from ladder import paired  # noqa: E402

RIGHT, SLEEP = 2, 6


def make_second():
    """The collector's second step -- NOOP from each first-step successor -- vmapped over the 17."""
    env, params = _env()
    from craftax.craftax_classic.constants import BlockType

    def one(state, key):
        _, nxt, _, _, _ = env.step(key, state, NOOP, params)
        lava = nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value
        return lava | (nxt.player_health <= 0)

    return jax.jit(jax.vmap(one, in_axes=(0, None)))


def outcome(fork, second, env_state, key1, key2):
    """death1 and death2 for all 17 actions under one pair of keys."""
    _, state1, _, dead1, trunc1, _, _ = fork(env_state, key1, jnp.arange(N_ACTIONS))
    dead1, trunc1 = np.asarray(dead1), np.asarray(trunc1)
    dead2 = np.asarray(second(state1, key2))
    alive = ~dead1 & ~trunc1
    return dead1, dead1 | (alive & dead2), alive, dead2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--keys", type=int, default=32)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="replay_pilot")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)

    partition = json.loads(args.partition.read_text())
    fit_seeds, judge_seeds = seeds_for(partition, FORK_STORE)
    pick = torch.randperm(len(judge_seeds), generator=torch.Generator().manual_seed(args.seed))
    pilot = sorted(judge_seeds[i] for i in pick[:args.seeds].tolist())
    paths = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}

    # Exactly the collector's configuration: the encoder under the 32-slot base config it was
    # written with, everything else under direct/attention (`collect_broad_forks.py:143-151`).
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    encoder, world, heads = load_policy(base)
    fork, second = make_fork(), make_second()

    records, checks = [], {"roots": 0, "frames_max_drift": 0, "death1_mismatch": 0,
                           "second_mismatch": 0}
    with torch.no_grad():
        for number, seed in enumerate(pilot):
            stored = {int(r["step"]): r for r in torch.load(paths[seed], weights_only=False)}
            last = max(stored)
            observation, env_state = reset(seed)
            state = None
            incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
            world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
            policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
            frames = []
            for index in range(last + 1):
                frames.append(observation.clone())
                patches = patchify(observation[None, None], config.patch).to(config.device)
                state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
                logits = heads(agent)["policy"][:, -1, 0]
                chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
                if index in stored:
                    row = stored[index]
                    drift = int((torch.stack(frames[-HISTORY:]).int() - row["frames"].int()).abs().max())
                    checks["frames_max_drift"] = max(checks["frames_max_drift"], drift)
                    d1, d2, alive, raw2 = outcome(fork, second, env_state,
                                                  jax.random.PRNGKey(seed + index + 1),
                                                  jax.random.PRNGKey(seed + index + 2))
                    checks["death1_mismatch"] += int((d1 != row["terminated"].numpy()).sum())
                    stored_alive = row["second_valid"].numpy()
                    checks["second_mismatch"] += int(
                        (raw2[stored_alive] != row["second_terminated"].numpy()[stored_alive]).sum()
                        + (alive != stored_alive).sum())
                    checks["roots"] += 1
                    if drift > 1 or checks["death1_mismatch"] or checks["second_mismatch"]:
                        raise SystemExit(f"replay does not reproduce the stored root {seed}:{index}: "
                                         f"{json.dumps(checks)}")
                    keys = jax.random.split(jax.random.PRNGKey(2**30 + seed * 1000 + index),
                                            2 * args.keys)
                    one, two = np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)
                    for k in range(args.keys):
                        a, b, _, _ = outcome(fork, second, env_state, keys[k], keys[args.keys + k])
                        one += a
                        two += b
                    records.append({"seed": seed, "step": index,
                                    "p_death1": one / args.keys, "p_death2": two / args.keys,
                                    "death1": d1, "death2": d2})
                observation, env_state, _, terminated, truncated = env_step(
                    env_state, chosen, seed + index + 1)
                incoming.fill_(chosen)
                if terminated or truncated:
                    break
            log(stage="seed", done=number + 1, of=len(pilot), roots=len(records), checks=checks)

    # ---- the FIT-root action prior, from the stored fit rows, for both outcomes ----
    # Labels only: keeping whole fit rows would hold ~0.8 MB of frames each, ~5.7 GB in all.
    labels = {"terminated": [], "second_valid": [], "second_terminated": []}
    for seed in sorted(fit_seeds):
        for row in torch.load(paths[seed], weights_only=False):
            if len(row["frames"]) >= 4:
                for key in labels:
                    labels[key].append(row[key].bool())
    first, valid, sec = (torch.stack(labels[k]) for k in labels)
    fit_truth = {"death1": first, "death2": first | (~first & valid & sec)}

    P = {o: torch.tensor(np.stack([r[f"p_{o}"] for r in records])) for o in ("death1", "death2")}
    Y = {o: torch.tensor(np.stack([r[o] for r in records])).bool() for o in ("death1", "death2")}
    seeds = torch.tensor([r["seed"] for r in records])
    summary = {}
    for o in ("death1", "death2"):
        ft = fit_truth[o]
        usable_fit = ft.any(1) & (~ft).any(1)
        prior = int((~ft[usable_fit]).float().mean(0).argmax())
        y, p = Y[o], P[o]
        usable = y.any(1) & (~y).any(1)
        choice = p.argmin(1)
        realized = (~y).gather(1, choice[:, None]).squeeze(1).float()
        prior_ok = (~y[:, prior]).float()
        right_ok = (~y[:, RIGHT]).float()
        random_pairs = ((p > 0) & (p < 1)).float()
        summary[o] = {
            "roots": int(len(y)), "opportunity_roots": int(usable.sum()),
            "fit_prior_action": prior,
            "oracle_realized_safe": float(realized[usable].mean()),
            "oracle_expected_safe": float((1 - p.gather(1, choice[:, None]).squeeze(1))[usable].mean()),
            "prior_realized_safe": float(prior_ok[usable].mean()),
            "always_right_realized_safe": float(right_ok[usable].mean()),
            "oracle_vs_prior": paired(realized[usable], prior_ok[usable], seeds[usable],
                                      draws=args.draws, seed=args.seed + 11),
            "oracle_vs_always_right": paired(realized[usable], right_ok[usable], seeds[usable],
                                             draws=args.draws, seed=args.seed + 13),
            "random_share_all_pairs": float(random_pairs.mean()),
            "random_share_opportunity_pairs": float(random_pairs[usable].mean()),
            "opportunity_roots_with_any_random_action": int((random_pairs[usable].sum(1) > 0).sum()),
            "mean_p": {"NOOP": float(p[:, NOOP].mean()), "RIGHT": float(p[:, RIGHT].mean()),
                       "SLEEP": float(p[:, SLEEP].mean())},
            "oracle_choice_histogram": np.bincount(choice[usable].numpy(), minlength=N_ACTIONS).tolist()}
        log(stage="summary", outcome=o, **{k: v for k, v in summary[o].items()
                                            if not isinstance(v, (dict, list))})

    rows_path = args.out / f"{args.name}_rows.pt"
    torch.save({"records": records}, rows_path)
    report = {"schema": "d4mj_replay_pilot_v1",
              "status": "PILOT, exploratory: judgement seeds already examined post hoc",
              "script_sha256": _sha256(Path(__file__)),
              "collector_sha256": _sha256(EDA / "collect_broad_forks.py"),
              "partition_sha256": _sha256(args.partition),
              "pilot_seeds": pilot, "keys_per_root": args.keys,
              "row_identity": hashlib.sha256(repr([(r["seed"], r["step"]) for r in records])
                                             .encode()).hexdigest(),
              "reproduction_checks": checks,
              "rows": {"path": rows_path.name, "sha256": _sha256(rows_path)},
              "summary": summary}
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    log(status="replay_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
