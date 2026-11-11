"""Native F5 generation on the registered hunger and cooldown forks, with a held-out digit reader."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
G = importlib.import_module("artifacts.experiments.20261111_change_forecast_gate.gate")
IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
DEV = torch.device("cuda")
DRAWS = 32


def hunger_pairs(limit):
    bank = torch.load(G.BANK, map_location="cpu", weights_only=False)
    for i, row in enumerate(bank[:limit]):
        q = row["end"]
        prefix = (row["prefix_frames"], row["prefix_actions"], row["prefix_reward"], row["prefix_term"])
        yield i, row["e"], prefix, row["frames"], row["reward"], row["term"], q, row["food"][:, q:q + 2]


def cooldown_pairs(limit):
    rows = json.loads((HERE / "FORECAST_GATE.json").read_text())["rows"]["C2"]
    registry = {r["e"]: r for r in json.loads((G.P / "CHECKS.json").read_text())["rows"]["C2"]}
    jax, jnp, _, _, roll = G.sim()
    labels = G.dev_labels()
    for i, row in enumerate(rows[:limit]):
        e = row["e"]
        ref = registry[e]
        t = ref["t0"]
        state = D.RP.advance_to(int(labels["ep_shard"][e]), int(labels["ep_slot"][e]), t)
        dist = np.abs(np.asarray(state.zombies.position) - np.asarray(state.player_position)).sum(-1)
        zi, = np.flatnonzero(np.asarray(state.zombies.mask) & (dist == 1))
        keys = jax.random.split(jax.random.PRNGKey(20261111 + 1000 * e + ref["key"]), 9)
        runs = []
        for cooldown in (0, 2):
            zombies = state.zombies.replace(attack_cooldown=state.zombies.attack_cooldown.at[zi].set(cooldown))
            runs.append(roll(state.replace(zombies=zombies), keys, [0] * 9))
        frames, states, rewards, terms = zip(*runs)
        frames, rewards, terms = map(np.stack, (frames, rewards, terms))
        health = np.array([[int(s.player_health) for s in branch] for branch in states])
        assert np.array_equal(frames[0, 3:7], frames[1, 3:7]) and not np.array_equal(frames[0, 1], frames[1, 1])
        assert (health[:, 7] - health[:, 6]).tolist() == [-2, 0]
        assert rewards[:, 6].tolist() == row["rewards"] and not terms.any()
        recorded = G.recorded(labels, e)
        assert np.array_equal(frames[0, 0], recorded["observations"][t].numpy())
        prefix = tuple(recorded[key][:t].numpy() for key in ("observations", "actions_taken", "rewards", "terminated"))
        yield i, e, prefix, frames, rewards, terms, 6, health[:, 6:8]


@torch.no_grad()
def readers(model, cfg):
    labels = dict(np.load(S.LABELS / "s0_train400_labels.npz"))
    offsets = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
    data, target, held = [], [], []
    for i, episode in enumerate(S.episodes(labels)[:100]):
        frames = D.RP.episode_fields(episode.shard, episode.slot)["observations"].numpy()
        ids = np.arange(0, len(frames), 2)
        for batch in np.array_split(ids, max(1, (len(ids) + 511) // 512)):
            with TR.autocast(DEV, cfg):
                p = model.perception(torch.as_tensor(frames[batch], device=DEV),
                                     torch.full((len(batch), 2), 4, device=DEV, dtype=torch.long))
            v = p.v[:, None].expand(-1, 2, -1)
            data.append(torch.cat([p.p[:, 63:65], v], -1).float())
            target.append(labels["s_body"][offsets[i] + batch, :2])
            held.extend([i % 5 == 4] * len(batch))
    data = torch.cat(data)
    target = torch.as_tensor(np.concatenate(target), device=DEV).long()
    valid = torch.as_tensor(held, device=DEV)
    result = {}
    for j in range(2):
        x, y = data[:, j], target[:, j]
        xt, xv, yt, yv = x[~valid], x[valid], y[~valid], y[valid]
        with torch.enable_grad():
            reader, _ = S.fit(lambda ids, val: (xv if val else xt)[ids],
                              lambda out, ids, val: cross_entropy(out["y"], (yv if val else yt)[ids]),
                              len(yt), len(yv), {"y": 10}, 110812)
        pred = torch.cat([reader(z)["y"].argmax(-1) for z in xv.split(8192)])
        accuracy = float((pred == yv).float().mean())
        assert accuracy >= .99, (j, accuracy)
        result[63 + j] = (reader.eval(), accuracy)
    return result


@torch.no_grad()
def generate(model, cfg, row, slot, reader):
    i, e, prefix, frames, rewards, terms, q, truth = row
    old_f, old_a, old_r, old_d = prefix
    n = len(old_f)
    f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
    a = np.r_[old_a, np.zeros(q + 1, dtype=np.int64)]
    rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
    td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
    fix = D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))
    query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=DEV)
    fix = torch.as_tensor(np.repeat(fix[None], 2, 0), device=DEV)
    act = torch.as_tensor(np.repeat(a[None], 2, 0), device=DEV)
    prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
    with TR.autocast(DEV, cfg):
        p = model.perception(torch.as_tensor(f, device=DEV), fix)
        inputs = model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=DEV, dtype=torch.float32),
                                     torch.as_tensor(td, device=DEV), act)
        y, _ = model.belief(inputs, model.belief.initial(2, DEV))
        bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
        common = 8 if slot == 64 else 4
        assert all(torch.equal(z[0, -common:], z[1, -common:]) for z in p.parts())
        gen = []
        for br in range(2):
            rng = torch.Generator(device=DEV).manual_seed(14000 + i)
            cache = model.denoiser.new_cache()
            for t in range(n + q - 7, n + q + 1):
                cache = M.context_pass(model.denoiser, tuple(z[br:br+1, t:t+1] for z in p.parts()),
                                       prev[br:br+1, t:t+1], model.belief.memory(bp[br:br+1, t:t+1]),
                                       fix[br:br+1, t:t+1], cache, cfg, rng)
            cache = {"t": cache["t"], "kv": [tuple(z.expand(DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
            z = M.sample(model.denoiser, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DRAWS, 1),
                         model.belief.memory(y[br:br+1, -1:]).expand(DRAWS, -1, -1, -1),
                         query[None, None].expand(DRAWS, 1, -1), cache, cfg,
                         torch.Generator(device=DEV).manual_seed(16000 + i))
            gen.append(torch.cat([z[0][:, 0, slot], z[2][:, 0]], 1).float())
        real = model.perception(torch.as_tensor(frames[:, q:q + 2], device=DEV), query[None, None].expand(2, 2, -1))
        rp = torch.cat([real.p[:, :, slot], real.v], -1).float()
    assert np.array_equal(reader(rp)["y"].argmax(-1).cpu().numpy(), truth)
    digits = [reader(z)["y"].argmax(-1).cpu().numpy() for z in gen]
    drops = [int((z < truth[br, 0]).sum()) for br, z in enumerate(digits)]
    return {"episode": e, "true": truth.tolist(), "drops": drops,
            "generated_digits": [np.bincount(z, minlength=10).tolist() for z in digits]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    model, cfg = IF.load(args.checkpoint.resolve())
    fit = readers(model, cfg)
    report = {"sha256": TR.file_sha256(args.checkpoint), "draws": DRAWS,
              "reader_accuracy": {k: v[1] for k, v in fit.items()}}
    limit = 2 if args.smoke else None
    for name, pairs, slot in (("hunger", hunger_pairs(limit), 64), ("cooldown", cooldown_pairs(limit), 63)):
        rows = []
        for row in pairs:
            rows.append(generate(model, cfg, row, slot, fit[slot][0]))
            print(name, rows[-1], flush=True)
        assert len(rows) == (2 if args.smoke else 24 if name == "hunger" else 32)
        rate = np.array([r["drops"] for r in rows], dtype=float) / DRAWS
        rng = np.random.default_rng(1111)
        diff = rate[:, 0] - rate[:, 1]
        ci = np.quantile(diff[rng.integers(len(diff), size=(5000, len(diff)))].mean(1), [.025, .975])
        report[name] = {"pairs": rows, "rates": rate.mean(0).tolist(),
                        "paired_scene_contrast_ci95": ci.tolist()}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
