"""F5 checkpoint trajectory (RESULTS.md, trajectory). Frozen readouts at each of the ten checkpoints, no training.

On the same exact-history pairs at every checkpoint (recovery, cooldown, energy from dissect.py and native_pairs.py;
hunger from native_pairs.py):
  heads   the forecast heads on the due transition, from the post-action belief alone. Reward forecast: log p of each
          member's actual reward bin under each member's belief; separation = lp[A, rA] + lp[B, rB] - lp[B, rA] -
          lp[A, rB] (log-odds, 0 = the branches are not told apart). TSP forecast at a 3x3 window containing the digit's
          cell: kappa = (f_A - f_B).(y_A - y_B) / |y_A - y_B|^2 at that cell. This is the fraction of the true due vs
          not-due pixel difference reproduced, where 0 means no timing; copying the last frame gives 0, since the
          recent frames are identical.
  native  generation, due vs not-due event rates, as dissect.generate (32 draws, full cache, unchanged sampler).
  probe   20261107_f4_followups/belief_probe.py (event APs from bp) per checkpoint.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.trajectory [checkpoint numbers...]
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.trajectory controls   (energy pairs on F4/HUD)
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection/trajectory"
DS = importlib.import_module("artifacts.experiments.20261111_p1f5.dissect")
NP = DS.NP
BP = importlib.import_module("artifacts.experiments.20261107_f4_followups.belief_probe")
DEV = DS.DEV
SLOTS = (63, 64, 66)
WINDOW = {63: ((7, 1), 3), 64: ((7, 1), 4), 66: ((7, 2), 5)}        # cell -> (window centre, position in the window)
PAIRS = {"recovery": (63, "rise"), "cooldown": (63, "drop"), "energy": (66, "drop"), "hunger": (64, "drop")}


def pair_rows(name):
    if name == "cooldown":
        return list(NP.cooldown_pairs(None))
    if name == "hunger":
        return list(NP.hunger_pairs(None))
    return list(DS.clock_pairs(name))


@torch.no_grad()
def readers(model, cfg, slots=SLOTS):
    """native_pairs.readers for any HUD slots (same episodes, held-out fifth, seed; 63/64 match it exactly)."""
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
            v = p.v[:, None].expand(-1, len(slots), -1)
            data.append(torch.cat([p.p[:, list(slots)], v], -1).float())
            target.append(labels["s_body"][offsets[i] + batch][:, [s - 63 for s in slots]])
            held.extend([i % 5 == 4] * len(batch))
    data = torch.cat(data)
    target = torch.as_tensor(np.concatenate(target), device=DEV).long()
    valid = torch.as_tensor(held, device=DEV)
    out = {}
    for j, slot in enumerate(slots):
        x, y = data[:, j], target[:, j]
        xt, xv, yt, yv = x[~valid], x[valid], y[~valid], y[valid]
        with torch.enable_grad():
            reader, _ = S.fit(lambda ids, val: (xv if val else xt)[ids],
                              lambda o, ids, val: cross_entropy(o["y"], (yv if val else yt)[ids]),
                              len(yt), len(yv), {"y": 10}, 110812)
        pred = torch.cat([reader(z)["y"].argmax(-1) for z in xv.split(8192)])
        accuracy = float((pred == yv).float().mean())
        assert accuracy >= .99, (slot, accuracy)
        out[slot] = (reader.eval(), accuracy)
    return out


@torch.no_grad()
def heads(model, cfg, row, slot):
    """Forecast heads on the due transition from the post-action belief (generate's exact inputs)."""
    i, e, prefix, frames, rewards, terms, q, truth = row
    old_f, old_a, old_r, old_d = prefix
    f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
    a = np.r_[old_a, np.zeros(q + 1, dtype=np.int64)]
    rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
    td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
    fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=DEV)
    act = torch.as_tensor(np.repeat(a[None], 2, 0), device=DEV)
    with TR.autocast(DEV, cfg):
        p = model.perception(torch.as_tensor(f, device=DEV), fix)
        inputs = model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=DEV, dtype=torch.float32),
                                     torch.as_tensor(td, device=DEV), act)
        y, _ = model.belief(inputs, model.belief.initial(2, DEV))
        b = y[:, -1:]                                                          # the generating (post-action) belief
        out = {}
        if getattr(model.outcome, "mask", None) is not None:
            h = model.outcome.features(b, None)
            r = torch.as_tensor(rewards[:, q], device=DEV, dtype=torch.float32)
            lp = np.array([[float(model.outcome.reward_logp(h[m:m + 1], r[k].view(1, 1))) for k in (0, 1)] for m in (0, 1)])
            out["reward_logp"] = lp.tolist()
            out["reward_separation"] = float(lp[0, 0] + lp[1, 1] - lp[1, 0] - lp[0, 1])
        if getattr(model.tsp, "mask", None) is not None:
            centre, pos = WINDOW[slot]
            fx = torch.tensor(centre, device=DEV).expand(2, 1, 2)
            fc = model.pixels.mean(model.tsp.predict(b, None, None, M.window(fx))).float()[:, 0, pos]
            tgt = M.cell_pixels(M.crop(torch.as_tensor(frames[:, q + 1:q + 2], device=DEV), fx)).float()[:, 0, pos] / 127.5 - 1
            dy = tgt[0] - tgt[1]
            out["tsp_kappa"] = float(((fc[0] - fc[1]) * dy).sum() / dy.square().sum().clamp(min=1e-9))
            out["tsp_err_due_notdue"] = [float((fc[m] - tgt[m]).square().mean()) for m in (0, 1)]
    return out


def checkpoint(k):
    ck = HERE / f"run/ckpt_{k:02d}.pt"
    path = OUT / f"ckpt_{k:02d}.json"
    if path.exists():
        return
    model, cfg = NP.IF.load(ck.resolve())
    fit = readers(model, cfg)
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "reader_accuracy": {s: v[1] for s, v in fit.items()}}
    for name, (slot, event) in PAIRS.items():
        rows = pair_rows(name)
        hs = [heads(model, cfg, r, slot) for r in rows]
        gen = [DS.generate(model, cfg, r, slot, fit[slot][0], [(0, 0), (1, 1)], event) for r in rows]
        summ = DS.summarise(gen, ("00", "11"))
        res[name] = {"native": summ, "pairs": len(rows),
                     "heads": {key: float(np.mean([h[key] for h in hs])) for key in ("reward_separation", "tsp_kappa") if key in hs[0]},
                     "heads_per_pair": hs, "native_per_pair": [g["routes"] for g in gen]}
        print(k, name, json.dumps(summ), json.dumps(res[name]["heads"]), flush=True)
    del model
    torch.cuda.empty_cache()
    probe_path = OUT / f"belief_probe_{k:02d}.json"
    if not probe_path.exists():
        BP.main(str(ck.resolve()), str(probe_path))
    res["belief_probe"] = {t: v["ap"]["belief_bp"]["ap"] for t, v in json.loads(probe_path.read_text())["targets"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")


def controls():
    """Energy pairs (and hunger for the reader check) on the F4 and HUD-Change checkpoints: native generation only."""
    path = OUT / "controls_energy.json"
    res = {}
    for arm in ("hud", "f4"):
        model, cfg = NP.IF.load(DS.CKPTS[arm].resolve())
        fit = readers(model, cfg)
        gen = [DS.generate(model, cfg, r, 66, fit[66][0], [(0, 0), (1, 1)], "drop") for r in pair_rows("energy")]
        res[arm] = {"sha256": TR.file_sha256(DS.CKPTS[arm]), "native": DS.summarise(gen, ("00", "11")),
                    "native_per_pair": [g["routes"] for g in gen]}
        print(arm, json.dumps(res[arm]["native"]), flush=True)
        del model
        torch.cuda.empty_cache()
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["controls"]:
        controls()
    else:
        for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else range(1, 11)):
            checkpoint(k)
