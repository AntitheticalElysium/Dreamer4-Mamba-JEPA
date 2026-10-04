"""Decision heads trained on the world's OWN generated states (MuZero / TD-MPC2 style), factual data only.

The interface worlds' trained continuation heads read death on REAL successors at within-root AUC ~0.996
and on GENERATED ones at ~0.65 (INTERFACE.md, both seeds), and their within-root choice is at the level
of actions_only. They were fitted mostly on real states: H2's bridge weights 0.5 observed prefix,
0.25 observed suffix, 0.25 generated suffix. MuZero, EfficientZero and TD-MPC2 fit every decision head
on the dynamics model's rolled-out latents instead, so heads never see a distribution they are not
used on. SPATIAL.md step 7 tried this on the T world and got the prior -- but T's generated state did
not carry action-specific content, and U's does (interface_shuffle: within-root permutation costs 0.31
on zombie roots). So the idea is re-tested here on the frozen U worlds, inside the factual contract.

Frozen worlds (U_s1, U_s2, Z_s1, Z_s2); nothing in them changes. Factual training examples: every pool
window (TRAIN split), each giving two depth-1 generated successors at the evaluator position (4 observed
frames then one advance with the LOGGED action): frame 4 from frames 0-3 and frame 5 from frames 1-4;
label = the logged continuation (dead only at a terminal window's frame 5). No fork data, no
counterfactual labels, no selection on forks: fixed 3,000 updates of 512, AdamW 1e-3 / 1e-4, class-
balanced BCE, three head seeds, final weights. Heads (one MLP 512 -> 1 each):
  gen_state      the generated latent alone
  gen_features   the world's agent readout of the generated successor (latent + history)
  real_state     CONTROL: the same head trained on the REAL successor latent (as H2's heads mostly are),
                 then applied to generated states at judgement
Judged within root on the 56k block (POST HOC: read three times), all 17 actions advanced once;
expected safe on 32-key P; paired episode-seed-clustered intervals. References: DOWN, actions_only and
the world's own trained head (recomputed), and the counterfactual probe (INTERFACE.md: 0.672 / 0.670).

Readings (committed before the run), per U world:
  gen_state - own_trained_head resolved > 0 overall AND on zombie roots   -> generated_training_helps
  and gen_state - actions_only resolved > 0 overall                       -> ... and beats_actions_only
  gen_state - real_state resolved > 0                                     -> training_distribution_matters
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N, STEPS, BATCH = 17, 3000, 512
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"
WORLDS = {"U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"), "U_s2": ("U", ROOT / "artifacts/eda/interface_worlds_v2/U.pt"),
          "Z_s1": ("Z", ROOT / "artifacts/eda/interface_worlds_v1/Z.pt"), "Z_s2": ("Z", ROOT / "artifacts/eda/interface_worlds_v2/Z.pt")}


@torch.no_grad()
def factual(bundle, pool, key, config, device, batch=256):
    """Depth-1 generated successors (latent, features) and real successors (latent, features) for every
    pool window at the evaluator position, with their logged continuation labels."""
    from d4mj.train import autocast_context
    out = {k: [] for k in ("gen_state", "gen_features", "real_state", "real_features", "alive")}
    n = len(pool[key])
    for start in (0, 1):
        for i in range(0, n, batch):
            s = pool[key][i:i + batch][:, start:start + 5].to(device)[:, :, None]
            a = pool["actions"][i:i + batch][:, start:start + 4].to(device)
            with autocast_context(config):
                state = bundle.world.teacher(s[:, :4], a[:, :3]).state
                advanced, features = bundle.advance(state, a[:, 3:4])
                _, real_features = bundle.world.observe_latent(state, a[:, 3:4], s[:, 4:5])
            out["gen_state"].append(advanced.latent[:, 0, 0].float().cpu())
            out["gen_features"].append(features[:, -1, 0].float().cpu())
            out["real_state"].append(s[:, 4, 0].float().cpu())
            out["real_features"].append(real_features[:, -1, 0].float().cpu())
            out["alive"].append(pool["alive"][i:i + batch][:, start + 4].float())
    return {k: torch.cat(v) for k, v in out.items()}


def fit(x, alive, device, seed):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    head = nn.Sequential(nn.Linear(x.shape[-1], 512), nn.ReLU(), nn.Linear(512, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    dead = 1 - alive
    pos = (alive.sum() / dead.sum()).to(device)
    for _ in range(STEPS):
        idx = torch.randint(len(x), (BATCH,), generator=g)
        logit = head(((x[idx] - mean) / scale).to(device))[:, 0]
        loss = F.binary_cross_entropy_with_logits(logit, dead[idx].to(device), pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()

    @torch.no_grad()
    def risk(v):
        flat = v.flatten(0, -2)
        return torch.cat([head(((flat[j:j + 4096] - mean) / scale).to(device))[:, 0].cpu()
                          for j in range(0, len(flat), 4096)]).view(v.shape[:-1])
    return risk


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from interface import KEY, POOL, branches, load_bridge, world_bundle
    from ladder import paired
    from observability import expected_safe, load

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    judge, manifest, _ = judge_store(JUDGE)
    pj, seeds = judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, _ = seeds_for(partition, FORK_STORE)
    fitr, dev = load(fit_seeds)["fit"], dev_rows(sorted(partition["fit_dev"]["seeds"]))
    pf, pd = fitr["p_death1"], dev["p_death1"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    actions4 = torch.cat([fitr["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    mean, scale = standardize(actions4, train_rows)
    runs = []
    for seed in range(3):
        model, _ = probe_train("vector", actions4.shape[1:], (actions4 - mean) / scale, p, train_rows, hold_rows,
                               seed=seed, device=device, steps=3000)
        runs.append(expected_safe(scores(model, (judge["actions"][:, -4:].flatten(1) - mean) / scale,
                                         torch.arange(len(pj)), device), pj)[0])
    safe["actions_only"] = torch.stack(runs).mean(0)
    del fitr, dev
    encoder, config = load_bridge()
    result, counts = {}, {}
    for name, (arm, path) in WORLDS.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        bundle = world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device)
        heads.load_state_dict(stored["heads"])
        heads.eval()
        train = factual(bundle, pool, KEY[arm], config, device)
        counts[name] = {"examples": len(train["alive"]), "dead": int((train["alive"] == 0).sum())}
        b = branches(bundle, heads, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device)
        del bundle, heads
        judge_in = {"gen_state": b["generated"][..., :192], "gen_features": b["features"][..., :256]}
        safe[f"{name}_own"] = expected_safe(b["p_dead"], pj)[0]
        for head, source in (("gen_state", "gen_state"), ("gen_features", "gen_features"), ("real_state", "real_state")):
            target = judge_in["gen_state" if head == "real_state" else head]
            r = [expected_safe(fit(train[source], train["alive"], device, s)(target), pj)[0] for s in range(3)]
            safe[f"{name}_{head}"] = torch.stack(r).mean(0)
            log(world=name, head=head, safe=round(float(safe[f"{name}_{head}"][opp].mean()), 4),
                zombie=round(float(safe[f"{name}_{head}"][zombie].mean()), 4))
        log(world=name, own=round(float(safe[f"{name}_own"][opp].mean()), 4), examples=counts[name])
    test = lambda a, bb, m: paired(safe[a][m], safe[bb][m], seeds[m], draws=1000, seed=20261014)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    contrasts, readings = {}, {}
    for name in WORLDS:
        for a, bb in ((f"{name}_gen_state", f"{name}_own"), (f"{name}_gen_state", "actions_only"), (f"{name}_gen_state", "DOWN"),
                      (f"{name}_gen_state", f"{name}_real_state"), (f"{name}_gen_features", f"{name}_own"),
                      (f"{name}_gen_features", "actions_only")):
            contrasts[f"{a}_vs_{bb}"] = {"overall": test(a, bb, opp), "zombie": test(a, bb, zombie)}
        if name.startswith("U"):
            c = lambda a, bb, s="overall": contrasts[f"{a}_vs_{bb}"][s]
            readings[name] = {
                "generated_training_helps": up(c(f"{name}_gen_state", f"{name}_own")) and up(c(f"{name}_gen_state", f"{name}_own", "zombie")),
                "beats_actions_only": up(c(f"{name}_gen_state", "actions_only")),
                "training_distribution_matters": up(c(f"{name}_gen_state", f"{name}_real_state"))}
    evidence = {"schema": "d4mj_genhead_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "judge_manifest": manifest, "roots": {"opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "training_examples": counts, "readings": readings, "contrasts": contrasts,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/genhead.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="genhead_complete", **{k: v for k, v in readings.items()})


if __name__ == "__main__":
    main()
