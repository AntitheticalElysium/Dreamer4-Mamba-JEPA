"""Can FACTUAL data teach the one-step consequence at all, given the world's own input?

ridge_oracle (not_linearly_predictable): closed-form least squares with state x action terms aligns with
the fatal direction no better than the Mamba worlds (0.60 vs 0.61-0.64) although it fits the tail
better. So allocation alone is not the story. Two explanations remain:
  allocation   a nonlinear model trained on the factual transitions CAN predict the consequence; the
               world does not because death is ~0.1% of its loss
  confounding  factual data (one logged action per state, chosen by a policy) does not identify the
               counterfactual consequence; only all-action outcomes do
Same input, same model, same objective, labels differ:
  input   the world's own input: 4 context u (768) + 3 past actions one-hot + the candidate action one-hot
  model   MLP 836 -> 512 -> 512 -> 1, class-balanced BCE on P(dead after this action), 3 seeds,
          5,000 updates of 512, AdamW 1e-3 / 1e-4, final weights (no fork selection)
  factual        the interface pool's TRAIN windows: (frames 0-3 -> 4) and (1-4 -> 5) with the LOGGED
                 action and its logged continuation (65,294 transitions, 8,155 deaths)
  counterfactual FIT fork roots, all 17 actions, 32-key P(death1) as the soft target (7,085 x 17)
Judged within root on the 56k block (POST HOC), all 17 actions: expected safe; zombie roots.
References on the same block: DOWN 0.591, actions_only 0.637, U's own head 0.656 / 0.658, U generated
probe 0.672 / 0.670, root_U ranking probe 0.708.

Reading (committed before the run):
  factual >= counterfactual - 0.03             -> factual_data_suffices (the world's shortfall is allocation)
  factual <= 0.657 (actions_only + 0.02)       -> factual_data_confounded
  otherwise                                    -> partial
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

N, STEPS, BATCH = 17, 5000, 512
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"


def inputs(ctx, past, action):
    return torch.cat([ctx.flatten(1), F.one_hot(past, N).float().flatten(1), F.one_hot(action, N).float()], 1)


def fit(x, y, device, seed, balanced):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(x.shape[1], 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    pos = ((1 - y).sum() / y.sum()).to(device) if balanced else None
    for _ in range(STEPS):
        idx = torch.randint(len(x), (BATCH,), generator=g)
        logit = net(((x[idx] - mean) / scale).to(device))[:, 0]
        loss = F.binary_cross_entropy_with_logits(logit, y[idx].to(device), pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return lambda v: torch.cat([net(((v[j:j + 8192] - mean) / scale).to(device))[:, 0].detach().cpu()
                                for j in range(0, len(v), 8192)])


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import strata
    from interface import POOL, encode, load_bridge, project
    from ladder import paired
    from observability import expected_safe, load

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    u, acts, alive = pool["u"], pool["actions"], pool["alive"]
    xf = torch.cat([inputs(u[:, s:s + 4], acts[:, s:s + 3], acts[:, s + 3]) for s in (0, 1)])
    yf = torch.cat([1 - alive[:, s + 4].float() for s in (0, 1)])
    encoder, _ = load_bridge()

    def ctx_of(frames):
        out = []
        for i in range(0, len(frames), 64):
            _, grid = encode(encoder, frames[i:i + 64, -4:], device)
            out.append(project(pool["pca"], grid))
        return torch.cat(out)
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    fitr = load(fit_seeds)["fit"]
    judge, manifest, _ = judge_store(JUDGE)
    fan = lambda ctx, past: torch.stack([inputs(ctx, past, torch.full((len(ctx),), a)) for a in range(N)], 1)
    xc = fan(ctx_of(fitr["frames"]), fitr["actions"][:, -3:].argmax(-1))
    yc = fitr["p_death1"]
    xj = fan(ctx_of(judge["frames"]), judge["actions"][:, -3:].argmax(-1))
    del encoder
    pj, seeds = judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    safe, per_seed = {}, {}
    for name, (x, y, balanced) in {"factual": (xf, yf, True), "counterfactual": (xc.flatten(0, 1), yc.flatten(), False)}.items():
        runs = [expected_safe(fit(x, y, device, s, balanced)(xj.flatten(0, 1)).view(-1, N), pj)[0] for s in range(3)]
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4),
            per_seed=[round(v, 4) for v in per_seed[name]])
    diff = paired(safe["factual"][opp], safe["counterfactual"][opp], seeds[opp], draws=1000, seed=20261018)
    diffz = paired(safe["factual"][zombie], safe["counterfactual"][zombie], seeds[zombie], draws=1000, seed=20261018)
    f, c = float(safe["factual"][opp].mean()), float(safe["counterfactual"][opp].mean())
    reading = "factual_data_suffices" if f >= c - 0.03 else "factual_data_confounded" if f <= 0.657 else "partial"
    evidence = {"schema": "d4mj_factual_direct_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "judge_manifest": manifest, "training": {"factual": [len(yf), int(yf.sum())], "counterfactual": list(yc.shape)},
                "reading": reading, "factual_minus_counterfactual": {"overall": diff, "zombie": diffz}, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/factual_direct.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="factual_direct_complete", reading=reading, diff=diff["difference"], interval=diff["interval"])


if __name__ == "__main__":
    main()
