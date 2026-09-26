"""Action gap: does a DUELING factual head rank actions better than a plain one?

factual_direct2: pointwise BCE fails on zombie roots even with all 17 counterfactual labels (zombie 0.48);
only the within-root ranking objective succeeds there (0.65). The hypothesis: pointwise objectives spend
capacity on BETWEEN-state variation (which states are dangerous), while the decision lives in the small
WITHIN-state, across-action differences -- the action gap (Bellemare et al. 2016). The dueling
architecture (Wang et al. 2016, Dueling DQN) was built for exactly this: Q(s,a) = V(s) + A(s,a) - mean A,
so the state value is absorbed by V and the shared advantage stream learns relative action effects, from
ordinary single-action (factual) updates.

Both arms: FACTUAL data only (the interface pool's TRAIN transitions, logged action, logged continuation),
the world's own input (4 context u + 3 past actions; candidate action), class-balanced BCE on the
logit of P(dead | s, a), 3 seeds, 5,000 updates of 512, selection every 250 on FIT-dev within-root
expected safe (as factual_direct2):
  plain     MLP([state, action one-hot]) -> 1                               (factual_direct2's model)
  dueling   V = MLP(state) -> 1;  A = MLP(state) -> 17;  logit = V + A[a] - mean(A)
Judged within root on the 56k block (POST HOC).

Reading (committed before the run):
  dueling - plain resolved > 0 overall AND on zombie roots -> action_gap_architecture_helps
  dueling - plain resolved > 0 overall only                -> helps_overall_only
  otherwise                                                -> no_evidence_dueling_helps
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


class Dueling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.a = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, N))

    def forward(self, x):                          # x = [state (width), action one-hot (17)]
        s, a = x[:, :-N], x[:, -N:]
        adv = self.a(s)
        return self.v(s)[:, 0] + ((adv - adv.mean(1, keepdim=True)) * a).sum(1)


def fit(x, y, device, seed, balanced, select, dueling):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    mean[-N:], scale[-N:] = 0.0, 1.0                 # keep the action one-hot exact for the dueling gather
    if dueling:
        net = Dueling(x.shape[1] - N).to(device)
        head = net
    else:
        net = nn.Sequential(nn.Linear(x.shape[1], 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1)).to(device)
        head = lambda v: net(v)[:, 0]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    pos = ((1 - y).sum() / y.sum()).to(device) if balanced else None
    score = lambda v: torch.cat([head(((v[j:j + 8192] - mean) / scale).to(device)).detach().cpu()
                                 for j in range(0, len(v), 8192)])
    best, state = -1.0, None
    for step in range(STEPS):
        idx = torch.randint(len(x), (BATCH,), generator=g)
        logit = head(((x[idx] - mean) / scale).to(device))
        loss = F.binary_cross_entropy_with_logits(logit, y[idx].to(device), pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            net.eval()
            with torch.no_grad():
                value = select(score)
            net.train()
            if value > best:
                best, state = value, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    return lambda v: score(v)


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
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
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    dev = dev_rows(sorted(partition["fit_dev"]["seeds"]))
    fan = lambda ctx, past: torch.stack([inputs(ctx, past, torch.full((len(ctx),), a)) for a in range(N)], 1)
    xc = fan(ctx_of(fitr["frames"]), fitr["actions"][:, -3:].argmax(-1))
    yc = fitr["p_death1"]
    xj = fan(ctx_of(judge["frames"]), judge["actions"][:, -3:].argmax(-1))
    xd = fan(ctx_of(dev["frames"]), dev["actions"][:, -3:].argmax(-1))
    pd = dev["p_death1"]

    def select(score):
        s, o = expected_safe(score(xd.flatten(0, 1)).view(-1, N), pd)
        return float(s[o].mean())
    del encoder
    pj, seeds = judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    safe, per_seed = {}, {}
    for name, dueling in {"plain": False, "dueling": True}.items():
        runs = [expected_safe(fit(xf, yf, device, s, True, select, dueling)(xj.flatten(0, 1)).view(-1, N), pj)[0] for s in range(3)]
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4),
            per_seed=[round(v, 4) for v in per_seed[name]])
    diff = paired(safe["dueling"][opp], safe["plain"][opp], seeds[opp], draws=1000, seed=20261019)
    diffz = paired(safe["dueling"][zombie], safe["plain"][zombie], seeds[zombie], draws=1000, seed=20261019)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    reading = ("action_gap_architecture_helps" if up(diff) and up(diffz) else
               "helps_overall_only" if up(diff) else "no_evidence_dueling_helps")
    evidence = {"schema": "d4mj_dueling_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "judge_manifest": manifest, "training": {"factual": [len(yf), int(yf.sum())]},
                "reading": reading, "dueling_minus_plain": {"overall": diff, "zombie": diffz}, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/dueling.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="dueling_complete", reading=reading, diff=diff["difference"], interval=diff["interval"])


if __name__ == "__main__":
    main()
