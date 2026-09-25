"""Post hoc, sixth step: does the generated per-tile STATE carry the within-root decision?

spatial_why5: a fresh probe reads one-step damage from TH's generated health tile at transition-level
DEV AUC 0.91 (real tile 1.00), though the trained heads do not -- the world writes the forecast as a
small displacement, not the real state's large change. But transition-level AUC mixes between-context
differences, and the decision is WITHIN a root, across 17 counterfactual actions. The 2x2's fresh
probes read only pooled 256-d features (0.58-0.65). This reads the full generated state.

Exploratory observability roots (FIT fit, 50k judge; one-step opportunity roots only), frozen worlds,
4-frame context, all 17 actions advanced once. Ranking probes exactly as frozen_ladder (soft_rank over
32-key P, AdamW 1e-3 / 1e-4, batches of 128 roots, 3,000 updates, inner-FIT selection every 100, three
probe seeds), each branch scored by one shared per-branch head:
  generated_{Z,ZH}   generated z (normalized)          MLP 192 -> 512 -> 1
  generated_{T,TH}   generated 81 tiles (normalized)   spatial_why4's attention probe -> 1
  real_tokens        the REAL successor's 81 tiles     hindsight control (same head)
  real_z             the REAL successor's z            hindsight control
  root_tokens        frozen_ladder tokens_attn on the root's raw patch tokens -> 17 (the reference)
  prior              the FIT action prior

Readings (committed before the run; post hoc, exploratory):
  real_tokens < 0.90                                         -> void
  max(generated_T, generated_TH) >= root_tokens - 0.03       -> generated_state_carries_decision
      (the failure is in the trained heads / supervision layout, not the generated state)
  max(generated_T, generated_TH) <= prior + 0.05             -> generated_state_loses_decision
  otherwise                                                  -> partial
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from frozen_ladder import scores as root_scores, standardize, train as root_train  # noqa: E402
from ladder import paired  # noqa: E402
from observability import expected_safe, load, soft_rank  # noqa: E402
from spatial import ARMS, N, TOKENS, WIDTH, WORLDS, World, bridge, encode  # noqa: E402
from spatial_why4 import Probe  # noqa: E402
from u_world import successors  # noqa: E402


@torch.no_grad()
def generate(world, config, encoder, frames, actions, device, state, batch=16):
    from d4mj.train import autocast_context
    out = []
    for i in range(0, len(frames), batch):
        z, tokens = encode(encoder, frames[i:i + batch, -4:], device)
        n = len(z)
        s = (z[:, :, None] if state == "z" else tokens.float()).to(device).repeat_interleave(N, 0)
        past = actions[i:i + batch, -3:].argmax(-1)
        a = torch.cat([past.repeat_interleave(N, 0), torch.arange(N).repeat(n)[:, None]], 1).to(device)
        with autocast_context(config):
            predicted, _ = world(s, a)
        g = predicted[:, 3].float().view(n, N, *predicted.shape[2:])
        out.append((g[:, :, 0] if state == "z" else g).half().cpu())
    return torch.cat(out)


class Branch(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.net = Probe(WIDTH, False) if tokens else nn.Sequential(nn.Linear(WIDTH, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):                                                  # [B, 17, ...] -> [B, 17] risk
        b = x.shape[0]
        y = self.net(x.flatten(0, 1))
        return (y if self.tokens else y[:, 0]).view(b, N)


def rank_train(tokens, x, p, train_rows, hold_rows, *, seed, device, steps=3000, batch=128, every=100):
    """frozen_ladder.train, verbatim in procedure, for a shared per-branch head."""
    torch.manual_seed(seed)
    model = Branch(tokens).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    score = lambda rows: torch.cat([model(x[rows[i:i + 256]].float().to(device)).cpu() for i in range(0, len(rows), 256)])
    best, state = -1.0, None
    for step in range(steps):
        idx = train_rows[torch.randint(len(train_rows), (batch,), generator=gen)]
        loss = soft_rank(model(x[idx].float().to(device)), p[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % every == 0 or step + 1 == steps:
            model.eval()
            with torch.no_grad():
                safe, opp = expected_safe(score(hold_rows), p[hold_rows])
            model.train()
            value = float(safe[opp].mean())
            if value > best or state is None:
                best, state = value, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def scorer(rows, data):
        return torch.cat([model(data[rows[i:i + 256]].float().to(device)).cpu() for i in range(0, len(rows), 256)])
    return scorer


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads  # noqa: F401  (config parity with spatial.py)
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    data = load(fit_seeds)
    succ = successors(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded or {s: succ[s][1] for s in succ} != recorded:
        raise SystemExit("rows are not the observability test's")
    keep = {s: torch.where(data[s]["p_death1"].amax(1) > data[s]["p_death1"].amin(1))[0] for s in data}
    for s in data:
        data[s] = {k: (v[keep[s]] if torch.is_tensor(v) else v) for k, v in data[s].items()}
        data[s]["successors"] = succ[s][0][keep[s]]
    del succ
    encoder, config = bridge()
    for s in data:
        rz, rt = encode(encoder, data[s]["successors"], device)
        data[s]["real_z"], data[s]["real_tokens"] = rz, rt
        with torch.no_grad():
            data[s]["root_tokens"] = torch.cat([encoder._hidden(data[s]["frames"][i:i + 64, -1:].to(device))[2].cpu()
                                                for i in range(0, len(data[s]["frames"]), 64)])
    for arm, (state, health) in ARMS.items():
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world = World(1 if state == "z" else TOKENS, health).to(device)
        world.load_state_dict(stored["world"])
        world.eval()
        for s in data:
            data[s][f"generated_{arm}"] = generate(world, config, encoder, data[s]["frames"], data[s]["actions"], device, state)
        del world
        log(stage="generated", arm=arm)
    del encoder
    torch.cuda.empty_cache()

    fit, judge = data["fit"], data["judge"]
    groups = fit["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(20260924))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in fit["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(judge["seed"]))
    pf, pj = fit["p_death1"], judge["p_death1"]
    prior = int(pf[~inner].mean(0).argmin())
    safe = {"prior": expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    mean, scale = standardize(fit["root_tokens"], train_rows)
    xf, xj = (fit["root_tokens"] - mean) / scale, (judge["root_tokens"] - mean) / scale
    runs = []
    for seed in range(3):
        model, _ = root_train("tokens_attn", xf.shape[1:], xf, pf, train_rows, hold_rows, seed=seed, device=device, steps=3000)
        runs.append(expected_safe(root_scores(model, xj, judge_rows, device), pj)[0])
    safe["root_tokens"] = torch.stack(runs).mean(0)
    log(arm="root_tokens", safe=round(float(safe["root_tokens"].mean()), 4))
    for name in ("real_tokens", "real_z", *(f"generated_{a}" for a in ARMS)):
        tokens = fit[name].ndim == 4
        runs = []
        for seed in range(3):
            scorer = rank_train(tokens, fit[name], pf, train_rows, hold_rows, seed=seed, device=device)
            runs.append(expected_safe(scorer(judge_rows, judge[name]), pj)[0])
        safe[name] = torch.stack(runs).mean(0)
        log(arm=name, safe=round(float(safe[name].mean()), 4))
    seeds = judge["seed"]
    report = {k: float(v.mean()) for k, v in safe.items()}
    contrasts = {f"{a}_vs_root_tokens": paired(safe[a], safe["root_tokens"], seeds, draws=1000, seed=20260937)
                 for a in ("generated_T", "generated_TH", "generated_Z", "generated_ZH")}
    best = max(report["generated_T"], report["generated_TH"])
    reading = ("void" if report["real_tokens"] < 0.90 else
               "generated_state_carries_decision" if best >= report["root_tokens"] - 0.03 else
               "generated_state_loses_decision" if best <= report["prior"] + 0.05 else "partial")
    evidence = {"schema": "d4mj_spatial_why6_v1", "status": "POST HOC, EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "opportunity_roots": {"fit": len(pf), "judge": len(pj)},
                "reading": reading, "expected_safe": report, "contrasts": contrasts}
    (HERE / "evidence/spatial_why6.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why6_complete", reading=reading)


if __name__ == "__main__":
    main()
