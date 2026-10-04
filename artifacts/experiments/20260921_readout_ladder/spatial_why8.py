"""Post hoc, eighth step: the population, or the labels?

spatial_why7 (factual_training_does_not_transfer): a continuation head fitted on FACTUAL generated
states from corpus windows chose at the prior within root (T 0.628, prior 0.628), while a ranking probe
fitted on FORK roots with all 17 counterfactual outcomes read the same world's generated states at
0.724 (spatial_why6). Two things differ: the population (corpus windows vs hazard fork roots) and the
labels (one logged action vs all 17). Hold the population fixed and cross the labels:

Frozen T and Z worlds; generated one-step successors at position 4 for all 17 actions at every FIT
fork root (the observability FIT set, 7,085 roots). One head per arm and label set, the same
objective (BCE on the 32-key P(death1) as a soft target), architecture (T: spatial_why4's attention
probe on the generated tiles; Z: MLP), budget (3,000 updates of 256 branches) and three seeds:
  factual          only the branch of the collector's own action at each root (`bc_action`)
  counterfactual   all 17 branches of every root
Judged within root on the exploratory observability judge opportunity roots, as steps 6 and 7.

Readings for T (committed before the run; post hoc; root_tokens 0.724, prior 0.628):
  counterfactual >= 0.674 and factual <= 0.678     -> labels (counterfactual outcomes are what teach it)
  counterfactual >= 0.674 and factual >= 0.674     -> population (hazard roots suffice, labels do not matter)
  counterfactual < 0.674                           -> objective (BCE does not recover what ranking did)
  otherwise                                        -> mixed
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from observability import FIT_STATE, expected_safe, load  # noqa: E402
from spatial import N, TOKENS, WORLDS, World, bridge  # noqa: E402
from spatial_why6 import generate  # noqa: E402
from spatial_why7 import head  # noqa: E402

STEPS = 3000


def bc_actions(fit_seeds):
    """The collector's action at each FIT root, in observability.load's row order."""
    store = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    out = []
    for s in sorted(fit_seeds):
        pixels = {int(r["step"]): r for r in torch.load(store[s], weights_only=False)}
        for step in {int(r["step"]): r for r in torch.load(state[s], weights_only=False)}:
            out.append(int(pixels[step]["bc_action"]))
    return torch.tensor(out)


def fit_and_judge(x, y, xj, tokens, device, seed):
    torch.manual_seed(seed)
    model = head(x.shape[-1], tokens).to(device)
    score = (lambda v: model(v)) if tokens else (lambda v: model(v)[:, 0])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = torch.randint(len(x), (256,), generator=gen)
        loss = F.binary_cross_entropy_with_logits(score(x[idx].float().to(device)), y[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        flat = xj.flatten(0, 1)
        risk = torch.cat([score(flat[i:i + 1024].float().to(device)).cpu() for i in range(0, len(flat), 1024)])
    return risk.view(xj.shape[0], N)


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded:
        raise SystemExit("rows are not the observability test's")
    bc = bc_actions(fit_seeds)
    if len(bc) != len(data["fit"]["seed"]):
        raise SystemExit("bc_action rows misaligned")
    fit, judge = data["fit"], data["judge"]
    opp = judge["p_death1"].amax(1) > judge["p_death1"].amin(1)
    pj = judge["p_death1"][opp]
    encoder, config = bridge()
    result = {}
    for arm in ("Z", "T"):
        state = "z" if arm == "Z" else "tokens"
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world = World(1 if state == "z" else TOKENS, False).to(device)
        world.load_state_dict(stored["world"])
        world.eval()
        xf = generate(world, config, encoder, fit["frames"], fit["actions"], device, state)
        xj = generate(world, config, encoder, judge["frames"][opp], judge["actions"][opp], device, state)
        del world
        torch.cuda.empty_cache()
        rows = torch.arange(len(bc))
        sets = {"factual": (xf[rows, bc], fit["p_death1"][rows, bc]),
                "counterfactual": (xf.flatten(0, 1), fit["p_death1"].flatten())}
        for name, (x, y) in sets.items():
            runs = [float(expected_safe(fit_and_judge(x, y, xj, state == "tokens", device, s), pj)[0].mean())
                    for s in range(3)]
            result[f"{arm}_{name}"] = {"expected_safe": float(np.mean(runs)), "per_seed": runs, "examples": len(y),
                                       "positive_mass": float(y.sum())}
            log(arm=f"{arm}_{name}", safe=round(result[f"{arm}_{name}"]["expected_safe"], 4), examples=len(y))
        del xf, xj
    c, f = result["T_counterfactual"]["expected_safe"], result["T_factual"]["expected_safe"]
    reading = ("labels" if c >= 0.674 and f <= 0.678 else "population" if c >= 0.674 and f >= 0.674 else
               "objective" if c < 0.674 else "mixed")
    evidence = {"schema": "d4mj_spatial_why8_v1", "status": "POST HOC, EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "judge_opportunity_roots": int(opp.sum()),
                "reading": reading, "result": result}
    (HERE / "evidence/spatial_why8.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why8_complete", reading=reading)


if __name__ == "__main__":
    main()
