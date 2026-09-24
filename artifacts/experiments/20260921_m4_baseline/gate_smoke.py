"""Production-geometry CUDA smoke for the gate evaluator, after the code freeze.

The earlier smoke exercised bridge/actor TRAINING only. It cannot establish feasibility for the
17-way action-blind rollout, the H16 recursive evaluation, the all-action fork gate or the
retention panel, which are the new and by far the most memory-hungry parts. This runs each at the
real recipe's geometry and records peak memory and wall time.
"""
import json, sys, time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
from d4mj.agent import Heads
from d4mj.config import load_recipe
from d4mj.lewm_diagnostics import (_action_blind_rollout, _action_outcomes, _critical_retention,
                                   fork_population, retention_addresses)
from d4mj.world_api import ModelBundle


def measure(name, function):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    value = function()
    torch.cuda.synchronize()
    row = {"stage": name, "seconds": round(time.time() - started, 1),
           "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    print(json.dumps(row), flush=True)
    return row, value


def main():
    config = load_recipe(ROOT / "d4mj/recipes/lewm_mamba_raw_m4.json")
    screen = load_recipe(ROOT / "d4mj/recipes/joint_screen.json")
    settings = config.agent
    bundle = ModelBundle.create(config)
    bundle.encoder.freeze(); bundle.world.eval()
    for module in bundle.world.predictor_projector.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    heads = Heads(config).to(config.runtime.device)
    prior = Heads(config).to(config.runtime.device)
    rows = []

    # the heaviest evaluation path: 17-way fan at the real batch and the long window
    def blind():
        device = bundle.device
        z = torch.randint(0, 256, (settings.batch, settings.sequence_long, 63, 63, 3),
                          dtype=torch.uint8, device=device)
        with torch.no_grad():
            latents = bundle.encoder(z)
            actions = torch.zeros(settings.batch, settings.sequence_long - 1,
                                  dtype=torch.long, device=device)
            anchor = settings.sequence_long - 1 - settings.recursive_depth_final
            prefix = bundle.world.teacher(latents[:, :anchor + 1], actions[:, :anchor]).state
            return _action_blind_rollout(bundle, prefix, settings.batch,
                                         settings.recursive_depth_final)
    rows.append(measure(f"action_blind_H{settings.recursive_depth_final}_"
                        f"b{settings.batch}x{settings.sequence_long}", blind)[0])

    forks = fork_population(config, roots=512, seed=config.seed + 51)
    rows.append(measure("all_action_forks_512_roots",
                        lambda: _action_outcomes(bundle, heads, prior, forks,
                                                 draws=screen.bootstrap_draws, seed=1,
                                                 output="/tmp/gate-smoke/forks"))[0])
    panel = retention_addresses()
    rows.append(measure("critical_retention_panel",
                        lambda: _critical_retention(bundle, None, panel, screen,
                                                    "/tmp/gate-smoke/retention"))[0])
    out = ROOT / "artifacts/experiments/20260921_m4_baseline/gate_smoke.json"
    out.write_text(json.dumps({"schema": "d4mj_gate_smoke_v1", "device":
                               torch.cuda.get_device_name(0), "recipe": "lewm_mamba_raw_m4",
                               "fork_roots": forks["roots"], "panel_rows": len(panel["frames"]),
                               "stages": rows,
                               "budget_mib": 6144}, indent=2) + "\n")
    worst = max(r["peak_mib"] for r in rows)
    print(json.dumps({"status": "gate_smoke_complete", "worst_peak_mib": worst,
                      "fits_6gb": worst < 6144}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
