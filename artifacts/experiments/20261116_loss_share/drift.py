"""Loss-share question (PLAN.md): token drift between checkpoints, F4 / F5 / F6 / F6.1, CPU perception only.

    GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261116_loss_share.drift
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
EXP = HERE.parent
RUNS = {"F4": EXP / "20261106_p1f4/run", "F5": EXP / "20261111_p1f5/run", "F6": EXP / "20261113_p1f6/run",
        "F6.1": EXP / "20261114_p1f61/run"}


@torch.no_grad()
def main():
    torch.set_num_threads(6)
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    eps = G.S.episodes(L)
    rng = np.random.default_rng(1116)
    pick = rng.choice(len(eps), 40, replace=False)
    frames = []
    for e in pick:
        obs = G.D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"]
        t = rng.choice(len(obs), min(30, len(obs)), replace=False)
        frames.append(obs[np.sort(t)])
    frames = torch.cat(frames)
    res = {"frames": int(len(frames))}
    for run, path in RUNS.items():
        toks = []
        for k in range(1, 11):
            model, _ = G.CR.NP.IF.load((path / f"ckpt_{k:02d}.pt").resolve())
            toks.append(torch.cat([G.C.coarse(model.perception, z)[0] for z in frames.split(256)]).float())
            del model
        var = float((toks[-1] - toks[-1].mean(0)).square().mean())
        drift = [float((toks[k] - toks[k - 1]).square().mean()) / var for k in range(1, 10)]   # into checkpoints 2..10
        to_final = [float((toks[k] - toks[-1]).square().mean()) / var for k in range(10)]
        early, late = float(np.mean(drift[0:3])), float(np.mean(drift[6:9]))
        res[run] = {"drift_into_2_to_10": drift, "distance_to_final_1_to_10": to_final, "early_2_4": early,
                    "late_8_10": late, "late_over_early": late / early, "token_variance_ckpt10": var}
        print(run, json.dumps({k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v]) for k, v in res[run].items()}), flush=True)
    (HERE / "drift.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
