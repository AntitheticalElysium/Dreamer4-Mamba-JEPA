"""F6.1 dissection, check 5 (PLAN.md): why the queue's mining stage stopped. Its pre-check asserts that each mining
pair's frame-(t-1) periphery tokens, recomputed by the world's perception from the bank's stored frame (batch 2), match
S1's encoding of the recorded episode (batches of 512) to within 0.04. Per pair of the coal and wood banks: the stored
frame against the recorded episode's frame (exact), and the largest token difference, for F6.1 and F6 (which passed).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.mining_check      (GPU)
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
from saccade import data as D, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
CKPTS = {"F6.1": "artifacts/experiments/20261114_p1f61/run/ckpt_10.pt", "F6": "artifacts/experiments/20261113_p1f6/run/ckpt_10.pt"}


@torch.no_grad()
def main():
    bank = [r for r in torch.load(RT.HERE / "readings/bank.pt", weights_only=False, map_location="cpu") if r["kind"] in RT.SLOT]
    L = dict(np.load(RT.CV.LABELS / "s0_dev_labels.npz"))
    eps = sorted({r["e"] for r in bank})
    sub = SL.subset(L, eps)
    where = {e: i for i, e in enumerate(eps)}
    episodes = S.episodes(L)
    same_frame = [bool(np.array_equal(np.asarray(r["frame"])[0], D.RP.episode_fields(episodes[r["e"]].shard, episodes[r["e"]].slot)["observations"][r["t"] - 1].numpy()))
                  for r in bank]
    res = {"pairs": len(bank), "stored_frame_equals_recorded": float(np.mean(same_frame))}
    for name, ck in CKPTS.items():
        model, cfg = RT.CV.IF.load(ck)
        enc = S.encode(model, sub, cfg)
        diffs, scale = [], []
        for r in bank:
            E = enc[where[r["e"]]]
            with TR.autocast(RT.CV.DEV, cfg):
                P2 = model.perception(torch.as_tensor(r["frame"], device=RT.CV.DEV), E["fix"][r["t"] - 1].to(RT.CV.DEV).expand(2, -1))
            real = E["p"][r["t"] - 1].to(RT.CV.DEV).float()
            diffs.append(float((P2.p[0].float() - real).abs().max()))
            scale.append(float(real.abs().max()))
        d = np.array(diffs)
        res[name] = {"max_abs_diff": {"max": float(d.max()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99))},
                     "pairs_over_0.04": int((d >= .04).sum()), "first_failing_pair": int(np.argmax(d >= .04)) if (d >= .04).any() else None,
                     "token_abs_max_median": float(np.median(scale)), "token_abs_max_max": float(np.max(scale))}
        print(name, json.dumps(res[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    (HERE / "mining_check.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
