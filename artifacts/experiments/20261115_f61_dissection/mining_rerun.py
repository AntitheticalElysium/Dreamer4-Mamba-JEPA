"""F6.1 dissection, check 5b (PLAN.md): the queue's mining stage (`../20261111_p1f5/followups.py`, `mining`), its code
line for line, with its 0.04 token pre-check recorded per pair instead of asserted (check 5: the stored frames equal
the recorded ones on all 402 pairs; F6.1 exceeds 0.04 on one pair, by 0.0008, F6's largest difference is 0.0398).
Written to this folder, never to the pinned queue's readings.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.mining_rerun      (GPU)
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

FU = importlib.import_module("artifacts.experiments.20261111_p1f5.followups")
HERE = Path(__file__).resolve().parent
CKPT = (HERE.parent / "20261114_p1f61/run/ckpt_10.pt").resolve()


def main():
    out = HERE / "run/mining"
    out.mkdir(parents=True, exist_ok=True)
    rf = FU.routes(CKPT, out)
    rt, cv = rf.RT, rf.CV
    model, cfg, labels, enc = rf.setup()
    cv.HUD = {slot: cv.HUD[slot] for slot in rt.SLOT.values()}
    readers, valid = cv.fit_readers(model, cfg, labels, enc, 0)
    bank_path = rt.HERE / "readings/bank.pt"
    bank = torch.load(bank_path, weights_only=False, map_location="cpu")
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    lo = cfg["edm"]["inference_context_sigma"]
    result = {"bank_sha256": FU.sha256(bank_path), "draws": rt.K, "checkpoint_sha256": FU.sha256(CKPT),
              "conditions": ["DO/DO", "NOOP/DO", "DO/NOOP", "NOOP/NOOP"], "readers": valid, "precheck_max_abs_diff": {}}
    with torch.no_grad():
        for kind, slot in rt.SLOT.items():
            rows, episode, rec, pre = [], [], [], []
            for r in bank:
                if r["kind"] != kind:
                    continue
                p, successors, _, bp, caches, fx = rt.pair_state(model, cfg, enc, r)
                real = enc[r["e"]]["p"][r["t"] - 1].to(cv.DEV).float()
                pre.append(float((p.p[0].float() - real).abs().max()))          # the stage asserts < .04 here
                read = lambda p, v: readers[slot](torch.cat([p.float(), v.float()], -1))["y"].softmax(-1).argmax(-1)
                count = r["truth"]["count"][1]
                rec.append(read(successors.p[:, slot], successors.v).tolist() == [count + 1, count])
                rates = []
                for action, memory in rt.COND:
                    xs = rf.sample(model, cfg, shapes, r["actions"][action], bp[memory:memory + 1],
                                   fx, caches[lo][action], rt.K, r["j"])
                    rates.append(float((read(xs[0][:, 0, slot], xs[2][:, 0]) == count + 1).float().mean()))
                rows.append(rates)
                episode.append(r["e"])
            assert len(rows) == rt.KINDS[kind], (kind, len(rows))
            result["precheck_max_abs_diff"][kind] = {"max": max(pre), "over_0.04": int(sum(d >= .04 for d in pre))}
            result[kind] = {"pairs": len(rows), "episode_ids": episode, "per_pair_rates": rows,
                            "recorded_reader_check": float(np.mean(rec)),
                            "factorial": rt.factorial(np.asarray(rows), np.full(len(rows), -1.), np.asarray(episode))}
            print(kind, json.dumps({k: v for k, v in result[kind].items() if k not in ("per_pair_rates", "episode_ids")})[:600], flush=True)
    (HERE / "mining_rerun.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
