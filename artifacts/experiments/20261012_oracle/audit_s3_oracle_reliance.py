"""Audit (10-14), stage 3: did generated-prefix training change how much the world relies on the oracle versus copying its history?
On the 200 DEV roots, teacher-forced (15 TRUE frames), depths 1-16, for the shared start (loss_intervention/D_u3000) and the two
generated-prefix arms at 3,000 updates: the world's output with the true next-layout oracle and with the oracle zeroed. Reader:
scoreboard.AllProbes (cow channel). Reported per arm, at arrival cells (cow now, not before), vacated cells (cow before, not now)
and stay cells: mean cow score with / without the oracle, and the share above threshold. Large oracle effects at arrival and
vacated cells = the oracle drives the edit; small effects = the edit comes from the history (copy).
Also self-fed (H16, true oracle every step): the same split, at depths 9-16, to see where the extra prefix false positives sit.
Usage: audit_s3_oracle_reliance.py -> stdout + artifacts/eda/oracle_v1/generated_prefix/audit_oracle_reliance.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
for d in ("20260927_levers", "20260926_diagnosis", "20260921_readout_ladder", "20261012_oracle"):
    sys.path.insert(0, str(ROOT / "artifacts/experiments" / d))
import oracle_eval as OE  # noqa: E402

OUT = OE.OUT
H, W = 16, 15


class ZeroOracle(torch.nn.Module):
    def __init__(self, oc):
        super().__init__(); self.oc = oc

    def forward(self, layout, frames):
        return self.oc(torch.zeros_like(layout), frames)


def main():
    import teval as T
    import scoreboard as SB
    from d4mj.config import config_from_dict
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    panel = torch.load(OUT / "panel.pt", weights_only=False)
    tok, lay, act = panel["tokens"], panel["layouts"], panel["actions"]
    R = len(tok)
    meta, tr, ts = T.split(); cache = T.build_cache("raw", torch.device("cpu"))
    probes = SB.AllProbes(T, cache, meta, tr, ts)
    thr = float(probes.thr[OE.PROBE_CH["cow"]])
    truth = lay[..., OE.LAYOUT_CH["cow"]] > 0
    truth[:, :, 31] = False
    tru, prv = truth[:, 16:32], truth[:, 15:31]
    cells = {"arrive": tru & ~prv, "vacated": prv & ~tru, "stay": tru & prv, "empty": ~tru & ~prv}

    def cow_scores(pred):                                                   # [R,16,81,192] -> [R,16,63]
        sc = torch.full((R, H, 63), -9.0)
        sc[:, :, OE.MAPC] = probes.mob(pred[:, :, OE.MAPC].flatten(0, 2).float()).view(R, H, 62, 4)[..., OE.PROBE_CH["cow"]]
        return sc
    arms = {"start_D_u3000": OUT / "loss_intervention/D_u3000.pt", "teacher_u3000": OUT / "generated_prefix/teacher_u3000.pt",
            "prefix_u3000": OUT / "generated_prefix/prefix_u3000.pt"}
    res = {"threshold": thr, "cells": {k: int(v.sum()) for k, v in cells.items()}}
    for name, path in arms.items():
        world, oc = OE.load(path, device)
        assert oc is not None, name
        out = {}
        for tag, cond in (("oracle", oc), ("zeroed", ZeroOracle(oc))):
            tf = torch.empty(R, H, 81, 192); gen = torch.empty(R, H, 81, 192)
            for i in range(0, R, 8):
                sl = slice(i, min(i + 8, R))
                for k in range(1, H + 1):
                    tf[sl, k - 1] = OE.step(world, cond, tok[sl, k:k + W], act[sl, k:k + W], lay[sl, k + 1:k + 1 + W], device, config)
                if tag == "oracle":
                    frames = [tok[sl, j].float() for j in range(1, 16)]
                    for k in range(1, H + 1):
                        g = OE.step(world, cond, torch.stack(frames[-W:], 1), act[sl, k:k + W], lay[sl, k + 1:k + 1 + W], device, config)
                        gen[sl, k - 1] = g; frames.append(g)
            sc = cow_scores(tf)
            out[f"tf_{tag}"] = {c: {"mean": round(float(sc[m].mean()), 3), "above": round(float((sc[m] > thr).float().mean()), 3)} for c, m in cells.items()}
            if tag == "oracle":
                sg = cow_scores(gen)
                late = torch.zeros(R, H, 63, dtype=torch.bool); late[:, 8:] = True
                out["gen_oracle_late"] = {c: {"mean": round(float(sg[m & late].mean()), 3), "above": round(float((sg[m & late] > thr).float().mean()), 3),
                                              "n": int((m & late).sum())} for c, m in cells.items()}
                out["gen_oracle_late"]["false_cells"] = int(((sg > thr) & ~tru & late).sum())
        res[name] = out
        print(name, json.dumps(out), flush=True)
        del world; torch.cuda.empty_cache()
    (OUT / "generated_prefix/audit_oracle_reliance.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
