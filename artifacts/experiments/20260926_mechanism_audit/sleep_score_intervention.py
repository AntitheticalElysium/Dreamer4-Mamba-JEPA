"""Post-hoc intervention on only the SLEEP score of existing trained heads.

This tests how much of U/W's action gap is due to the unusually low SLEEP
score. It is not a training-cause experiment or a deployable policy.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/experiments/20260921_readout_ladder")]
from frozen_ladder import strata

def score(p, death):
    chosen = p.argmin(1)
    safe = 1 - death.gather(1, chosen[:, None]).squeeze(1)
    return {"safe": float(safe.mean()), "sleep_share": float((chosen == 6).float().mean()),
            "chosen": chosen, "safe_rows": safe}

def main():
    d = ROOT / "artifacts/eda/diagnosis_dump_v1"
    out = {}
    for b in ("55k", "56k", "57k", "58k"):
        meta = torch.load(d / f"{b}_meta.pt", weights_only=False)
        zombie = strata(meta["visible"])["zombie_adjacent"]
        death = meta["p_death1"][zombie]
        arms = {}
        for a in ("U", "W"):
            p = torch.load(d / f"{b}_{a}.pt", weights_only=False)["p_dead"][zombie]
            q = p.clone(); q[:, 6] = q[:, 0]
            m = p.clone(); m[:, 6] = float("inf")
            arms[a] = {"actual": score(p, death), "sleep_score_equal_noop": score(q, death),
                       "sleep_forbidden": score(m, death),
                       "pdead_sleep": float(p[:, 6].mean()), "pdead_noop": float(p[:, 0].mean()),
                       "truth_sleep": float(death[:, 6].mean()), "truth_noop": float(death[:, 0].mean())}
        out[b] = {"n_zombie": len(death), "arms": {
            a: {k: {kk: vv for kk, vv in v.items() if kk in ("safe", "sleep_share")}
                if isinstance(v, dict) else v for k, v in arm.items()}
            for a, arm in arms.items()}}
        w_u = arms["W"]["actual"]["safe_rows"] - arms["U"]["actual"]["safe_rows"]
        u_fix = arms["U"]["sleep_forbidden"]["safe_rows"] - arms["U"]["actual"]["safe_rows"]
        out[b]["W_minus_U"] = float(w_u.mean())
        out[b]["U_gain_from_forbidding_SLEEP"] = float(u_fix.mean())
        out[b]["share_of_gap"] = float(u_fix.mean() / w_u.mean()) if abs(float(w_u.mean())) > 1e-6 else None
        print(b, json.dumps(out[b]), flush=True)
    (HERE / "sleep_score_intervention.json").write_text(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__":
    main()
