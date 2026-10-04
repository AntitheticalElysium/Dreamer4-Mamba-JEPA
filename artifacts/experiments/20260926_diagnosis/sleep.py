"""D7. Why do the heads rate SLEEP as safe next to a zombie? What the training data says about sleep frames.

craftax_classic/renderer.py applies sleep last to the map area: pixels = (0.5 lum, 0.5 lum, 0.5 lum + 8),
so a sleeping frame's map has R == G and B - R == 8 (+-rounding) everywhere, day or night. Detector
validated on the observe store's 32-frame histories, whose sleep flag comes from the simulator state.
Then, over corpus frames: share asleep, P(death at the next step | asleep) vs awake, and the zombie-root
fact the heads face: P(the SLEEP successor is rendered asleep) = P(energy < 9 at the root).
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
from frozen_ladder import strata  # noqa: E402


def asleep(frames):
    x = frames[:, :49].float()
    rg = (x[..., 0] - x[..., 1]).abs().mean((1, 2))
    br = (x[..., 2] - x[..., 0]).mean((1, 2))
    return (rg < 1.5) & (br > 6) & (br < 10)


def validate():
    rows = [r for f in sorted((ROOT / "artifacts/eda/observe_fresh_v6").glob("seed-*.pt"))[:300]
            for r in torch.load(f, weights_only=False)]
    frames = torch.cat([r["frames"] for r in rows])
    flag = torch.cat([r["history"][:, 4] > 0.5 for r in rows])
    got = asleep(frames)
    return {"frames": len(flag), "asleep_share": float(flag.float().mean()),
            "recall": float(got[flag].float().mean()), "false_positive_rate": float(got[~flag].float().mean())}


def corpus(path, shards):
    n = {"frames": 0, "asleep": 0, "dead_next_asleep": 0, "dead_next_awake": 0, "deaths": 0,
         "sleep_action": 0, "sleep_action_dead_next": 0}
    for s in sorted(Path(path).glob("shard-*.pt"))[:shards]:
        for e in torch.load(s, weights_only=False, mmap=True)["episodes"]:
            obs, act, term = e["observations"][:-1], e["actions_taken"], e["terminated"]
            flag = torch.cat([asleep(obs[i:i + 4096]) for i in range(0, len(obs), 4096)])
            n["frames"] += len(flag); n["asleep"] += int(flag.sum())
            n["dead_next_asleep"] += int((term & flag).sum()); n["dead_next_awake"] += int((term & ~flag).sum())
            n["deaths"] += int(term.sum())
            s_act = (act == 6) & ~flag
            n["sleep_action"] += int(s_act.sum()); n["sleep_action_dead_next"] += int((term & s_act).sum())
    awake = n["frames"] - n["asleep"]
    return n | {"p_dead_next_given_asleep": n["dead_next_asleep"] / max(n["asleep"], 1),
                "p_dead_next_given_awake": n["dead_next_awake"] / max(awake, 1),
                "p_dead_next_given_sleep_action_while_awake": n["sleep_action_dead_next"] / max(n["sleep_action"], 1)}


def zombie_roots():
    out = {}
    for b, store in (("55k", "observe_fresh_v6"), ("56k", "observe_fresh_v7")):
        rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
        rows = [r for r in rows if r["p_death1"].max() > r["p_death1"].min()]
        vis = torch.stack([r["visible"].float() for r in rows])
        zom = strata(vis)["zombie_adjacent"]
        energy = vis[:, 1515] * 9
        succ_asleep = asleep(torch.stack([r["successors"][6] for r in rows]))
        out[b] = {"zombie_roots": int(zom.sum()), "energy_below_9": float((energy[zom] < 9).float().mean()),
                  "sleep_successor_rendered_asleep": float(succ_asleep[zom].float().mean()),
                  "p_death1_sleep": float(torch.stack([r["p_death1"][6] for r in rows])[zom].mean()),
                  "p_death1_noop": float(torch.stack([r["p_death1"][0] for r in rows])[zom].mean())}
    return out


if __name__ == "__main__":
    out = {"validation": validate()}
    print(json.dumps(out), flush=True)
    out["zombie_roots"] = zombie_roots()
    print(json.dumps(out["zombie_roots"]), flush=True)
    out["expert_v1"] = corpus(ROOT / "artifacts/craftax_expert_store_v1", 12)
    print(json.dumps(out["expert_v1"]), flush=True)
    out["support_v2"] = corpus(ROOT / "artifacts/craftax_support_v2", 60)
    print(json.dumps(out["support_v2"]), flush=True)
    (HERE / "sleep.json").write_text(json.dumps(out, indent=2) + "\n")
