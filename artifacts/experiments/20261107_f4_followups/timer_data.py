"""Timer data, exactly (PLAN.md 11): from the replayed counters (counters.py) and the labels of P1's training episodes
(the train split less the monitor episodes, saccade.data.load_split). A counter's reset is any change other than its
normal step (hunger, thirst: +1 awake / +0.5 asleep; fatigue: +1 / -1; recover: +1, +2, -1, -0.5); it is visible when
its digit changes at that step (fatigue: or the sleep state). A timer event is a reset that moves its digit by the
rule. Per event: the steps since the counter's previous reset and since its previous VISIBLE reset (the episode start
counts as one), and the chance that the latter falls in the same training chunk (64 / 256 steps, uniform offset).
Usage: python -m artifacts.experiments.20261107_f4_followups.timer_data   (CPU)"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import yaml

from saccade import data as D

HERE = Path(__file__).resolve().parent
LABELS = HERE.parents[1] / "eda/s0"
NORMAL = {"hunger": (1.0, 0.5), "thirst": (1.0, 0.5), "fatigue": (1.0, -1.0), "recover": (1.0, 2.0, -1.0, -0.5)}
DIGIT = {"hunger": 1, "thirst": 2, "fatigue": 3, "recover": 0}         # s_body columns: health food drink energy
EVENTS = {"food_down": ("hunger", 1, -1), "drink_down": ("thirst", 2, -1), "energy_down_awake": ("fatigue", 3, -1),
          "energy_up_asleep": ("fatigue", 3, 1), "health_up_recovery": ("recover", 0, 1), "health_down_needs": ("recover", 0, -1)}


def main():
    L, C = np.load(LABELS / "s0_train_labels.npz"), np.load(HERE / "readings/counters_train.npz")
    cfg = yaml.safe_load((HERE.parents[2] / "saccade/m0.yaml").read_text())
    split = D.load_split(cfg)
    keep = {(e.shard, e.slot) for e in split["train"]}
    eps_in = [(int(s), int(t)) in keep for s, t in zip(L["ep_shard"], L["ep_slot"])]
    T, s_off = L["ep_T"], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    body, sleep, inv = L["s_body"], L["s_sleep"], L["s_inv"]
    res = {"episodes": int(sum(eps_in)), "episodes_in_labels": int(len(T)), "transitions": int(sum(T[e] for e in range(len(T)) if eps_in[e])),
           "counter_resets": {}, "events": {}}
    ev = {k: [] for k in EVENTS}
    resets = {k: {"visible": 0, "invisible": 0} for k in NORMAL}
    other = {"health_down_other": 0, "coal_up": 0}
    for e in np.nonzero(eps_in)[0]:
        lo, hi = s_off[e], s_off[e + 1]
        b, sl = body[lo:hi], sleep[lo:hi]
        d = np.diff(b, axis=0)
        recover_reset = None
        for k in NORMAL:
            c = C[k][lo:hi]
            dc = np.diff(c)
            reset = ~np.isin(np.round(dc * 2) / 2, NORMAL[k])                # step s -> s + 1, indexed by s + 1
            vis = d[:, DIGIT[k]] != 0
            if k == "fatigue":
                vis |= sl[1:] != sl[:-1]
            resets[k]["visible"] += int((reset & vis).sum()); resets[k]["invisible"] += int((reset & ~vis).sum())
            steps = np.nonzero(reset)[0] + 1
            vsteps = np.r_[0, np.nonzero(reset & vis)[0] + 1]
            if k == "recover":
                recover_reset = reset
            for name, (kk, col, sgn) in EVENTS.items():
                if kk != k:
                    continue
                for s in np.nonzero(reset & (d[:, col] == sgn))[0] + 1:
                    if name == "energy_up_asleep" and not sl[s - 1]:
                        continue
                    if name == "energy_down_awake" and sl[s - 1]:
                        continue
                    prev = steps[steps < s]
                    pv = vsteps[vsteps < s]
                    ev[name].append((s - (prev[-1] if len(prev) else 0), s - pv[-1], float(L["ep_eps"][e])))
        other["health_down_other"] += int(((d[:, 0] < 0) & ~recover_reset).sum())
        other["coal_up"] += int((np.diff(inv[lo:hi, 2]) > 0).sum())
    res["counter_resets"] = resets
    for name, rows in ev.items():
        a = np.array(rows)
        gap, gv, eps = a[:, 0], a[:, 1], a[:, 2]
        mode = np.bincount(gap.astype(int)).argmax()
        res["events"][name] = {
            "count": int(len(a)), "per_1000_transitions": 1000 * len(a) / res["transitions"],
            "gap since previous reset: mode, share at mode, median": [int(mode), float((gap == mode).mean()), float(np.median(gap))],
            "share whose previous reset was invisible": float((gv > gap).mean()),
            "gap since previous VISIBLE reset: median, p90": [float(np.median(gv)), float(np.percentile(gv, 90))],
            "share with previous visible reset <= 8 steps back": float((gv <= 8).mean()),
            "chance the previous visible reset is in the same chunk (64 / 256)": [float(np.clip(1 - gv / 64, 0, 1).mean()), float(np.clip(1 - gv / 256, 0, 1).mean())],
            "per 1000 transitions by policy epsilon": {str(x): float(1000 * (eps == x).sum() / max(1, sum(T[i] for i in np.nonzero(eps_in)[0] if L["ep_eps"][i] == x)))
                                                       for x in np.unique(L["ep_eps"])}}
    res["comparison"] = {k: {"count": v, "per_1000_transitions": 1000 * v / res["transitions"]} for k, v in other.items()}
    (HERE / "readings/danger/timer_data.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
