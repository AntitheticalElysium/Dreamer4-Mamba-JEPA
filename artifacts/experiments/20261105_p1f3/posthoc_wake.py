"""Post-hoc (11-06, user: the reward mismatch): where the own-history pilot's real-history ("teacher") false rewards
come from, on existing outputs only (recursion raw.npz at 16 steps, F2 and F3) against the dev labels (simulator state:
s_sleep, s_body energy, s_ach). Craftax Classic (game_logic.py): asleep, every action becomes NOOP; the agent wakes when
energy >= 9 (WAKE_UP unlock, first time only) or when hit. Writes readings/posthoc_wake.json.

Usage: python posthoc_wake.py"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RAW = {"f2": ROOT / "artifacts/experiments/20261105_consequence_trace/readings/recursion/f2_steps16/raw.npz",
       "f3": HERE / "readings/recursion_steps16/raw.npz"}
WAKE_UP = 15                                                             # constants.Achievement.WAKE_UP


def main():
    L = np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")
    ao, so = np.r_[0, np.cumsum(L["ep_T"])], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"]); s = so[ep] + np.arange(len(ep)) - ao[ep]
    asleep, woke, energy = L["s_sleep"][s], L["s_sleep"][s] & ~L["s_sleep"][s + 1], L["s_body"][s][:, 3]
    out = {"labels_rule": {"p_wake_asleep_energy9": float(woke[asleep & (energy >= 9)].mean()),
                           "p_wake_asleep_energy_below9": float(woke[asleep & (energy < 9)].mean())}}
    for name, path in RAW.items():
        z = np.load(path)
        st = so[z["episode"]][:, None] + z["t0"][:, None] + np.arange(16)[None]   # state before generated step j
        tp, pos = z["truth_reward"] > 0.5, z["teacher__positive"]
        sl, stay = L["s_sleep"][st], L["s_sleep"][st] & L["s_sleep"][st + 1]
        done, en = L["s_ach"][st][..., WAKE_UP], L["s_body"][st][..., 3]
        fp = pos * ~tp
        m = stay & ~tp
        out[name] = {"mean_positive": float(pos.mean()), "truth_positive": float(tp.mean()),
                     "false_mass_share_asleep": float(fp[sl].sum() / fp.sum()),
                     "positive_truth_negative_asleep": float(pos[~tp & sl].mean()), "positive_truth_negative_awake": float(pos[~tp & ~sl].mean()),
                     "asleep_stays_asleep": {"n": int(m.sum()), "wake_up_not_yet": float(pos[m & ~done].mean()), "wake_up_done": float(pos[m & done].mean()),
                                             "by_energy": {int(k): [int((m & (en == k)).sum()), float(pos[m & (en == k)].mean())] for k in np.unique(en[m])}},
                     "true_wake_steps": {"n": int((sl & ~L["s_sleep"][st + 1]).sum()), "positive": float(pos[sl & ~L["s_sleep"][st + 1]].mean()),
                                         "truth_rewarded": float(tp[sl & ~L["s_sleep"][st + 1]].mean())}}
    (HERE / "readings/posthoc_wake.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
