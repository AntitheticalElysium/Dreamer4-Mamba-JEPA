"""S2's terminal-inclusive DEV bank (S2_PROPOSAL L1; admission contract of ../20261102_p1f15_readiness/P2_DATA.md,
applied to DEV). CPU only, no simulator stepping, no model.

A start is admitted from its PRECOMMITTED exogenous flags over the whole H-step horizon, regenerated from the original
2,500-step seed tree (so flags after the recorded death count too), whatever the recorded future's length: it must lie
before the recorded end; an episode that ended by time truncation admits only complete windows (a truncation is never
padded as a death). The planned actions of the whole horizon are saved with each start, so a branch that outlives the
recorded death continues the same schedule. Nothing about the future (reward, survival, events) enters admission.

Saved: every admitted sliding start, the seeded uniform evaluation sample (SAMPLE starts; uniform over starts keeps
the terminal share), and the legacy stride-H complete-window bank as a labelled comparator. Checks: regenerated random
actions and flags equal the stored labels; the validator accepts the terminal-inclusive sample and REJECTS the legacy
bank as survival-censored.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261114_s2_harness.bank
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import importlib
import json
from pathlib import Path

import numpy as np

from saccade import data as D
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
A = importlib.import_module("artifacts.experiments.20261023_system_design.eda_exogenous_steps")
H, SAMPLE, SEED = 16, 4000, 20261114
COLUMNS = ("episode", "shard", "slot", "start", "recorded_T", "recorded_death")


def validate(rows, h=H):
    """A bank must contain recorded deaths at every position 1..h of its windows (P2_DATA: reject a survival-censored
    bank). Returns (accepted, deaths by position)."""
    rel = rows[:, 4] - rows[:, 3]
    dead = rows[:, 5].astype(bool) & (rel >= 1) & (rel <= h)
    by_pos = np.bincount(rel[dead] - 1, minlength=h)
    return bool((by_pos > 0).all()), by_pos


def main():
    f = np.load(S.LABELS / "s0_dev_labels.npz")
    L = {k: f[k] for k in ("ep_shard", "ep_slot", "ep_T", "t_exogenous", "t_terminated", "t_action")}
    ao = np.r_[0, np.cumsum(L["ep_T"])]
    rows, plans, legacy = [], [], []
    checked = matched = flag_steps = 0
    last, cache = -1, {}
    for e, (sh, sl, T) in enumerate(zip(L["ep_shard"], L["ep_slot"], L["ep_T"])):
        sh, sl, T = int(sh), int(sl), int(T)
        if sh != last:
            _, keys = D.RP.round_keys(sh)                                # always the full 2,500 keys: the same key tree
            cache, last = {}, sh
        eps = float(D.RP.episode_fields(sh, sl)["epsilon"])
        if eps not in cache:
            cache[eps] = tuple(np.asarray(x) for x in A.flags(keys, eps))
        ex, acts = cache[eps][0][:, sl], cache[eps][1][:, sl]
        observed = L["t_action"][ao[e]:ao[e + 1]]
        assert np.array_equal(ex[:T], L["t_exogenous"][ao[e]:ao[e + 1]])
        checked += int(ex[:T].sum()); matched += int((acts[:T][ex[:T]] == observed[ex[:T]]).sum()); flag_steps += T
        done = bool(L["t_terminated"][ao[e + 1] - 1])
        bad = np.r_[0, np.cumsum(~ex)]
        starts = np.flatnonzero((bad[H:] - bad[:-H]) == 0)               # H precommitted random flags, inside 2,500 steps
        starts = starts[starts < T]
        if not done:
            starts = starts[starts + H <= T]
        for t in starts:
            rows.append((e, sh, sl, int(t), T, int(done)))
            plans.append(acts[t:t + H])
        legacy += [(e, sh, sl, t0, T, int(done)) for t0 in range(0, T - H + 1, H) if L["t_exogenous"][ao[e] + t0:ao[e] + t0 + H].all()]
    assert checked == matched
    rows, plans, legacy = np.array(rows, np.int32), np.array(plans, np.int8), np.array(legacy, np.int32)
    pick = np.sort(np.random.default_rng(SEED).choice(len(rows), SAMPLE, replace=False))
    ok, by_pos = validate(rows[pick])
    ok_legacy, legacy_pos = validate(legacy)
    assert ok and not ok_legacy, (by_pos, legacy_pos)
    for i in pick[:200]:                                                 # planned actions equal the recorded ones while alive
        e, _, _, t, T, _ = rows[i]
        n = min(H, T - t)
        assert np.array_equal(plans[i][:n], L["t_action"][ao[e] + t:ao[e] + t + n])
    early = rows[:, 3] + H > rows[:, 4]
    out = {"horizon": H, "episodes": int(len(L["ep_T"])), "episodes_with_starts": int(len(np.unique(rows[:, 0]))),
           "random_steps_checked": checked, "random_action_matches": matched, "exogenous_flag_steps_checked": flag_steps,
           "admitted_starts": int(len(rows)), "complete_window_starts": int((~early).sum()),
           "early_terminal_starts": int(early.sum()), "deaths_by_position_all": validate(rows)[1].tolist(),
           "sample": {"starts": SAMPLE, "seed": SEED, "episodes": int(len(np.unique(rows[pick, 0]))),
                      "early_terminal": int(early[pick].sum()), "deaths_by_position": by_pos.tolist(), "accepted": ok},
           "legacy_stride_bank": {"windows": int(len(legacy)), "deaths_by_position": legacy_pos.tolist(),
                                  "accepted": ok_legacy}}
    np.savez_compressed(HERE / "dev_bank_h16.npz", rows=rows, plans=plans, sample=pick, legacy=legacy,
                        columns=np.array(COLUMNS))
    (HERE / "dev_bank_h16.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
