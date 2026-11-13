"""How often did each F5 training loss see each clock's event? (RESULTS.md, data counts.) CPU, no model.

Rebuilds F5's training stream exactly from run/gaze.bin: per update, 16 streams x S steps in chunk order, S = 64 for
updates 0..12499 and 256 after. Each row's TSP forecast window is rebuilt from the seed the trainer used,
default_rng([615001, world seed 1, update]).integers(49, size=(16, S)). Exact labels and counters come from
s0_train_labels.npz and 20261107_f4_followups/readings/counters_train.npz.

Per event, at transition t-1 -> t, with t >= 1 (forecasts skip an episode's first frame):
  - seen by the reward/continuation forecast: every frame;
  - seen by the TSP forecast: the forecast window covers the digit's cell;
  - seen by the reactive TSP: the learned fixation covers the cell. Context only; reactive TSP reads the frame.
  - credit reaches the evidence: the counter's previous reset (counter -> 0) lies inside the same chunk, i.e. its lag
    <= the event's step in the chunk. The belief carry crosses chunks, but gradients stop at the chunk boundary.
Cumulative counts are given at each of the ten checkpoints (2.56M frames each).

    python -m artifacts.experiments.20261111_p1f5.data_counts
"""
import json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, train as TR, chooser as C

HERE = Path(__file__).resolve().parent
STREAMS, S0, S1, U0, UPDATES = 16, 64, 256, 12500, 15625
CELL = {"health": 63, "food": 64, "drink": 65, "energy": 66}


def forecast_windows():
    out = np.empty(U0 * STREAMS * S0 + (UPDATES - U0) * STREAMS * S1, np.uint8)
    a = 0
    for u in range(UPDATES):
        S = S0 if u < U0 else S1
        w = np.random.default_rng([615001, 1, u]).integers(49, size=(STREAMS, S)).reshape(-1)
        out[a:a + len(w)] = w
        a += len(w)
    assert a == len(out) == 25_600_000
    return out


def row_layout(n):
    r = np.arange(n, dtype=np.int64)
    late = r >= U0 * STREAMS * S0
    rr = np.where(late, r - U0 * STREAMS * S0, r)
    S = np.where(late, S1, S0)
    u = np.where(late, U0 + rr // (STREAMS * S1), rr // (STREAMS * S0))
    step = rr % S
    return u, step


def last_reset(counter, ep_off):
    """Per state index: the most recent index j <= it where the counter was reset to 0 (c[j] == 0, c[j-1] != 0)."""
    reset = np.zeros(len(counter), bool)
    reset[1:] = (counter[1:] == 0) & (counter[:-1] != 0)
    reset[ep_off[:-1]] = True                                          # the episode start is the earliest evidence
    idx = np.where(reset, np.arange(len(counter)), -1)
    return np.maximum.accumulate(idx)


def main():
    ck = torch.load(HERE / "run/ckpt_10.pt", map_location="cpu", weights_only=False, mmap=True)
    eps = D.load_split(ck["cfg"])["train"]
    off = D.frame_offsets(eps)
    L = dict(np.load(TR.ROOT / "artifacts/eda/s0/s0_train_labels.npz"))
    Cn = dict(np.load(TR.ROOT / "artifacts/experiments/20261107_f4_followups/readings/counters_train.npz"))
    loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    lab_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    mapping = np.array([lab_of[(e.shard, e.slot)] for e in eps])
    for k in ("hunger", "thirst", "fatigue", "recover"):
        assert len(Cn[k]) == len(L["s_body"])
    reset = {k: last_reset(Cn[k], loff) for k in ("hunger", "thirst", "fatigue", "recover")}
    win = C.WIN.numpy()
    z = np.memmap(HERE / "run/gaze.bin", dtype=C.TRACE, mode="r")
    assert len(z) == 25_600_000
    fw = forecast_windows()
    u_all, step_all = row_layout(len(z))
    # Alignment check: inside a chunk row, consecutive steps are consecutive frames except at episode boundaries.
    tg0 = np.asarray(z["target"][:4096]).reshape(4, 16, 64)
    assert np.mean(np.diff(tg0, axis=-1) == 1) > .95

    events = {  # name: (digit column, sign, sleep condition, counter governing the clock)
        "food_drop_awake": (1, -1, False, "hunger"), "drink_drop_awake": (2, -1, False, "thirst"),
        "energy_drop_awake": (3, -1, False, "fatigue"), "energy_rise_asleep": (3, +1, True, "fatigue"),
        "health_rise": (0, +1, None, "recover"), "health_drop": (0, -1, None, None)}
    cell_of = {1: "food", 2: "drink", 3: "energy", 0: "health"}
    out = {name: {"all": 0, "credit": 0, "tsp_forecast": 0, "tsp_forecast_credit": 0, "fixated": 0,
                  "cumulative": {k: [0] * 10 for k in ("all", "credit", "tsp_forecast", "tsp_forecast_credit")},
                  "lag": []} for name in events}
    frames_counted = 0
    for a in range(0, len(z), 2_000_000):
        q = np.asarray(z[a:a + 2_000_000])
        u, step = u_all[a:a + len(q)], step_all[a:a + len(q)]
        rows = np.arange(a, a + len(q))
        tg = q["target"]
        ei = np.searchsorted(off, tg, side="right") - 1
        t = tg - off[ei]
        ok = (tg >= 0) & (t >= 1)
        q, u, step, rows, ei, t = q[ok], u[ok], step[ok], rows[ok], ei[ok], t[ok]
        frames_counted += len(q)
        li = loff[mapping[ei]] + t
        body, prev = L["s_body"][li].astype(int), L["s_body"][li - 1].astype(int)
        asleep = L["s_sleep"][li - 1].astype(bool)
        fwin, lwin = win[fw[rows]], win[q["window"].astype(int)]
        ck_ix = np.minimum(rows // 2_560_000, 9)
        for name, (col, sign, sleep, counter) in events.items():
            m = np.sign(body[:, col] - prev[:, col]) == sign
            if sleep is not None:
                m &= asleep == sleep
            if not m.any():
                continue
            cell = CELL[cell_of[col]]
            fcov = (fwin[m] == cell).any(-1)
            lcov = (lwin[m] == cell).any(-1)
            if counter is not None:
                lag = (li[m] - 1) - reset[counter][li[m] - 1]                # steps since the previous reset, at t-1
                lag = lag + 1                                                 # the reset frame's input is lag steps before t
                credit = lag <= step[m]
                out[name]["lag"].append(lag)
            else:
                credit = np.zeros(int(m.sum()), bool)
            o = out[name]
            o["all"] += int(m.sum()); o["credit"] += int(credit.sum())
            o["tsp_forecast"] += int(fcov.sum()); o["tsp_forecast_credit"] += int((fcov & credit).sum())
            o["fixated"] += int(lcov.sum())
            ci = ck_ix[m]
            for key, mask in (("all", np.ones(len(ci), bool)), ("credit", credit), ("tsp_forecast", fcov),
                              ("tsp_forecast_credit", fcov & credit)):
                o["cumulative"][key] = (np.array(o["cumulative"][key]) + np.bincount(ci[mask], minlength=10)).tolist()
        print("rows", a + len(q), {k: v["all"] for k, v in out.items()}, flush=True)
    res = {"frames_counted": int(frames_counted), "passes_over_corpus": float(frames_counted / off[-1]),
           "corpus_frames": int(off[-1])}
    for name, o in out.items():
        lag = np.concatenate(o.pop("lag")) if o["lag"] else np.array([])
        o["cumulative"] = {k: np.cumsum(v).tolist() for k, v in o["cumulative"].items()}
        o["rate_per_frame"] = o["all"] / frames_counted
        if len(lag):
            o["lag_quantiles_10_50_90"] = np.quantile(lag, [.1, .5, .9]).tolist()
        res[name] = o
    (HERE / "readings/dissection/data_counts.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "cumulative"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
