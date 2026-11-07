"""Belief gradient (PLAN.md 12): what each loss asks of the belief at health changes, against the wake. Frozen F4 on
training chunks built as P1 builds them (Streams over P1's training split, carried Mamba state, change targets, one
bootstrap row in four past bootstrap_start), the training's loss terms (train.p1_losses) each backpropagated alone with
its harmoniser weight; G1 split by output token through a gradient mask on the denoiser's x-prediction. Per chunk
frame: |dL/d bp_t| per pass and the pass token's incoming error |dL/d x_hat|; per pass and frame, the gradient on the
belief's input values per periphery token (|dL/d x_token| entering percept_in; PLAN 12 amendment: its weight gradient
scales with each column's activation and cannot say which digit a loss asks for). Frames classed from the train labels and the replayed counters. The bootstrap row is left out of
the readings (its G1 memory is detached, as in training).
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.belief_gradient CKPT OUT N_MICRO"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
HERE = Path(__file__).resolve().parent
TOKENS = {"health": 63, "food": 64, "drink": 65, "energy": 66, "player": 31}
CLASSES = ("recovery", "need_damage", "damage", "wake", "sleep_onset", "food_down", "energy_up_asleep", "ordinary")
STREAMS, STEPS = 4, 64


def main(ckpt, out, n_micro):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(ckpt)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    dev, P = CV.DEV, cfg["data"]["context_prefix"]
    eps = D.load_split(cfg)["train"]
    targets = np.load(TR.ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    streams = D.Streams(eps, STREAMS, P, seed=2117, targets_offsets=D.frame_offsets(eps))
    L, C = np.load(CV.LABELS / "s0_train_labels.npz"), np.load(HERE / "readings/counters_train.npz")
    lab = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    body, sleep, inv, rec = L["s_body"], L["s_sleep"], L["s_inv"], C["recover"]
    w = (1 / model.harmoniser.log_sigma.detach().exp()).tolist()        # G1, G2, G3 (train.groups order)
    carry = model.belief.initial(STREAMS, dev)
    cache = torch.zeros(STREAMS, P + 1, cfg["belief"]["width"], device=dev)
    st = {"mode": None, "err": None, "y": None, "recs": None, "xin": None}

    gather = D.gather

    def gather_(episodes, recs, offsets=None):                           # the records behind each chunk frame
        st["recs"] = recs
        return gather(episodes, recs, offsets)

    def keep_y(mod, args, output):
        output[0].retain_grad()
        st["y"] = output[0]

    def keep_x(mod, args, output):                                       # the belief's input vector, [B, T, ...]
        args[0].retain_grad()
        st["xin"] = args[0]

    def mask(i):
        def f(g):
            m = st["mode"]
            if m is None or m == "total":
                return g
            if i != 0:
                return g if m == "rest" else torch.zeros_like(g)
            keep = torch.ones(g.shape[-2], device=g.device, dtype=g.dtype)
            if m == "rest":
                keep[list(TOKENS.values())] = 0
            else:
                keep.zero_()[TOKENS[m]] = 1
                st["err"] = g[..., TOKENS[m], :].float().norm(dim=-1).detach().cpu()   # [B, P + T]
            return g * keep[:, None]
        return f

    fwd = model.denoiser.forward

    def fwd_(*a, **k):
        o = fwd(*a, **k)
        if torch.is_grad_enabled():
            for i, t in enumerate(o):
                if t.requires_grad:
                    t.register_hook(mask(i))
        return o

    def classify(recs, boot_row):
        f = {c: np.zeros((STREAMS, STEPS), bool) for c in CLASSES}
        for i in range(STREAMS):
            for k in range(1, STEPS):
                r = recs[i][P + k]
                if i == boot_row or r is None or r[2] == 0:
                    continue
                ep = eps[r[0]]
                s = s_off[lab[(ep.shard, ep.slot)]] + r[2]
                dh = body[s, 0] - body[s - 1, 0]
                reset = not np.isin(np.round((rec[s] - rec[s - 1]) * 2) / 2, (1.0, 2.0, -1.0, -0.5))
                f["recovery"][i, k] = dh == 1 and reset
                f["need_damage"][i, k] = dh == -1 and reset
                f["damage"][i, k] = dh < 0 and not reset
                f["wake"][i, k] = sleep[s - 1] and not sleep[s]
                f["sleep_onset"][i, k] = sleep[s] and not sleep[s - 1]
                f["food_down"][i, k] = body[s, 1] - body[s - 1, 1] == -1
                f["energy_up_asleep"][i, k] = body[s, 3] - body[s - 1, 3] == 1 and sleep[s - 1]
                f["ordinary"][i, k] = (body[s] == body[s - 1]).all() and (inv[s] == inv[s - 1]).all() and sleep[s] == sleep[s - 1]
        return {c: torch.as_tensor(v) for c, v in f.items()}

    passes = ([(f"g1/{k}", ("g1", 0), k) for k in TOKENS] + [("g1/rest", ("g1", 0), "rest")]
              + [(t, (t, g), None) for t, g in (("change", 1), ("tsp", 1), ("reward", 2), ("cont", 2))] + [("total", None, "total")])
    acc = {p[0]: {c: [0.0, 0.0, 0] for c in CLASSES} for p in passes}   # sum |dL/d bp|, sum token error, frames
    sens = {p[0]: {c: torch.zeros(M.N) for c in CLASSES + ("all",)} for p in passes}   # sum |dL/d x_token| per token
    n_all = 0
    D.gather, model.denoiser.forward = gather_, fwd_
    hb = model.belief.register_forward_hook(keep_y)
    hx = model.belief.percept_in.register_forward_hook(keep_x)
    try:
        for n in range(n_micro):
            chunk = streams.next(STEPS)
            boot = np.zeros(STREAMS, bool)
            boot[n % STREAMS] = True
            terms, c_new, cache_new = TR.p1_losses(model, chunk, carry, cache, targets, cfg, boot, 10 ** 6)
            y, flags = st["y"], classify(st["recs"], n % STREAMS)
            for name, term, mode in passes:
                st["mode"], st["err"] = mode, None
                model.zero_grad(set_to_none=True)
                y.grad = None
                st["xin"].grad = None
                loss = model.harmoniser(TR.groups(terms)) if term is None else terms[term[0]] * w[term[1]]
                loss.backward(retain_graph=True)
                gb = torch.zeros(STREAMS, STEPS)
                if y.grad is not None:
                    gb[:, 1:] = y.grad[:, 1::2][:, :-1].float().norm(dim=-1).cpu()  # bp of new frame k = nxt[k - 1]
                for c in CLASSES:
                    m = flags[c]
                    acc[name][c][0] += float(gb[m].sum())
                    acc[name][c][2] += int(m.sum())
                    if st["err"] is not None:
                        acc[name][c][1] += float(st["err"][:, P:][m].sum())
                gx = torch.zeros(STREAMS, STEPS, M.N)
                if st["xin"].grad is not None:
                    gx = st["xin"].grad[..., :M.N * 16].float().reshape(STREAMS, STEPS, M.N, 16).norm(dim=-1).cpu()
                live = torch.ones(STREAMS, STEPS, dtype=torch.bool)
                live[n % STREAMS] = False
                sens[name]["all"] += gx[live].sum(0)
                for c in CLASSES:
                    sens[name][c] += gx[flags[c]].sum(0)
            M.carry_put(carry, list(range(STREAMS)), c_new)
            cache = cache_new
            n_all += int((STREAMS - 1) * STEPS)
            del terms, c_new, y
            st["y"], st["xin"] = None, None
            if n % 10 == 0:
                print(f"micro {n}/{n_micro}", {c: acc["total"][c][2] for c in CLASSES}, flush=True)
    finally:
        D.gather, model.denoiser.forward = gather, fwd
        hb.remove(), hx.remove()
    ordinary = {name: acc[name]["ordinary"][0] / max(1, acc[name]["ordinary"][2]) for name, _, _ in passes}
    res = {"checkpoint": ckpt, "microbatches": n_micro, "harmoniser_weights": w, "frames": {c: acc["total"][c][2] for c in CLASSES},
           "mean |dL/d bp| (pass / class)": {}, "excess over ordinary (pass / class)": {}, "transmission (single-token G1 pass / class)": {},
           "mean token error (single-token G1 pass / class)": {}, "input sensitivity, all frames (pass / token)": {},
           "input sensitivity at class frames (pass / class / token)": {}}
    for name, _, mode in passes:
        for c in CLASSES:
            s, e, k = acc[name][c]
            key = f"{name} / {c}"
            res["mean |dL/d bp| (pass / class)"][key] = s / max(1, k)
            res["excess over ordinary (pass / class)"][key] = s / max(1, k) - ordinary[name]
            if mode in TOKENS:
                res["transmission (single-token G1 pass / class)"][key] = s / e if e > 0 else None
                res["mean token error (single-token G1 pass / class)"][key] = e / max(1, k)
        map_tokens = [c for c in range(63) if c != 31]
        tok = lambda v: {**{k: float(v[c]) for k, c in TOKENS.items()}, "map median": float(v[map_tokens].median())}
        res["input sensitivity, all frames (pass / token)"][name] = tok(sens[name]["all"] / n_all)
        for c in CLASSES:
            res["input sensitivity at class frames (pass / class / token)"][f"{name} / {c}"] = tok(sens[name][c] / max(1, acc[name][c][2]))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("frames", "input sensitivity, all frames (pass / token)")}, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
