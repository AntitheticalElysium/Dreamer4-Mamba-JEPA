"""Denoiser probe (PLAN.md 10): does the hit reach the health token inside F4's denoiser? The production path
(interface.futures through determinant_curves.generate, one future per row) on every adjacent-zombie dev row, with
forward hooks that leave every output untouched: the health and lighting tokens' residual at the input and after each
block of the first evaluation (pure noise), the health token after the last block and its x-prediction at each of the
K evaluations. Linear probes as belief_probe.py; the generated health digit by exp 2's readers.
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.denoiser_probe CKPT OUT"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

DC = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_checks")
CV, RT = DC.CV, DC.RT
HEALTH, LIGHT = 63, M.N + M.G


def main(ckpt, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1" and not DC.OLD
    model, cfg = DC.load(ckpt, 4)
    den, n_eval = model.denoiser, cfg["edm"]["sampler"]["steps"]
    n_tok, n_blk = M.N + M.G + 1 + den.K, len(den.blocks)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    with torch.enable_grad():
        R, _ = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(ckpt).parent / "s1/results.reader.pt", TR.ROOT / ckpt, CV.DEV)
    adj = T["adjacent"] & (T["t"] >= CV.P)
    pops = {"awake": np.nonzero(adj & ~T["asleep"])[0], "asleep": np.nonzero(adj & T["asleep"])[0]}
    CV.K = 1                                                             # one future per row (generate reads CV.K)
    sham = CV.generate(model, cfg, enc, L, T, pops["awake"][:256], 4600, R, s1)   # before any hook: batch 0 alone

    st = {"on": False, "pass": -1, "got": []}
    embed, head, sample = den.embed, den.head, M.sample

    def embed_(*a):                                                      # counts the evaluations of one sample call
        tok, ang = embed(*a)
        if st["on"]:
            st["pass"] += 1
            if st["pass"] == 0:
                st["cur"] = {"input": tok[:, 0, [HEALTH, LIGHT]].float().cpu()}
        return tok, ang

    def hook(i):
        def f(mod, args, output):
            x = output[0]
            if st["on"] and x.shape[2] == n_tok and st["pass"] < n_eval:     # the K physical evaluations, not the glimpse stage
                if st["pass"] == 0:
                    st["cur"][f"block{i + 1}"] = x[:, 0, [HEALTH, LIGHT]].float().cpu()
                if i == n_blk - 1:
                    st["cur"][f"last_block_eval{st['pass']}"] = x[:, 0, HEALTH].float().cpu()
        return f

    def head_(h, c):
        o = head(h, c)
        if st["on"] and st["pass"] < n_eval:
            st["cur"][f"output_eval{st['pass']}"] = o[0][:, 0, HEALTH].float().cpu()
        return o

    def sample_(*a, **k):
        st["on"], st["pass"] = True, -1
        try:
            return sample(*a, **k)
        finally:
            st["on"] = False
            st["got"].append(st.pop("cur", None))

    den.embed, den.head, M.sample = embed_, head_, sample_
    hooks = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(den.blocks)]
    res = {"checkpoint": ckpt, "tree": "canonical", "evaluations": n_eval, "populations": {}}
    try:
        for i, (name, rows) in enumerate(pops.items()):
            st["got"] = []
            o = CV.generate(model, cfg, enc, L, T, rows, 4600 + i, R, s1)
            cap = {k: torch.cat([g[k] for g in st["got"]]).double() for k in st["got"][0]}
            j = o["j"]
            assert np.array_equal(j, rows) and len(cap["input"]) == len(j), "capture misaligned with the rows"
            if i == 0:
                res["sham_batch0_equal"] = {k: bool(np.array_equal(sham[k], o[k][:256])) for k in ("gen_hud63", "gen_asleep")}
                print("sham", res["sham_batch0_equal"], flush=True)
            y, ep = (T["true_body"][j, 0] < T["body"][j, 0]).astype(float), T["ep"][j]
            frame = lambda key, dt: torch.stack([enc[int(T["ep"][x])][key][int(T["t"][x]) + dt] for x in j])
            bp = frame("bp", 0)
            with torch.no_grad():
                mem = torch.cat([model.belief.memory(bp[a:a + 4096].float().to(CV.DEV)).flatten(1).float().cpu()
                                 for a in range(0, len(j), 4096)]).double()
            feats = {"belief_bp": bp.double(), "memory_tokens": mem,
                     "action_only": torch.nn.functional.one_hot(torch.as_tensor(T["action"][j]), 17).double()}
            for place in ["input"] + [f"block{b + 1}" for b in range(n_blk)]:
                feats[f"health@{place}"], feats[f"lighting@{place}"] = cap[place][:, 0], cap[place][:, 1]
            feats.update({f"health@last_block_eval{e}": cap[f"last_block_eval{e}"] for e in range(1, n_eval)})
            feats.update({f"health_output_eval{e}": cap[f"output_eval{e}"] for e in range(n_eval)})

            def probe(X):
                folds, s = RT.folds_of(ep), np.zeros(len(y))
                for f in range(5):
                    s[folds == f] = RT.logistic(X[folds != f], y[folds != f], X[folds == f])
                return s
            scores, ap = {}, {}
            for k, X in feats.items():
                scores[k] = probe(X)
                ap[k] = RT.ap_boot(y, scores[k], ep)
                print(name, k, round(ap[k]["ap"], 3), flush=True)
            base, ref = ap["action_only"]["ap"], ap["memory_tokens"]["ap"]
            copy, truth = frame("p", -1)[:, HEALTH].double(), frame("p", 0)[:, HEALTH].double()
            d2 = (truth - copy).square().sum(-1)
            hit = (y > 0) & (d2.numpy() > 1e-6)
            lean = lambda xh: (((xh - copy) * (truth - copy)).sum(-1) / d2.clamp(min=1e-12)).numpy()[hit]
            drawn = (o["gen_hud63"] < T["body"][j, 0]).astype(float)
            res["populations"][name] = {
                "rows": int(len(j)), "episodes": int(len(np.unique(ep))), "prevalence": float(y.mean()),
                "ap": ap, "share (ap - action) / (memory tokens - action)": {k: (v["ap"] - base) / (ref - base) for k, v in ap.items()},
                "lean on hit rows (0 = copy, 1 = the hit's token)": {f"eval{e}": RT.boot(lean(cap[f"output_eval{e}"]), ep[hit]) for e in range(n_eval)},
                "calibrated lean at eval0 (mean out-of-fold memory-token probability on hit rows)": RT.boot(scores["memory_tokens"][hit], ep[hit]),
                "hit rows with a token change": int(hit.sum()),
                "drawn drop": {"hit rows": RT.boot(drawn[y > 0], ep[y > 0]), "no-hit rows": RT.boot(drawn[y == 0], ep[y == 0])}}
            print(name, json.dumps({k: round(v, 2) for k, v in res["populations"][name]["share (ap - action) / (memory tokens - action)"].items()}), flush=True)
    finally:
        den.embed, den.head, M.sample = embed, head, sample
        for h in hooks:
            h.remove()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
