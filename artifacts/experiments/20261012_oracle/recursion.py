"""10-12 oracle-innovation test, recursion analysis (POST HOC; the predeclared recursion reading failed for oracle_deltairis: completion
is at the reader ceiling teacher-forced, but self-fed with TRUE layouts the decoded cow count grows to 1.4-1.5x by depth 5-16).
Where do the extra mobs come from? On the self-fed predictions saved by oracle_eval.py (pred_<name>.pt), scroll-aligned steps
only, every decoded-present mob (zombie, skeleton, cow) at depth k that the truth (= the oracle input) does not have is classified:
  persist_true    the imagined previous frame (k - 1; the true root at k = 1) decoded it at the same cell AND it was true there
                  (the residue of a mob that left)
  persist_false   ... decoded it at the same cell and it was already false there (a self-sustaining false positive)
  adjacent        not persisting, but a true mob of that type is within one cell (4-neighbourhood) at k
  spontaneous     neither
Also the probe score of false vs true positives (how strongly they are drawn), and new false positives per step.
Usage: recursion.py <name> ...  -> artifacts/eda/oracle_v1/recursion_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
for d in ("20260927_levers", "20260926_diagnosis", "20260921_readout_ladder"):
    sys.path.insert(0, str(ROOT / "artifacts/experiments" / d))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20261012_oracle"))
import oracle_eval as OE  # noqa: E402
OUT = OE.OUT
H = 16


def main():
    import teval as T
    import scoreboard as SB
    panel = torch.load(OUT / "panel.pt", weights_only=False)
    tok, lay = panel["tokens"], panel["layouts"]
    R = len(tok)
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache("raw", torch.device("cpu"))
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    thr = torch.tensor(probes.thr)

    def scores(frames):                                                    # [n,81,192] -> probe scores [n,63,3] (zombie, skeleton, cow)
        cells = frames[:, OE.MAPC].float().flatten(0, 1)
        m = probes.mob(cells).view(len(frames), 62, 4)
        out = torch.full((len(frames), 63, 3), -9.0)
        for j, t in enumerate(OE.TYPES):
            out[:, OE.MAPC, j] = m[..., OE.PROBE_CH[t]]
        return out
    thr3 = torch.tensor([float(thr[OE.PROBE_CH[t]]) for t in OE.TYPES])
    truth = torch.stack([lay[..., OE.LAYOUT_CH[t]] > 0 for t in OE.TYPES], -1)          # [R,32,63,3]
    truth[:, :, 31] = False

    def nb4(x):                                                            # [...,63,3] bool -> any 4-neighbour true
        g = x.view(*x.shape[:-2], 7, 9, x.shape[-1])
        o = torch.zeros_like(g)
        o[..., 1:, :, :] |= g[..., :-1, :, :]; o[..., :-1, :, :] |= g[..., 1:, :, :]
        o[..., :, 1:, :] |= g[..., :, :-1, :]; o[..., :, :-1, :] |= g[..., :, 1:, :]
        return o.view_as(x)
    for name in sys.argv[1:]:
        pr = torch.load(OUT / f"pred_{name}.pt", weights_only=False)
        gen, aligned = pr["gen"].float(), pr["aligned"]
        sc = scores(gen.flatten(0, 1)).view(R, H, 63, 3)
        dec = sc > thr3
        root_dec = scores(tok[:, 15].float()) > thr3                                    # [R,63,3]
        prev_dec = torch.cat([root_dec[:, None], dec[:, :-1]], 1)
        tru = truth[:, 16:32]; prv_tru = truth[:, 15:31]
        res = {"world": name}
        for j, t in enumerate(OE.TYPES):
            rows = {}
            for g, ks in {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}.items():
                sel = aligned[:, ks][..., None].expand(-1, -1, 63)
                d, tr = dec[:, ks][..., j] & sel, tru[:, ks][..., j]
                fp = d & ~tr
                pdec = prev_dec[:, ks][..., j]
                pdec_true = pdec & prv_tru[:, ks][..., j]
                persist_t = fp & pdec_true
                persist_f = fp & pdec & ~pdec_true
                adj = fp & ~pdec & nb4(tru[:, ks])[..., j]
                spont = fp & ~pdec & ~nb4(tru[:, ks])[..., j]
                n_fp = int(fp.sum()); n_tp = int((d & tr).sum())
                s = sc[:, ks][..., j]
                rows[g] = {"true_positives": n_tp, "false_positives": n_fp,
                           "fp_per_true_mob": round(n_fp / max(int((tr & sel).sum()), 1), 3),
                           "persist_true": round(int(persist_t.sum()) / max(n_fp, 1), 3),
                           "persist_false": round(int(persist_f.sum()) / max(n_fp, 1), 3),
                           "adjacent": round(int(adj.sum()) / max(n_fp, 1), 3),
                           "spontaneous": round(int(spont.sum()) / max(n_fp, 1), 3),
                           "score_fp_mean": round(float(s[fp].mean()), 3) if n_fp else None,
                           "score_tp_mean": round(float(s[d & tr].mean()), 3) if n_tp else None}
            res[t] = rows
        res["threshold"] = [round(float(x), 3) for x in thr3]
        (OUT / f"recursion_{name}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
