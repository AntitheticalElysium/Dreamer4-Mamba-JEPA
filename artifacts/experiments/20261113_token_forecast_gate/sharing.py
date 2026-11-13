"""Does a head's forecast mode hurt or help its reactive mode? (PLAN.md round 2, sharing; user 11-13.) F5 checkpoint
10's frozen inputs, as ../20261111_p1f5/readout_fit.py builds them; no world training.

The earlier check (readout_fit.sharing) compared a fresh single-mode head with F5's trained shared head, which also
differs in online training and window distribution. Here every regime is a fresh head of the original classes with
the same initial weights (seed per replicate), the same rows, windows and budget; only the objective differs:
  TSP (M.TSP + M.PixelDecoder, squared error of the pixel mean, as readout_fit): reactive only (belief, window tokens,
      lighting), forecast only (belief and window position, as F5's forecast mode), joint (the sum, one network).
  Outcome (M.Outcome, reward log-loss as trained): reactive only (belief, real next percept embedding), forecast only
      (belief), joint (the sum).
Scored on held-out episodes: TSP squared error on uniform windows (each mode), the forecast mode's error at the HUD
cells on frames where the digit does not change (readout_fit's health control), and the reward log-loss of each mode,
overall and on frames with a non-zero reward. Two initialisation replicates give the seed-to-seed spread. This tests
sharing in the head, on frozen inputs; it does not test what the shared objective does to the belief in training.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_token_forecast_gate.sharing
"""
import importlib
import json
import os

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

G = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate")
RF = importlib.import_module("artifacts.experiments.20261111_p1f5.readout_fit")
DEV, CV, DS = RF.DEV, RF.CV, RF.DS
OUT = G.HERE / os.environ.get("GATE_OUT", "readings/round2")
TSP_STEPS, OUT_STEPS, BATCH = int(os.environ.get("SHARING_TSP_STEPS", 20000)), int(os.environ.get("SHARING_OUT_STEPS", 60000)), 512
SEEDS = (7, 8)
REGIMES = ("reactive", "forecast", "joint")


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.manual_seed(0)
    model, cfg = CV.IF.load(str(DS.CKPTS["f5"]))
    model.eval()
    assert cfg["heads"].get("forecast")
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    enc = S.encode(model, L, cfg)
    bps, frames, ep_of, changed, ps, vs, embs, rws = [], [], [], [], [], [], [], []
    for e, ep in enumerate(S.episodes(L)):                              # readout_fit.main's inputs, unchanged
        fr = D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()
        bps.append(enc[e]["bp"][1:].float()); frames.append(torch.as_tensor(fr[1:]))
        ps.append(enc[e]["p"][1:].half()); vs.append(enc[e]["v"][1:].float()); rws.append(enc[e]["reward"][1:].float())
        with torch.no_grad(), TR.autocast(DEV, cfg):
            pc = M.Percept(*(enc[e][k][1:].to(DEV).float() for k in ("p", "g", "v")), enc[e]["fix"][1:].to(DEV))
            embs.append(model.belief.percept_embed(pc).float().cpu())
        body = L["s_body"][off[e]:off[e + 1]]
        changed.append(torch.as_tensor(body[1:] != body[:-1]))
        ep_of.append(torch.full((len(fr) - 1,), e))
    del enc
    bp, fr, ep_of, changed = torch.cat(bps), torch.cat(frames), torch.cat(ep_of), torch.cat(changed)
    pt, vt, emb, rew = torch.cat(ps), torch.cat(vs), torch.cat(embs), torch.cat(rws)
    held = ep_of % 5 == 4
    tr_ix, te_ix = torch.nonzero(~held)[:, 0], torch.nonzero(held)[:, 0]
    C49 = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))

    def crop(ix, c):
        return M.cell_pixels(M.crop(fr[ix].to(DEV), c.to(DEV))).float() / 127.5 - 1

    def tsp_err(tsp, pix, ix, c, mode):
        b, cells = bp[ix].to(DEV), M.window(c.to(DEV))
        p, v = (pt[ix].to(DEV).float(), vt[ix].to(DEV)) if mode == "reactive" else (None, None)
        return (pix.mean(tsp.predict(b, p, v, cells)) - crop(ix, c)).square()

    res = {"tsp_steps": TSP_STEPS, "outcome_steps": OUT_STEPS, "batch": BATCH, "seeds": SEEDS,
           "train_frames": int(len(tr_ix)), "heldout_frames": int(len(te_ix))}
    for seed in SEEDS:
        for regime in REGIMES:
            torch.manual_seed(seed)
            tsp, pix = M.TSP(cfg).to(DEV), M.PixelDecoder(cfg).to(DEV)
            opt = torch.optim.Adam(list(tsp.parameters()) + list(pix.parameters()), 1e-3)
            g, gw = torch.Generator().manual_seed(2), torch.Generator().manual_seed(4)
            for s in range(TSP_STEPS):
                ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
                cr, cf = C49[torch.randint(49, (len(ix),), generator=gw)], C49[torch.randint(49, (len(ix),), generator=gw)]
                loss = 0.
                if regime in ("reactive", "joint"):
                    loss = loss + tsp_err(tsp, pix, ix, cr, "reactive").mean()
                if regime in ("forecast", "joint"):
                    loss = loss + tsp_err(tsp, pix, ix, cf, "forecast").mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if s % 5000 == 0:
                    print("tsp", seed, regime, s, float(loss), flush=True)
            row = {}
            with torch.no_grad():
                ge = torch.Generator().manual_seed(3)
                for mode in ("reactive", "forecast"):
                    tot = 0.
                    for a in range(0, len(te_ix), 4096):
                        ix = te_ix[a:a + 4096]
                        tot += float(tsp_err(tsp, pix, ix, C49[torch.randint(49, (len(ix),), generator=ge)], mode).mean()) * len(ix)
                    row[f"{mode}_mse"] = tot / len(te_ix)
                for digit, (centre, pos, col) in RF.HUD.items():
                    same = te_ix[~changed[te_ix, col]]
                    tot = 0.
                    for a in range(0, len(same), 4096):
                        ix = same[a:a + 4096]
                        c = torch.tensor(centre).expand(len(ix), 2)
                        tot += float(tsp_err(tsp, pix, ix, c, "forecast")[:, pos].mean()) * len(ix)
                    row[f"forecast_{digit}_unchanged"] = tot / len(same)
            res[f"tsp/{seed}/{regime}"] = row
            print("tsp", seed, regime, json.dumps(row), flush=True)
            del tsp, pix, opt
        for regime in REGIMES:
            torch.manual_seed(seed)
            head = M.Outcome(cfg).to(DEV)
            head.edges.copy_(model.outcome.edges); head.bin_value.copy_(model.outcome.bin_value)
            opt = torch.optim.Adam(head.parameters(), 1e-3)
            g = torch.Generator().manual_seed(2)
            nll = lambda ix, mode: -head.reward_logp(head.features(bp[ix].to(DEV), emb[ix].to(DEV) if mode == "reactive" else None),
                                                     rew[ix].to(DEV))
            for s in range(OUT_STEPS):
                ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
                loss = sum(nll(ix, m).mean() for m in ("reactive", "forecast") if regime in (m, "joint"))
                opt.zero_grad(); loss.backward(); opt.step()
                if s % 20000 == 0:
                    print("outcome", seed, regime, s, float(loss), flush=True)
            row = {}
            with torch.no_grad():
                for mode in ("reactive", "forecast"):
                    v = torch.cat([nll(te_ix[a:a + 8192], mode).float().cpu() for a in range(0, len(te_ix), 8192)])
                    nz = rew[te_ix] != 0
                    row[f"{mode}_nll"], row[f"{mode}_nll_reward_frames"] = float(v.mean()), float(v[nz].mean())
            res[f"outcome/{seed}/{regime}"] = row
            print("outcome", seed, regime, json.dumps(row), flush=True)
            del head, opt
    with torch.no_grad():                                                # F5's trained shared heads, for reference
        ge, row = torch.Generator().manual_seed(3), {}
        for mode in ("reactive", "forecast"):
            tot = 0.
            for a in range(0, len(te_ix), 4096):
                ix = te_ix[a:a + 4096]
                tot += float(tsp_err(model.tsp, model.pixels, ix, C49[torch.randint(49, (len(ix),), generator=ge)], mode).float().mean()) * len(ix)
            row[f"tsp_{mode}_mse"] = tot / len(te_ix)
        res["trained_f5"] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sharing.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
