"""Can a forecast readout recover what F5's frozen belief holds? (RESULTS.md, readout refit.) No world training.

Inputs are fixed: F5 checkpoint 10's pre-frame belief bp_t (S1 encode, DEV), the one the TSP forecast reads. Target:
frame t's pixels in a 3x3 window, in [-1, 1]. Readouts:
  trained  F5's own TSP forecast head and pixel decoder (reference);
  concat   a fresh head of the same form (belief -> 128, cell embedding, learned mask, MLP 256-256, decoder mean MLP
           256-256-147), trained only to forecast;
  attend   fresh cell queries attending over Mamba's memory tokens (model.belief.memory(bp), the denoiser's route)
           plus the belief vector, one 4-head attention layer, then the same decoder form.
The fresh readouts train on uniform windows (as F5's forecast was trained) on 80% of DEV episodes, with Adam 1e-3,
batch 512 and a fixed step budget. All readouts are scored on held-out episodes at the HUD windows: per-cell error on
frames where the digit does not change (health is the positive control: the belief holds it at 97% exact), against the
cell's digit-blind variance; and kappa on the energy, hunger and recovery pairs (trajectory.py). This tests what a
readout of each form can extract from the same belief. It does not test what it would teach in training.

Sharing check (both F5 heads have a reactive and a forecast mode sharing one network; both reactive modes are ~2x worse
than F4/HUD-Change): fresh single-mode heads of the original classes on the same frozen F5 inputs. TSP reactive-only
(window tokens, lighting, belief) is scored on held-out uniform windows (squared error) against the trained head's
reactive mode. The outcome head is refit reactive-only (bp, the real next percept embedding) and forecast-only (bp),
reward log-loss against the trained head's two modes. If a fresh single-mode head is clearly better than the trained
shared one on the same inputs, the shared network costs that mode; if not, the inputs limit it.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.readout_fit
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection"
TJ = importlib.import_module("artifacts.experiments.20261111_p1f5.trajectory")
DS = TJ.DS
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
DEV = DS.DEV
STEPS, BATCH = 20000, 512
HUD = {"health": ((7, 1), 3, 0), "food": ((7, 1), 4, 1), "drink": ((7, 1), 5, 2), "energy": ((7, 2), 5, 3)}


class Concat(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.cell = nn.Embedding(M.N, 32)
        self.belief = nn.Linear(W, 128)
        self.mask = nn.Parameter(torch.zeros(24))
        self.net = nn.Sequential(M.mlp([24 + 32 + 128, 256, 256]), nn.GELU())
        self.mean = M.mlp([256, 256, 147])

    def forward(self, bp, mem, cells):
        ctx = torch.cat([self.mask.expand(*cells.shape, 24), self.cell(cells),
                         self.belief(bp).unsqueeze(-2).expand(*cells.shape, 128)], -1)
        return self.mean(self.net(ctx))


class Attend(nn.Module):
    def __init__(self, W, Wm):
        super().__init__()
        self.cell = nn.Embedding(M.N, 128)
        self.kv = nn.Linear(Wm, 128)
        self.btok = nn.Linear(W, 128)
        self.att = nn.MultiheadAttention(128, 4, batch_first=True)
        self.net = nn.Sequential(M.mlp([256, 256, 256]), nn.GELU())
        self.mean = M.mlp([256, 256, 147])

    def forward(self, bp, mem, cells):
        kv = torch.cat([self.kv(mem), self.btok(bp).unsqueeze(-2)], -2)          # [B, K + 1, 128]
        q = self.cell(cells)                                                      # [B, 9, 128]
        a, _ = self.att(q, kv, kv)
        return self.mean(self.net(torch.cat([q, a], -1)))


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.manual_seed(0)
    model, cfg = CV.IF.load(str(DS.CKPTS["f5"]))
    model.eval()
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    enc = S.encode(model, L, cfg)
    bps, frames, ep_of, changed, ps, vs, embs, rws = [], [], [], [], [], [], [], []
    for e, ep in enumerate(S.episodes(L)):
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
    pt, vt, emb, rew_all = torch.cat(ps), torch.cat(vs), torch.cat(embs), torch.cat(rws)      # rw is reused by the pair loop
    held = (ep_of % 5 == 4)
    tr_ix, te_ix = torch.nonzero(~held)[:, 0], torch.nonzero(held)[:, 0]
    @torch.no_grad()
    def mem_all(b):
        return model.belief.memory(b.to(DEV)).float()

    W, Wm = bp.shape[-1], mem_all(bp[:2]).shape[-1]
    C49 = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))

    def batch(ix, fixed=None):
        b = bp[ix].to(DEV)
        c = C49[torch.randint(49, (len(ix),))] if fixed is None else torch.tensor(fixed).expand(len(ix), 2)
        c = c.to(DEV)
        y = M.cell_pixels(M.crop(fr[ix].to(DEV), c)).float() / 127.5 - 1
        return b, mem_all(b), M.window(c), y

    res = {"steps": STEPS, "batch": BATCH, "train_frames": int(len(tr_ix)), "heldout_frames": int(len(te_ix))}
    readouts = {"concat": Concat(W).to(DEV), "attend": Attend(W, Wm).to(DEV)}
    for name, net in readouts.items():
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(1)
        for s in range(STEPS):
            ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
            b, m, cells, y = batch(ix)
            loss = (net(b, m, cells) - y).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 5000 == 0:
                print(name, s, float(loss), flush=True)
        net.eval()
    torch.save({k: net.state_dict() for k, net in readouts.items()}, OUT / "readout_fit_nets.pt")   # for credit decomposition
    readouts["trained"] = lambda b, m, cells: model.pixels.mean(model.tsp.predict(b, None, None, cells))
    with torch.no_grad():
        for digit, (centre, pos, col) in HUD.items():
            same = te_ix[~changed[te_ix, col]]
            row = {}
            errs = {k: [] for k in readouts}
            ys = []
            for a in range(0, len(same), 4096):
                b, m, cells, y = batch(same[a:a + 4096], centre)
                ys.append(y[:, pos].cpu())
                for k, net in readouts.items():
                    errs[k].append(((net(b, m, cells).float()[:, pos] - y[:, pos]).square().mean(-1)).cpu())
            yall = torch.cat(ys)
            row["digit_blind"] = float(yall.var(0).mean())
            for k in readouts:
                row[k] = float(torch.cat(errs[k]).mean())
            res[digit] = row
            print(digit, json.dumps(row), flush=True)
        # Due vs not-due separation on the exact-history pairs, from the same frozen beliefs.
        for pname, slot in (("recovery", 63), ("energy", 66), ("hunger", 64)):
            centre, pos = TJ.WINDOW[slot]
            kap = {k: [] for k in readouts}
            for r in TJ.pair_rows(pname):
                i, e, prefix, frs, rewards, terms, q, truth = r
                old_f, old_a, old_r, old_d = prefix
                f = np.concatenate([np.repeat(old_f[None], 2, 0), frs[:, :q + 1]], 1)
                aa = np.r_[old_a, np.zeros(q + 1, dtype=np.int64)]
                rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
                td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
                fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=DEV)
                with TR.autocast(DEV, cfg):
                    p = model.perception(torch.as_tensor(f, device=DEV), fix)
                    y_, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=DEV, dtype=torch.float32),
                                                             torch.as_tensor(td, device=DEV), torch.as_tensor(np.repeat(aa[None], 2, 0), device=DEV)),
                                         model.belief.initial(2, DEV))
                b = y_[:, -1].float()
                cfix = torch.tensor(centre, device=DEV).expand(2, 2)
                cells = M.window(cfix)
                tgt = M.cell_pixels(M.crop(torch.as_tensor(frs[:, q + 1], device=DEV), cfix)).float()[:, pos] / 127.5 - 1
                dy = tgt[0] - tgt[1]
                for k, net in readouts.items():
                    fc = net(b, mem_all(b), cells).float()[:, pos]
                    kap[k].append(float(((fc[0] - fc[1]) * dy).sum() / dy.square().sum().clamp(min=1e-9)))
            res[f"kappa/{pname}"] = {k: float(np.mean(v)) for k, v in kap.items()}
            print(pname, json.dumps(res[f"kappa/{pname}"]), flush=True)
    sharing(model, cfg, bp, fr, pt, vt, emb, rew_all, tr_ix, te_ix, C49, res)
    (OUT / "readout_fit.json").write_text(json.dumps(res, indent=1) + "\n")


def sharing(model, cfg, bp, fr, pt, vt, emb, rw, tr_ix, te_ix, C49, res):
    single = {"tsp_reactive": M.TSP(cfg).to(DEV), "pixels": M.PixelDecoder(cfg).to(DEV),
              "outcome_reactive": M.Outcome(cfg).to(DEV), "outcome_forecast": M.Outcome(cfg).to(DEV)}
    for k in ("outcome_reactive", "outcome_forecast"):
        single[k].edges.copy_(model.outcome.edges); single[k].bin_value.copy_(model.outcome.bin_value)

    def tsp_batch(ix, gen=None):
        c = C49[torch.randint(49, (len(ix),), generator=gen)].to(DEV)
        y = M.cell_pixels(M.crop(fr[ix].to(DEV), c)).float() / 127.5 - 1
        return bp[ix].to(DEV), pt[ix].to(DEV).float(), vt[ix].to(DEV), M.window(c), y

    groups = {"tsp": (list(single["tsp_reactive"].parameters()) + list(single["pixels"].parameters())),
              "outcome_reactive": list(single["outcome_reactive"].parameters()),
              "outcome_forecast": list(single["outcome_forecast"].parameters())}
    for name, params in groups.items():
        opt = torch.optim.Adam(params, 1e-3)
        g = torch.Generator().manual_seed(2)
        for s in range(STEPS):
            ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
            if name == "tsp":
                b, p, v, cells, y = tsp_batch(ix, g)
                loss = (single["pixels"].mean(single["tsp_reactive"].predict(b, p, v, cells)) - y).square().mean()
            else:
                head = single[name]
                e = emb[ix].to(DEV) if name == "outcome_reactive" else None
                loss = -head.reward_logp(head.features(bp[ix].to(DEV), e), rw[ix].to(DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 5000 == 0:
                print("sharing", name, s, float(loss), flush=True)
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        err = {"fresh": [], "trained": []}
        nll = {k: [] for k in ("fresh_reactive", "trained_reactive", "fresh_forecast", "trained_forecast")}
        for a in range(0, len(te_ix), 4096):
            ix = te_ix[a:a + 4096]
            b, p, v, cells, y = tsp_batch(ix, g)
            err["fresh"].append((single["pixels"].mean(single["tsp_reactive"].predict(b, p, v, cells)) - y).square().mean((-1, -2)).cpu())
            err["trained"].append((model.pixels.mean(model.tsp.predict(b, p, v, cells)).float() - y).square().mean((-1, -2)).cpu())
            e, r = emb[ix].to(DEV), rw[ix].to(DEV)
            for tag, head, ee in (("fresh_reactive", single["outcome_reactive"], e), ("trained_reactive", model.outcome, e),
                                  ("fresh_forecast", single["outcome_forecast"], None), ("trained_forecast", model.outcome, None)):
                nll[tag].append((-head.reward_logp(head.features(b, ee), r)).float().cpu())
        res["sharing"] = {"tsp_reactive_mse": {k: float(torch.cat(v).mean()) for k, v in err.items()},
                          "reward_nll": {k: float(torch.cat(v).mean()) for k, v in nll.items()}}
        print("sharing", json.dumps(res["sharing"]), flush=True)


if __name__ == "__main__":
    main()
