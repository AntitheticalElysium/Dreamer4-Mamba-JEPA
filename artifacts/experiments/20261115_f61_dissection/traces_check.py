"""Check 12 instrument check: do the due and delayed branches' traces differ at the HUD cell, and by how much after the
trained map (denoiser_2000.pt)? Prints per bank; output quoted in RESULTS.md, check 12."""
import importlib, numpy as np, torch
TF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.traces_ft")
from saccade import chooser as C
model, cfg = TF.DS.NP.IF.load(TF.CKPT.resolve()); dev = next(model.parameters()).device
den = model.denoiser; TF.install(den)
den.load_state_dict(torch.load(TF.OUT / "denoiser_2000.pt", map_location=dev)["denoiser"])
W = den.trace_proj
for name, rows, slot, sign in TF.pairs():
    d_slot, d_all, ratio, gaps = [], [], [], []
    for (prefix, frames, rewards, terms, q, acts, truth) in rows:
        old_f = prefix[0]
        f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
        ft = torch.as_tensor(f, device=dev)
        with torch.no_grad():
            pc = torch.stack([C.coarse(model.perception, ft[b])[0] for b in (0, 1)])
            first = torch.zeros(pc.shape[:2], dtype=torch.bool, device=dev); first[:, 0] = True
            tin, tst = TF.traces(pc, first, torch.zeros(2, 81, 2, device=dev))
            e = tst[:, -1]                                  # [2, 81, 2]
            d_slot.append((e[0, slot] - e[1, slot]).cpu().numpy())
            d_all.append((e[0] - e[1]).abs().sum(-1).cpu().numpy())
            a = W(torch.log1p(e))                           # [2, 81, W]
            ratio.append(float((a[0, slot] - a[1, slot]).norm() / a[0, slot].norm().clamp_min(1e-9)))
            # last change of the slot's token in each branch (frames ago)
            x = (pc[:, 1:, slot] - pc[:, :-1, slot]).float().square().sum(-1)
            g = []
            for b in (0, 1):
                nz = torch.nonzero(x[b] > 1e-6).flatten()
                g.append(int(x.shape[1] - 1 - nz[-1]) if len(nz) else -1)
            gaps.append(g)
    d_slot, d_all = np.array(d_slot), np.array(d_all)
    print(name, "pairs", len(rows), "slot", slot,
          "| trace due-delayed at slot: mean", d_slot.mean(0).round(4), "abs", np.abs(d_slot).mean(0).round(4),
          "| cells with any diff", float((d_all > 1e-6).sum(1).mean()),
          "| proj diff/norm at slot", round(float(np.mean(ratio)), 4),
          "| frames since slot change (due, delayed) first 6:", gaps[:6])
