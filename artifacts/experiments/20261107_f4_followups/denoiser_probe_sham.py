"""Sham for denoiser_probe.py: the same 256 awake adjacent rows (first 80 dev episodes) generated with and without the
probe's hooks (same wrappers, same seed) -> generated p, v bitwise; then the asleep reader on those rows alone vs inside
a larger batch (its float output's batch-size dependence)."""
import importlib, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, torch
from saccade import model as M
from saccade.eval import s1 as S

DC = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_checks")
CV = DC.CV
model, cfg = DC.load("artifacts/experiments/20261106_p1f4/run/ckpt_10.pt", 4)
den = model.denoiser
L = S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 80)
enc = S.encode(model, L, cfg)
T = CV.transitions(L)
rows = np.nonzero(T["adjacent"] & (T["t"] >= CV.P) & ~T["asleep"])[0][:256]
tuples = [(int(T["ep"][j]), int(T["t"][j])) for j in rows]
gen = lambda: CV.IF.futures(model, enc, L, tuples, 1, cfg, torch.Generator(device=CV.DEV).manual_seed(4600))
with CV.captured() as a:
    gen()
st, n_tok = {"on": False, "pass": -1}, M.N + M.G + 1 + den.K
embed, head, sample = den.embed, den.head, M.sample
def embed_(*x):
    tok, ang = embed(*x)
    if st["on"]:
        st["pass"] += 1; st["keep"] = tok[:, 0, [63, 90]].float().cpu()
    return tok, ang
def head_(h, c):
    o = head(h, c)
    if st["on"]:
        st["out"] = o[0][:, 0, 63].float().cpu()
    return o
def sample_(*x, **k):
    st["on"], st["pass"] = True, -1
    try:
        return sample(*x, **k)
    finally:
        st["on"] = False
hk = [b.register_forward_hook(lambda m, i, o: st.__setitem__("blk", o[0][:, 0, 63].float().cpu()) if st["on"] and o[0].shape[2] == n_tok else None) for b in den.blocks]
den.embed, den.head, M.sample = embed_, head_, sample_
with CV.captured() as b:
    gen()
den.embed, den.head, M.sample = embed, head, sample
for h in hk:
    h.remove()
pa, va = (torch.cat(x) for x in zip(*a)); pb, vb = (torch.cat(x) for x in zip(*b))
print("generated p, v bitwise equal with vs without hooks:", torch.equal(pa, pb), torch.equal(va, vb))
with torch.enable_grad():
    R, _ = CV.fit_readers(model, cfg, L, enc, 0)
s1 = S.load_reader(CV.TR.ROOT / "artifacts/experiments/20261106_p1f4/run/s1/results.reader.pt", CV.TR.ROOT / "artifacts/experiments/20261106_p1f4/run/ckpt_10.pt", CV.DEV)
cell = T["cell"][rows]
alone = CV.read(R, s1, pa, va, cell)
big = CV.read(R, s1, torch.cat([pa] * 64), torch.cat([va] * 64), np.tile(cell, 64))
print("asleep float equal alone vs in a 16k batch:", np.array_equal(alone["asleep"], big["asleep"][:256]),
      "max abs diff", float(np.abs(alone["asleep"] - big["asleep"][:256]).max()),
      "| hud63 equal:", np.array_equal(alone["hud63"], big["hud63"][:256]),
      "| asleep > 0.5 equal:", np.array_equal(alone["asleep"] > 0.5, big["asleep"][:256] > 0.5))
