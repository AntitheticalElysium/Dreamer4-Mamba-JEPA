"""EDA: eigen spectra of every world state in the interface line (TRAIN pool main windows, every 3rd frame).

LeJEPA's Lemma 1 is about the covariance EIGENVALUES of the embedding. SIGReg's sketch (1,024 random 1-D
projections) mostly constrains per-coordinate / random-direction marginals. This measures, for z (the
canonical SIGReg-trained latent), u (PCA patch state), w (u whitened per component) and SP (sigreg_patch's
learned SIGReg projection): eigenvalue spread, top/median and median/bottom ratios, effective rank
(participation ratio), and per-coordinate variance range. Output: evidence/eigen_spectra.txt.
"""
import sys, json, torch
from pathlib import Path
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import os; os.chdir(ROOT)
import interface as I
from sigreg_patch import projector
pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
main = ~pool["terminal"]
enc, cfg = I.load_bridge()
dev = torch.device("cuda")
P = projector(cfg, dev); P.load_state_dict(torch.load(ROOT / "artifacts/eda/sigreg_patch_v1/SP.pt", map_location="cpu", weights_only=False)["projector"]); P.eval()
u = pool["u"][main].reshape(-1, 192)[::3]
with torch.no_grad():
    s = torch.cat([P(u[i:i + 8192].to(dev)).cpu() for i in range(0, len(u), 8192)])
z = pool["z"][main].reshape(-1, 192)[::3]
w = u / u.std(0)
for x, n in ((z, "z"), (u, "u"), (w, "w"), (s, "SP")):
    e = torch.linalg.eigvalsh(torch.cov(x.T.double())).clamp_min(1e-12).flip(0)
    print(f"{n:3s} eigen spread {float(e[0]/e[-1]):10.1f}  top/median {float(e[0]/e[96]):7.1f}  median/bottom {float(e[96]/e[-1]):7.1f}"
          f"  effective rank {float(e.sum()**2/(e**2).sum()):6.1f}/192  per-coord var range {float(x.var(0).max()/x.var(0).min()):7.1f}")
