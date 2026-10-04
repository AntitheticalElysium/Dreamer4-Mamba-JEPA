"""EDA (post hoc, FIT roots only): where the fatal direction of U lives in the 4x4 pooled grid.

Fits the fatal direction on FIT real successor u (as transition_diag), maps it back through the PCA basis to the
16 pooled cells (each 192-d), and compares with the mean fatal-vs-safe grid difference and each cell's share of the
within-root action-effect energy. Output: evidence/pooled_cells.txt."""
import sys, os, json, torch
R = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"; sys.path.insert(0, R); sys.path.insert(0, R + "/artifacts/experiments/20260921_readout_ladder"); os.chdir(R)
import interface as I
from transition_diag import direction, centre
from observability import load
from confirm import seeds_for
from u_world import successors
from d4mj.lewm_diagnostics import FORK_STORE
dev = torch.device("cuda")
pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True); pca = pool["pca"]
fs, _ = seeds_for(json.load(open("artifacts/experiments/20260921_readout_ladder/evidence/root_partition.json")), FORK_STORE)
fit = load(fs)["fit"]; succ = successors(fs)["fit"][0]
enc, cfg = I.load_bridge()
grids = []
for i in range(0, len(succ), 16):
    z, g = I.encode(enc, succ[i:i+16].flatten(0, 1)[:, None], dev); grids.append(g[:, 0].view(-1, 17, 3072))
grid = torch.cat(grids); u = I.project(pca, grid)
fatal = fit["p_death1"] > 0.5; opp = fatal.any(1) & (~fatal).any(1)
keep = torch.ones_like(fatal)
w = direction(centre(u, keep)[opp].reshape(-1, 192), fatal[opp].reshape(-1).float())
# the fatal direction mapped back into grid space, and where the REAL fatal-vs-safe grid difference lives
d = pca["basis"] @ w                                  # [3072]
cell_w = (d.view(16, 192) ** 2).sum(-1); cell_w = cell_w / cell_w.sum()
diff = (centre(grid, keep)[opp] * fatal[opp][..., None].float()).sum((0, 1)) / fatal[opp].sum() - \
       (centre(grid, keep)[opp] * (~fatal[opp])[..., None].float()).sum((0, 1)) / (~fatal[opp]).sum()
cell_d = (diff.view(16, 192) ** 2).sum(-1); cell_d = cell_d / cell_d.sum()
# how much of the real fatal-vs-safe grid difference survives the PCA-192 projection
kept = float(((pca["basis"].T @ diff) ** 2).sum() / (diff ** 2).sum())
# total variance of each pooled cell across transitions (effect energy per cell)
eff = centre(grid, keep)[opp]
cell_e = (eff.view(-1, 16, 192) ** 2).sum((0, 2)); cell_e = cell_e / cell_e.sum()
print("pooled cell (row,col) : fatal-dir share | mean fatal-vs-safe diff share | effect-energy share")
for c in range(16):
    print(f"  ({c//4},{c%4})  {cell_w[c]:.3f}  {cell_d[c]:.3f}  {cell_e[c]:.3f}")
print("fraction of the mean fatal-vs-safe grid difference kept by PCA-192:", round(kept, 4))
