"""Post-hoc, CPU: split Codex's fine-only attenuation between SpatialFusion and the 6,569 -> 256 projection (percept_in),
on the bank's own renders (F3 ckpt 10, fixation [3, 4]). Linear algebra: d embed = W_canvas d canvas exactly when only
g changes; coarse-only: d embed = W_p d p + W_canvas d canvas(coarse change through the fusion)."""
import os; os.environ['JAX_PLATFORMS'] = 'cpu'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, numpy as np, torch
from saccade import model as M
H = 'artifacts/experiments/20261106_mamba_hostile_path/'
ck = torch.load('artifacts/experiments/20261105_p1f3/run/ckpt_10.pt', map_location='cpu', weights_only=False)
m = M.M0(ck['cfg']); m.load_state_dict(ck['model']); m.eval()
bank = torch.load(H + 'bank.pt', map_location='cpu', weights_only=False)
B = m.belief; Wp, Wc = None, None
W = B.percept_in.weight.detach()                                  # [256, 6569]: p 1296 | canvas 5184 | mask 81 | v 8
Wp, Wc = W[:, :1296], W[:, 1296:1296 + 5184]
rows = {'dg': [], 'dfused_fine': [], 'dcanvas_proj_fine': [], 'dp': [], 'dfused_coarse': [], 'dembed_coarse': [], 'dembed_fine': [],
        'p_bg': [], 'g_bg': []}
with torch.no_grad():
    for r in bank:
        x = torch.as_tensor(r['source'])                          # empty, zombie, cow renders
        fix = torch.tensor([3, 4]).expand(3, -1)
        P = m.perception(x, fix)
        cells = M.window(fix)
        def emb(pi, gi):
            Q = M.Percept(P.p[pi:pi + 1], P.g[gi:gi + 1], P.v[:1], fix[:1])
            fused = B.spatial_fusion(Q, cells[:1])
            canvas = fused.new_zeros(1, 81, 64).scatter(-2, cells[:1, :, None].expand_as(fused), fused)
            return fused, canvas, B.percept_embed(Q)
        f0, c0, e0 = emb(0, 0); ff, cf, ef = emb(0, 1); fc, cc, ec = emb(1, 0)
        rows['dg'].append(float((P.g[1] - P.g[0]).norm()))
        rows['dfused_fine'].append(float((ff - f0).norm()))
        rows['dcanvas_proj_fine'].append(float((Wc @ (cf - c0).flatten()).norm()))
        rows['dembed_fine'].append(float((ef - e0).norm()))
        rows['dp'].append(float((P.p[1] - P.p[0]).norm()))
        rows['dfused_coarse'].append(float((fc - f0).norm()))
        rows['dembed_coarse'].append(float((ec - e0).norm()))
        rows['p_bg'].append(P.p[0]); rows['g_bg'].append(P.g[0])
med = {k: float(np.median(v)) for k, v in rows.items() if k not in ('p_bg', 'g_bg')}
# operator gains of the projection's two blocks (largest singular values) and on the actual directions
med['gain_fine_path'] = med['dcanvas_proj_fine'] / med['dfused_fine']
med['fusion_gain_fine'] = med['dfused_fine'] / med['dg']
med['svd_top_Wp'] = float(torch.linalg.svdvals(Wp)[0]); med['svd_top_Wc'] = float(torch.linalg.svdvals(Wc)[0])
med['frob_Wp_per_col'] = float(Wp.norm() / Wp.shape[1] ** 0.5); med['frob_Wc_per_col'] = float(Wc.norm() / Wc.shape[1] ** 0.5)
# per-cell column norms at the zombie's cells: how much the projection weighs one cell's fine vs coarse channels
cell = [r['meta']['cell'] for r in bank]
cn_p = Wp.reshape(256, 81, 16).norm(dim=(0, 2)); cn_c = Wc.reshape(256, 81, 64).norm(dim=(0, 2))
med['colnorm_p_at_cells'] = float(cn_p[cell].median()); med['colnorm_canvas_at_cells'] = float(cn_c[cell].median())
print(json.dumps(med, indent=1))
json.dump(med, open('artifacts/experiments/20261106_mamba_hostile_path_review/fusion_split.json', 'w'), indent=1)
