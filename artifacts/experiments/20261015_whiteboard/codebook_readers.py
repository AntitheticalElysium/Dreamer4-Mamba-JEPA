"""Track B groundwork (2026-10-15 night, CPU): do the E2 Raw k-means codebooks (levers_codebooks_v1, K 1024 / 4096) carry
mobs and tiles through the frozen readers on the CURRENT pool? 300 seeded training windows; map cells replaced by their nearest
centroid; reader decisions vs the real tokens. Usage: codebook_readers.py (prints)."""
import sys, torch, json
sys.path.insert(0, '/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/experiments/20261012_oracle')
import direct_train as BASE, audit_s5_direct as A5, direct_coherence as C
torch.set_num_threads(8)
_, windows, actions, tokens = BASE.load_data()
sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(5))[:300]
frames, _, _ = BASE.batch(tokens, actions, windows, sel, device='cpu')       # [300,16,81,192]
bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
d0 = C.decode(frames, bank, head)
m0, t0 = d0['mob']['real'][..., :3], d0['tile']
for k in (1024, 4096):
    c = torch.load(f'/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/levers_codebooks_v1/raw_K{k}.pt', map_location='cpu', weights_only=False)['codes'].float()
    x = frames.reshape(-1, 192); q = torch.empty_like(x)
    for i in range(0, len(x), 20000):
        q[i:i+20000] = c[torch.cdist(x[i:i+20000], c).argmin(-1)]
    q = q.view(frames.shape)
    q[..., 63:, :] = frames[..., 63:, :]                                      # map cells only
    d1 = C.decode(q, bank, head)
    m1, t1 = d1['mob']['real'][..., :3], d1['tile']
    out = {'K': k}
    for t, n in enumerate(('zombie', 'cow', 'skeleton')):
        pos = m0[..., t]
        out[n] = {'real_mobs': int(pos.sum()), 'kept_after_quantization': round(float(m1[..., t][pos].float().mean()), 4),
                  'false_mobs_per_1000_cells': round(1000 * float(m1[..., t][~pos].float().mean()), 3)}
    valid = t0 >= 0
    out['tile_agreement'] = round(float((t1[valid] == t0[valid]).float().mean()), 4)
    print(json.dumps(out), flush=True)
