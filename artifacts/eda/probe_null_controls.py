import sys, torch, numpy as np
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
from probe_hidden_ceiling import cache_hidden
from probe_hidden_forensics import fit_and_score, row_null_bases, standardise
from train_phase1b_fork import MixerWorld, fork_actions, load_forkset, seed_split
from d4mj.checkpoint import load
from d4mj.config import Config
DEVICE="cuda"
rows = load_forkset(Path("forkset_abm0_n64"))
splits = np.array([seed_split(r["seed"]) for r in rows])
masks = tuple(torch.from_numpy(splits==s) for s in ("fit","tune","test"))
history = torch.stack([r["z_history"] for r in rows])
branch = torch.stack([r["z_branch"] for r in rows]).float()
labels = torch.stack([r["label"] for r in rows]).numpy()
config = replace(Config(transition="direct", time_mixer="attention"), n_latents=64, d_bottleneck=16, seed=Config().seed)
world = MixerWorld(config).to(DEVICE); load(Path("phase1b_abm0_n64/world_020000.pt"), config, part0=world)
world.eval()
for p in world.parameters(): p.requires_grad_(False)
raw = cache_hidden(world, config, history, fork_actions(rows))
with torch.no_grad():
    hidden = torch.cat([world.mix_norm(raw[lo:lo+64].to(DEVICE).float()).half().cpu() for lo in range(0,len(raw),64)])
del raw
_, v_null = row_null_bases(world.readout[2].weight.detach().cpu())
hc = hidden.float() - hidden.float().mean(1, keepdim=True)
null = (hc @ v_null).reshape(*hc.shape[:2], -1).half()
g = torch.Generator().manual_seed(4)
proj = torch.linalg.qr(torch.randn(null.shape[-1], 512, generator=g))[0].half()
nullp = standardise(null @ proj, masks[0])
true_eff = standardise((branch - branch.mean(1, keepdim=True)).half(), masks[0])
order = torch.stack([torch.randperm(17, generator=g) for _ in range(len(nullp))])
null_shuf = torch.gather(nullp, 1, order[...,None].expand_as(nullp))
for name, x in (("true+null_SHUFFLED", torch.cat([true_eff, null_shuf], -1)),
                ("true+gaussian_noise", torch.cat([true_eff, torch.randn(*true_eff.shape[:2],512).half()], -1))):
    r = fit_and_score(x, labels, masks, seed=11)
    print(f"  {name:<22} AUC {r['test_auc']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]", flush=True)
