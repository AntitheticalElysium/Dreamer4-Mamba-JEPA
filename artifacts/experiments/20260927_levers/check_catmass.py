"""E13e check (2026-10-01): categorical world, class-level probability mass at the faced tile (held-out pool windows). Result: changed transitions: mass on the true next class 0.067, on the current class 0.828, next class most likely 0.004."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS, quantize
from stageprobe import ACT, FACED
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as Sp
dev = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
meta, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, tr, ts)
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
world, st = T.load_world("artifacts/eda/levers_tworlds_v1/categorical_raw_teacher_s7_K4096.pt", dev)
code_cls = probes.tile(world.codes.float().cpu()).argmax(-1)                     # [K] tile class of each code
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
res = {"changed": [], "unchanged": []}
with torch.no_grad():
    for i in range(0, len(held), 64):
        r = held[i:i + 64]; s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
        s_in = world.codes[quantize(s.to(dev), world.codes)]
        with autocast_context(config):
            _, _, logits = world(s_in, F.pad(a, (0, 1)).to(dev))
        p = logits.float().softmax(-1).cpu()
        face = probes.facing(s[:, :5, 31].flatten(0, 1)).argmax(-1).view(len(r), 5)
        for t in range(5):
            m = torch.isin(a[:, t], torch.tensor(ACT)) & alive[:, t + 1]
            for j in torch.nonzero(m)[:, 0].tolist():
                cell = FACED[int(face[j, t])]
                x0, x1 = s[j, t, cell], s[j, t + 1, cell]
                c0, c1 = probes.tile(torch.stack([x0, x1])).argmax(-1).tolist()
                ch = c0 != c1 and float(((x1 - x0) ** 2).sum()) > 120
                pc = torch.zeros(17).index_add_(0, code_cls, p[j, t, cell])     # class mass
                res["changed" if ch else "unchanged"].append((float(pc[c1]), float(pc[c0]), int(pc.argmax() == c1)))
for k, v in res.items():
    v = torch.tensor(v)
    print(f"{k:9s} n {len(v):5d} | mass on true NEXT class {v[:, 0].mean():.3f} | mass on CURRENT class {v[:, 1].mean():.3f} | next class has most mass {v[:, 2].float().mean():.3f}")
