"""E14a check (2026-10-02): where does the all-token L1 cost of a consequence-catching head land? (headfit addendum, s7: linear
mask1 catches 0.54 at +21% all-token L1, mlp_mask1 0.58 at +7%.) Held-out pool windows (tworld's 2,048), teacher-forced, the
trained world vs the same frozen backbone with a re-trained head (headfit_heads_v1). Per token class: mean L1 (192-d mean |err|),
its share of the total L1 increase, and the corr head's mean "generate" mixture weight.
Classes (exclusive, this priority): consequence (faced tile, strict change), attempt_nochange (faced tile of a DO / place with
no strict change), hud (tokens 63-80), player (31), entering (leading row / column of a scrolled frame, scroll.estimate on the
TRUE frames), near (the 4 neighbours of the player), static (every other map token).
Usage: check_costwhere.py <world.pt> <arm> [<arm> ...]
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS
from scroll import SHIFTS, estimate
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as Sp
dev = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
lab = torch.load("artifacts/eda/headfit_labels_v1.pt")
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
CLASSES = ("consequence", "attempt_nochange", "hud", "player", "entering", "near", "static")


def classes(rows, s):
    n = len(rows)
    c = torch.full((n, 5, 81), CLASSES.index("static"))
    for t in range(5):
        sh = estimate(s[:, t], s[:, t + 1])
        for k, (dr, dc) in enumerate(SHIFTS):
            m = sh == k
            if dr == 1: c[m, t, 54:63] = CLASSES.index("entering")
            if dr == -1: c[m, t, 0:9] = CLASSES.index("entering")
            if dc == 1: c[m, t, 8:63:9] = CLASSES.index("entering")
            if dc == -1: c[m, t, 0:63:9] = CLASSES.index("entering")
    c[..., [22, 30, 32, 40]] = CLASSES.index("near")
    c[..., 31] = CLASSES.index("player")
    c[..., 63:] = CLASSES.index("hud")
    face, att, cons = lab["faced"][rows], lab["attempt"][rows], lab["cons"][rows]
    r = torch.arange(n)[:, None].expand(n, 5); t = torch.arange(5)[None].expand(n, 5)
    c[r[att & ~cons], t[att & ~cons], face[att & ~cons]] = CLASSES.index("attempt_nochange")
    c[r[cons], t[cons], face[cons]] = CLASSES.index("consequence")
    return c


def load_head(world, path):
    sd = torch.load(path)
    for nm in ("proj", "choose", "logits"):
        if f"{nm}.0.weight" in sd:                     # an MLP head (headfit mlp_ arms)
            lin = getattr(world, nm)
            setattr(world, nm, torch.nn.Sequential(torch.nn.Linear(lin.in_features, 512), torch.nn.GELU(), torch.nn.Linear(512, lin.out_features)).to(dev))
    world.load_state_dict({**world.state_dict(), **{k: v.to(dev) for k, v in sd.items()}})
    return world


@torch.no_grad()
def run(world):
    L = torch.zeros(len(CLASSES)); N = torch.zeros(len(CLASSES)); G = torch.zeros(len(CLASSES))
    for i in range(0, len(held), 64):
        r = held[i:i + 64]; s = pool["tokens"][r].float(); a = pool["actions"][r]
        with autocast_context(config):
            pred = world(s.to(dev), F.pad(a, (0, 1)).to(dev))[0][:, :5].float().cpu()
        err = (pred - s[:, 1:]).abs().mean(-1)
        gw = world.last_weights[:, :5, :, 5].float().cpu() if hasattr(world, "last_weights") else torch.zeros_like(err)
        c = classes(r, s)
        for k in range(len(CLASSES)):
            m = c == k
            L[k] += err[m].sum(); N[k] += m.sum(); G[k] += gw[m].sum()
    return L, N, G


if __name__ == "__main__":
    name = sys.argv[1]
    base, st = T.load_world(name, dev)
    wname = st["name"]
    Lb, N, Gb = run(base)
    out = {"world": wname, "tokens": {c: int(N[k]) for k, c in enumerate(CLASSES)},
           "trained": {c: {"l1": round(float(Lb[k] / N[k]), 5), "gen_weight": round(float(Gb[k] / N[k]), 4)} for k, c in enumerate(CLASSES)}}
    for arm in sys.argv[2:]:
        w, _ = T.load_world(name, dev)
        w = load_head(w, f"artifacts/eda/headfit_heads_v1/{wname}_{arm}.pt")
        L, _, G = run(w)
        inc = L - Lb
        out[arm] = {"all_l1_ratio": round(float(L.sum() / Lb.sum()), 4),
                    **{c: {"l1": round(float(L[k] / N[k]), 5), "share_of_increase": round(float(inc[k] / inc.sum()), 4),
                           "gen_weight": round(float(G[k] / N[k]), 4)} for k, c in enumerate(CLASSES)}}
        print(json.dumps({arm: out[arm]}), flush=True)
    print(json.dumps(out))
