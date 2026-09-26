"""A dueling decision head on the world's IMAGINED siblings, factual labels only.

dueling.py (action_gap_architecture_helps, post hoc 56k): on the world's input, a dueling factual head
(V(s) + mean-zero A(s,a)) beats a plain one by +0.050 [+0.024, +0.075] (zombie +0.033*), reaching 0.670.
The world-model translation, inside the factual contract: for each training state, the frozen world
imagines the successor of ALL 17 actions (4 observed frames, one advance each); the head's advantage
A is a shared MLP on each imagined successor, centred across the 17 siblings; V is an MLP on the context.
logit(dead | s, a) = V(s) + A(g(s,a)) - mean_a' A(g(s,a')). Trained only on the LOGGED action's
continuation label (the interface pool's TRAIN transitions: frames 0-3 -> 4 and 1-4 -> 5). The
counterfactual siblings come from the world's imagination, never from fork outcomes.
Within a root V cancels: the choice depends only on A over the imagined successors.

Compared, same data / budget / selection (dueling.py's: class-balanced BCE, 3 seeds, 5,000 updates of
512, selection every 250 on FIT-dev within-root expected safe):
  plain_gen    MLP(imagined successor of the logged action) -> logit      (genhead's gen_state)
  duel_gen     the dueling head above
Frozen worlds U_s1, U_s2. Judged within root on the 56k block (POST HOC). References on that block:
DOWN 0.591, actions_only 0.637, own heads 0.656 / 0.658, dueling on the world's INPUT 0.670, counterfactual
probe on generated states 0.672 / 0.670, root_U ranking probe 0.708.

Reading, per world (committed before the run):
  duel_gen - plain_gen resolved > 0 overall AND zombie -> dueling_on_imagination_helps, else no_evidence
  (actions_only is reported unpaired, from interface_seed1_56k.json)
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N, STEPS, BATCH = 17, 5000, 512
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"
WORLDS = {"U_s1": ROOT / "artifacts/eda/interface_worlds_v1/U.pt", "U_s2": ROOT / "artifacts/eda/interface_worlds_v2/U.pt"}


@torch.no_grad()
def siblings(bundle, ctx, past, config, device, batch=128):
    """ctx [n,4,192], past [n,3] -> imagined successors of all 17 actions [n,17,192] (fp16)."""
    from d4mj.train import autocast_context
    out = []
    for i in range(0, len(ctx), batch):
        s = ctx[i:i + batch].to(device)[:, :, None]
        n = len(s)
        with autocast_context(config):
            state = bundle.world.teacher(s, past[i:i + batch].to(device)).state
            advanced, _ = bundle.advance(bundle.repeat_state(state, N), torch.arange(N, device=device).repeat(n)[:, None])
        out.append(advanced.latent[:, 0, 0].float().view(n, N, -1).half().cpu())
    return torch.cat(out)


class Head(nn.Module):
    def __init__(self, dueling):
        super().__init__()
        self.dueling = dueling
        mlp = lambda i: nn.Sequential(nn.Linear(i, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.a = mlp(192)
        self.v = mlp(768 + 3 * N) if dueling else None

    def all_actions(self, ctx, sib):
        """[n, 17] logits for every action."""
        adv = self.a(sib.flatten(0, 1)).view(len(sib), N)
        if not self.dueling:
            return adv
        return self.v(ctx)[:, 0, None] + adv - adv.mean(1, keepdim=True)


def fit(train, dev_in, pd, device, seed, dueling):
    from observability import expected_safe
    torch.manual_seed(seed)
    ctx, sib, act, dead = train
    cm, cs = ctx.mean(0), ctx.std(0).clamp_min(1e-6)
    flat = sib.flatten(0, 1).float()
    sm, ss = flat.mean(0), flat.std(0).clamp_min(1e-6)
    del flat
    head = Head(dueling).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    pos = ((1 - dead).sum() / dead.sum()).to(device)
    norm = lambda c, s: (((c - cm) / cs).to(device), ((s.float() - sm) / ss).to(device))

    def logits(c, s):
        c, s = norm(c, s)
        return head.all_actions(c, s)
    best, state = -1.0, None
    for step in range(STEPS):
        idx = torch.randint(len(dead), (BATCH,), generator=g)
        lg = logits(ctx[idx], sib[idx]).gather(1, act[idx].to(device)[:, None])[:, 0]
        loss = F.binary_cross_entropy_with_logits(lg, dead[idx].to(device), pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            head.eval()
            with torch.no_grad():
                r = torch.cat([logits(dev_in[0][j:j + 1024], dev_in[1][j:j + 1024]).cpu() for j in range(0, len(pd), 1024)])
            s_, o_ = expected_safe(r, pd)
            value = float(s_[o_].mean())
            head.train()
            if value > best:
                best, state = value, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    head.eval()
    return lambda c, s: torch.cat([logits(c[j:j + 1024], s[j:j + 1024]).detach().cpu() for j in range(0, len(c), 1024)])


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads  # noqa: F401
    from boundary import judge_store
    from frozen_heads import dev_rows
    from frozen_ladder import strata
    from interface import POOL, encode, load_bridge, project, world_bundle
    from ladder import paired
    from observability import expected_safe

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    u, acts, alive = pool["u"], pool["actions"], pool["alive"]
    ctx_t = torch.cat([u[:, s:s + 4] for s in (0, 1)])
    past_t = torch.cat([acts[:, s:s + 3] for s in (0, 1)])
    act_t = torch.cat([acts[:, s + 3] for s in (0, 1)])
    dead_t = torch.cat([1 - alive[:, s + 4].float() for s in (0, 1)])
    encoder, config = load_bridge()

    def ctx_of(frames):
        out = []
        for i in range(0, len(frames), 64):
            _, grid = encode(encoder, frames[i:i + 64, -4:], device)
            out.append(project(pool["pca"], grid))
        return torch.cat(out)
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    dev = dev_rows(sorted(partition["fit_dev"]["seeds"]))
    judge, manifest, _ = judge_store(JUDGE)
    ctx_d, past_d = ctx_of(dev["frames"]), dev["actions"][:, -3:].argmax(-1)
    ctx_j, past_j = ctx_of(judge["frames"]), judge["actions"][:, -3:].argmax(-1)
    vin = lambda c, p: torch.cat([c.flatten(1), F.one_hot(p, N).float().flatten(1)], 1)
    pd, pj, seeds = dev["p_death1"], judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    safe, per_seed = {}, {}
    for name, path in WORLDS.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        bundle = world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        sib_t = siblings(bundle, ctx_t, past_t, config, device)
        sib_d = siblings(bundle, ctx_d, past_d, config, device)
        sib_j = siblings(bundle, ctx_j, past_j, config, device)
        del bundle
        log(stage="siblings", world=name, train=tuple(sib_t.shape))
        train = (vin(ctx_t, past_t), sib_t, act_t, dead_t)
        for arm, dueling in (("plain_gen", False), ("duel_gen", True)):
            runs = []
            for seed in range(3):
                score = fit(train, (vin(ctx_d, past_d), sib_d), pd, device, seed, dueling)
                runs.append(expected_safe(score(vin(ctx_j, past_j), sib_j), pj)[0])
            key = f"{name}_{arm}"
            safe[key] = torch.stack(runs).mean(0)
            per_seed[key] = [float(r[opp].mean()) for r in runs]
            log(arm=key, safe=round(float(safe[key][opp].mean()), 4), zombie=round(float(safe[key][zombie].mean()), 4),
                per_seed=[round(v, 4) for v in per_seed[key]])
        del sib_t, sib_d, sib_j
    ref = json.loads((HERE / "evidence/interface_seed1_56k.json").read_text())
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261020)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    contrasts, readings = {}, {}
    for name in WORLDS:
        d_all, d_z = test(f"{name}_duel_gen", f"{name}_plain_gen", opp), test(f"{name}_duel_gen", f"{name}_plain_gen", zombie)
        contrasts[name] = {"duel_minus_plain": {"overall": d_all, "zombie": d_z}}
        readings[name] = "dueling_on_imagination_helps" if up(d_all) and up(d_z) else "no_evidence"
    evidence = {"schema": "d4mj_dueling_world_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "judge_manifest": manifest, "readings": readings, "contrasts": contrasts, "per_seed": per_seed,
                "reference_actions_only": ref["reported"]["expected_safe"]["actions_only"]["overall"],
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/dueling_world.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="dueling_world_complete", **readings)


if __name__ == "__main__":
    main()
