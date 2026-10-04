"""Data scale, isolated: does the full TRAIN corpus teach a factual head the consequence the pool cannot?

Every factual learner so far (the worlds, ridge, direct MLPs) sits near 0.62-0.67 within-root on 56k and
at DOWN on zombie roots, and every one saw only the interface pool (~1.6% of TRAIN transitions; all 8,071
deaths but ~500 near-zombie damage events). The full corpus shows the mechanism ~60x more often
(exposure.json: 33,115 near-zombie damage events; staying 21.9% vs moving 7.1%).

Same model, input, budget and selection; only the data and the label change:
  model     dueling.py's winner: V(s) + mean-zero A(s,a) (per-action outputs from the state), MLPs 512x2
  input     4 context u (768) + 3 past actions one-hot; action = the logged action (factual)
  data      pool   the interface pool's TRAIN windows' transitions (frames 3->4 and 4->5), from the cache
            full   every TRAIN transition with 4 context frames (u_cache_train_v1)
  label     dead   the logged continuation;  harm = dead OR health -2 or worse (the dense form)
  budget    20,000 updates of 1,024, class-balanced BCE, AdamW 1e-3 / 1e-4, 3 seeds, selection every 500
            on FIT-dev within-root expected safe
Judged within root on the 56k block (POST HOC), one-step death, all 17 actions.

Readings (committed before the run):
  full_dead - pool_dead on zombie roots resolved > 0            -> data_scale_helps
  full_harm - full_dead on zombie roots resolved > 0            -> dense_label_helps
  best arm beats actions_only overall (0.637, unpaired) and DOWN on zombie roots (0.539, unpaired) -> reported
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

N, STEPS, BATCH, EVERY = 17, 20_000, 1024, 500
CACHE = ROOT / "artifacts/eda/u_cache_train_v1"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"


class Dueling(nn.Module):
    def __init__(self, width):
        super().__init__()
        mlp = lambda o: nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, o))
        self.v, self.a = mlp(1), mlp(N)

    def forward(self, s):                       # -> [n, 17] logits of P(harm | s, a)
        adv = self.a(s)
        return self.v(s) + adv - adv.mean(1, keepdim=True)


def load_cache():
    rows = [r for p in sorted(CACHE.glob("shard-*.pt")) for r in torch.load(p, weights_only=False)]
    lengths = torch.tensor([len(r["actions"]) for r in rows])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), (lengths + 1).cumsum(0)[:-1]])
    U = torch.cat([r["u"] for r in rows])
    A = torch.cat([torch.cat([r["actions"], torch.zeros(1, dtype=torch.long)]) for r in rows])     # pad the last frame
    dead = torch.cat([torch.cat([r["dead"], torch.zeros(1, dtype=torch.bool)]) for r in rows])
    dh = torch.cat([torch.cat([r["dh"], torch.zeros(1, dtype=torch.int8)]) for r in rows])
    ids = {r["id"]: (int(o), int(l)) for r, o, l in zip(rows, offsets, lengths)}
    # transition t (current frame t, t >= 3, action t exists) as a global frame index
    valid = torch.cat([torch.arange(3, int(l)) + int(o) for o, l in zip(offsets, lengths)])
    return U, A, dead, dh, ids, valid


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from boundary import judge_store
    from frozen_heads import dev_rows
    from frozen_ladder import strata
    from interface import POOL, encode, load_bridge, project
    from ladder import paired
    from observability import expected_safe

    manifest = json.loads((CACHE / "manifest.json").read_text())
    U, A, dead, dh, ids, valid = load_cache()
    harm = dead | (dh <= -2)
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    src = torch.load(ROOT / "artifacts/eda/spatial_pool_v1/pool.pt", weights_only=False, mmap=True)["ids"]
    pool_t = torch.tensor([ids[e][0] + s + k for e, s in src for k in (3, 4) if e in ids and s + k < ids[e][1]])
    log(stage="cache", frames=len(U), transitions=len(valid), pool_transitions=len(pool_t),
        dead=int(dead[valid].sum()), harm=int(harm[valid].sum()))

    def features(t):
        ctx = U[t[:, None] + torch.arange(-3, 1)].float().flatten(1)
        past = F.one_hot(A[t[:, None] + torch.arange(-3, 0)], N).float().flatten(1)
        return torch.cat([ctx, past], 1)

    encoder, _ = load_bridge()

    def ctx_of(frames, actions):
        out = []
        for i in range(0, len(frames), 64):
            _, grid = encode(encoder, frames[i:i + 64, -4:], device)
            out.append(project(pool["pca"], grid))
        ctx = torch.cat(out)
        return torch.cat([ctx.flatten(1), F.one_hot(actions[:, -3:].argmax(-1), N).float().flatten(1)], 1)
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    dev = dev_rows(sorted(partition["fit_dev"]["seeds"]))
    judge, jmanifest, _ = judge_store(JUDGE)
    xd, xj = ctx_of(dev["frames"], dev["actions"]), ctx_of(judge["frames"], judge["actions"])
    del encoder
    pd, pj, seeds = dev["p_death1"], judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    sample = features(valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:200_000]])
    mean, scale = sample.mean(0), sample.std(0).clamp_min(1e-6)

    def run(transitions, label, seed):
        torch.manual_seed(seed)
        net = Dueling(mean.shape[0]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        y_all = label[transitions].float()
        pos = ((1 - y_all).sum() / y_all.sum()).to(device)
        score = lambda x: torch.cat([net(((x[j:j + 4096] - mean) / scale).to(device)).detach().cpu() for j in range(0, len(x), 4096)])
        best, state = -1.0, None
        for step in range(STEPS):
            t = transitions[torch.randint(len(transitions), (BATCH,), generator=g)]
            logits = net(((features(t) - mean) / scale).to(device)).gather(1, A[t].to(device)[:, None])[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, label[t].float().to(device), pos_weight=pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % EVERY == 0:
                net.eval()
                with torch.no_grad():
                    s_, o_ = expected_safe(score(xd), pd)
                value = float(s_[o_].mean())
                net.train()
                if value > best:
                    best, state = value, {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            return expected_safe(score(xj), pj)[0]

    safe, per_seed = {}, {}
    for name, transitions, label in (("pool_dead", pool_t, dead), ("full_dead", valid, dead),
                                     ("pool_harm", pool_t, harm), ("full_harm", valid, harm)):
        runs = [run(transitions, label, s) for s in range(3)]
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4),
            per_seed=[round(v, 4) for v in per_seed[name]])
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261021)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    contrasts = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)} for a, b in (
        ("full_dead", "pool_dead"), ("full_harm", "full_dead"), ("full_harm", "pool_harm"), ("pool_harm", "pool_dead"))}
    readings = {"data_scale": "data_scale_helps" if up(contrasts["full_dead_vs_pool_dead"]["zombie"]) else "no_evidence_scale_helps",
                "dense_label": "dense_label_helps" if up(contrasts["full_harm_vs_full_dead"]["zombie"]) else "no_evidence_dense_label_helps"}
    evidence = {"schema": "d4mj_factual_scale_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "cache_manifest_sha256": _sha256(CACHE / "manifest.json"), "cache_episodes": manifest["episodes"],
                "judge_manifest": jmanifest, "readings": readings, "contrasts": contrasts, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/factual_scale.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="factual_scale_complete", **readings)


if __name__ == "__main__":
    main()
