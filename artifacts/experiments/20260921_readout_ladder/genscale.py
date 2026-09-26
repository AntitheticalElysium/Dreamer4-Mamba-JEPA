"""Does the world's IMAGINATION carry the consequence once its head is trained on full data with the dense label?

factual_scale (dense_label_helps): a factual dueling head on the world's INPUT reaches 0.689 within root on
56k (zombie 0.601) when trained on the full TRAIN corpus with the harm label (death OR 2+ health lost);
neither the full corpus with the death label nor the pool with the harm label moves it. Every world and
head in the interface line was trained on the pool with the sparse continuation label.

Frozen seed-1 U world (interface_worlds_v1/U.pt); nothing in it changes. For every TRAIN transition in the
u cache (3,086,463), the world imagines the successor of the LOGGED action at the evaluator position
(4 observed frames through teacher, one advance). Heads, factual labels only, same budget and selection as
factual_scale (20,000 updates of 1,024, class-balanced BCE, AdamW 1e-3 / 1e-4, 3 seeds, selection every
500 on FIT-dev within-root expected safe):
  gen_harm   MLP(imagined successor) -> P(harm)          harm = death OR health -2 or worse
  gen_dead   MLP(imagined successor) -> P(dead)
  root_harm  factual_scale's dueling head on the world's input (recomputed here for pairing)
Judged within root on 56k (POST HOC), one-step death, all 17 actions imagined.

Readings (committed before the run):
  gen_harm - root_harm not resolved < 0 (overall AND zombie)   -> imagination_keeps_consequence
  gen_harm - gen_dead resolved > 0 on zombie roots             -> dense_label_helps_on_imagination
  gen_harm vs the world's own head (0.656 on this block)       -> reported
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
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"
WORLD = ROOT / "artifacts/eda/interface_worlds_v1/U.pt"


@torch.no_grad()
def imagine_logged(bundle, U, A, t, config, device, batch=2048):
    from d4mj.train import autocast_context
    out = torch.empty(len(t), 192, dtype=torch.float16)
    for i in range(0, len(t), batch):
        tt = t[i:i + batch]
        ctx = U[tt[:, None] + torch.arange(-3, 1)].float().to(device)[:, :, None]
        past = A[tt[:, None] + torch.arange(-3, 0)].to(device)
        with autocast_context(config):
            state = bundle.world.teacher(ctx, past).state
            advanced, _ = bundle.advance(state, A[tt].to(device)[:, None])
        out[i:i + len(tt)] = advanced.latent[:, 0, 0].float().half().cpu()
    return out


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from boundary import judge_store
    from dueling_world import siblings
    from factual_scale import CACHE, Dueling, load_cache
    from frozen_heads import dev_rows
    from frozen_ladder import strata
    from interface import POOL, encode, load_bridge, project, world_bundle
    from ladder import paired
    from observability import expected_safe

    U, A, dead, dh, ids, valid = load_cache()
    harm = dead | (dh <= -2)
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    encoder, config = load_bridge()
    bundle = world_bundle(config, encoder, device)
    bundle.world.load_state_dict(torch.load(WORLD, map_location="cpu", weights_only=False)["world"])
    bundle.world.eval()
    G = imagine_logged(bundle, U, A, valid, config, device)
    log(stage="imagined", transitions=len(G))

    def ctx_of(frames):
        out = []
        for i in range(0, len(frames), 64):
            _, grid = encode(encoder, frames[i:i + 64, -4:], device)
            out.append(project(pool["pca"], grid))
        return torch.cat(out)
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    dev = dev_rows(sorted(partition["fit_dev"]["seeds"]))
    judge, jmanifest, _ = judge_store(JUDGE)
    cd, pdst = ctx_of(dev["frames"]), dev["actions"][:, -3:].argmax(-1)
    cj, pjst = ctx_of(judge["frames"]), judge["actions"][:, -3:].argmax(-1)
    sib_d, sib_j = siblings(bundle, cd, pdst, config, device), siblings(bundle, cj, pjst, config, device)
    del bundle, encoder
    torch.cuda.empty_cache()
    root_in = lambda c, p: torch.cat([c.flatten(1), F.one_hot(p, N).float().flatten(1)], 1)
    rd, rj = root_in(cd, pdst), root_in(cj, pjst)
    pd, pj, seeds = dev["p_death1"], judge["p_death1"], judge["seed"]
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]

    gmean, gscale = G[:200_000].float().mean(0), G[:200_000].float().std(0).clamp_min(1e-6)
    feats = lambda t: torch.cat([U[t[:, None] + torch.arange(-3, 1)].float().flatten(1),
                                 F.one_hot(A[t[:, None] + torch.arange(-3, 0)], N).float().flatten(1)], 1)
    sample = feats(valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:200_000]])
    rmean, rscale = sample.mean(0), sample.std(0).clamp_min(1e-6)
    del sample

    def run(kind, label, seed):
        torch.manual_seed(seed)
        if kind == "gen":
            net = nn.Sequential(nn.Linear(192, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1)).to(device)
            fwd = lambda idx: net(((G[idx].float() - gmean) / gscale).to(device))[:, 0]
            score = lambda sib: torch.cat([net(((sib[j:j + 256].float() - gmean) / gscale).to(device))[..., 0].detach().cpu()
                                           for j in range(0, len(sib), 256)])
            dev_in, judge_in = sib_d, sib_j
        else:
            net = Dueling(rmean.shape[0]).to(device)
            fwd = lambda idx: net(((feats(valid[idx]) - rmean) / rscale).to(device)).gather(1, A[valid[idx]].to(device)[:, None])[:, 0]
            score = lambda x: torch.cat([net(((x[j:j + 4096] - rmean) / rscale).to(device)).detach().cpu() for j in range(0, len(x), 4096)])
            dev_in, judge_in = rd, rj
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        y_all = label[valid].float()
        pos = ((1 - y_all).sum() / y_all.sum()).to(device)
        best, state = -1.0, None
        for step in range(STEPS):
            idx = torch.randint(len(valid), (BATCH,), generator=g)
            loss = F.binary_cross_entropy_with_logits(fwd(idx), label[valid[idx]].float().to(device), pos_weight=pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % EVERY == 0:
                net.eval()
                with torch.no_grad():
                    s_, o_ = expected_safe(score(dev_in), pd)
                value = float(s_[o_].mean())
                net.train()
                if value > best:
                    best, state = value, {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            return expected_safe(score(judge_in), pj)[0]

    safe, per_seed = {}, {}
    for name, kind, label in (("gen_harm", "gen", harm), ("gen_dead", "gen", dead), ("root_harm", "root", harm)):
        runs = [run(kind, label, s) for s in range(3)]
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4),
            per_seed=[round(v, 4) for v in per_seed[name]])
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261022)
    down = lambda r: r["difference"] < 0 and r["excludes_zero"]
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    contrasts = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)}
                 for a, b in (("gen_harm", "root_harm"), ("gen_harm", "gen_dead"))}
    c = contrasts["gen_harm_vs_root_harm"]
    readings = {"imagination": ("imagination_keeps_consequence" if not down(c["overall"]) and not down(c["zombie"])
                                else "imagination_loses_consequence"),
                "dense_label": ("dense_label_helps_on_imagination" if up(contrasts["gen_harm_vs_gen_dead"]["zombie"])
                                else "no_evidence")}
    evidence = {"schema": "d4mj_genscale_v1", "status": "POST HOC on the 56k block", "script_sha256": _sha256(Path(__file__)),
                "world_sha256": _sha256(WORLD), "cache_manifest_sha256": _sha256(CACHE / "manifest.json"),
                "judge_manifest": jmanifest, "readings": readings, "contrasts": contrasts, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()}}
    (HERE / "evidence/genscale.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="genscale_complete", **readings)


if __name__ == "__main__":
    main()
