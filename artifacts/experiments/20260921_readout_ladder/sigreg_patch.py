"""The canonical-path version of the fix: SIGReg on a patch-derived world state (LeWM's joint objective, moved
off CLS).

whiten_rep (sealed 59k, two seeds): the per-component-whitened patch state makes the world's own trained
system pass. WHY.md explains it with LeJEPA's Lemma 1 (anisotropy amplifies downstream bias): SIGReg
exists to make embeddings isotropic Gaussian for the heads, canonical z has that isotropy but lacks the
mob, the PCA patch state has the mob but breaks isotropy. Post-hoc whitening is a fixed stand-in. The
LeJEPA-consistent design learns it: a projector trained jointly with the world under LeWM's own
objective, so the state is isotropic Gaussian by construction and still derived from the patch grid.

Arm SP (seed-1 seeds, the interface pool, 1x budget):
  joint phase  s = P(u), P = LeWMProjector(192 -> projector_hidden -> 192) (the canonical projector class)
               on the frozen encoder's PCA patch state (a fixed linear map of the pooled grid, 99.0% of its
               variance); LeWM's joint objective exactly: both-sided next-latent MSE over 4-frame windows +
               0.09 x SIGReg(s) (17 knots, 1,024 directions, seeded projection RNG); P and the world trained
               together; 10,000 updates of 128 (phase-1 budget), AdamW at the M4 joint settings, constant LR
  bridge       P frozen (eval, BN statistics fixed); the SAME world continues through interface.py's phase 2
               (alias-free bridge, H2 heads, 9,333 updates) on s = P(u) of the pool windows
Reference: W seed 1 (whitened, the sealed pass).

Judge: a NEW block, `observe.py collect --seed-start 61000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v12`, collected after this commit, read once. Each world's OWN
continuation head on its imagined successors; actions_only (frozen_ladder, FIT-train / FIT-dev); DOWN.

DECLARED READINGS (committed before collection and training):
  primary    SP - DOWN overall, SP - actions_only overall, SP - DOWN on zombie roots all resolved > 0
             -> sp_trained_system_passes
  secondary  SP - W_s1 on zombie roots: resolved > 0 -> sigreg_better; resolved < 0 -> whitening_better;
             else -> equivalent
Reported: the isotropy of s (eigenvalue range of its covariance on TRAIN), SLEEP choices, strata.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

import interface as I  # noqa: E402

N = 17
OUT = ROOT / "artifacts/eda/sigreg_patch_v1"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v12"


def projector(config, device):
    from d4mj.lewm import LeWMProjector
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(7 + 11)
        return LeWMProjector(192, config.encoder).to(device)


def install(P, device):
    """Adds arm SP to interface: state_of(SP) = P(u)."""
    base = I.state_of

    def state_of(arm, pca, z, grid):
        if arm != "SP":
            return base(arm, pca, z, grid)
        u = base("U", pca, z, grid)
        with torch.no_grad():
            return P(u.flatten(0, -2).to(device)).view(*u.shape[:-1], -1).cpu()
    I.state_of = state_of
    I.KEY["SP"] = "s"


def train(device, log):
    from d4mj.lewm import SIGReg
    encoder, config = I.load_bridge()
    pool = dict(torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True))
    bundle = I.world_bundle(config, encoder, device)
    world = bundle.world
    P = projector(config, device)
    reg = SIGReg().to(device)
    world.train().requires_grad_(True)
    world.agent_readout.requires_grad_(False)
    P.train()
    j = config.joint
    params = [p for p in list(world.parameters()) + list(P.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=j.learning_rate, betas=tuple(j.betas), eps=j.optimizer_eps, weight_decay=j.weight_decay)
    order, proj_rng = torch.Generator().manual_seed(11), torch.Generator(device=device).manual_seed(29)
    main = torch.where(~pool["terminal"])[0]
    history = []
    for step in range(I.PHASE1_UPDATES):
        idx = main[torch.randint(len(main), (128,), generator=order)]
        off = torch.randint(3, (128,), generator=order)
        ar = torch.arange(128)[:, None]
        u = pool["u"][idx][ar, off[:, None] + torch.arange(4)].to(device)                   # [128, 4, 192]
        acts = pool["actions"][idx][ar, off[:, None] + torch.arange(3)].to(device)
        s = P(u.flatten(0, 1)).view(128, 4, 1, -1)
        pred = world.teacher(s, acts).predicted
        pred_loss = (pred.float() - s[:, 1:].float()).square().mean()
        reg_loss = reg(s[:, :, 0].float().transpose(0, 1), proj_rng)
        loss = pred_loss + j.sigreg_weight * reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, j.grad_clip)
        opt.step()
        if (step + 1) % 1000 == 0:
            history.append({"phase": "joint", "update": step + 1, "pred": float(pred_loss), "sigreg": float(reg_loss)})
            log(stage="joint", update=step + 1, pred=round(float(pred_loss), 5), sigreg=round(float(reg_loss), 4))
    P.eval()
    joint_world = {k: v.detach().clone() for k, v in world.state_dict().items()}
    with torch.no_grad():
        pool["s"] = torch.cat([P(pool["u"][i:i + 4096].flatten(0, 1).to(device)).view(-1, 6, 192).cpu()
                               for i in range(0, len(pool["u"]), 4096)])
    main_s = pool["s"][~pool["terminal"]].reshape(-1, 192)
    eig = torch.linalg.eigvalsh(torch.cov(main_s.T))
    var = pool["s"][~pool["terminal"]][:, 1:].reshape(-1, 192).var(0)
    lam = var.rsqrt()
    pool["weights"] = dict(pool["weights"]) | {"SP": lam / lam.mean()}
    install(P, device)
    I.PHASE1_UPDATES = 0
    I.WORLD_HOOK = lambda w: w.load_state_dict(joint_world)
    world2, heads, bridge_history, counts = I.train("SP", pool, device, log)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": "SP", "world": world2.state_dict(), "heads": heads.state_dict(), "projector": P.state_dict(),
                "history": history + bridge_history, "depth_counts": counts,
                "state_eigen_range": float(eig.max() / eig.clamp_min(1e-12).min()),
                "script_sha256": _sha256(HERE / "interface.py"), "sigreg_patch_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, OUT / "SP.pt")
    log(status="train_complete", arm="SP", state_eigen_range=float(eig.max() / eig.clamp_min(1e-12).min()))


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(JUDGE)
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in range(1, 12) for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < 61_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    mean, scale = standardize(actions4, train_rows)
    runs = []
    for seed in range(3):
        model, _ = probe_train("vector", actions4.shape[1:], (actions4 - mean) / scale, p, train_rows, hold_rows,
                               seed=seed, device=device, steps=3000)
        runs.append(expected_safe(scores(model, (judge["actions"][:, -4:].flatten(1) - mean) / scale,
                                         torch.arange(len(pj)), device), pj)[0])
    safe["actions_only"] = torch.stack(runs).mean(0)
    encoder, config = I.load_bridge()
    stored = torch.load(OUT / "SP.pt", map_location="cpu", weights_only=False)
    P = projector(config, device)
    P.load_state_dict(stored["projector"])
    P.eval()
    install(P, device)
    sleep = {}
    for name, arm, path in (("SP", "SP", OUT / "SP.pt"), ("W_s1", "W", ROOT / "artifacts/eda/interface_worlds_white/W.pt")):
        st = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(st["world"])
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(st["heads"])
        h.eval()
        risk = I.branches(b, h, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device)["p_dead"]
        safe[name] = expected_safe(risk, pj)[0]
        sleep[name] = int((risk[opp].argmin(1) == 6).sum())
        log(world=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4))
    test = lambda a, c, m: paired(safe[a][m], safe[c][m], seeds[m], draws=1000, seed=20261028)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    down = lambda r: r["difference"] < 0 and r["excludes_zero"]
    rules = {"SP_vs_DOWN": test("SP", "DOWN", opp), "SP_vs_actions_only": test("SP", "actions_only", opp),
             "SP_vs_DOWN_zombie": test("SP", "DOWN", zombie), "SP_vs_W_zombie": test("SP", "W_s1", zombie)}
    readings = {"primary": ("sp_trained_system_passes" if up(rules["SP_vs_DOWN"]) and up(rules["SP_vs_actions_only"])
                            and up(rules["SP_vs_DOWN_zombie"]) else "sp_trained_system_fails"),
                "secondary": ("sigreg_better" if up(rules["SP_vs_W_zombie"]) else
                              "whitening_better" if down(rules["SP_vs_W_zombie"]) else "equivalent")}
    reported = {f"{a}_vs_{c}": {"overall": test(a, c, opp), "zombie": test(a, c, zombie)} for a, c in (
        ("SP", "W_s1"), ("W_s1", "DOWN"), ("W_s1", "actions_only"))}
    evidence = {"schema": "d4mj_sigreg_patch_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "judge_seed_files": files,
                "state_eigen_range": stored["state_eigen_range"], "roots": {"judge": len(pj), "opportunity": int(opp.sum()),
                                                                            "zombie": int(zombie.sum())},
                "rules": rules, "readings": readings, "reported": reported, "sleep_choices": sleep,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                                      **{s_: float(v[m & opp].mean()) for s_, m in strat.items() if (m & opp).any()}}
                                  for k, v in safe.items()}}
    (HERE / "evidence/sigreg_patch.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="sigreg_patch_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "smoke"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    device = torch.device("cuda")
    if args.command == "smoke":
        global OUT
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 20, 20
        OUT = Path("/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/453c4e0b-57dd-4ecb-a41b-c69a33183014/scratchpad/sp_smoke")
        train(device, log)
    else:
        (train if args.command == "train" else score)(device, log)


if __name__ == "__main__":
    raise SystemExit(main())
