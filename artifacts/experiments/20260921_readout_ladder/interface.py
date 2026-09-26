"""Representation-interface comparison: patch-derived u->u Mamba vs CLS z->z Mamba, both alias-free.

After the reviewing agent's challenge (`20260925_h2_bound_challenge/README.md`, `a2c1aee`): CLS/z
encodes zombie adjacency weakly (linear AUC ~0.60) and lava well (~0.95); patch tokens encode both
(~1.00); H2's transition degrades zombie choice below its own input; and H2's trained head is aliased
to generation depth. This compares the two world INTERFACES with the terminal-layout repair held
fixed. u->u changes the world's input AND its prediction target, so this is a representation-interface
comparison, not an input-only ablation. Nothing here is a demonstrated LeWM repair until judged.

HELD FIXED (both arms)
  encoder   the frozen canonical Raw H2 encoder (`raw/bridge/step-002000.pt`, sha pinned)
  state     Z: projected z of each frame (192). U: the same encoder's 4x4 pooled patch grid (3,072)
            projected on a PCA-192 fitted on the TRAIN pool's main-window frames only (the
            construction of COMPACTNESS.md's Raw H2 arm, 0.783 at the root; fit on TRAIN here)
  world     canonical LeWMWorld (Mamba, M4 raw config: depth 6, width 256), init seed 7, from scratch
  data      the 2x2's windows (`spatial_pool_v1`, pool sha 7662de45): 24,576 uniform 6-frame main
            windows from the M4 corpus TRAIN split + 8,071 windows ending in each TRAIN death
  loss      MSE weighted per component by TRAIN-target variance^(-1/2), normalized to mean one: A/B/C's
            arm C rule (C - B +0.070* on identical data), applied identically to both arms (for z,
            which SIGReg keeps near-isotropic, it is near-uniform; its range is logged)
  phase 1   A/B/C's joint recipe (`abc.train_world`): teacher-only weighted MSE on 4-frame sub-windows
            of the main windows, 10,000 updates of 128, AdamW at the M4 joint settings, constant LR,
            batch seed 11, predictor BN in train mode
  phase 2   H2's bridge (`train_bridge` / `bridge_losses`): encoder frozen, predictor BN statistics
            frozen, agent readout trained, H2's own `Heads` (policy head masked: it supervises the
            logged action), `bridge_rollout`, `_bridge_head_losses` (0.5 observed prefix / 0.25 observed
            suffix / 0.25 generated suffix), 0.8 / 0.2 `paired_terminal_loss` on the terminal rows,
            `_phase_balance` over dynamics / reward / continuation, `phase_optimizer`, `_phase_lr`, bf16.
            H2's frame budget: 2,000 updates x ~1,120 frames = 9,333 updates of 32 main + 8 terminal
            6-frame windows.
  ALIAS-FREE TERMINAL LAYOUT (the only departure from H2's bridge): every update draws its generation
            depth d in {1, 2} (seeded). The window is frames (2-d)..5 of each 6-frame window: 4 observed
            frames, then the last d generated (for death rows the death is always the last frame). Death rows
            (terminal windows) and alive rows (main windows) are generated at the same depths, so neither
            depth nor position predicts the label; the evaluation depth (4 observed + 1 generated) is
            trained half the time.

JUDGEMENT -- `observe.py collect --seed-start 55000 --target-opportunity 800 --max-seeds 1500 --out
artifacts/eda/observe_fresh_v6`, collected after this commit, read once. Fit / selection for every probe:
the partition's FIT-train roots (fit) and FIT-dev fork rows (selection); never the gate-reserved or
unallocated seeds. Evaluator protocol: 4 observed frames through `teacher`, one `advance` per action.
Metric: expected safe = 1 - P(death1 | chosen), 32-key P, one-step opportunity roots; paired
episode-seed-clustered 95% intervals, 1,000 draws. Probes: frozen_ladder's all-action ranking harness,
3 seeds, 3,000 updates, one shared per-branch head (512 -> 1) for every branch input:
  root_A        4 observed states + 3 past actions + the candidate action   (the reviewer's matched root)
  generated_A   the generated successor state + the candidate action
  features_A    H2-style agent readout of the generated successor + the candidate action (reported)
  trained_A     the arm's OWN continuation head on its generated successor (no refit)
Controls: DOWN (the FIT action prior), actions_only, tokens_attn on the root's patch tokens.

DECLARED READINGS (committed before collection)
  P0         tokens_attn - DOWN on zombie opportunity roots not resolved > 0         -> void
  interface  generated_U - generated_Z on zombie roots: resolved > 0 -> patch_interface_better;
             resolved < 0 -> patch_interface_worse; else -> no_difference
  retention  per arm, generated_A - root_A on zombie roots resolved < 0 -> world_degrades_A, else retains_A
  world      u_world_state_passes iff generated_U beats DOWN overall and on zombie roots, beats
             actions_only overall (all resolved), and retains_U. A fork-probed WORLD-STATE claim only.
  trained    per arm, trained_system_passes iff trained_A beats DOWN overall and on zombie roots and
             actions_only overall (all resolved). Declared prediction: fails in BOTH arms -- factual
             outcome labels teach a context shortcut (SPATIAL.md steps 7-8).
Reported, not ruled: every arm by zombie / night / lava / day / stay-kills-move-survives stratum;
features_A; trained heads' SLEEP choices and chosen-action histograms; per-seed means; fidelity --
the trained head's within-root death AUC on REAL vs GENERATED successors (judge opportunity roots), and
on 400 DEV terminal episodes (corpus DEV split) its paired death vs alive-10 / pre-death AUC on the
generated and the real successor at the evaluator position (canonical H2 there: 0.51 generated).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N, W, WIDTH = 17, 6, 192
CHECKPOINT = ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt"
DATASET = ROOT / "artifacts/lewm_m4_canonical/raw/dataset.json"
POOL_IN = ROOT / "artifacts/eda/spatial_pool_v1"
POOL = ROOT / "artifacts/eda/interface_pool_v1"
WORLDS = ROOT / "artifacts/eda/interface_worlds_v1"
SEALED = ROOT / "artifacts/eda/observe_fresh_v6"
# Replication knobs (replicate.py sets them); these defaults ARE the seed-1 run, unchanged.
SEEDS = {"init": 7, "phase1": 11, "heads": 2, "phase2": 17, "depth": 13}
MIN_JUDGE_SEED, USED_STORES, EVIDENCE = 55_000, range(1, 6), "interface.json"
SEED1_TRAIN_SHA = "e2e3578b00ab884d2362acb713ba3a810a509fccbfc6a0b2cf3452bf17c63882"   # interface.py @ 9aefa86d, which trained the seed-1 worlds
PHASE1_UPDATES, PHASE1_BATCH = 10_000, 128
PHASE2_UPDATES, MAIN, TERMINAL = 9_333, 32, 8
ARMS = ("U", "Z")
KEY = {"U": "u", "Z": "z"}


def load_bridge():
    from d4mj.experiments import _load_bridge_parent
    if _sha256(CHECKPOINT) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("checkpoint differs from the one every earlier rung used")
    bundle, heads, _ = _load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    return bundle.encoder, bundle.config


@torch.no_grad()
def encode(encoder, frames, device, batch=32):
    """uint8 [n, t, 63, 63, 3] -> raw z [n, t, 192] and raw 4x4 pooled grid [n, t, 3072] (fp32, cpu)."""
    zs, gs = [], []
    for i in range(0, len(frames), batch):
        f = torch.as_tensor(np.asarray(frames[i:i + batch])).to(device)
        z, _, pooled = encoder.export(f, grid=4)
        zs.append(z[:, :, 0].float().cpu())
        gs.append(pooled.flatten(2).float().cpu())
    return torch.cat(zs), torch.cat(gs)


def project(pca, grid):
    return (grid - pca["mean"]) @ pca["basis"]


def state_of(arm, pca, z, grid):
    return project(pca, grid) if arm == "U" else z


# ------------------------------------------------------------------------------------------ pool
def build_pool(device, log):
    from d4mj.data import load_joint_corpus
    source = torch.load(POOL_IN / "pool.pt", weights_only=False, mmap=True)
    if _sha256(POOL_IN / "pool.pt") != json.loads((POOL_IN / "pool.json").read_text())["pool_sha256"]:
        raise SystemExit("the 2x2 pool changed")
    encoder, config = load_bridge()
    record = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    by_id = {e.episode_id: e for e in episodes}
    ids = source["ids"]
    n = len(ids)
    z = torch.empty(n, W, WIDTH)
    grid = torch.empty(n, W, 16 * WIDTH, dtype=torch.float16)
    for b in range(0, n, 256):
        chunk = ids[b:b + 256]
        zz, gg = encode(encoder, np.stack([np.asarray(by_id[e].observations[s:s + W]) for e, s in chunk]), device)
        z[b:b + len(chunk)], grid[b:b + len(chunk)] = zz, gg.half()
        if b % 8192 == 0:
            log(stage="encoding", done=b + len(chunk), of=n)
    main = ~source["terminal"]
    x = grid[main].flatten(0, 1)
    mean = x.float().mean(0, keepdim=True)
    cov = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=device)
    for j in range(0, len(x), 8192):
        c = (x[j:j + 8192].float() - mean).to(device).double()
        cov += c.T @ c
    values, vectors = torch.linalg.eigh(cov / (len(x) - 1))
    order = values.argsort(descending=True)[:WIDTH]
    scale = values[order].clamp_min(0).sqrt()
    rank = int((scale > scale[0] * 1e-3).sum())
    if rank < WIDTH:
        raise SystemExit(f"PCA rank {rank} < {WIDTH}")
    pca = {"mean": mean, "basis": vectors[:, order].float().cpu(), "explained": float(values[order].sum() / values.sum())}
    del x, cov, vectors
    u = torch.cat([project(pca, grid[j:j + 2048].float()) for j in range(0, n, 2048)])
    weights, meta = {}, {}
    for arm, s in (("U", u), ("Z", z)):
        var = s[main][:, 1:].reshape(-1, WIDTH).var(0)
        lam = var.rsqrt()
        weights[arm] = lam / lam.mean()
        meta[arm] = {"variance_range": float(var.max() / var.min()), "weight_range": float(weights[arm].max() / weights[arm].min()),
                     "total_variance": float(var.sum())}
    pool = {"u": u, "z": z, "pca": pca, "weights": weights,
            **{k: source[k].clone() for k in ("actions", "reward_led", "reward_valid", "alive", "dh", "terminal")}}
    POOL.mkdir(parents=True, exist_ok=True)
    torch.save(pool, POOL / "pool.pt")
    meta = {"windows": n, "pca_explained": pca["explained"], "arms": meta, "source_pool_sha256": _sha256(POOL_IN / "pool.pt"),
            "checkpoint_sha256": _sha256(CHECKPOINT), "pool_sha256": _sha256(POOL / "pool.pt")}
    (POOL / "pool.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ----------------------------------------------------------------------------------------- train
def world_bundle(config, encoder, device):
    from d4mj.lewm import LeWMWorld
    from d4mj.world_api import ModelBundle
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(SEEDS["init"])
        torch.cuda.manual_seed_all(SEEDS["init"])
        world = LeWMWorld(config).to(device)
    return ModelBundle.from_models(config, encoder, world)


def head_targets(pool, rows, shift, config, device):
    """H2's head_targets fields for windows (shift..5); the policy head is masked out."""
    from d4mj.agent import _leads
    leads, length, b = config.agent.mtp_leads, W - shift, len(rows)
    zeros = torch.zeros(b, length, leads, device=device)
    reward = pool["reward_led"][rows][:, shift:].to(device)
    return {"action": zeros, "action_valid": zeros, "reward": _leads(reward, leads, fill=0.0),
            "valid": _leads(pool["reward_valid"][rows][:, shift:].float().to(device), leads, fill=0.0),
            "continuation": pool["alive"][rows][:, shift:].float().to(device)[..., None],
            "continuation_valid": torch.ones(b, length, 1, device=device),
            "policy_rows": zeros[:, :1, :1], "reward_rows": torch.ones(b, 1, 1, device=device)}


def train(arm, pool, device, log):
    from d4mj.agent import Heads, paired_terminal_loss
    from d4mj.train import (_bridge_head_losses, _phase_balance, _phase_lr, autocast_context, bridge_rollout,
                            optimizer_step, phase_optimizer)
    encoder, config = load_bridge()
    bundle = world_bundle(config, encoder, device)
    world, key, lam = bundle.world, KEY[arm], pool["weights"][arm].to(device)
    main_rows, terminal_rows = torch.where(~pool["terminal"])[0], torch.where(pool["terminal"])[0]
    weighted = lambda p, t: ((p.float() - t.float()).square() * lam).mean()
    history = []

    # ---- phase 1: A/B/C's joint recipe, teacher-only ----
    world.train().requires_grad_(True)
    world.agent_readout.requires_grad_(False)
    j = config.joint
    parameters = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=j.learning_rate, betas=tuple(j.betas), eps=j.optimizer_eps,
                                  weight_decay=j.weight_decay)
    order = torch.Generator().manual_seed(SEEDS["phase1"])
    for step in range(PHASE1_UPDATES):
        idx = main_rows[torch.randint(len(main_rows), (PHASE1_BATCH,), generator=order)]
        off = torch.randint(3, (PHASE1_BATCH,), generator=order)
        ar = torch.arange(PHASE1_BATCH)[:, None]
        src = pool[key][idx][ar, off[:, None] + torch.arange(4)].to(device)[:, :, None]
        acts = pool["actions"][idx][ar, off[:, None] + torch.arange(3)].to(device)
        loss = weighted(world.teacher(src, acts).predicted, src[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, j.grad_clip)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            history.append({"phase": 1, "update": step + 1, "dynamics": float(loss)})
            log(stage="phase1", arm=arm, update=step + 1, loss=round(float(loss), 5))

    # ---- phase 2: H2's bridge with the alias-free terminal layout ----
    world.train().requires_grad_(True)
    world.agent_readout.requires_grad_(True)
    for module in world.predictor_projector.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(config.seed + SEEDS["heads"])
        heads = Heads(config).to(device)
    optimizer = phase_optimizer([world, heads], config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    order, depths, balance = torch.Generator().manual_seed(SEEDS["phase2"]), torch.Generator().manual_seed(SEEDS["depth"]), {}
    counts = {1: 0, 2: 0}
    for update in range(PHASE2_UPDATES):
        d = 1 + int(torch.randint(2, (), generator=depths))
        counts[d] += 1
        shift = 2 - d
        main = main_rows[torch.randint(len(main_rows), (MAIN,), generator=order)]
        term = terminal_rows[torch.randint(len(terminal_rows), (TERMINAL,), generator=order)]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config):
            parts = {}
            for name, rows in (("main", main), ("terminal", term)):
                s = pool[key][rows][:, shift:].to(device)[:, :, None]
                a = pool["actions"][rows][:, shift:].to(device)
                teacher, generated, generated_features, anchor = bridge_rollout(bundle, s, a, d, None)
                parts[name] = (s, teacher, generated, generated_features, anchor, head_targets(pool, rows, shift, config, device))
            s, teacher, generated, generated_features, anchor, targets = parts["main"]
            losses = _bridge_head_losses(heads, teacher.features, generated_features, targets, config, anchor)
            losses.pop("policy")
            losses["dynamics"] = weighted(teacher.predicted, s[:, 1:]) + weighted(generated, s[:, anchor + 1:])
            _, t_teacher, _, t_features, t_anchor, t_targets = parts["terminal"]
            observed = heads(t_teacher.features) | {"centers": heads.centers}
            recursive = heads(torch.cat((t_teacher.features[:, :t_anchor + 1], t_features), 1)) | {"centers": heads.centers}
            losses["continuation"] = 0.8 * losses["continuation"] + 0.2 * paired_terminal_loss(recursive, observed, t_targets)
            objective = _phase_balance(losses, balance, config.agent.rms_decay)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"nonfinite objective at update {update}")
        norm = optimizer_step(optimizer, objective, parameters, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=False)
        if (update + 1) % 1000 == 0 or update + 1 == PHASE2_UPDATES:
            row = {"phase": 2, "update": update + 1, "gradient_norm": float(norm),
                   **{k: float(v.detach()) for k, v in losses.items()}}
            history.append(row)
            log(stage="phase2", arm=arm, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    return world, heads, history, counts


# ----------------------------------------------------------------------------------------- score
@torch.no_grad()
def branches(bundle, heads, pca, arm, encoder, frames, actions, device, successors=None, batch=16):
    """Evaluator protocol: 4 observed frames through teacher, one advance per action."""
    from d4mj.train import autocast_context
    config = bundle.config
    out = {k: [] for k in ("root", "generated", "features", "p_dead", "p_dead_real")}
    for i in range(0, len(frames), batch):
        z, grid = encode(encoder, frames[i:i + batch, -4:], device)
        n = len(z)
        s = state_of(arm, pca, z, grid).to(device)
        past = actions[i:i + batch, -3:].argmax(-1)
        if bool((past >= N).any()):
            raise SystemExit("a context action is BOS: fewer than four real frames")
        past = past.to(device)
        acts = torch.arange(N, device=device).repeat(n)[:, None]
        candidate = F.one_hot(torch.arange(N), N).float().repeat(n, 1)
        with autocast_context(config):
            state = bundle.world.teacher(s[:, :, None], past).state
            fan = bundle.repeat_state(state, N)
            advanced, features = bundle.advance(fan, acts)
            dead = lambda f: (1 - torch.sigmoid(heads(f)["continuation"][:, -1, 0].float())).view(n, N).cpu()
            out["p_dead"].append(dead(features))
            if successors is not None:
                rz, rgrid = encode(encoder, successors[i:i + batch].flatten(0, 1)[:, None], device)
                real = state_of(arm, pca, rz, rgrid).to(device)[:, :, None]
                _, real_features = bundle.world.observe_latent(fan, acts, real)
                out["p_dead_real"].append(dead(real_features))
        root = torch.cat([s.flatten(1).cpu(), F.one_hot(past.cpu(), N + 1).float().flatten(1)], 1)
        out["root"].append(torch.cat([root.repeat_interleave(N, 0), candidate], 1).view(n, N, -1))
        out["generated"].append(torch.cat([advanced.latent[:, 0, 0].float().cpu(), candidate], 1).view(n, N, -1))
        out["features"].append(torch.cat([features[:, -1, 0].float().cpu(), candidate], 1).view(n, N, -1))
    return {k: torch.cat(v) for k, v in out.items() if v}


@torch.no_grad()
def dev_fidelity(bundle, heads, pca, arm, encoder, device, episodes, batch=32):
    """Trained P(dead) at the evaluator position for death / pre-death / alive-10 frames."""
    from d4mj.train import autocast_context
    result = {}
    for name, back in (("death", 0), ("pre_death", 1), ("alive_10", 10)):
        gen, real = [], []
        for i in range(0, len(episodes), batch):
            chunk = episodes[i:i + batch]
            ends = [len(e) - back for e in chunk]
            z, grid = encode(encoder, np.stack([np.asarray(e.observations[t - 4:t + 1]) for e, t in zip(chunk, ends)]), device)
            s = state_of(arm, pca, z, grid).to(device)[:, :, None]
            past = torch.stack([torch.as_tensor(np.asarray(e.actions_taken[t - 4:t - 1])).long() for e, t in zip(chunk, ends)]).to(device)
            act = torch.tensor([int(e.actions_taken[t - 1]) for e, t in zip(chunk, ends)], device=device)[:, None]
            with autocast_context(bundle.config):
                state = bundle.world.teacher(s[:, :4], past).state
                _, features = bundle.advance(state, act)
                _, real_features = bundle.world.observe_latent(state, act, s[:, 4:5])
                dead = lambda f: (1 - torch.sigmoid(heads(f)["continuation"][:, -1, 0].float())).cpu()
                gen.append(dead(features))
                real.append(dead(real_features))
        result[name] = (torch.cat(gen), torch.cat(real))
    pair = lambda a, b: float(((a > b).float() + 0.5 * (a == b).float()).mean())
    return {f"{kind}_{stat}": v for kind, k in (("generated", 0), ("real", 1)) for stat, v in (
        ("death_mean", float(result["death"][k].mean())), ("pre_death_mean", float(result["pre_death"][k].mean())),
        ("auc_vs_alive_10", pair(result["death"][k], result["alive_10"][k])),
        ("auc_vs_pre_death", pair(result["death"][k], result["pre_death"][k])))}


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.data import load_joint_corpus
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from diagnose import within_auc
    from frozen_heads import dev_rows
    from frozen_ladder import arm_input, scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    pca = pool["pca"]
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(SEALED)
    judge["successors"] = torch.stack([r["successors"] for f in sorted(SEALED.glob("seed-*.pt"))
                                       for r in torch.load(f, weights_only=False)])
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in USED_STORES
            for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < MIN_JUDGE_SEED or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")

    encoder, config = load_bridge()
    record = json.loads(DATASET.read_text())
    episodes, _ = load_joint_corpus(record["paths"], config)
    dev_terminal = [e for e in episodes if e.split == "dev" and bool(e.terminated[-1]) and len(e) >= 15]
    dev_terminal = [dev_terminal[i] for i in torch.randperm(len(dev_terminal), generator=torch.Generator().manual_seed(20261008))[:400].tolist()]
    del episodes
    data, fidelity, histories = {}, {}, {}
    for arm in ARMS:
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        if stored["pool_sha256"] != json.loads((POOL / "pool.json").read_text())["pool_sha256"] or \
                stored["script_sha256"] not in (_sha256(Path(__file__)), SEED1_TRAIN_SHA):
            raise SystemExit(f"{arm} was not trained by this script on this pool")
        bundle = world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device)
        heads.load_state_dict(stored["heads"])
        heads.eval()
        data[arm] = {"fit": branches(bundle, heads, pca, arm, encoder, fit["frames"], fit["actions"], device),
                     "dev": branches(bundle, heads, pca, arm, encoder, dev["frames"], dev["actions"], device),
                     "judge": branches(bundle, heads, pca, arm, encoder, judge["frames"], judge["actions"], device,
                                       judge["successors"])}
        fidelity[arm] = dev_fidelity(bundle, heads, pca, arm, encoder, device, dev_terminal)
        histories[arm] = {"history": stored["history"], "depth_counts": stored["depth_counts"]}
        log(stage="branches", arm=arm, fidelity={k: round(v, 4) for k, v in fidelity[arm].items()})
        del bundle, heads
    with torch.no_grad():
        for d in (fit, dev, judge):
            d["tokens1"] = torch.cat([encoder._hidden(d["frames"][i:i + 64, -1:].to(device))[2].cpu()
                                      for i in range(0, len(d["frames"]), 64)])
    del encoder
    torch.cuda.empty_cache()

    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    per_seed = {}

    def probe(name, kind, xf, xj):
        mean, scale = standardize(xf, train_rows)
        xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
        runs = []
        for seed in range(3):
            model, _ = probe_train(kind, xf.shape[1:], xf, p, train_rows, hold_rows, seed=seed, device=device, steps=3000)
            runs.append(expected_safe(scores(model, xj, torch.arange(len(pj)), device), pj)[0])
            del model
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), per_seed=[round(v, 4) for v in per_seed[name]])

    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    probe("actions_only", "vector", actions4, judge["actions"][:, -4:].flatten(1))
    probe("tokens_attn", "tokens_attn", torch.cat([fit["tokens1"], dev["tokens1"]]), judge["tokens1"])
    for arm in ARMS:
        for rung in ("root", "generated", "features"):
            probe(f"{rung}_{arm}", "branch", torch.cat([data[arm]["fit"][rung], data[arm]["dev"][rung]]),
                  data[arm]["judge"][rung])
        safe[f"trained_{arm}"] = expected_safe(data[arm]["judge"]["p_dead"], pj)[0]
        log(arm=f"trained_{arm}", safe=round(float(safe[f"trained_{arm}"][opp].mean()), 4))

    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    stay = opp & (pj[:, 0] > 0.5) & (pj[:, 1:5].amin(1) < 0.5)
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261009)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    down = lambda r: r["difference"] is not None and r["difference"] < 0 and r["excludes_zero"]
    rules = {"P0_tokens_attn_vs_DOWN_zombie": test("tokens_attn", "DOWN", zombie),
             "interface_zombie": test("generated_U", "generated_Z", zombie),
             "interface_overall": test("generated_U", "generated_Z", opp),
             **{f"retention_{a}_zombie": test(f"generated_{a}", f"root_{a}", zombie) for a in ARMS},
             "world_U_vs_DOWN": test("generated_U", "DOWN", opp), "world_U_vs_DOWN_zombie": test("generated_U", "DOWN", zombie),
             "world_U_vs_actions_only": test("generated_U", "actions_only", opp),
             **{f"trained_{a}_{c}": test(f"trained_{a}", ref, m) for a in ARMS for c, ref, m in
                (("vs_DOWN", "DOWN", opp), ("vs_DOWN_zombie", "DOWN", zombie), ("vs_actions_only", "actions_only", opp))}}
    retains = {a: not down(rules[f"retention_{a}_zombie"]) for a in ARMS}
    readings = {
        "P0": "ok" if up(rules["P0_tokens_attn_vs_DOWN_zombie"]) else "void",
        "interface": ("patch_interface_better" if up(rules["interface_zombie"]) else
                      "patch_interface_worse" if down(rules["interface_zombie"]) else "no_difference"),
        "retention": {a: ("retains" if retains[a] else "world_degrades") for a in ARMS},
        "world": ("u_world_state_passes" if up(rules["world_U_vs_DOWN"]) and up(rules["world_U_vs_DOWN_zombie"]) and
                  up(rules["world_U_vs_actions_only"]) and retains["U"] else "u_world_state_fails"),
        "trained": {a: ("trained_system_passes" if all(up(rules[f"trained_{a}_{c}"]) for c in
                                                       ("vs_DOWN", "vs_DOWN_zombie", "vs_actions_only")) else "trained_system_fails")
                    for a in ARMS}}
    fatal = pj > 0.5
    within = lambda key, arm: within_auc([s for s in data[arm]["judge"][key][opp]], [f for f in fatal[opp]])
    names = ["DOWN", "actions_only", "tokens_attn", *[f"{r}_{a}" for a in ARMS for r in ("root", "generated", "features", "trained")]]
    reported = {
        "expected_safe": {k: {"overall": float(safe[k][opp].mean()), "zombie": float(safe[k][zombie].mean()),
                              "stay_kills_move_survives": float(safe[k][stay].mean()),
                              **{s: float(safe[k][m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                          for k in names},
        "per_seed": per_seed,
        "contrasts": {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie), "night": test(a, b, opp & strat["night"])}
                      for a, b in (("generated_U", "generated_Z"), ("root_U", "root_Z"), ("features_U", "features_Z"),
                                   ("trained_U", "trained_Z"), ("generated_U", "tokens_attn"), ("root_U", "tokens_attn"))},
        "fidelity": {a: {"trained_within_root_auc_real": within("p_dead_real", a),
                         "trained_within_root_auc_generated": within("p_dead", a), **fidelity[a]} for a in ARMS},
        "trained_sleep_choices": {a: int((data[a]["judge"]["p_dead"][opp].argmin(1) == 6).sum()) for a in ARMS},
        "trained_chosen_histogram": {a: torch.bincount(data[a]["judge"]["p_dead"][opp].argmin(1), minlength=N).tolist()
                                     for a in ARMS},
        "training": histories}
    evidence = {"schema": "d4mj_interface_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "pool": json.loads((POOL / "pool.json").read_text()),
                "judge_store": str(SEALED), "judge_manifest": manifest, "judge_seed_files": files,
                "roots": {"fit": len(pf), "dev": len(pd), "judge": len(pj), "opportunity": int(opp.sum()),
                          "zombie_opportunity": int(zombie.sum()), "stay_kills_move_survives": int(stay.sum())},
                "prior_action": prior, "rules": rules, "readings": readings, "reported": reported}
    (HERE / f"evidence/{EVIDENCE}").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="interface_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pool", "train", "score", "smoke"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    if args.command == "pool":
        log(status="pool_complete", **build_pool(device, log))
    elif args.command in ("train", "smoke"):
        global PHASE1_UPDATES, PHASE2_UPDATES
        if args.command == "smoke":
            PHASE1_UPDATES, PHASE2_UPDATES = 20, 20
        pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
        world, heads, history, counts = train(args.arm, pool, device, log)
        if args.command == "train":
            WORLDS.mkdir(parents=True, exist_ok=True)
            torch.save({"arm": args.arm, "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                        "depth_counts": counts, "script_sha256": _sha256(Path(__file__)),
                        "pool_sha256": json.loads((POOL / "pool.json").read_text())["pool_sha256"]}, WORLDS / f"{args.arm}.pt")
        log(status=f"{args.command}_complete", arm=args.arm, depth_counts=counts)
    else:
        score(device, log)


if __name__ == "__main__":
    raise SystemExit(main())
