"""Frozen-representation ladder in the observability harness, with the pixel controls it lacked.

OBSERVABILITY.md found that a from-scratch CNN on the last four frames (+ the last four actions)
reaches 0.768 expected safe choice on one-step death against a 0.628 action prior, while every
representation of the model's own that RANKPROBE tried sat at or below the prior -- but those were
scored with a different head on different roots. This scores both in ONE harness.

Data: exactly the observability test's roots, asserted by content hash against
`evidence/observability.json` for both splits (no directory glob can add a seed), with every
shard file hashed and the Raw H2 checkpoint asserted equal to the one CONFIRM used. The judgement
roots were fresh for the observability test and have since been inspected: EXPLORATORY here. Any
repair this motivates is judged on a new seed block.

Every arm: same objective (expected-risk pair ranking over 32-key P), AdamW 1e-3 / 1e-4, batches of
128 roots, 3,000 updates, selection every 100 on the same inner 15% of FIT seeds, three probe seeds,
and a 512-wide hidden layer before the scores.

  pixel controls (trained from scratch, Nature-DQN CNN trunk)
    actions_only        the last 4 actions alone
    pixels1 / pixels4   the last 1 / 4 frames, NO actions
    pixels4_actions     the observability arm itself, through its own code -- reproduction check
  frozen encoder (Raw H2 checkpoint, last root frame unless noted)
    tokens1_flat        the 81 unpooled patch tokens (one per tile), flattened -> 512
    tokens1_attn        the same tokens, position-aware attention readout (2026-09-19 TokenHead)
    cls_pooled1         CLS + the 4x4 pooled grid
    cls1 / cls4         CLS of the last 1 / 4 frames
    z1 / z4             projected z of the last 1 / 4 frames -- what the world consumes
  frozen world (the gate's 4-frame context)
    root_features       the world's root readout
    u                   Mamba output per action branch   } one head shared across the 17
    generated_z         predicted successor per branch   } branches, scoring each
    generated_features  agent readout per branch         }

Reported, not ruled on: expected safe per arm and probe seed, fit / inner / judgement, parameter
count; paired seed-clustered contrasts along the chain; day / night and hazard strata (lava,
adjacent zombie, arrow or skeleton); chosen-action histograms. These rungs are not a nested
information chain, so a falling score is a place to look, not a verdict: if patch tokens fail, the
next check is whether they decode the visible tiles, mobs and HUD -- night-aware -- before anything
about the encoder is concluded.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from ladder import paired  # noqa: E402
from observability import FIT_STATE, FRESH, expected_safe, fit as obs_fit, load, soft_rank  # noqa: E402

N, WIDTH, TOKENS = 17, 192, 81
LAVA, NEIGHBOURS = 14, ((2, 4), (4, 4), (3, 3), (3, 5))


@torch.no_grad()
def materialize(bundle, frames, actions, batch=16):
    """Every frozen rung for every root, from its last four frames and the three actions between them."""
    world, enc, device, count = bundle.world, bundle.encoder, bundle.device, bundle.n_actions
    keys = ("tokens1", "cls_pooled1", "cls1", "cls4", "z1", "z4", "root_features", "u",
            "generated_z", "generated_features")
    out = {k: [] for k in keys}
    for i in range(0, len(frames), batch):
        f = frames[i:i + batch, -4:].to(device)
        past = actions[i:i + batch, -3:].argmax(-1)
        if bool((past >= N).any()):
            raise SystemExit("a context action is BOS: fewer than four real frames")
        past = past.to(device)
        n = len(f)
        z, cls, tokens, _, _ = enc._hidden(f)
        if i == 0 and not torch.allclose(z.reshape(n, 4, 1, -1), enc(f), atol=1e-5, rtol=1e-4):
            raise SystemExit("encoder._hidden disagrees with the encoder's forward")
        z, cls, tokens = z.reshape(n, 4, -1), cls.reshape(n, 4, -1), tokens.reshape(n, 4, TOKENS, WIDTH)
        last = tokens[:, -1]
        grid = nn.functional.adaptive_avg_pool2d(last.transpose(1, 2).reshape(n, WIDTH, 9, 9), 4)
        grid = grid.flatten(2).transpose(1, 2).reshape(n, -1)
        state = world.teacher(z[:, :, None], past).state
        acts = torch.arange(count, device=device).repeat(n)[:, None]
        advanced, generated = bundle.advance(bundle.repeat_state(state, count), acts)
        branch = lambda x: x.reshape(n, count, -1).cpu()
        for key, value in (("tokens1", last.half().cpu()), ("cls_pooled1", torch.cat((cls[:, -1], grid), -1).cpu()),
                           ("cls1", cls[:, -1].cpu()), ("cls4", cls.flatten(1).cpu()),
                           ("z1", z[:, -1].cpu()), ("z4", z.flatten(1).cpu()),
                           ("root_features", world.features(state)[:, -1, 0].cpu()),
                           ("u", branch(advanced.history)), ("generated_z", branch(advanced.latent)),
                           ("generated_features", branch(generated[:, -1, 0]))):
            out[key].append(value)
    return {k: torch.cat(v) for k, v in out.items()}


class Head(nn.Module):
    """Every arm's scorer: an input trunk, then [.] -> 512 -> 17 (or -> 1 per branch)."""

    def __init__(self, kind, shape):
        super().__init__()
        self.kind = kind
        if kind == "vector":
            self.net = nn.Sequential(nn.Linear(shape[-1], 512), nn.ReLU(), nn.Linear(512, N))
        elif kind == "wide":
            # CLS capacity control (boundary.py): ~40x the parameters of the patch-attention head.
            self.net = nn.Sequential(nn.Linear(shape[-1], 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(),
                                     nn.Linear(2048, N))
        elif kind == "branch":
            self.net = nn.Sequential(nn.Linear(shape[-1], 512), nn.ReLU(), nn.Linear(512, 1))
        elif kind == "tokens_flat":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(TOKENS * WIDTH, 512), nn.ReLU(), nn.Linear(512, N))
        elif kind == "tokens_attn":
            # 2026-09-19 `readout.TokenHead`: learned positions + attention pooling keep position
            # explicit without a 15,552-wide first layer; its 128-wide output feeds the shared head.
            self.position = nn.Parameter(torch.zeros(TOKENS, WIDTH))
            self.value, self.score = nn.Linear(WIDTH, 128), nn.Linear(WIDTH, 1)
            self.net = nn.Sequential(nn.GELU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, N))
        elif kind.startswith("pixels"):
            k = int(shape[0])
            self.trunk = nn.Sequential(nn.Conv2d(3 * k, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            self.net = nn.Sequential(nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, N))

    def forward(self, x):
        if self.kind == "branch":
            return self.net(x).squeeze(-1)
        if self.kind == "tokens_attn":
            flat = x + self.position
            pooled = (self.value(flat) * self.score(flat).softmax(1)).sum(1)
            return self.net(pooled)
        if self.kind.startswith("pixels"):
            return self.net(self.trunk(x.permute(0, 1, 4, 2, 3).flatten(1, 2).float() / 255.0))
        return self.net(x)


def arm_input(data, arm, train_rows):
    """(kind, tensor, shape) for an arm, standardized on FIT non-inner rows where it is a feature."""
    if arm == "actions_only":
        return "vector", data["actions"][:, -4:].flatten(1), None
    if arm in ("pixels1", "pixels4"):
        k = int(arm[-1])
        return f"pixels{k}", data["frames"][:, -k:], (k,)
    x = data[arm].float() if arm == "tokens1" else data[arm]
    kind = {"tokens1": None, "u": "branch", "generated_z": "branch", "generated_features": "branch"}.get(arm, "vector")
    return kind, x, None


def standardize(x, rows):
    flat = x[rows].reshape(-1, x.shape[-1]).float()
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def train(kind, shape, x, p, train_rows, hold_rows, *, seed, device, steps, batch=128, every=100):
    torch.manual_seed(seed)
    model = Head(kind, shape).to(device)
    params = sum(t.numel() for t in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    best, best_step, state, curve = -1.0, 0, None, []
    for step in range(steps):
        idx = train_rows[torch.randint(len(train_rows), (batch,), generator=gen)]
        loss = soft_rank(model(x[idx].to(device)), p[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % every == 0 or step + 1 == steps:
            model.eval()
            safe, opp = expected_safe(scores(model, x, hold_rows, device), p[hold_rows])
            model.train()
            value = float(safe[opp].mean()) if bool(opp.any()) else float("-inf")
            curve.append(round(value, 4))
            if value > best or state is None:
                best, best_step = value, step + 1
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), {"selected_step": best_step, "inner_safe": best, "curve": curve, "parameters": params}


@torch.no_grad()
def scores(model, x, rows, device, batch=256):
    return torch.cat([model(x[rows[i:i + batch]].to(device)).cpu() for i in range(0, len(rows), batch)])


def strata(visible):
    """Day/night and non-exclusive hazard strata from the (pre-darkness) visible state vector."""
    tiles = visible[:, :7 * 9 * 17].reshape(-1, 7, 9, 17)
    mobs = visible[:, 1071:1071 + 7 * 9 * 7].reshape(-1, 7, 9, 7)
    lava = torch.stack([tiles[:, r, c, LAVA] > 0 for r, c in NEIGHBOURS], 1).any(1)
    zombie = torch.stack([mobs[:, r, c, 0] > 0 for r, c in NEIGHBOURS], 1).any(1)
    ranged = (mobs[:, 2:5, 3:6, 3:].sum((1, 2, 3)) > 0) | (mobs[..., 2].sum((1, 2)) > 0)
    night = visible[:, 1521] < 0.5
    return {"day": ~night, "night": night, "lava_adjacent": lava, "zombie_adjacent": zombie,
            "arrow_or_skeleton": ranged, "none_of_these": ~(lava | zombie | ranged)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="frozen_ladder")
    parser.add_argument("--smoke", type=int, default=0, help="smoke only: subsample, skip the identity pin")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    # ---- pin the data and the checkpoint ----
    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())
    if not args.smoke and {s: data[s]["identity"] for s in data} != recorded["identity"]:
        raise SystemExit("data differs from what the observability test measured")
    shards = {name: hashlib.sha256("".join(f"{p.name}:{_sha256(p)}" for p in sorted(folder.glob("seed-*.pt")))
                                   .encode()).hexdigest()
              for name, folder in (("fresh", FRESH), ("fit_state", FIT_STATE))}
    checkpoint = args.run / "bridge/step-002000.pt"
    checkpoint_sha = _sha256(checkpoint)
    confirm_sha = json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]
    if checkpoint_sha != confirm_sha:
        raise SystemExit("checkpoint differs from the one every earlier rung used")
    if args.smoke:
        for split in data:
            keep = torch.arange(0, len(data[split]["seed"]), max(1, len(data[split]["seed"]) // args.smoke))
            data[split] = {k: (v[keep] if torch.is_tensor(v) else v) for k, v in data[split].items()}

    from d4mj.experiments import _load_bridge_parent
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    for split in data:
        data[split].update(materialize(bundle, data[split]["frames"], data[split]["actions"]))
    del bundle, heads
    torch.cuda.empty_cache()
    log(stage="materialized", fit=len(data["fit"]["seed"]), judge=len(data["judge"]["seed"]))

    # The observability test's inner split, rebuilt the same way.
    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(data["judge"]["seed"]))
    strat = {k: v for k, v in strata(data["judge"]["visible"]).items()}
    seeds = data["judge"]["seed"]

    arms = ("actions_only", "pixels1", "pixels4", "pixels4_actions", "tokens1_flat", "tokens1_attn",
            "cls_pooled1", "cls1", "cls4", "z1", "z4", "root_features", "u", "generated_z",
            "generated_features")
    report, kept = {}, {}
    for outcome in ("death1", "death2"):
        pf, pj = data["fit"][f"p_{outcome}"], data["judge"][f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        block = {"judge_opportunity_roots": int(opp.sum()), "prior_action": prior,
                 "prior_expected_safe": float(prior_safe[opp].mean()),
                 "oracle_expected_safe": float((1 - pj.amin(1))[opp].mean()),
                 "strata_opportunity": {k: int((v & opp).sum()) for k, v in strat.items()},
                 "prior_by_stratum": {k: float(prior_safe[v & opp].mean()) for k, v in strat.items() if (v & opp).any()},
                 "arms": {}}
        per_arm = {}
        for arm in arms:
            runs, safes, hist = [], [], np.zeros(N, int)
            for probe in range(args.probe_seeds):
                if arm == "pixels4_actions":
                    model, (m, sd), trace = obs_fit("pixels4", data["fit"], outcome, inner, seed=probe,
                                                    device=device, steps=args.steps)
                    from observability import score as obs_score
                    sj = obs_score(model, data["judge"], judge_rows, None, device)
                    sf = obs_score(model, data["fit"], train_rows, None, device)
                    trace["parameters"] = sum(t.numel() for t in model.parameters())
                else:
                    source = "tokens1" if arm.startswith("tokens1") else arm
                    kind, xf, shape = arm_input(data["fit"], source, train_rows)
                    _, xj, _ = arm_input(data["judge"], source, None)
                    if arm.startswith("tokens1"):
                        kind = arm.split("_", 1)[1] == "flat" and "tokens_flat" or "tokens_attn"
                    if not kind.startswith("pixels"):
                        mean, scale = standardize(xf, train_rows)
                        xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
                    model, trace = train(kind, shape if shape else xf.shape[1:], xf, pf, train_rows, hold_rows,
                                         seed=probe, device=device, steps=args.steps)
                    sj = scores(model, xj, judge_rows, device)
                    sf = scores(model, xf, train_rows, device)
                safe, _ = expected_safe(sj, pj)
                fsafe, fopp = expected_safe(sf, pf[train_rows])
                hist += np.bincount(sj[opp].argmin(1).numpy(), minlength=N)
                runs.append({**{k: v for k, v in trace.items() if k != "curve"},
                             "judge_expected_safe": float(safe[opp].mean()),
                             "fit_expected_safe": float(fsafe[fopp].mean())})
                safes.append(safe)
                kept[f"{outcome}|{arm}|{probe}"] = sj
                del model
                torch.cuda.empty_cache()
            mean_safe = torch.stack(safes).mean(0)
            per_arm[arm] = mean_safe
            block["arms"][arm] = {
                "runs": runs, "judge_expected_safe": float(mean_safe[opp].mean()),
                "vs_prior": paired(mean_safe[opp], prior_safe[opp], seeds[opp], draws=args.draws, seed=args.seed + 11),
                "by_stratum": {k: float(mean_safe[v & opp].mean()) for k, v in strat.items() if (v & opp).any()},
                "chosen_actions": hist.tolist()}
            t = block["arms"][arm]["vs_prior"]
            log(stage="arm", outcome=outcome, arm=arm, safe=round(block["arms"][arm]["judge_expected_safe"], 4),
                vs_prior=round(t["difference"], 4), resolved=t["excludes_zero"],
                params=runs[0]["parameters"], fit=round(runs[0]["fit_expected_safe"], 3))
        chain = {"actions_only_vs_prior": ("actions_only", None),
                 "pixels4_actions_vs_pixels4": ("pixels4_actions", "pixels4"),
                 "pixels4_vs_pixels1": ("pixels4", "pixels1"),
                 "tokens1_flat_vs_pixels1": ("tokens1_flat", "pixels1"),
                 "tokens1_attn_vs_pixels1": ("tokens1_attn", "pixels1"),
                 "cls_pooled1_vs_tokens1_flat": ("cls_pooled1", "tokens1_flat"),
                 "cls1_vs_cls_pooled1": ("cls1", "cls_pooled1"),
                 "z1_vs_cls1": ("z1", "cls1"), "z4_vs_cls4": ("z4", "cls4"),
                 "root_features_vs_z4": ("root_features", "z4"),
                 "u_vs_root_features": ("u", "root_features"),
                 "generated_z_vs_u": ("generated_z", "u"),
                 "generated_features_vs_generated_z": ("generated_features", "generated_z")}
        block["contrasts"] = {name: paired(per_arm[a][opp], (prior_safe if b is None else per_arm[b])[opp],
                                           seeds[opp], draws=args.draws, seed=args.seed + 13)
                              for name, (a, b) in chain.items()}
        report[outcome] = block

    rows_path = args.out / f"{args.name}_rows.pt"
    torch.save({"p_death1": data["judge"]["p_death1"], "p_death2": data["judge"]["p_death2"],
                "seed": seeds, "strata": strat, "scores": kept}, rows_path)
    reproduction = {"observability_pixels4_death1": recorded["outcomes"]["death1"]["arms"]["pixels4"]["judge_expected_safe"],
                    "here_pixels4_actions_death1": report["death1"]["arms"]["pixels4_actions"]["judge_expected_safe"]}
    evidence = {"schema": "d4mj_frozen_ladder_v1", "status": "EXPLORATORY: judgement roots already inspected",
                "script_sha256": _sha256(Path(__file__)), "observability_sha256": _sha256(HERE / "observability.py"),
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                "identity": {s: data[s]["identity"] for s in data}, "shard_manifests": shards,
                "roots": {s: len(data[s]["seed"]) for s in data}, "smoke": bool(args.smoke),
                "reproduction": reproduction,
                "rows": {"path": rows_path.name, "sha256": _sha256(rows_path)}, "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="frozen_ladder_complete", reproduction=reproduction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
