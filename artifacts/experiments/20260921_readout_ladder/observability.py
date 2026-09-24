"""Observability test: can the state that decides death be recovered from what the agent sees?

The replay pilot showed one-step death is fixed by (full state, action); the ranking probes found no
representation of the root observation that beats the action prior. This separates the two
remaining explanations -- the deciding state is not drawn, or it is drawn and lost -- with matched,
ranking-trained predictors on different inputs, developed on existing roots and judged on fresh
seeds nothing has touched (`observe.py collect`, seeds from 50,000).

Arms. Every arm maps the root to 17 death scores through the same head, [trunk] -> 512 -> 17, is
trained with the same objective, optimizer and batches, three probe seeds, and selected on an inner
15% of FIT seeds (never on judgement roots). Only the input and its trunk differ:

  visible           the drawn state now: 7x9 tiles, on-screen mobs and arrows, facing, sleep, light,
                    HUD stats, inventory                                             (MLP trunk)
  visible_history   + 32 steps of HUD stats and the threats in the 3x3 cells around the player
  full              visible + hidden: mob cooldowns and health, player accumulators -- the
                    positive control; the full-state oracle is ~1.0 one-step
  pixels4           the last 4 frames + last 4 actions            (Nature-DQN CNN trunk, Mnih 2015)
  pixels32          all 32 stored frames + 32 actions
  pixels32_hidden   pixels32 + the hidden vector at the head
  prior             the FIT-root action prior: the single action with the lowest mean P(death)

Objective: within-root ranking by EXPECTED risk -- softplus(s_safe - s_risky) over action pairs,
weighted by their difference in P(death | s, a) over 32 keys, averaged per root. For one-step death,
where P is essentially 0 or 1, this is the 2026-09-19 pair-ranking loss; for two-step death it does
not train on a single stored draw.

Scoring, on judgement roots whose P(death) varies across actions: expected safe choice
1 - P(death | s, chosen), which avoids the single-draw selection effect REPLAY.md found; paired,
episode-seed-clustered intervals against the prior and between arms.

Declared reading (measured-outcome names only):
  full does not beat the prior                  -> evaluator_inadequate: read nothing else
  visible or visible_history beats the prior    -> drawn_state_suffices
  only full / pixels32_hidden beat the prior    -> only_hidden_state_suffices
  pixels arms fail where visible_history holds  -> pixel_learning_gap (reported, not a verdict)
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

N = 17
# The stores `observe.py` writes. Not imported from it: `observe` puts artifacts/eda first on sys.path,
# where a different `replay.py` lives, so importing it after `confirm` resolves the wrong module.
FIT_STATE = ROOT / "artifacts/eda/observe_fit_v1"
FRESH = ROOT / "artifacts/eda/observe_fresh_v1"
ARMS = ("visible", "visible_history", "full", "pixels4", "pixels32", "pixels32_hidden")


def load(fit_seeds, partial=False):
    """FIT: fork-store frames/actions joined to observe_fit_v1 state by (seed, step). JUDGE: fresh."""
    store = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    missing = sorted(set(fit_seeds) - set(state))
    if partial:
        fit_seeds = sorted(set(fit_seeds) & set(state))
    elif missing:
        raise SystemExit(f"observe replay incomplete: {len(missing)} FIT seeds missing")
    out = {}
    for split, files in (("fit", [(store[s], state[s]) for s in sorted(fit_seeds)]),
                         ("judge", [(p, None) for p in sorted(FRESH.glob("seed-*.pt"))])):
        rows = []
        for pixel_file, state_file in files:
            pixels = {int(r["step"]): r for r in torch.load(pixel_file, weights_only=False)}
            features = (pixels if state_file is None else
                        {int(r["step"]): r for r in torch.load(state_file, weights_only=False)})
            for step, f in features.items():
                p = pixels[step]
                rows.append({"seed": int(f["seed"]), "frames": p["frames"], "actions": p["led_to_action"],
                             "visible": f["visible"], "hidden": f["hidden"], "history": f["history"],
                             "p1": f["p_death1"], "p2": f["p_death2"]})
        stack = lambda k: torch.stack([r[k] for r in rows])
        actions = nn.functional.one_hot(stack("actions").clamp(max=N), N + 1).float()   # BOS = 17
        out[split] = {"seed": torch.tensor([r["seed"] for r in rows]), "frames": stack("frames"),
                      "actions": actions, "visible": stack("visible").float(),
                      "hidden": stack("hidden").float(), "history": stack("history").flatten(1).float(),
                      "p_death1": stack("p1").float(), "p_death2": stack("p2").float(),
                      "identity": hashlib.sha256(repr([(r["seed"], int(r["frames"].sum()))
                                                       for r in rows]).encode()).hexdigest()}
    return out


def vectors(data, arm):
    if arm == "visible":
        return data["visible"]
    if arm == "visible_history":
        return torch.cat((data["visible"], data["history"]), -1)
    if arm == "full":
        return torch.cat((data["visible"], data["hidden"]), -1)
    return data["hidden"] if arm == "pixels32_hidden" else None


class Scorer(nn.Module):
    """[trunk] -> 512 -> 17. CNN trunk = Nature DQN (Mnih et al. 2015): 8x8/4, 4x4/2, 3x3/1."""

    def __init__(self, arm, extra):
        super().__init__()
        self.frames = {"pixels4": 4, "pixels32": 32, "pixels32_hidden": 32}.get(arm)
        if self.frames:
            self.trunk = nn.Sequential(
                nn.Conv2d(3 * self.frames, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            width = 64 * 4 * 4 + self.frames * (N + 1) + extra
        else:
            self.trunk, width = nn.Identity(), extra
        self.head = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, N))

    def forward(self, frames, actions, vector):
        parts = []
        if self.frames:
            x = frames.permute(0, 1, 4, 2, 3).flatten(1, 2).float() / 255.0
            parts += [self.trunk(x), actions.flatten(1)]
        if vector is not None:
            parts.append(vector)
        return self.head(torch.cat(parts, -1))


def soft_rank(scores, p):
    """softplus(s_safe - s_risky) weighted by P(risky) - P(safe), mean per root over its pairs."""
    gap = p[:, :, None] - p[:, None, :]                          # [B, risky i, safer j]
    weight = gap.clamp(min=0)
    pair = nn.functional.softplus(scores[:, None, :] - scores[:, :, None])   # s_j - s_i
    total = weight.sum((1, 2))
    usable = total > 0
    if not bool(usable.any()):
        return scores.sum() * 0.0
    return ((pair * weight).sum((1, 2))[usable] / total[usable]).mean()


def inputs(model, data, idx, vec, device):
    """Only what the arm reads goes to the GPU: MLP arms never touch frames, pixels4 gets four."""
    k = model.frames
    return (data["frames"][idx][:, -k:].to(device) if k else None,
            data["actions"][idx][:, -k:].to(device) if k else None,
            None if vec is None else vec[idx].to(device))


@torch.no_grad()
def score(model, data, rows, vec, device, batch=256):
    return torch.cat([model(*inputs(model, data, rows[i:i + batch], vec, device)).cpu()
                      for i in range(0, len(rows), batch)])


def expected_safe(scores, p):
    """Per root: 1 - P(death | chosen); opportunity = P varies across actions."""
    chosen = scores.argmin(1)
    return 1.0 - p.gather(1, chosen[:, None]).squeeze(1), p.amax(1) > p.amin(1)


def fit(arm, data, outcome, inner, *, seed, device, steps, batch=128, every=100):
    torch.manual_seed(seed)
    vec = vectors(data, arm)
    mean = vec[~inner].mean(0) if vec is not None else None
    scale = vec[~inner].std(0).clamp_min(1e-6) if vec is not None else None
    vec = None if vec is None else (vec - mean) / scale
    model = Scorer(arm, 0 if vec is None else vec.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    p = data[f"p_{outcome}"]
    train, hold = torch.where(~inner)[0], torch.where(inner)[0]
    best, best_step, state, trace = -1.0, 0, None, []
    for step in range(steps):
        idx = train[torch.randint(len(train), (batch,), generator=gen)]
        s = model(*inputs(model, data, idx, vec, device))
        loss = soft_rank(s, p[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % every == 0:
            model.eval()
            safe, opp = expected_safe(score(model, data, hold, vec, device), p[hold])
            model.train()
            value = float(safe[opp].mean())
            trace.append(round(value, 4))
            if value > best:
                best, best_step = value, step + 1
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), (mean, scale), {"selected_step": best_step, "inner_safe": best, "curve": trace}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="observability")
    parser.add_argument("--partial", action="store_true", help="smoke only: use the FIT seeds replayed so far")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds, partial=args.partial)
    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])
    log(stage="loaded", fit=len(data["fit"]["seed"]), judge=len(data["judge"]["seed"]))

    report, kept = {}, {}
    for outcome in ("death1", "death2"):
        pf, pj = data["fit"][f"p_{outcome}"], data["judge"][f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        oracle = 1.0 - pj.amin(1)
        seeds = data["judge"]["seed"]
        block = {"judge_opportunity_roots": int(opp.sum()), "prior_action": prior,
                 "prior_expected_safe": float(prior_safe[opp].mean()),
                 "oracle_expected_safe": float(oracle[opp].mean()), "arms": {}}
        per_arm = {}
        for arm in ARMS:
            runs, safes = [], []
            for probe in range(args.probe_seeds):
                model, (mean, scale), trace = fit(arm, data["fit"], outcome, inner, seed=probe,
                                                  device=device, steps=args.steps)
                vec = vectors(data["judge"], arm)
                vec = None if vec is None else (vec - mean) / scale
                sj = score(model, data["judge"], torch.arange(len(seeds)), vec, device)
                safe, _ = expected_safe(sj, pj)
                fit_vec = vectors(data["fit"], arm)
                fit_vec = None if fit_vec is None else (fit_vec - mean) / scale
                train_rows = torch.where(~inner)[0]
                fit_safe, fit_opp = expected_safe(score(model, data["fit"], train_rows, fit_vec, device),
                                                  pf[train_rows])
                runs.append({**trace, "judge_expected_safe": float(safe[opp].mean()),
                             "fit_expected_safe": float(fit_safe[fit_opp].mean())})
                safes.append(safe)
                kept[f"{outcome}|{arm}|{probe}"] = sj
                del model
                torch.cuda.empty_cache()
            mean_safe = torch.stack(safes).mean(0)
            per_arm[arm] = mean_safe
            block["arms"][arm] = {"runs": runs, "judge_expected_safe": float(mean_safe[opp].mean()),
                                  "vs_prior": paired(mean_safe[opp], prior_safe[opp], seeds[opp],
                                                     draws=args.draws, seed=args.seed + 11)}
            t = block["arms"][arm]["vs_prior"]
            log(stage="arm", outcome=outcome, arm=arm, safe=round(block["arms"][arm]["judge_expected_safe"], 4),
                vs_prior=round(t["difference"], 4), resolved=t["excludes_zero"])
        contrasts = {"full_vs_visible": ("full", "visible"),
                     "history_vs_visible": ("visible_history", "visible"),
                     "pixels32_vs_visible_history": ("pixels32", "visible_history"),
                     "pixels32_hidden_vs_pixels32": ("pixels32_hidden", "pixels32"),
                     "pixels32_vs_pixels4": ("pixels32", "pixels4")}
        block["contrasts"] = {name: paired(per_arm[a][opp], per_arm[b][opp], seeds[opp],
                                           draws=args.draws, seed=args.seed + 13)
                              for name, (a, b) in contrasts.items()}
        beats = {arm: bool(block["arms"][arm]["vs_prior"]["difference"] > 0
                           and block["arms"][arm]["vs_prior"]["excludes_zero"]) for arm in ARMS}
        block["beats_prior"] = beats
        block["reading"] = ("evaluator_inadequate" if not beats["full"] else
                            "drawn_state_suffices" if beats["visible"] or beats["visible_history"] else
                            "only_hidden_state_suffices" if beats["full"] or beats["pixels32_hidden"] else
                            "mixed")
        block["pixel_learning_gap"] = bool(beats["visible_history"] and not beats["pixels32"])
        report[outcome] = block

    rows_path = args.out / f"{args.name}_rows.pt"
    torch.save({"p_death1": data["judge"]["p_death1"], "p_death2": data["judge"]["p_death2"],
                "seed": data["judge"]["seed"], "scores": kept}, rows_path)
    evidence = {"schema": "d4mj_observability_v1", "script_sha256": _sha256(Path(__file__)),
                "observe_sha256": _sha256(HERE / "observe.py"),
                "partition_sha256": _sha256(args.partition),
                "identity": {s: data[s]["identity"] for s in data},
                "roots": {s: len(data[s]["seed"]) for s in data},
                "judge_seeds": sorted(set(data["judge"]["seed"].tolist())),
                "rows": {"path": rows_path.name, "sha256": _sha256(rows_path)},
                "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="observability_complete",
        readings={o: report[o]["reading"] for o in report})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
