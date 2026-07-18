"""C-matrix: depth-indexed task heads (C0/C1/C2/C3).

Protocol: reviews/2026-07-18-cmatrix-depth-head-protocol.md (registered
before implementation). Bases: committed X-FLM_s505 / X-FLG_s505. Heads-only
training on the pinned replay; evaluation on the Stage-1 fresh bundles
(spent for selection — cannot grant planner GO).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from fork_oracle_v2 import sha256_file  # noqa: E402
from model import assert_encoder_frozen  # noqa: E402
from stage1_head_adaptation import (  # noqa: E402
    ARTIFACTS, BATCH, BUNDLE, LR, MANIFEST, NATURAL, TERMINAL, UPDATES,
    freeze_world_except_heads, load_base)
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage1c_head_depth_ceiling import (  # noqa: E402
    PREFIX, WINDOW, build_schedule, make_batch, window_index)
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402
import phase_e_continuation_depth as cont_depth  # noqa: E402
import phase_e_same_target as same_target  # noqa: E402
from phase_e_same_target import prediction_metrics, bootstrap_indices  # noqa: E402

PROTOCOL = REPO_ROOT / "reviews/2026-07-18-cmatrix-depth-head-protocol.md"
REPORT_PATH = ARTIFACTS / "cmatrix_report.json"
RAW_PATH = ARTIFACTS / "cmatrix_raw.json"
SEED = 505
KINDS = ("X-FLM", "X-FLG")
DEPTH = 8
DISTANCES = DEPTH + 1        # 0 = teacher-forced, 1..8 = generated
CAL_UPDATES = 1_000
GAMMA = 0.997


class DepthIndexedHead(nn.Module):
    """Dreamer-4/MTP-INSPIRED (paper Eq. 9: one output layer per forecast
    distance; NOT faithful — no task conditioning): shared trunk from the
    base head, one final linear per distance, all initialized from the base
    head's final layer so C1 starts exactly at the shared-head function."""

    def __init__(self, base_net: nn.Sequential, out_features: int,
                 distances: int = DISTANCES):
        super().__init__()
        self.trunk = nn.Sequential(base_net[0], base_net[1], base_net[2])
        final: nn.Linear = base_net[3]
        self.outputs = nn.ModuleList()
        for _ in range(distances):
            layer = nn.Linear(final.in_features, out_features)
            with torch.no_grad():
                layer.weight.copy_(final.weight)
                layer.bias.copy_(final.bias)
            self.outputs.append(layer)

    def forward_at(self, x: torch.Tensor, distance: int) -> torch.Tensor:
        return self.outputs[distance](self.trunk(x))

    def forward_per_position(self, x: torch.Tensor,
                             distances: list[int]) -> torch.Tensor:
        h = self.trunk(x)                       # [B, P, hidden]
        return torch.stack([self.outputs[d](h[:, i])
                            for i, d in enumerate(distances)], dim=1)


def event_index(train) -> list[tuple[int, int]]:
    """Windows whose GENERATED-position rewards (7..14) contain an event."""
    picks = []
    for e, ep in enumerate(train):
        rewards = np.asarray(ep["rewards"])
        for start in range(len(ep["obs"]) - WINDOW + 1):
            if np.abs(rewards[start + PREFIX - 1:start + WINDOW - 1]).max() > 1e-6:
                picks.append((e, start))
    return picks


def rollout_contexts(world, batch, depth=DEPTH):
    state = world.initial_state(batch["obs"].shape[0], batch["obs"].device)
    pooled, dists, r_t, c_t = [], [], [], []
    for t in range(PREFIX):
        state = world.observe_step(batch["obs"][:, t],
                                   batch["previous_actions"][:, t], state)
        if t >= 1:
            pooled.append(world.pool(state.tokens))
            dists.append(0)
            r_t.append(batch["rewards"][:, t - 1])
            c_t.append(batch["continues"][:, t - 1])
    for k in range(depth):
        a = PREFIX - 1 + k
        state, _, _, _ = world.imagine_step(state, batch["actions"][:, a],
                                            deterministic_mode=True)
        pooled.append(world.pool(state.tokens))
        dists.append(k + 1)
        r_t.append(batch["rewards"][:, a])
        c_t.append(batch["continues"][:, a])
    return (torch.stack(pooled, 1), dists,
            torch.stack(r_t, 1), torch.stack(c_t, 1))


def head_loss(world, reward_head, cont_head, pooled, dists, r_t, c_t,
              reward_only=False):
    if isinstance(reward_head, DepthIndexedHead):
        r_logits = reward_head.forward_per_position(pooled, dists)
        c_logits = cont_head.forward_per_position(pooled, dists).squeeze(-1)
    else:
        r_logits = reward_head(pooled)
        c_logits = cont_head(pooled)
    loss = world.reward.loss(r_logits, r_t).mean() if not isinstance(
        reward_head, DepthIndexedHead) else \
        world.reward_loss_fn(r_logits, r_t).mean()
    if not reward_only:
        loss = loss + F.binary_cross_entropy_with_logits(c_logits, c_t)
    return loss


def train_arm(world, arm, schedule, ev_index, train, device, rng):
    freeze_world_except_heads(world)
    reward_head, cont_head = world.reward, world.continuation
    if arm in ("C1", "C2", "C3"):
        reward_head = DepthIndexedHead(world.reward.net,
                                       world.cfg.reward_bins).to(device)
        cont_head = DepthIndexedHead(world.continuation.net, 1).to(device)
        world.reward_loss_fn = world.reward.loss   # two-hot loss reuse
    trainable = ([p for p in world.parameters() if p.requires_grad]
                 if arm == "C0" else
                 list(reward_head.parameters()) + list(cont_head.parameters()))
    if arm != "C0":
        for p in world.parameters():
            p.requires_grad_(False)
        # the trunk modules are SHARED objects with the frozen base head;
        # re-enable exactly the new heads' parameters (trunk + outputs)
        for p in trainable:
            p.requires_grad_(True)
        world.encoder_frozen = True
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    before = state_digest(world, exclude_heads=True)
    phases = [("main", UPDATES)] + ([("cal", CAL_UPDATES)] if arm == "C3" else [])
    losses = []
    for phase, updates in phases:
        for u in range(updates):
            picks = schedule[(u * BATCH) % (len(schedule) - BATCH):][:BATCH]
            batch = make_batch(train, picks, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled, dists, r_t, c_t = rollout_contexts(world, batch)
                loss = head_loss(world, reward_head, cont_head,
                                 pooled, dists, r_t, c_t)
                if arm in ("C2", "C3") and phase == "main":
                    epicks = [ev_index[int(rng.integers(len(ev_index)))]
                              for _ in range(BATCH // 2)]
                    ebatch = make_batch(train, epicks, device)
                    ep, ed, er, ec = rollout_contexts(world, ebatch)
                    loss = loss + head_loss(world, reward_head, cont_head,
                                            ep, ed, er, ec, reward_only=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 100.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    assert_encoder_frozen(world, optimizer)
    assert state_digest(world, exclude_heads=True) == before
    world.eval()
    return reward_head, cont_head, {
        "loss_first_last": [float(np.mean(losses[:100])),
                            float(np.mean(losses[-100:]))]}


@torch.no_grad()
def evaluate(world, reward_head, cont_head, natural_arrays, cont_arrays,
             actual_continue, anchors, device):
    def reward_logits_at(pooled, k):
        if isinstance(reward_head, DepthIndexedHead):
            return reward_head.forward_at(pooled, min(k, DISTANCES - 1))
        return reward_head(pooled)

    def cont_prob_at(pooled, k):
        if isinstance(cont_head, DepthIndexedHead):
            return torch.sigmoid(cont_head.forward_at(
                pooled, min(k, DISTANCES - 1)).squeeze(-1))
        return torch.sigmoid(cont_head(pooled))

    from model import WorldState
    from phase_e_taskheads import clone_world_state

    def depth_predictions(arrays, kind):
        actual = arrays["rewards"]
        boot = bootstrap_indices(arrays["episodes"])
        out = {}
        raw = {}
        for K in (0, 1, 2, 4, 8):
            decoded_all, nll_all, probs = [], [], []
            for start in range(0, len(actual), 64):
                stop = min(start + 64, len(actual))
                obs = torch.from_numpy(arrays["obs"][start:stop]).to(device)
                acts = torch.from_numpy(arrays["actions"][start:stop]).to(device)
                prev = torch.from_numpy(
                    arrays["previous_actions"][start:stop]).to(device)
                b = stop - start
                state = world.initial_state(b, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in range(16 - K):
                        state = world.observe_step(obs[:, t], prev[:, t], state)
                    for j in range(K):
                        state, _, _, _ = world.imagine_step(
                            state, acts[:, 16 - 1 - K + j],
                            deterministic_mode=True)
                    pooled = world.pool(state.tokens)
                    logits = reward_logits_at(pooled, K).float()
                    prob = cont_prob_at(pooled, K).float()
                target = torch.from_numpy(actual[start:stop]).to(device)
                decoded_all.append(world.reward.decode(logits).cpu().numpy())
                nll_all.append(world.reward.loss(logits, target).cpu().numpy())
                probs.append(prob.cpu().numpy())
            decoded = np.concatenate(decoded_all)
            if kind == "reward":
                out[f"k{K}"] = prediction_metrics(
                    decoded, np.concatenate(nll_all), actual,
                    arrays["episodes"], boot)
            else:
                out[f"k{K}"] = cont_depth.continuation_metrics(
                    np.concatenate(probs), actual_continue, boot)
            raw[f"k{K}"] = decoded.tolist()
        return out, raw

    reward_metrics, reward_raw = depth_predictions(natural_arrays, "reward")
    cont_metrics, _ = depth_predictions(cont_arrays, "continuation")

    # ranking with distance-aware per-step decode + zero-suffix false reward
    rows = []
    for anchor in anchors:
        obs = torch.from_numpy(anchor["obs_hist"][None]).to(device)
        state = world.initial_state(1, device)
        for i in range(len(anchor["act_hist"])):
            prev = torch.tensor([int(anchor["act_hist"][i])], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = world.observe_step(obs[:, i], prev, state)
        j_g, actual = {}, {}
        pred_sum = {}
        for name, suffix in anchor["suffixes"].items():
            branch = clone_world_state(state)
            alive = torch.ones(1, device=device)
            total = torch.zeros(1, device=device)
            total_raw = torch.zeros(1, device=device)
            for k, av in enumerate(suffix):
                a = torch.tensor([int(av)], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    branch, _, _, _ = world.imagine_step(
                        branch, a, deterministic_mode=True)
                    pooled = world.pool(branch.tokens)
                    r = world.reward.decode(
                        reward_logits_at(pooled, k + 1).float())
                    c = cont_prob_at(pooled, k + 1).float()
                total = total + (GAMMA ** k) * alive * r
                total_raw = total_raw + r
                alive = alive * c
            j_g[name] = float(total)
            pred_sum[name] = float(total_raw)
            actual[name] = float(np.mean(
                [o["reward_sum"] for o in anchor["branches"][name]["outcomes"]]))
        names = list(anchor["suffixes"])
        av = np.array([actual[n] for n in names])
        chosen = names[int(np.array([j_g[n] for n in names]).argmax())]
        rows.append({"env_seed": int(anchor["env_seed"]),
                     "differs": bool(av.std() > 1e-9),
                     "chosen_minus_random": float(actual[chosen] - av.mean()),
                     "regret": float(av.max() - actual[chosen]),
                     "zero_suffix_abs_pred": [abs(pred_sum[n]) for n in names
                                              if abs(actual[n]) < 1e-9]})
    diff = [r for r in rows if r["differs"]]
    zero_preds = [v for r in rows for v in r["zero_suffix_abs_pred"]]
    ranking = {
        "chosen_minus_random_mean": float(np.mean(
            [r["chosen_minus_random"] for r in diff])) if diff else None,
        "regret_mean": float(np.mean([r["regret"] for r in diff])) if diff else None,
        "n_differing": len(diff),
        "zero_suffix_abs_pred_mean": float(np.mean(zero_preds)) if zero_preds else None,
    }
    return ({"reward_depth": reward_metrics, "continuation_depth": cont_metrics,
             "ranking": ranking},
            {"reward_decoded": reward_raw, "ranking_rows": rows})


def main():
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    for key, path in (("natural", NATURAL), ("terminal", TERMINAL),
                      ("bundle", BUNDLE)):
        assert sha256_file(path) == manifest[key]["sha256"], key
    natural_eps = torch.load(NATURAL, weights_only=False)
    terminal_eps = torch.load(TERMINAL, weights_only=False)
    anchors = torch.load(BUNDLE, weights_only=False)
    nrows = same_target.target_rows(natural_eps)
    narrays = same_target.window_arrays(natural_eps, nrows)
    crows = same_target.target_rows(terminal_eps)
    carrays = same_target.window_arrays(terminal_eps, crows)
    actual_c = cont_depth.continuation_targets(terminal_eps, crows)
    train, _ = load_scaled_data()
    schedule, sched_digest = build_schedule(train)
    ev_index = event_index(train)

    report = {"provenance": {
        "protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head": git_head(), "source_digest": source_digest(),
        "versions": software_versions(),
        "schedule_sha256": sched_digest,
        "event_pool": len(ev_index),
        "data_sha256": {k: manifest[k]["sha256"]
                        for k in ("natural", "terminal", "bundle")},
        "replay_sha256": sha256_file(TRAIN_40K_CACHE)},
        "results": {}}
    raw_out = {"results": {}}
    for kind in KINDS:
        for arm in ("C0", "C1", "C2", "C3"):
            tag = f"{kind}_s{SEED}_{arm}"
            world = load_base(kind, SEED, device)
            rng = np.random.default_rng(30_000 + SEED)
            reward_head, cont_head, info = train_arm(
                world, arm, schedule, ev_index, train, device, rng)
            torch.save({"reward": {k: v.cpu() for k, v in
                                   reward_head.state_dict().items()},
                        "continuation": {k: v.cpu() for k, v in
                                         cont_head.state_dict().items()},
                        "arm": arm, "kind": kind},
                       ARTIFACTS / f"cmatrix_heads_{tag}.pt")
            metrics, raw = evaluate(world, reward_head, cont_head, narrays,
                                    carrays, actual_c, anchors, device)
            report["results"][tag] = {**info, **metrics}
            raw_out["results"][tag] = raw
            REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
            RAW_PATH.write_text(json.dumps(raw_out, default=str))
            print(f"[{tag}] k1_auroc "
                  f"{metrics['reward_depth']['k1']['event_auroc']:.3f} "
                  f"k8_auroc {metrics['reward_depth']['k8']['event_auroc']:.3f} "
                  f"zero_pred {metrics['ranking']['zero_suffix_abs_pred_mean']:.4f} "
                  f"rank_adv {metrics['ranking']['chosen_minus_random_mean']:.4f}",
                  flush=True)
            del world
            torch.cuda.empty_cache()
    print("cmatrix complete")


if __name__ == "__main__":
    main()
