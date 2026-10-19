"""Audit (10-14), stage 4: is the learned-innovation negative on the 26-case optimization control a property of learned interfaces,
or of this implementation? The control (innovation_overfit.py) used the SAME four FIT roots / 26 distinct futures; supplied IDs reached
MSE/floor 0.015 while learned hard / soft posteriors stayed at the deterministic floor (1.05). Three separating tests, same data,
parent world, channel module and optimizer family as the control:
  A  posterior spread: the saved final soft / hard channels (optimization_control/state.pt): within-root spread of the posterior's
     logits and probabilities over each root's distinct futures (how much the encoder distinguishes them at all)
  B  supervised separability: a FRESH posterior (same architecture, frozen parent world for the belief) trained with cross-entropy
     to output the supplied 6-digit IDs of the 26 cases (the assigned arm's codes); 500 updates, AdamW lr 1e-3. Accuracy of the
     joint code. If B succeeds, the posterior can represent the futures; the control's failure is the joint unsupervised path.
  C  initialization: joint training exactly as the control's soft arm (world + channel trainable, full D loss, phase AdamW and lr
     schedule, seed-919 case batches), except the adapter's zero initialization: C1 output Linear at PyTorch's default
     initialization; C2 C1 + code value vectors at std 1 (instead of 0.02). 800 updates (the assigned arm was below the floor by
     800). MSE / exact conditional-mean floor and within-root prediction variance / target variance, every 100 updates.
  D  dilution: where in the posterior the within-root distinctions vanish. For fresh posteriors (5 init seeds) and the control's
     final soft / hard posteriors: share of within-root variance in the input tile vectors, the projected tiles, the pooled factors
     and the logits; and the attention mass the factor queries put on the tiles where a root's futures differ.
  E  pooling control for B: same inputs (per-tile LayerNorm of [belief, current, successor - current]), CE targets, batches and
     optimizer as B, with two linear encoders: "flat" (all 81 tiles flattened -> 6x8 logits) and "mean" (tile mean -> 6x8).
  B accepts an optional update count (B 3000 -> audit_channel_B3000.json).
Usage: audit_s4_channel.py A|B [updates]|C1|C2|D|E  -> artifacts/eda/oracle_v1/learned_innovation/audit_channel_<test>.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import innovation_model as I  # noqa: E402
import innovation_train as T  # noqa: E402
import loss_intervention as L  # noqa: E402
from d4mj.config import config_from_dict  # noqa: E402
from d4mj.train import autocast_context, phase_optimizer, optimizer_step, _phase_lr  # noqa: E402

DIRECTORY = T.OUT / "optimization_control"


def cases():
    """the control's exact selection: four FIT day roots with >1 distinct future, all distinct futures as cases"""
    panel = torch.load(L.OUT / "panel.pt", map_location="cpu", weights_only=False)
    fan = torch.load(T.OUT / "real_futures/fan.pt", map_location="cpu", weights_only=False)
    visible = torch.load(L.OUT / "snap_visible.pt", map_location="cpu", weights_only=False)["visible"]
    split = json.loads((L.OUT / "generated_prefix/readout_refit/contract.json").read_text())["split"]
    fit = set(split["FIT"]); selected = []
    for i, root in enumerate(panel["roots"]):
        if root["episode_id"] in fit and bool((visible[i, 15:17, 1521] >= .5).all()):
            if len(fan["layouts"][i].flatten(1).unique(dim=0)) > 1:
                selected.append(i)
        if len(selected) == 4:
            break
    frames, actions, targets, root_index = [], [], [], []
    for j, root in enumerate(selected):
        for target in fan["tokens"][root].flatten(1).unique(dim=0).view(-1, 81, 192):
            frames.append(panel["tokens"][root, 1:16]); actions.append(panel["actions"][root, 1:16])
            targets.append(target); root_index.append(j)
    frames, actions, targets = torch.stack(frames).float(), torch.stack(actions), torch.stack(targets).float()
    root_index = torch.tensor(root_index)
    means = torch.stack([targets[root_index == j].mean(0) for j in range(4)])
    floor = float((targets - means[root_index]).square().mean())
    ids = torch.arange(len(frames)); assigned = torch.stack([(ids // (8 ** k)) % 8 for k in range(6)], -1)
    return frames, actions, targets, root_index, floor, assigned


def within_root_var(x, root_index):
    """sum over roots of within-root variance of x (flattened), relative helper"""
    tot = 0.0
    for j in root_index.unique():
        v = x[root_index == j].flatten(1)
        tot += float(((v - v.mean(0)) ** 2).sum())
    return tot


def main():
    test = sys.argv[1]
    updates = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    frames, actions, targets, root_index, floor, assigned = cases()
    n = len(frames)
    config = config_from_dict(torch.load(L.TW.S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    parent = torch.load(T.INIT, map_location="cpu", weights_only=False)
    res = {"test": test, "cases": n, "floor": floor}
    if test == "A":
        st = torch.load(DIRECTORY / "state.pt", map_location="cpu", weights_only=False)
        for arm in ("soft", "hard"):
            w = L.TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16); w.load_state_dict(st["worlds"][arm]); w = w.cuda().eval()
            ch = I.Innovation(); ch.load_state_dict(st["channels"][arm]); ch = ch.cuda().eval()
            with torch.no_grad():
                belief = I.native_belief(w, frames.cuda(), actions.cuda(), config)
                logits = ch.posterior(belief[:, -1], frames[:, -1].cuda(), targets.cuda()).float().cpu()
            p = logits.softmax(-1)
            res[arm] = {"logit_within_root_var_over_total_var": within_root_var(logits, root_index) / float(((logits - logits.mean(0)) ** 2).sum()),
                        "logit_within_root_max_abs_diff": float(max((logits[root_index == j] - logits[root_index == j][0]).abs().max() for j in range(4))),
                        "prob_within_root_max_abs_diff": float(max((p[root_index == j] - p[root_index == j][0]).abs().max() for j in range(4))),
                        "max_prob_mean": float(p.max(-1).values.mean()),
                        "value_vector_norm_mean": float(ch.values.norm(dim=-1).mean()), "adapter_output_weight_norm": float(ch.output.weight.norm())}
    elif test == "B":
        w = L.TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16); w.load_state_dict(parent["world"]); w = w.cuda().eval()
        with torch.no_grad():
            belief = I.native_belief(w, frames.cuda(), actions.cuda(), config)[:, -1]
        post = I.Posterior().cuda()
        opt = torch.optim.AdamW(post.parameters(), lr=1e-3, weight_decay=0.01)
        g = torch.Generator().manual_seed(919)
        curve = []
        for u in range(updates):
            idx = torch.randint(n, (16,), generator=g)
            logits = post(belief[idx], frames[idx, -1].cuda(), targets[idx].cuda())
            loss = F.cross_entropy(logits.reshape(-1, 8).float(), assigned[idx].cuda().reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if (u + 1) % (50 if updates <= 500 else 250) == 0:
                with torch.no_grad():
                    pred = post(belief, frames[:, -1].cuda(), targets.cuda()).argmax(-1).cpu()
                curve.append({"update": u + 1, "loss": round(float(loss), 4), "joint_code_accuracy": float((pred == assigned).all(-1).float().mean()),
                              "distinct_joint_codes": int(pred.unique(dim=0).shape[0])})
        res["curve"] = curve
        if updates != 500:
            test = f"B{updates}"
    elif test == "E":
        w = L.TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16); w.load_state_dict(parent["world"]); w = w.cuda().eval()
        with torch.no_grad():
            belief = I.native_belief(w, frames.cuda(), actions.cuda(), config)[:, -1]
        inp = torch.cat([belief, frames[:, -1].cuda(), targets.cuda() - frames[:, -1].cuda()], -1)
        width = inp.shape[-1]
        for name in ("flat", "mean"):
            torch.manual_seed(7)
            norm = nn.LayerNorm(width).cuda()
            head = nn.Linear(81 * width if name == "flat" else width, 6 * 8).cuda()
            enc = (lambda x: head(norm(x).flatten(1))) if name == "flat" else (lambda x: head(norm(x).mean(1)))
            opt = torch.optim.AdamW(list(norm.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01)
            g = torch.Generator().manual_seed(919)
            curve = []
            for u in range(updates):
                idx = torch.randint(n, (16,), generator=g)
                loss = F.cross_entropy(enc(inp[idx]).view(-1, 8).float(), assigned[idx].cuda().reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                if (u + 1) % 50 == 0:
                    with torch.no_grad():
                        pred = enc(inp).view(n, 6, 8).argmax(-1).cpu()
                    curve.append({"update": u + 1, "loss": round(float(loss.detach()), 4),
                                  "joint_code_accuracy": float((pred == assigned).all(-1).float().mean())})
            res[name] = curve
    elif test == "D":
        w = L.TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16); w.load_state_dict(parent["world"]); w = w.cuda().eval()
        with torch.no_grad():
            belief = I.native_belief(w, frames.cuda(), actions.cuda(), config)[:, -1]
        cur, suc = frames[:, -1].cuda(), targets.cuda()
        # Raw tokens are contextual (every tile moves a little when one changes), so "changed" = the fewest tiles holding half of the
        # root's within-root target variance
        changed = torch.zeros(n, 81, dtype=torch.bool); res["within_root_variance_concentration"] = []
        for j in range(4):
            t = targets[root_index == j]
            v = ((t - t.mean(0)) ** 2).sum(-1).sum(0)                        # [81] within-root variance per tile
            order = v.argsort(descending=True); cum = v[order].cumsum(0) / v.sum()
            k = int((cum < .5).sum()) + 1
            changed[root_index == j] = torch.zeros(81, dtype=torch.bool).index_fill_(0, order[:k], True)
            res["within_root_variance_concentration"].append({"futures": len(t), "tiles_for_half": k, "top1_share": round(float(cum[0]), 3),
                                                              "top5_share": round(float(cum[4]), 3)})
        res["changed_tiles_per_case"] = round(float(changed.sum(1).float().mean()), 2)
        share = lambda x: round(within_root_var(x.float().cpu(), root_index) / float(((x.float().cpu() - x.float().cpu().mean(0)) ** 2).sum()), 6)

        def probe(post, belief):
            enc = post.encoder
            with torch.no_grad():
                inp = torch.cat([belief, cur, suc - cur], -1)
                x = enc.project(inp)
                q = enc.queries[None].expand(n, -1, -1)
                att, wts = enc.attention(q, x, x, need_weights=True, average_attn_weights=True)
                pooled = enc.norm(q + att)
                logits = post.logits(pooled)
            cos = F.normalize(pooled, dim=-1); cos = (cos @ cos.transpose(1, 2))[:, ~torch.eye(6, dtype=torch.bool)].mean()
            return {"input": share(inp), "projected_tiles": share(x), "pooled_factors": share(pooled), "logits": share(logits),
                    "slot_pairwise_cosine": round(float(cos), 4),
                    "same_argmax_all_slots": round(float((logits.argmax(-1) == logits.argmax(-1)[:, :1]).all(-1).float().mean()), 3),
                    "attention_mass_on_changed_tiles": round(float((wts.cpu() * changed[:, None]).sum(-1).mean()), 4),
                    "uniform_attention_mass": round(float(changed.sum(1).float().mean() / 81), 4)}
        res["fresh"] = []
        for seed in range(5):
            torch.manual_seed(100 + seed)
            res["fresh"].append(probe(I.Posterior().cuda().eval(), belief))
        st = torch.load(DIRECTORY / "state.pt", map_location="cpu", weights_only=False)
        for arm in ("soft", "hard"):
            ch = I.Innovation(); ch.load_state_dict(st["channels"][arm])
            w.load_state_dict(st["worlds"][arm])                               # the arm's own trained world gives its belief
            with torch.no_grad():
                own = I.native_belief(w, frames.cuda(), actions.cuda(), config)[:, -1]
            res[arm] = probe(ch.posterior.cuda().eval(), own)
    else:
        world = L.TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16); world.load_state_dict(parent["world"]); world = world.cuda().train()
        ch = I.Innovation()
        nn.init.kaiming_uniform_(ch.output.weight, a=5 ** 0.5)                 # PyTorch's default Linear init (C1, C2)
        bound = 1 / ch.output.weight.shape[1] ** 0.5
        nn.init.uniform_(ch.output.bias, -bound, bound)
        if test == "C2":
            with torch.no_grad():
                ch.values.normal_(0, 1.0)
        ch = ch.cuda().train()
        opt = phase_optimizer([world], config)
        opt.add_param_group({"params": ch.completion_parameters(), "lr": opt.param_groups[0]["lr"], "weight_decay": opt.param_groups[0]["weight_decay"]})
        sampler = torch.Generator().manual_seed(919)

        def predict(idx):
            fr, ac = frames[idx].cuda(), actions[idx].cuda()
            belief = I.native_belief(world, fr, ac, config)
            logits = ch.posterior(belief[:, -1], fr[:, -1], targets[idx].cuda())
            delta = F.pad(ch.condition(belief[:, -1], logits.softmax(-1))[:, None], (0, 0, 0, 0, 14, 0))
            with I.conditioned(world, delta), autocast_context(config):
                out = world(fr, ac)[0][:, -1].float()
            return out, logits
        curve = []
        for u in range(800):
            idx = torch.randint(n, (16,), generator=sampler)
            out, _ = predict(idx)
            loss = L.objective(L.components(out[:, None], targets[idx].cuda()[:, None]), "D")
            params = [p for gr in opt.param_groups for p in gr["params"]]
            optimizer_step(opt, loss, params, learning_rate=_phase_lr(config, u), grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
            if (u + 1) % 100 == 0:
                with torch.no_grad():
                    outs, lg = zip(*[predict(torch.arange(f, min(n, f + 8))) for f in range(0, n, 8)])
                out = torch.cat(outs).cpu(); lg = torch.cat(lg).float().cpu()
                tvar = within_root_var(targets, root_index)
                curve.append({"update": u + 1, "mse_over_floor": round(float((out - targets).square().mean() / floor), 4),
                              "within_root_pred_var_over_target": round(within_root_var(out, root_index) / tvar, 5),
                              "joint_argmax_codes": int(lg.argmax(-1).unique(dim=0).shape[0])})
                print(json.dumps(curve[-1]), flush=True)
        res["curve"] = curve
    out = T.OUT / f"audit_channel_{test}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:3000], flush=True)


if __name__ == "__main__":
    main()
