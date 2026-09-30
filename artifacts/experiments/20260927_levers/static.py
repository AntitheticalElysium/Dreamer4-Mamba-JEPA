"""Why copy-head worlds drift past copying on rollouts whose TRUE view never scrolls (corrt 1.42x copy at depth 16).

The futures roots whose factual 16-step future never scrolls (scroll.estimate on consecutive TRUE frames; 98 roots).
The world imagines the 16 steps (teval's windows). Per depth k:
  false_scroll   fraction of roots whose IMAGINED frame k scrolled relative to imagined frame k-1 (it should never)
  err / V, copy / V   imagined error and copy-root error, over the 81 tokens
  per group      imagined error / the group's variance (where.py groups)
  by action      mean imagined error at depth 16 split by whether the 16 factual actions include a move attempt
Usage: static.py <world.pt> [...] -> static_<name>.json (CPU is enough for the `full` backbone)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
import teval as T  # noqa: E402
from scroll import estimate  # noqa: E402
from where import GROUPS  # noqa: E402


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    for path in map(Path, sys.argv[1:]):
        world, st = T.load_world(path, device)
        cache = T.build_cache(st["args"]["pool"], device)
        fut = torch.cat([cache["ctx"][:, -1:].float(), cache["fut"].float()], 1)
        true_scroll = torch.cat([estimate(fut[i:i + 32, :-1], fut[i:i + 32, 1:]) for i in range(0, len(fut), 32)])
        alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
        roots = torch.where((true_scroll == 0).all(1) & alive[:, 15])[0]
        var = ((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum(-1).mean((0, 1))
        V = float(var.sum())
        ctx, ca, fa = cache["ctx"][roots].float(), cache["ctx_a"][roots], cache["fut_a"][roots]
        frames, hist = [ctx[:, j] for j in range(4)], [ca[:, j] for j in range(3)]
        gen = []
        for k in range(T.H):
            w_ = 4 if k == 0 else 5
            a = torch.stack(hist[-(w_ - 1):] + [fa[:, k]], 1)
            g = torch.cat([T.step(world, torch.stack(frames[-w_:], 1)[i:i + 32], a[i:i + 32], device, config)
                           for i in range(0, len(roots), 32)])
            gen.append(g); frames.append(g); hist.append(fa[:, k])
        gen = torch.stack(gen, 1)                                                              # [n,16,81,192]
        imag = torch.cat([ctx[:, -1:], gen], 1)
        false_scroll = (estimate(imag[:, :-1], imag[:, 1:]) != 0).float()                    # [n,16]
        true_f = fut[roots, 1:]
        err = ((gen - true_f) ** 2).sum(-1)                                                    # [n,16,81]
        cp = ((fut[roots, :1] - true_f) ** 2).sum(-1)
        out = {"roots": len(roots), "depth": {}}
        for k in (1, 2, 4, 8, 16):
            out["depth"][k] = {"false_scroll": float(false_scroll[:, k - 1].mean()),
                               "err_over_V": float(err[:, k - 1].sum(-1).mean() / V),
                               "copy_over_V": float(cp[:, k - 1].sum(-1).mean() / V)} | \
                              {g: float(err[:, k - 1][:, tok].sum(-1).mean() / var[tok].sum()) for g, tok in GROUPS.items()}
        moves = ((fa >= 1) & (fa <= 4)).any(1)
        out["any_false_scroll_in_16"] = float((false_scroll.sum(1) > 0).float().mean())
        out["err16_over_V_with_move_attempts"] = float(err[moves, 15].sum(-1).mean() / V) if moves.any() else None
        out["err16_over_V_without_move_attempts"] = float(err[~moves, 15].sum(-1).mean() / V) if (~moves).any() else None
        out["n_with_move_attempts"] = int(moves.sum())
        # mechanism: at each imagined step with a (necessarily blocked) move attempt, the INPUT frame's error at the
        # tile the move targets (player 31 + direction), vs whether that step's output falsely scrolled
        target = torch.tensor([31, 30, 32, 22, 40])
        inputs = torch.cat([ctx[:, -1:], gen[:, :-1]], 1)                                      # imagined input of step k
        true_in = torch.cat([fut[roots, :1], fut[roots, 1:16]], 1)
        is_move = (fa >= 1) & (fa <= 4)
        t_idx = target[fa.clamp(max=4)]
        t_err = ((inputs - true_in) ** 2).sum(-1).gather(2, t_idx[..., None])[..., 0]           # [n,16]
        e, f = t_err[is_move], false_scroll[is_move]
        qs = e.quantile(torch.tensor([0.5, 0.75, 0.9]))
        out["move_steps"] = int(is_move.sum())
        out["false_scroll_rate_on_move_steps"] = float(f.mean())
        out["false_scroll_rate_by_target_input_error"] = {
            "below_median": float(f[e <= qs[0]].mean()), "median_to_p75": float(f[(e > qs[0]) & (e <= qs[1])].mean()),
            "p75_to_p90": float(f[(e > qs[1]) & (e <= qs[2])].mean()), "above_p90": float(f[e > qs[2]].mean())}
        out["target_input_error_quantiles"] = [float(q) for q in qs]
        out["target_input_error_mean_false_vs_not"] = [float(e[f > 0].mean()) if (f > 0).any() else None,
                                                       float(e[f == 0].mean())]
        # first false scroll only (later ones may be consistent with an already-diverged imagined world), and the
        # head's own move-decision logit on each blocked-move step: imagined window vs the TRUE window
        before = (false_scroll.cumsum(1) - false_scroll) == 0                                  # no earlier false scroll
        first = is_move & before
        out["first_false_scroll_hazard_on_move_steps"] = float(false_scroll[first].mean())
        out["first_false_scroll_hazard_by_depth"] = {k: float(false_scroll[first[:, k - 1], k - 1].mean())
                                                     for k in (2, 4, 8, 16) if first[:, k - 1].any()}
        if hasattr(world, "frame"):
            from d4mj.train import autocast_context
            imag_logit, true_logit = torch.zeros(len(roots), 16), torch.zeros(len(roots), 16)
            true_hist = [ctx[:, j] for j in range(4)] + [fut[roots, 1 + j] for j in range(16)]
            imag_hist = [ctx[:, j] for j in range(4)] + [gen[:, j] for j in range(16)]
            acts = [ca[:, j] for j in range(3)] + [fa[:, j] for j in range(16)]
            for k in range(16):
                w_ = 4 if k == 0 else 5
                a = torch.stack(acts[3 + k - (w_ - 1):3 + k] + [fa[:, k]], 1).to(device)
                for hist, store in ((imag_hist, imag_logit), (true_hist, true_logit)):
                    frames_k = torch.stack(hist[4 + k - w_:4 + k], 1).to(device).float()
                    with autocast_context(config):
                        h, ha = world.backbone_full(frames_k, a)
                        tgt = target[fa[:, k].clamp(max=4)].to(device)
                        ar = torch.arange(len(roots), device=device)
                        logit = world.frame(ha[:, -1]).float()[:, 0]
                        if hasattr(world, "target_gate"):
                            logit = logit + world.target_gate(h[ar, -1, tgt]).float()[:, 0]
                    store[:, k] = logit.cpu()
            sel = first
            out["decision_logit_blocked_first_steps"] = {
                "true_window_mean": float(true_logit[sel].mean()), "imagined_window_mean": float(imag_logit[sel].mean()),
                "imagined_on_false_scroll": float(imag_logit[sel & (false_scroll > 0)].mean()) if (sel & (false_scroll > 0)).any() else None,
                "true_on_false_scroll_steps": float(true_logit[sel & (false_scroll > 0)].mean()) if (sel & (false_scroll > 0)).any() else None,
                "share_true_window_positive": float((true_logit[sel] > 0).float().mean()),
                "share_imagined_window_positive": float((imag_logit[sel] > 0).float().mean())}
        fs = false_scroll.sum(1) > 0
        out["err16_over_V_roots_with_false_scroll"] = float(err[fs, 15].sum(-1).mean() / V) if fs.any() else None
        out["err16_over_V_roots_without_false_scroll"] = float(err[~fs, 15].sum(-1).mean() / V) if (~fs).any() else None
        (HERE / f"static_{st['name']}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(st["name"], json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
