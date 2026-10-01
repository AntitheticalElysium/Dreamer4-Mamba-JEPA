"""E13. Diagnosis by substitution: which components CAUSE the imagined error and the wrong move decisions? (2026-10-01, after the
user's audit request: E11 established the chain by correlation and onset attribution; these are the causal tests.)

Data: the diagnosis futures (1,002 roots; five sampled 16-step futures per root under the same actions; sample 0 = factual).
Worlds: corrt per-tile worlds (their move gate can be set). Rollout: teval's convention (4 context frames, then a 5-frame window).

PART A -- oracle components, each a counterfactual rollout of the same world on the same roots (sample 0 supplies the truth):
  scroll   the move gate (corrt: frame logit + target gate) is set to +20 / -20 at every step from the TRUE scroll of that step,
           so the world itself draws the frame with the right "did the view move" decision (it still chooses directions and
           content). Reported: how often the drawn frame then scrolls as the truth does.
  cons     after a DO / place step (actions 5, 7-10), the faced tile's predicted token is replaced by the true one
  enter    on a scroll step, the cells entering the view are replaced by the true ones
  cons / enter substitute only while the imagined view position is still right, in every arm (afterwards true tokens would
  land on the wrong world cells); with `scroll`, the position goes wrong only if the world draws the wrong direction.
  Arms: base, scroll, scroll+cons, scroll+enter, scroll+cons+enter, cons, enter, cons+enter.
  Per arm: depth-16 excess (stochdiag: |g - mu|^2 - s^2/5, / V; all-five-alive depths) by class; share of roots whose imagined
  view position is ever wrong by 16 (all-five-alive and agreeing depths); first wrong decisions per depth.
PART B -- the decision step (base arm). For each first wrong move decision (depth >= 2), the world is re-run on the window that
  produced it with the CURRENT frame (a) as imagined, (b) only the target tile (the cell the move enters) replaced by the true
  token, (c) one random other map cell replaced (seeded), (d) every map cell replaced except the target tile, (e) fully true.
  Outcome: the drawn frame's scroll (scroll.estimate against the input current frame) equals the true scroll.
PART C -- one-step or self-feeding? (base arm) At DO / place steps whose imagined position is right, the faced tile is predicted
  from the imagined window and from the TRUE window (sample 0's frames, same actions). Per window: squared distance to the true
  next token, and the tile class (teval's ridge probe, test-seed roots only) vs the true class: change missed / change
  hallucinated / changed and right / unchanged and right.
Readings, declared before any run (per world):
  R1 position_causal       base -> scroll removes >= 50% of depth-16 excess
  R2 triggers_causal       cons+enter reduces the share of roots ever position-wrong by >= 30% (relative) vs base; reported
                           separately for cons and enter
  R3 residual              depth-16 excess of scroll+cons+enter / base (reported)
  R4 target_tile_causal    (b) restores the right decision on >= 50% of cases, (c) on <= 15%, and (d) on <= 30%
  R5 consequence_one_step  on the true window, the faced-tile class error rate (missed + hallucinated) is >= 50% of that on the
                           imagined window (the consequence error exists without self-feeding)
Usage: subst16.py <world.pt> ... -> evals/subst16_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import driftanat as DA  # noqa: E402
SD, T = DA.SD, DA.T
H, S = SD.H, SD.S
FACED = {0: 30, 1: 32, 2: 22, 3: 40}                     # facing one-hot index (left, right, up, down) -> cell
TARGET = {1: 30, 2: 32, 3: 22, 4: 40}                    # move action -> the cell it enters
ACT = (5, 7, 8, 9, 10)                                   # DO, place stone / table / furnace / plant
ARMS = (("base", ()), ("scroll", ("scroll",)), ("scroll+cons", ("scroll", "cons")), ("scroll+enter", ("scroll", "enter")),
        ("scroll+cons+enter", ("scroll", "cons", "enter")), ("cons", ("cons",)), ("enter", ("enter",)),
        ("cons+enter", ("cons", "enter")))


def entering_cells(shift_idx):
    """[63] bool mask of the map cells entering the view for scroll.SHIFTS index `shift_idx` (stochdiag's convention)."""
    from scroll import SHIFTS
    dr, dc = SHIFTS[int(shift_idx)]
    m = torch.zeros(7, 9, dtype=torch.bool)
    if dr == 1: m[6, :] = True
    if dr == -1: m[0, :] = True
    if dc == 1: m[:, 8] = True
    if dc == -1: m[:, 0] = True
    return m.flatten()


class Gate:
    """Forward hooks that set the corrt move gate to +-20 per batch row (frame logit) and 0 (target gate)."""

    def __init__(self, world):
        self.world, self.value, self.handles = world, None, []

    def __enter__(self):
        def frame_hook(_, __, out):
            return out.new_full(out.shape, 0.0) + self.value.to(out.device, out.dtype).view(-1, 1, 1)
        self.handles.append(self.world.frame.register_forward_hook(frame_hook))
        if hasattr(self.world, "target_gate"):
            self.handles.append(self.world.target_gate.register_forward_hook(lambda _, __, out: torch.zeros_like(out)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    test_roots = ~train_roots
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    cls, _ = SD.classes(meta, fut5[:, 0], root)                                           # [R,16,81]
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)                                 # [R,16]
    shifts = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        shifts.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(shifts, 1)                                                  # [R,5,16]
    ts0 = true_shift[:, 0]
    true_off = DA.offsets(true_shift).long()
    agree = (true_off == true_off[:, :1]).all(-1).all(1)
    valid = alive & agree
    allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()   # [R,17,1534] sample 0
    facing = allv[:, :, 1516:1520].argmax(-1)                                            # [R,17]: 0 = root
    tiles_true = allv[:, :, :1071].reshape(R, 17, 63, 17).argmax(-1)                     # [R,17,63]
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    fut0 = fut5[:, 0]
    out_dir = HERE / "evals"

    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        if world.head != "corrt":
            print(json.dumps({"world": name, "skipped": "oracle gate needs corrt"}), flush=True)
            continue
        batch = 16
        res = {"world": name, "V": V, "arms": {}}
        gens = {}
        for arm, flags in ARMS:
            gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
            drawn_right = torch.zeros(R, H, dtype=torch.bool)
            subs = {"cons": 0, "enter": 0}
            part_c = []
            for i in range(0, R, batch):
                b = min(batch, R - i)
                rows = torch.arange(i, i + b)
                c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
                frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
                tframes = [c4[:, j] for j in range(4)]
                on_track = torch.ones(b, dtype=torch.bool)                               # imagined position right so far
                for k in range(H):
                    w = 4 if k == 0 else 5
                    win, acts = torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                    if "scroll" in flags:
                        with Gate(world) as gate:
                            gate.value = torch.where(ts0[rows, k] != 0, 20.0, -20.0)
                            g = T.step(world, win, acts, device, config)
                    else:
                        g = T.step(world, win, acts, device, config)
                    est = estimate(frames[-1].float(), g)
                    ok = est == ts0[rows, k]
                    drawn_right[i:i + b, k] = ok
                    allowed = ok & on_track                                              # true tokens land on the right cells
                    truth = fut0[i:i + b, k].float()
                    if arm == "base":                                                    # PART C on DO / place steps
                        m = torch.isin(fk[:, k], torch.tensor(ACT)) & on_track & ok & valid[i:i + b, k]
                        if m.any():
                            tw = torch.stack(tframes[-w:], 1)[m]
                            gt = T.step(world, tw, acts[m], device, config)
                            for jj, r in enumerate(torch.nonzero(m)[:, 0].tolist()):
                                cell = FACED[int(facing[i + r, k])]
                                part_c.append((i + r, k, cell, float(((g[r, cell] - truth[r, cell]) ** 2).sum()),
                                               float(((gt[jj, cell] - truth[r, cell]) ** 2).sum()),
                                               g[r, cell].clone(), gt[jj, cell].clone()))
                    if "cons" in flags:
                        m = torch.isin(fk[:, k], torch.tensor(ACT)) & allowed
                        for r in torch.nonzero(m)[:, 0].tolist():
                            cell = FACED[int(facing[i + r, k])]
                            g[r, cell] = truth[r, cell]; subs["cons"] += 1
                    if "enter" in flags:
                        m = (ts0[rows, k] != 0) & allowed
                        for r in torch.nonzero(m)[:, 0].tolist():
                            cells = torch.nonzero(entering_cells(ts0[i + r, k]))[:, 0]
                            g[r, cells] = truth[r, cells]; subs["enter"] += 1
                    on_track &= ok
                    gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fk[:, k]); tframes.append(truth)
            gens[arm] = gen if arm == "base" else None
            # metrics
            gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
            img_off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])).long()
            aligned = (img_off == true_off[:, 0]).all(-1)
            wrong_ever = ((~aligned) & valid).any(1)
            excess = torch.zeros(len(SD.CLASSES)); n16 = int(alive[:, -1].sum())
            exd = torch.zeros(H)
            for j in range(0, R, 16):
                bb = min(16, R - j)
                x = fut5[j:j + bb].float(); mu = x.mean(1)
                s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
                ex = ((gen[j:j + bb].float() - mu) ** 2).sum(-1) - s2 / S                  # [bb,16,81]
                for k in range(H):
                    exd[k] += float((ex[:, k].sum(-1) * alive[j:j + bb, k]).sum())
                m16 = alive[j:j + bb, -1]
                for ci in range(len(SD.CLASSES)):
                    excess[ci] += float((ex[:, -1] * (cls[j:j + bb, -1] == ci) * m16[:, None]).sum())
            res["arms"][arm] = {
                "excess16": float(excess.sum() / n16 / V), "excess16_by_class": {c: float(excess[ci] / n16 / V) for ci, c in enumerate(SD.CLASSES)},
                "excess_by_depth": (exd / alive.sum(0).float() / V).tolist(),
                "ever_position_wrong": float(wrong_ever[valid.any(1)].float().mean()),
                "position_wrong_at16": float((~aligned[:, -1])[valid[:, -1]].float().mean()),
                "drawn_scroll_right_rate": float(drawn_right[valid].float().mean()),
                "substitutions": subs}
            if arm == "base":
                res["part_c_raw"] = part_c
            print(json.dumps({"world": name, "arm": arm, **{k: v for k, v in res["arms"][arm].items() if k not in ("excess_by_depth", "excess16_by_class")}}), flush=True)
        base = res["arms"]["base"]
        # PART B: first wrong decisions in the base rollout
        gen = gens["base"]
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_shift = torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])
        aligned = (DA.offsets(img_shift).long() == true_off[:, 0]).all(-1)
        prev_ok = torch.cat([torch.ones(R, 1, dtype=torch.bool), aligned[:, :-1]], 1)
        move = (fa >= 1) & (fa <= 4)
        first = valid & prev_ok & ~aligned & move & (torch.cumsum((valid & ~aligned).int(), 1) == 1)
        first[:, 0] = False
        rng = torch.Generator().manual_seed(20261001)
        cases = torch.nonzero(first).tolist()
        outcome = {v: [] for v in ("imagined", "target_true", "random_true", "all_but_target", "all_true")}
        kinds = []
        for j in range(0, len(cases), 64):
            chunk = cases[j:j + 64]
            wins, acts, tg, rc, truth_s, kd = [], [], [], [], [], []
            for r, k in chunk:
                full = torch.cat([ctx[r].float(), gen[r, :k].float()], 0)[-5:]
                wins.append(full); acts.append(torch.cat([ca[r], fa[r, :k + 1]], 0)[-5:])
                t = TARGET[int(fa[r, k])]; tg.append(t)
                others = [c for c in range(63) if c not in (t, 31)]
                rc.append(others[int(torch.randint(len(others), (1,), generator=rng))])
                truth_s.append(int(ts0[r, k])); kd.append("missed" if int(ts0[r, k]) != 0 else "false")
            wins, acts = torch.stack(wins), torch.stack(acts)
            cur_true = torch.stack([fut0[r, k - 1].float() for r, k in chunk])
            truth_s = torch.tensor(truth_s)
            kinds += kd
            for v in outcome:
                wv = wins.clone()
                cur = wv[:, -1]
                for q in range(len(chunk)):
                    if v == "target_true":
                        cur[q, tg[q]] = cur_true[q, tg[q]]
                    elif v == "random_true":
                        cur[q, rc[q]] = cur_true[q, rc[q]]
                    elif v == "all_but_target":
                        keep = cur[q, tg[q]].clone(); cur[q, :63] = cur_true[q, :63]; cur[q, tg[q]] = keep
                    elif v == "all_true":
                        cur[q] = cur_true[q]
                out = T.step(world, wv, acts, device, config)
                outcome[v].append(estimate(wv[:, -1].float(), out) == truth_s)
        outcome = {v: torch.cat(x) for v, x in outcome.items()} if cases else {}
        kinds = torch.tensor([kd == "missed" for kd in kinds])
        res["part_b"] = {"n": len(cases), "n_missed": int(kinds.sum()) if cases else 0}
        for lab, m in (("all", torch.ones(len(kinds), dtype=torch.bool)), ("missed", kinds), ("false", ~kinds)):
            if cases and m.any():
                res["part_b"][lab] = {v: float(x[m].float().mean()) for v, x in outcome.items()}
        # PART C summary (test-seed roots for the class readout)
        pc = res.pop("part_c_raw")
        dist_img = torch.tensor([x[3] for x in pc]); dist_true = torch.tensor([x[4] for x in pc])
        on_test = torch.tensor([bool(test_roots[x[0]]) for x in pc])
        pred_img = probes.tile(torch.stack([x[5] for x in pc])).argmax(-1); pred_true = probes.tile(torch.stack([x[6] for x in pc])).argmax(-1)
        before = torch.tensor([int(tiles_true[x[0], x[1], x[2]]) for x in pc])          # state before the action (index k)
        after = torch.tensor([int(tiles_true[x[0], x[1] + 1, x[2]]) for x in pc])        # true state after (index k+1)
        changed = before != after
        conf = {}
        for lab, pred in (("imagined", pred_img), ("true", pred_true)):
            m = on_test
            conf[lab] = {"n": int(m.sum()), "n_changed": int((changed & m).sum()),
                         "change_missed": float(((pred != after) & changed & m).sum() / (changed & m).sum().clamp(min=1)),
                         "change_hallucinated": float(((pred != after) & ~changed & m).sum() / (~changed & m).sum().clamp(min=1)),
                         "class_error": float(((pred != after) & m).sum() / m.sum().clamp(min=1))}
        res["part_c"] = {"steps": len(pc), "faced_distance": {"imagined_window": float(dist_img.mean()), "true_window": float(dist_true.mean())},
                         "faced_distance_changed": {"imagined_window": float(dist_img[changed].mean()) if changed.any() else None,
                                                    "true_window": float(dist_true[changed].mean()) if changed.any() else None},
                         "class": conf, "change_rate": float(changed.float().mean())}
        a = res["arms"]
        pb = res["part_b"].get("all", {})
        res["readings"] = {
            "R1_scroll_removes": 1 - a["scroll"]["excess16"] / a["base"]["excess16"],
            "R1_position_causal": 1 - a["scroll"]["excess16"] / a["base"]["excess16"] >= 0.5,
            "R2_cons_reduces_wrong": 1 - a["cons"]["ever_position_wrong"] / a["base"]["ever_position_wrong"],
            "R2_enter_reduces_wrong": 1 - a["enter"]["ever_position_wrong"] / a["base"]["ever_position_wrong"],
            "R2_cons_enter_reduces_wrong": 1 - a["cons+enter"]["ever_position_wrong"] / a["base"]["ever_position_wrong"],
            "R2_triggers_causal": 1 - a["cons+enter"]["ever_position_wrong"] / a["base"]["ever_position_wrong"] >= 0.3,
            "R3_residual": a["scroll+cons+enter"]["excess16"] / a["base"]["excess16"],
            "R4_target_tile_causal": bool(pb) and pb["target_true"] >= 0.5 and pb["random_true"] <= 0.15 and pb["all_but_target"] <= 0.3,
            "R5_consequence_one_step": conf["true"]["class_error"] >= 0.5 * conf["imagined"]["class_error"]}
        (out_dir / f"subst16_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "readings": res["readings"], "part_b": res["part_b"], "part_c": res["part_c"]}), flush=True)
        del world, gens
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
