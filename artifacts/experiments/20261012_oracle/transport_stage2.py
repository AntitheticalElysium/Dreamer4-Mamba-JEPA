"""Stage 2 (2026-10-15): feasibility controls for retained-content transport. Simulator layouts define the TRUE assignment T*
here only as a diagnostic; nothing trains on them.
  rawswap  F1a, no learned component. The 104 bright DEV roots x 32 real keys (direct_eval cohort). T* per key: in the
           afterstate frame (root layout shifted by the exact recorded camera offset), every disappearance matched to an adjacent
           same-type appearance (greedy one-to-one) is a swap of the two cells. Applied to
             model  the frozen D parent's deterministic afterstate (PairedWorld.reference, pre-LN; paired_u0000 = exact parent):
                    zhat = LN(T* b)
             true   the real root tokens aligned by the offset (entering cells unknown -> excluded from fidelity)
           Both frozen readers (real ridge, bank) decode. Reported: on the 14 strict sources (audit S5 selection), the decoded
           event vs the true event, separately for keys whose true event is a move and keys that stay, for T* and for T = I;
           over every interior move in the panel, destination detection and source clearing; the readers on the real successor
           tokens as the ceiling; token RMS distance to the real successor at moved endpoints (T* vs T = I) against the
           real static floor (unchanged, non-entering cells).
  rawswap3k  the same with the paired study's 3,000-update reference (paired_u3000: the reference trained jointly with a
           stochastic completion), to test whether joint training turns the D parent's post-chance mean into an afterstate.
  Both also break down the model-swap failures on strict move keys by decoded event (stay/move-wrong-direction/absent/multiple).
  fit <arm>  F1b, a supplied-assignment fitting control (diagnostic: T* from layouts never enters the main experiment). From the
           exact D parent (paired_u0000; fresh PairedWorld + load_state_dict, so upstream Mamba no-decay flags are kept; the unused
           completion head is frozen), 1,500 updates, batch 16, L16, the parent's own D loss, phase AdamW/lr schedule, seeded
           windows of the training ledger restricted to rows with layouts (identical across arms):
             assigned  output LN(T*_t b_t): the reference is trained only through the true reassignment of its own tokens
             identity  output LN(b_t): the parent's objective continued for the same budget (matched control)
           T*_t: camera shift estimated from the real tokens (parameter-free scroll reader), layouts aligned to the afterstate,
           greedy adjacent same-type swaps. Checkpoints at 500 / 1,500 -> transport/fit_<arm>_u<N>.pt
  fiteval <arm> <N>  the rawswap readings with that fitted reference
Usage: transport_stage2.py rawswap|rawswap3k|fit <arm>|fiteval <arm> <N> -> artifacts/eda/oracle_v1/transport/stage2_<mode>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_s5_direct as A5  # noqa: E402

OUT = A5.BASE / "transport"
CROSS = A5.CROSS
NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def afterstate_layout(root, dr, dc):
    """[63,4] -> [63,4] int, -1 where the afterstate cell's source is outside the view (entering)"""
    out = -torch.ones(63, root.shape[-1], dtype=torch.long)
    for r in range(7):
        for c in range(9):
            if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                out[r * 9 + c] = root[(r + dr) * 9 + c + dc].long()
    return out


def swaps(prev, nxt):
    """greedy one-to-one adjacent same-type matches -> list of (source, destination, type)"""
    pairs, used_src, used_dst = [], set(), set()
    for k in range(prev.shape[-1]):
        gone = [x for x in range(63) if prev[x, k] == 1 and not nxt[x, k]]
        came = [x for x in range(63) if prev[x, k] == 0 and nxt[x, k]]
        for x in gone:
            r, c = divmod(x, 9)
            for y in came:
                if y not in used_dst and x not in used_src and abs(y // 9 - r) + abs(y % 9 - c) == 1:
                    pairs.append((x, y, k)); used_src.add(x); used_dst.add(y); break
    return pairs


def apply(tokens, pairs):
    out = tokens.clone()
    for s, d, _ in pairs:
        out[d], out[s] = tokens[s], tokens[d]
    return out


@torch.no_grad()
def rawswap(checkpoint="paired_u0000", state=None):
    import paired_model as PM
    import direct_train as T
    import direct_eval as E
    import direct_coherence as C
    from d4mj.train import autocast_context
    config = T.settings()
    data, _ = E.cohort("exploratory")
    world = PM.PairedWorld().cuda().eval()
    world.load_state_dict(state if state is not None else torch.load(A5.BASE / f"paired_event/training/{checkpoint}.pt", map_location="cpu", weights_only=False)["world"])
    bank = torch.load(A5.BASE / "snap_bank_readers.pt", map_location="cpu", weights_only=False)
    head = torch.load(A5.BASE / "learned_innovation/eval_u3000_h01/mob_heads.pt", map_location="cpu", weights_only=False)["heads"]["real"]
    n = len(data["roots"])
    refs = []
    for s0 in range(0, n, 4):
        ctx = data["context_tokens"][s0:s0 + 4].cuda().float(); ac = data["actions"][s0:s0 + 4].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(ac, (0, 1))
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            refs.append(world.reference(frames, ac, belief, ha)[:, 14].float().cpu())
    b = torch.cat(refs)                                                 # [104,81,192] pre-LN parent afterstate
    shifts, root_lay, fan_lay = data["shifts"], data["root_layout"], data["fan_layout"]
    real_tok, root_tok = data["fan_tokens"].float(), data["root_tokens"].float()
    K = real_tok.shape[1]
    model_swap = torch.empty(n, K, 81, 192); true_swap = torch.empty(n, K, 81, 192); entering = torch.zeros(n, 63, dtype=torch.bool)
    all_pairs = {}
    for i in range(n):
        dr, dc = (int(x) for x in shifts[i, 0])
        prev = afterstate_layout(root_lay[i], dr, dc)
        entering[i] = prev[:, 0] < 0
        aligned = root_tok[i].clone()
        for r in range(7):
            for c in range(9):
                if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                    aligned[r * 9 + c] = root_tok[i, (r + dr) * 9 + c + dc]
        for k in range(K):
            pr = swaps(prev, fan_lay[i, k].bool())
            all_pairs[(i, k)] = pr
            model_swap[i, k] = F.layer_norm(apply(b[i], pr), (192,))
            true_swap[i, k] = apply(aligned, pr)
    ident = F.layer_norm(b, (192,))[:, None].expand(-1, K, -1, -1)
    dec = {name: C.decode(x, bank, head)["mob"] for name, x in (("model_swap", model_swap), ("true_swap", true_swap),
                                                                ("identity", ident.contiguous()), ("real", real_tok))}
    units = json.loads((A5.OUT / "audit_recompute.json").read_text())["sources_list"]
    res = {"checkpoint": checkpoint, "roots": n, "keys": K, "strict_sources": {}, "panel_moves": {}, "fidelity": {}, "model_swap_move_key_events": {}}
    for reader in ("real", "bank"):
        strict = {v: {"move_keys_correct": 0, "move_keys": 0, "stay_keys_correct": 0, "stay_keys": 0} for v in dec}
        for root, source, kindname in units:
            kind = ("zombie", "cow", "skeleton").index(kindname)
            r, c = divmod(source, 9)
            pts = [(r + a) * 9 + c + e for a, e in CROSS]
            truth = A5.event(fan_lay[root][:, pts, kind].bool())
            for v, d in dec.items():
                ev = A5.event(d[reader][root][:, pts, kind])
                mv = truth > 0
                if v == "model_swap":
                    tally = res["model_swap_move_key_events"].setdefault(reader, {"correct": 0, "stay": 0, "other_direction": 0, "absent": 0, "multiple": 0})
                    for e, tr in zip(ev[mv].tolist(), truth[mv].tolist()):
                        tally["correct" if e == tr else "stay" if e == 0 else "absent" if e == 5 else "multiple" if e == 6 else "other_direction"] += 1
                strict[v]["move_keys"] += int(mv.sum()); strict[v]["move_keys_correct"] += int((ev == truth)[mv].sum())
                strict[v]["stay_keys"] += int((~mv).sum()); strict[v]["stay_keys_correct"] += int((ev == truth)[~mv].sum())
        res["strict_sources"][reader] = {v: {**s, "move_correct_rate": round(s["move_keys_correct"] / max(s["move_keys"], 1), 4),
                                             "stay_correct_rate": round(s["stay_keys_correct"] / max(s["stay_keys"], 1), 4)} for v, s in strict.items()}
        moves = {v: {"moves": 0, "destination_detected": 0, "source_cleared": 0, "both": 0} for v in dec}
        for (i, k), pr in all_pairs.items():
            for s, d, kind in pr:
                if kind > 2 or 31 in (s, d) or entering[i, s] or entering[i, d]:
                    continue
                for v, dd in dec.items():
                    m = dd[reader][i, k]
                    det, clr = bool(m[d, kind]), not bool(m[s, kind])
                    moves[v]["moves"] += 1; moves[v]["destination_detected"] += det; moves[v]["source_cleared"] += clr; moves[v]["both"] += det and clr
        res["panel_moves"][reader] = {v: {**m, "both_rate": round(m["both"] / max(m["moves"], 1), 4)} for v, m in moves.items()}
    # token fidelity at moved endpoints (interior, non-entering) and the static floor
    rms = lambda x, y: float((x - y).square().mean().sqrt())
    end = {"model_swap": [], "true_swap": [], "identity": []}; floor = []
    for (i, k), pr in all_pairs.items():
        moved = set()
        for s, d, kind in pr:
            if kind > 2 or 31 in (s, d) or entering[i, s] or entering[i, d]:
                continue
            for x in (s, d):
                moved.add(x)
                end["model_swap"].append(rms(model_swap[i, k, x], real_tok[i, k, x]))
                end["true_swap"].append(rms(true_swap[i, k, x], real_tok[i, k, x]))
                end["identity"].append(rms(ident[i, k, x], real_tok[i, k, x]))
        if k == 0:
            nxt = fan_lay[i, k].bool().any(-1); prv = (afterstate_layout(root_lay[i], *(int(x) for x in shifts[i, 0])) > 0).any(-1)
            for x in range(63):
                if x != 31 and not entering[i, x] and not nxt[x] and not prv[x] and x not in moved:
                    floor.append(rms(true_swap[i, k, x], real_tok[i, k, x]))
    res["fidelity"] = {v: {"n": len(x), "median_rms": round(float(np.median(x)), 4), "p90_rms": round(float(np.quantile(x, .9)), 4)} for v, x in end.items()}
    res["fidelity"]["static_floor_true_aligned_vs_real"] = {"n": len(floor), "median_rms": round(float(np.median(floor)), 4), "p90_rms": round(float(np.quantile(floor, .9)), 4)}
    return res


def fit(arm, updates=1500):
    import paired_model as PM
    import direct_train as T
    import loss_intervention as L
    from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr
    assert arm in ("assigned", "identity")
    config = T.settings()
    _, windows, actions, tokens = T.load_data()
    lay = np.load(A5.BASE / "layouts.npy", mmap_mode="r"); valid = torch.from_numpy(np.load(A5.BASE / "layouts_valid.npy"))
    windows = windows[valid[windows[:, 0]]]
    scroll = A5.load_scroll(); shifts = scroll.SHIFTS
    world = PM.PairedWorld()
    world.load_state_dict(torch.load(A5.BASE / "paired_event/training/paired_u0000.pt", map_location="cpu", weights_only=False)["world"])
    world = world.cuda().train()
    for name, prm in world.named_parameters():
        prm.requires_grad_(not name.startswith("completion."))
    opt = phase_optimizer([world], config)
    names = {id(prm): name for name, prm in world.named_parameters()}
    groups = [{"weight_decay": gr["weight_decay"], "parameters": len(gr["params"]), "names_sha": __import__("hashlib").sha256(
        ",".join(sorted(names[id(q)] for q in gr["params"])).encode()).hexdigest()} for gr in opt.param_groups]
    params = [prm for prm in world.parameters() if prm.requires_grad]
    g = torch.Generator().manual_seed(20261015)
    log, swaps_total = [], 0
    for u in range(updates):
        idx = torch.randint(len(windows), (16,), generator=g)
        frames, ac, choices = T.batch(tokens, actions, windows, idx)
        perm = torch.arange(81).repeat(16, 15, 1)
        if arm == "assigned":
            est = scroll.estimate(frames[:, :-1], frames[:, 1:]).cpu()
            for b, (row, start) in enumerate(choices.tolist()):
                Lw = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
                for t in range(15):
                    prev = afterstate_layout(Lw[t], *shifts[int(est[b, t])])
                    for src, dst, _ in swaps(prev, Lw[t + 1]):
                        perm[b, t, dst], perm[b, t, src] = src, dst; swaps_total += 1
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            ref = world.reference(frames, ac, belief, ha)[:, :15].float()
        ref = ref.gather(2, perm.cuda()[..., None].expand(-1, -1, -1, 192))
        out = F.layer_norm(ref, (192,))
        loss = L.objective(L.components(out, frames[:, 1:]), "D")
        optimizer_step(opt, loss, params, learning_rate=_phase_lr(config, u), grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
        if (u + 1) % 100 == 0:
            log.append({"update": u + 1, "loss": round(float(loss.detach()), 6), "swaps_so_far": swaps_total})
            print(json.dumps(log[-1]), flush=True)
        if u + 1 in (500, updates):
            torch.save({"arm": arm, "update": u + 1, "world": L.cpu_state(world), "groups": groups, "log": log},
                       OUT / f"fit_{arm}_u{u + 1:04d}.pt")
    return {"arm": arm, "updates": updates, "optimizer_groups": groups, "log": log, "eligible_windows": len(windows)}


if __name__ == "__main__":
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "fit":
        res = fit(sys.argv[2]); mode = f"fit_{sys.argv[2]}"
    elif mode == "fiteval":
        state = torch.load(OUT / f"fit_{sys.argv[2]}_u{int(sys.argv[3]):04d}.pt", map_location="cpu", weights_only=False)["world"]
        res = rawswap(f"fit_{sys.argv[2]}_u{int(sys.argv[3]):04d}", state); mode = f"fiteval_{sys.argv[2]}_u{int(sys.argv[3]):04d}"
    else:
        res = {"rawswap": rawswap, "rawswap3k": lambda: rawswap("paired_u3000")}[mode]()
    (OUT / f"stage2_{mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
