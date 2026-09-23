"""Frozen internal transition ladder: the first point inside `advance()` where safety goes missing.

CONFIRM.md found no fresh readout recovering within-root safety from the generated successor
latent. Reading `LeWMWorld` before building this settles what can differ between the two paths:
`observe_latent` runs `advance` UNCHANGED and then swaps only the latent slot, so the Mamba output
`u_t` is identical for the real and generated successor, and `features = agent_readout(cat(latent,
u_t))`. The 0.848 -> 0.464 gap between real and generated features is therefore entirely one input
slot, z_true versus z_hat. Every rung below is shared by both paths; there is no separate real
internal ladder to run.

Rungs, captured by forward hooks during the ONE `scan_pairs` that `advance()` runs for the fanned
root, so no file in the source closure is edited:

  pair_projection     pair_projection(cat(z_t, embed(a)))   root latent + action, NO history
  block_1..block_6    the residual stream after each Mamba block
  u                   final_norm(block_6) = `history`, what the agent readout reads beside the latent
  predictor_hidden    GELU(BN(Linear(u))) inside predictor_projector, 2048 wide
  generated_z         predictor_projector output = `advanced.latent`
  generated_features  agent_readout(cat(generated_z, u)), tying this ladder to CONFIRM.md

Information bar. Every rung is a deterministic function of the root state and the action -- what
the `context_action` control sees. The real successor is NOT a fair ceiling: Craftax draws zombie
movement (75% chase, otherwise random) and mob spawns from the step RNG, so z_true can carry
outcome randomness no function of (state, action) can. A rung beating root+action has COMPUTED the
consequence into a probe-accessible form; one that does not, has not.

Declared rules, fixed and committed before launch:

  A rung CARRIES safety when its terminal safe-choice on the judgement roots beats the
  `context_action` control with a paired, episode-seed-clustered 95% interval above zero, read in
  the ADAPTER family -- the only probe that applies identically to every rung and to the control.
  The exact deployed-capacity head runs on every 256-wide rung; where the two families disagree the
  rung is `probe_dependent` and counts as evidence for neither side. (Exact-vs-adapter-control is
  conservative: the control has the extra adapter layer.)

  u carries and generated_z does not                  -> predictor_projector_bottleneck
  every stack rung through u is a clean non-carry     -> dynamics_never_construct_it
  a block carries and u does not                      -> built_then_lost_inside_the_stack
  u and generated_z carry, generated_features does not -> agent_readout_combination
  any rung with no terminal test at all               -> insufficient_coverage
  anything else                                       -> mixed

Post-hoc root-side controls, added AFTER the declared run returned `dynamics_never_construct_it`
and committed before they were run. They never feed `verdict`. That call presumes the stack's
INPUT holds the information; even root+action sat below the marginal, so it was not established
that anything observable at the root predicts which action kills. Three rungs, adapter probe:

  root_z_action            the 4 root z + action: exactly what the transition consumes
  root_cls_action          the 4 root CLS + action: the encoder before the projector
  root_cls_patches_action  + the last root frame's 4x4 pooled patch grid (TC-LeWM's policy input)

  root_z_action beats root+action              -> input_carries_it: the stack fails to compute
                                                  it; training the transition is well-posed
  only the CLS/patch rungs beat root+action    -> projector_drops_it: lost before the transition
                                                  sees it; training on z cannot recover it
  none beats root+action                       -> not_predictable_from_root: the real successor's
                                                  edge is realized outcome, incl. step randomness

Capture checks fail the RUN, not the verdict: the `final_norm` capture must equal
`advanced.history` and the predictor output must equal `advanced.latent`, exactly; and the three
rungs shared with confirm (generated_z, generated_features, context_action) are checked against
its recorded numbers.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import Exact, seeds_for  # noqa: E402
from ladder import Readout, _gap, decisions, fit, paired, predict, rows_for  # noqa: E402

BLOCKS = tuple(f"block_{i}" for i in range(1, 7))
STACK = ("pair_projection", *BLOCKS, "u")
RUNGS = (*STACK, "predictor_hidden", "generated_z", "generated_features")
CONTROL = "context_action"
ROOT_CONTROLS = ("root_z_action", "root_cls_action", "root_cls_patches_action")
WIDE = (*STACK, "generated_features")          # 256 wide: the deployed head takes them unmodified


@torch.no_grad()
def materialize(bundle, rows, span, batch=16):
    """Every rung for all 17 actions of every root, captured inside the single `advance` scan."""
    world, device, count = bundle.world, bundle.device, bundle.n_actions
    captured, live = {}, [False]

    def grab(name, pick=lambda out: out):
        def hook(module, args, out):
            if live[0]:
                captured[name] = pick(out).detach()
        return hook

    handles = [world.pair_projection.register_forward_hook(grab("pair_projection"))]
    handles += [layer.register_forward_hook(grab(name, lambda out: out[0]))
                for name, layer in zip(BLOCKS, world.layers)]
    handles += [world.final_norm.register_forward_hook(grab("u")),
                world.predictor_projector[2].register_forward_hook(grab("predictor_hidden")),
                world.predictor_projector.register_forward_hook(grab("predictor_out"))]
    packs = {name: [] for name in (CONTROL, *ROOT_CONTROLS, *RUNGS)}
    reward, terminated, seeds, identity = [], [], [], hashlib.sha256()
    try:
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            n = len(chunk)
            for row in chunk:
                identity.update(repr((int(row["seed"]), int(row["step"]))).encode())
            frames = torch.stack([r["frames"][-span:] for r in chunk]).to(device)
            past = torch.stack([r["led_to_action"][-span + 1:] for r in chunk]).to(device)
            # `export` returns the same z `encoder(frames)` does, plus CLS and the pooled patch
            # grid, from one pass; the shared-rung reproduction check verifies z is unchanged.
            z_root, cls_root, grid_root = bundle.encoder.export(frames)
            state = world.teacher(z_root, past).state
            root_features = world.features(state)[:, -1, 0]
            actions = torch.arange(count, device=device).repeat(n)[:, None]
            fan = bundle.repeat_state(state, count)
            captured.clear()
            live[0] = True
            advanced, generated = bundle.advance(fan, actions)
            live[0] = False
            if not torch.equal(captured["u"], advanced.history):
                raise SystemExit("capture: final_norm output is not the state's history")
            if not torch.equal(captured["predictor_out"].reshape(advanced.latent.shape),
                               advanced.latent):
                raise SystemExit("capture: predictor output is not the generated latent")
            onehot = nn.functional.one_hot(actions[:, 0], count).float()
            packs[CONTROL].append(
                torch.cat((root_features.repeat_interleave(count, 0), onehot), -1).cpu())
            fanned = lambda x: torch.cat((x.flatten(1).repeat_interleave(count, 0), onehot), -1).cpu()
            packs["root_z_action"].append(fanned(z_root))
            packs["root_cls_action"].append(fanned(cls_root))
            packs["root_cls_patches_action"].append(
                fanned(torch.cat((cls_root.flatten(1), grid_root[:, -1].flatten(1)), -1)))
            for name in STACK:
                packs[name].append(captured[name][:, 0].cpu())
            packs["predictor_hidden"].append(captured["predictor_hidden"].cpu())
            packs["generated_z"].append(advanced.latent[:, 0, 0].cpu())
            packs["generated_features"].append(generated[:, -1, 0].cpu())
            reward.append(torch.stack([r["reward"] for r in chunk]).reshape(-1))
            terminated.append(torch.stack([r["terminated"] for r in chunk]).reshape(-1))
            seeds.append(torch.tensor([int(r["seed"]) for r in chunk]).repeat_interleave(count))
    finally:
        for handle in handles:
            handle.remove()
    data = {name: torch.cat(value).float() for name, value in packs.items()}
    data.update(reward=torch.cat(reward).float(), terminated=torch.cat(terminated).bool(),
                seed=torch.cat(seeds), roots=len(rows), row_identity=identity.hexdigest())
    return data


def verdict(matrix):
    """The declared rules. Returns the call and the per-rung reading it was made from."""
    def carries(cell):
        # None when there is no test at all -- too few terminal-opportunity roots. A missing test
        # is not a failed one: reading it as False let a smoke run with no terminal coverage
        # return `dynamics_never_construct_it`, a verdict made from no data.
        test = (cell or {}).get("judge", {}).get("terminal", {}).get("vs_context_action")
        if not test or test.get("difference") is None:
            return None
        return bool(test["difference"] > 0 and test["excludes_zero"])

    reading = {}
    for rung in RUNGS:
        adapter = carries(matrix[rung].get("adapter"))
        exact = carries(matrix[rung]["exact"]) if "exact" in matrix[rung] else adapter
        if adapter is None or exact is None:
            reading[rung] = None
        else:
            reading[rung] = "probe_dependent" if exact != adapter else adapter
    u, gz, gf = reading["u"], reading["generated_z"], reading["generated_features"]
    if any(value is None for value in reading.values()):
        call = "insufficient_coverage"
    elif u is True and gz is False:
        call = "predictor_projector_bottleneck"
    elif all(reading[r] is False for r in STACK):
        call = "dynamics_never_construct_it"
    elif any(reading[r] is True for r in BLOCKS) and u is False:
        call = "built_then_lost_inside_the_stack"
    elif u is True and gz is True and gf is False:
        call = "agent_readout_combination"
    else:
        call = "mixed"
    lost = None
    carried = [i for i, r in enumerate(STACK) if reading[r] is True]
    if carried and carried[-1] + 1 < len(STACK):
        lost = STACK[carried[-1] + 1]
    return {"call": call, "reading": reading, "first_rung_after_last_carry": lost}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--confirm", type=Path, default=HERE / "evidence/confirm.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--fit-roots", type=int, default=8000)
    parser.add_argument("--judge-roots", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--name", default="transition",
                        help="evidence file stem; the declared run is `transition`")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from d4mj.experiments import _load_bridge_parent
    from d4mj.lewm_config import window_layout

    checkpoint = args.checkpoint or args.run / "bridge/step-002000.pt"
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    heads.eval()
    config = bundle.config
    span = window_layout(config.joint)[0][-1] + 1
    device, count = bundle.device, bundle.n_actions
    width, bins = config.dynamics.width, config.agent.bins
    centers = heads.centers.detach().clone().to(device)

    # The SAME fit and judgement roots as confirm: fit on the retired fit seeds, judge on the 405
    # the partition left unallocated. No head here is fitted or selected on a judgement root.
    partition = json.loads(args.partition.read_text())
    fit_seeds, judge_seeds = seeds_for(partition, FORK_STORE)
    splits = {"fit": materialize(bundle, rows_for(fit_seeds, span, args.fit_roots), span),
              "judge": materialize(bundle, rows_for(judge_seeds, span, args.judge_roots), span)}
    print(json.dumps({"stage": "materialized", **{k: v["roots"] for k, v in splits.items()},
                      "seconds": round(time.time() - started, 1)}), flush=True)
    del bundle, heads
    torch.cuda.empty_cache()

    fit_roots = splits["fit"]["roots"]
    inner = torch.zeros(fit_roots, dtype=torch.bool)
    inner[torch.randperm(fit_roots, generator=torch.Generator().manual_seed(args.seed + 3))
          [: max(1, int(round(0.15 * fit_roots)))]] = True
    holdout = inner.repeat_interleave(count)

    control, matrix, rows_out = {}, {name: {} for name in (CONTROL, *ROOT_CONTROLS, *RUNGS)}, {}
    plan = [("adapter", Readout, CONTROL)] + [("adapter", Readout, r) for r in ROOT_CONTROLS] \
        + [("adapter", Readout, r) for r in RUNGS] \
        + [("exact", Exact, r) for r in WIDE]
    for variant, make, rung in plan:
        model = fit(splits["fit"][rung], splits["fit"]["reward"], ~splits["fit"]["terminated"],
                    centers, steps=args.steps, seed=args.seed, device=device, width=width,
                    bins=bins, holdout=holdout, make=make)
        cell = {"variant": variant, "rung": rung}
        for split in ("fit", "judge"):
            data = splits[split]
            pr, pd = predict(model, data[rung], centers, device)
            if split == "judge":
                rows_out[f"{variant}:{rung}"] = {"reward": pr, "death": pd}
            cell[split] = decisions(data["reward"], data["terminated"], pr, pd, data["seed"],
                                    count, draws=args.draws, seed=args.seed + 7)
        if rung == CONTROL:
            control = {(s, k): v for s in ("fit", "judge") for k, v in cell[s]["_vectors"].items()}
        for split in ("fit", "judge"):
            for key, higher in (("reward", False), ("terminal", True)):
                mine, theirs = cell[split]["_vectors"].get(key), control.get((split, key))
                if mine is None or theirs is None or rung == CONTROL:
                    continue
                left, right = (mine[0], theirs[0]) if higher else (theirs[0], mine[0])
                cell[split][key]["vs_context_action"] = paired(
                    left, right, mine[1], draws=args.draws, seed=args.seed + 11)
            del cell[split]["_vectors"]
        cell["fit_judge_gap"] = _gap({"train": cell["fit"], "dev": cell["judge"]})
        cell["fit_trace"] = model.trace
        matrix[rung][variant] = cell
        test = cell["judge"]["terminal"].get("vs_context_action") or {}
        print(json.dumps({"stage": "scored", "variant": variant, "rung": rung,
                          "safe": cell["judge"]["terminal"].get("safe_choice"),
                          "vs_context": test.get("difference"), "resolved": test.get("excludes_zero"),
                          "seconds": round(time.time() - started, 1)}), flush=True)
        del model
        torch.cuda.empty_cache()

    # The rungs shared with confirm must reproduce it: same checkpoint, roots, seeds and fit.
    recorded = json.loads(args.confirm.read_text())["matrix"]
    shared = {"adapter:generated_z": ("generated_z", "adapter:generated_z"),
              "adapter:generated_features": ("generated_features", "adapter:generated_features"),
              "exact:generated_features": ("generated_features", "exact:generated_features"),
              "adapter:context_action": ("context_action", "adapter:context_action")}
    reproduction = {}
    for name, (target, key) in shared.items():
        variant, rung = name.split(":")
        then = recorded[target][key]["judge"]["terminal"].get("safe_choice")
        now = matrix[rung][variant]["judge"]["terminal"].get("safe_choice")
        reproduction[name] = {"confirm": then, "transition": now,
                              "abs_diff": None if None in (then, now) else abs(then - now)}

    rows_path = args.out / f"{args.name}_rows.pt"
    torch.save({"truth": {k: splits["judge"][k] for k in ("reward", "terminated", "seed")},
                "count": count, "predictions": rows_out}, rows_path)
    report = {"schema": "d4mj_transition_ladder_v1",
              "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              "partition_sha256": _sha256(args.partition), "script_sha256": _sha256(Path(__file__)),
              "ladder_sha256": _sha256(HERE / "ladder.py"),
              "confirm_script_sha256": _sha256(HERE / "confirm.py"),
              "row_identity": {k: v["row_identity"] for k, v in splits.items()},
              "roots": {k: v["roots"] for k, v in splits.items()},
              "steps": args.steps, "draws": args.draws, "seed": args.seed,
              "reproduces_confirm": reproduction,
              "judge_rows": {"path": rows_path.name, "sha256": _sha256(rows_path),
                             "layout": "predictions['<variant>:<rung>']['reward'|'death'], "
                                       "root-major, 17 actions per root"},
              "matrix_layout": "matrix[rung][variant][split]",
              "matrix": matrix}
    report["verdict"] = verdict(matrix)
    tests = {r: matrix[r]["adapter"]["judge"]["terminal"].get("vs_context_action")
             for r in ROOT_CONTROLS}
    beats = {r: None if not t else bool(t["difference"] > 0 and t["excludes_zero"])
             for r, t in tests.items()}
    report["post_hoc_root"] = {
        "beats_root_action": beats,
        "call": ("insufficient_coverage" if None in beats.values() else
                 "input_carries_it" if beats["root_z_action"] else
                 "projector_drops_it" if beats["root_cls_action"] or beats["root_cls_patches_action"]
                 else "not_predictable_from_root"),
        "note": "post hoc; added after the declared call and never read by `verdict`. "
                "not_predictable_from_root is a statement about this probe family, not a proof "
                "that no function of the root observation predicts termination"}
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "transition_complete", "seconds": round(time.time() - started, 1),
                      **report["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
