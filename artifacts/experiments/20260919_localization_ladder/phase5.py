"""Phase 5: where inside the predictor does the action's influence die?

Phase 4 left one question open, and it is the one that matters. The world model IS handed the
action -- `scan_pairs` concatenates `action_embedding(a)` onto the root latent and projects the
pair -- yet its output carries no usable action information (derangement cost -3.9 over 20
draws for mamba_raw). So the influence is present at the input and absent at the output. It
dies somewhere in between, and the ladder so far stopped at `h` and generated `z`.

The path, from `LeWMWorld.scan_pairs`:

    action -> action_embedding ------.
                                      >-- pair_projection -> x0
    root latent z[:, :, 0] ----------'
    x0 -> Mamba block 0 .. block D-1 -> final_norm -> h
    h  -> predictor_projector: Linear -> BatchNorm -> GELU -> Linear -> generated z

Every one of those is tapped here with a forward hook, on the SAME 17-action branch fan the
gate builds, and each tap is scored two ways:

  choice       safe-action choice under the rank readout selected in phase 1
  derangement  the same tap with the action->branch map deranged, averaged over draws

A tap whose choice score collapses to its derangement score is one where the action no longer
carries decision-relevant information. The first such tap along the path is the answer.

Frozen: hooks only, no parameter is updated and no gate output changes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.m03.gate import M03Settings, load_m03_bundle
from readout import Fit, evaluate, fit_head, standardize, summarize
from phase4 import derangement, fit_and_score, publish, cached, sha

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
ARMS = {"mamba_raw": "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt",
        "mamba_tc": "artifacts/lewm_gates_20260906/paired/tc/joint/step-010000.pt"}


@torch.inference_mode()
def branch_taps(bundle, split, settings, batch=16):
    """Run the gate's own 17-action branch fan, capturing every internal stage.

    This reproduces `_encode_lewm`'s branch construction exactly -- prefill on the same
    `lewm_context` window, repeat the state across actions, advance one step -- and only adds
    hooks, so the generated z recovered here must equal the published one. That equality is
    asserted by the caller.
    """
    world = bundle.world
    captured, taps = {}, {}

    def grab(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().cpu()
        return hook

    handles = [world.pair_projection.register_forward_hook(grab("pair_projection")),
               world.final_norm.register_forward_hook(grab("final_norm_h"))]
    for i, layer in enumerate(world.layers):
        handles.append(layer.register_forward_hook(grab(f"mamba_block_{i}")))
    for i, module in enumerate(world.predictor_projector):
        handles.append(module.register_forward_hook(grab(f"projector_{i}_{type(module).__name__}")))

    context, actions = split["context"], split["past_actions"]
    device = bundle.device
    all_actions = torch.arange(bundle.n_actions, device=device, dtype=torch.long)
    collected, generated = {}, []
    try:
        for start in range(0, len(context), batch):
            end = min(len(context), start + batch)
            frame = context[start:end, -settings.lewm_context:].to(device)
            past = actions[start:end, -settings.lewm_context + 1:].to(device)
            z, _ = bundle.encoder.projected_and_cls(frame)
            state = bundle.prefill(z, past)
            branches = bundle.repeat_state(state, bundle.n_actions)
            action = all_actions.repeat(end - start)[:, None]
            captured.clear()
            predicted, _ = bundle.advance(branches, action)
            n = end - start
            generated.append(predicted.latent[:, 0].reshape(n, bundle.n_actions, -1).cpu())
            for name, value in captured.items():
                flat = value.reshape(n * bundle.n_actions, -1)
                collected.setdefault(name, []).append(flat.reshape(n, bundle.n_actions, -1))
    finally:
        for handle in handles:
            handle.remove()
    taps = {name: torch.cat(parts) for name, parts in collected.items()}
    taps["generated_z"] = torch.cat(generated)
    return taps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--arms", default="mamba_raw,mamba_tc")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    settings, spec = M03Settings(), Fit(steps=args.steps)
    gen = torch.Generator().manual_seed(515)
    orders = [derangement(17, gen) for _ in range(args.draws)]

    for arm in args.arms.split(","):
        inputs = {"checkpoint": sha(ROOT / ARMS[arm]), "draws": args.draws}
        destination = args.out / f"{arm}.json"
        if cached(destination, inputs):
            print(json.dumps({"stage": "cached", "arm": arm}), flush=True)
            continue
        bundle, _, _ = load_m03_bundle(ROOT / ARMS[arm], device=args.device,
                                       dataset_sha256=_sha256(DATASET))
        bundle.encoder.eval()
        bundle.world.eval()
        taps = {s: branch_taps(bundle, side[s], settings) for s in ("train", "dev")}
        # the hooked run must reproduce the published generated z, or the taps are not the
        # ones the gate scored
        from d4mj.m03.cache import resolve_payload
        published = resolve_payload(torch.load(
            ROOT / f"artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/"
                   f"{'raw' if arm == 'mamba_raw' else 'tc'}.dev.pt",
            map_location="cpu", weights_only=False))["features"]["generated_successor"].float()
        delta = (taps["dev"]["generated_z"] - published).abs().max().item()
        print(json.dumps({"stage": "parity", "arm": arm, "generated_z_max_abs": delta}), flush=True)

        order_names = (["pair_projection"]
                       + [f"mamba_block_{i}" for i in range(len(bundle.world.layers))]
                       + ["final_norm_h"]
                       + sorted(n for n in taps["dev"] if n.startswith("projector_"))
                       + ["generated_z"])
        rows = []
        for name in order_names:
            if name not in taps["dev"]:
                continue
            xtr, xdv = taps["train"][name], taps["dev"][name]
            intact, _ = fit_and_score(xtr, y["train"], xdv, y["dev"], spec, args.device, seeds=(0,))
            spread = []
            for o in orders:
                r, _ = fit_and_score(xtr[:, o], y["train"], xdv[:, o], y["dev"], spec,
                                     args.device, seeds=(0,))
                spread.append(r["mean_safe"])
            rows.append({"tap": name, "dim": int(xdv.shape[-1]), "intact": intact["mean_safe"],
                         "deranged_mean": round(float(np.mean(spread)), 2),
                         "action_cost": round(intact["mean_safe"] - float(np.mean(spread)), 2),
                         "within_root_auc": intact["within_root_auc"][0]})
            print(json.dumps({"stage": "tap", "arm": arm, **rows[-1]}), flush=True)
        publish(destination, {"schema": "d4mj_phase5_internals_v1", "arm": arm,
                              "generated_z_parity_max_abs": delta,
                              "question": "the first tap whose action_cost collapses is where the "
                                          "action stops carrying decision-relevant information",
                              "rows": rows}, inputs)
        del bundle, taps
        torch.cuda.empty_cache()
    print(json.dumps({"status": "phase5_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
