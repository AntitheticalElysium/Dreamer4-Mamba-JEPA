"""Phase 6: the two diagnostics phase 5 should have run, and the one it overstated.

An audit made three fair objections to phase 5:

1. It instrumented the original Raw/TC worlds, not the `u->u` world -- even though the diagnosis
   itself named that the cheapest next experiment. Real u is 36/36 and generated u is 23.0 on ONE
   lineage, which brackets the loss far more tightly than anything in Raw/TC.
2. Its "HUD tokens" are final ViT tokens after global self-attention, subset by original patch
   position. A token positioned over the HUD may carry information gathered from the whole image.
   Subsetting contextualized tokens cannot establish that HUD PIXELS explain the patch advantage.
   Only re-encoding actually masked images can.
3. The derangement test refits a head on permuted TRAIN and DEV, so a null result shows the
   representation lacks robust root-specific action correspondence under refitting -- not that
   no action information exists. A frozen head evaluated on permuted DEV actions is the
   stricter, promised test.

So this runs: masked-image re-encoding (a real causal test), `u->u` internals through every
block and projector stage, and BOTH permutation tests, with three seeds, per-root scores
retained, and within-root AUC reported beside safe choice -- because phase 5's own AUC column
told a different story from its headline.
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
from d4mj.m03.cache import resolve_payload
from d4mj.m03.gate import M03Settings, load_m03_bundle
from readout import Fit, fit_head, scores_of, standardize, summarize
from phase4 import derangement, publish, cached, sha

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
MATCHED = ROOT / "artifacts/lewm_gates_20260918/m03_matched"
WORLDS = ROOT / "artifacts/experiments/20260918_matched_10k/evidence"
SEEDS = (0, 1, 2)
HUD_ROWS = slice(49, 63)     # measured: health correlates with pixel rows 49-62
MAP_ROWS = slice(0, 49)


def score_both(xtr, ytr, xdv, ydv, spec, device, order=None):
    """Safe choice and within-root AUC, over three seeds, with BOTH permutation tests.

    refit_permuted   head refitted on permuted TRAIN, evaluated on permuted DEV (phase 4's test)
    frozen_permuted  head fitted on INTACT TRAIN, evaluated on permuted DEV (the stricter test:
                     a head that truly uses the action->branch map must degrade here)
    """
    xtr_s, xdv_s = standardize(xtr, xdv)
    intact, frozen_perm, refit_perm, kept = [], [], [], []
    for seed in SEEDS:
        model, _, params = fit_head(xtr_s, ytr, family="mlp128", objective="rank", seed=seed,
                                    spec=spec, device=device)
        s = scores_of(model, xdv_s, device)
        kept.append(s)
        intact.append(summarize(s, ydv))
        if order is not None:
            frozen_perm.append(summarize(scores_of(model, xdv_s[:, order], device), ydv))
            xp_tr, xp_dv = standardize(xtr[:, order], xdv[:, order])
            m2, _, _ = fit_head(xp_tr, ytr, family="mlp128", objective="rank", seed=seed,
                                spec=spec, device=device)
            refit_perm.append(summarize(scores_of(m2, xp_dv, device), ydv))
    out = {"parameters": params, "dim": int(xdv.shape[-1]),
           "safe": [m["safe_choice"] for m in intact],
           "mean_safe": round(float(np.mean([m["safe_choice"] for m in intact])), 2),
           "auc": [round(m["within_root_auc"], 4) for m in intact],
           "mean_auc": round(float(np.mean([m["within_root_auc"] for m in intact])), 4)}
    if order is not None:
        out["frozen_permuted_mean_safe"] = round(float(np.mean([m["safe_choice"] for m in frozen_perm])), 2)
        out["frozen_permuted_mean_auc"] = round(float(np.mean([m["within_root_auc"] for m in frozen_perm])), 4)
        out["refit_permuted_mean_safe"] = round(float(np.mean([m["safe_choice"] for m in refit_perm])), 2)
        out["frozen_auc_cost"] = round(out["mean_auc"] - out["frozen_permuted_mean_auc"], 4)
    return out, kept


@torch.inference_mode()
def encode_masked(encoder, frames, mask, batch=64):
    """Re-encode ACTUAL masked images. This is the causal test token-subsetting cannot give."""
    out = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].clone()
        if mask == "hud_only":
            chunk[:, MAP_ROWS] = 0
        elif mask == "map_only":
            chunk[:, HUD_ROWS] = 0
        elif mask != "intact":
            raise ValueError(mask)
        z, cls, tokens, b, t = encoder._hidden(chunk.unsqueeze(1).to(encoder.pixel_mean.device))
        out.append(torch.cat((cls.reshape(b, -1).cpu(), tokens.mean(1).reshape(b, -1).cpu()), -1))
    return torch.cat(out)


def stage_masked(side, y, spec, out, device):
    """Does masking HUD pixels actually remove the patch advantage?"""
    inputs = {"test": "masked_reencode_v1"}
    got = cached(out / "masked_reencode.json", inputs)
    if got:
        return got
    bundle, _, _ = load_m03_bundle(
        ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt",
        device=device, dataset_sha256=_sha256(DATASET))
    encoder = bundle.encoder.eval()
    rows = []
    for mask in ("intact", "map_only", "hud_only"):
        feats = {}
        for split in ("train", "dev"):
            succ = side[split]["successors"]
            n, a = succ.shape[:2]
            flat = encode_masked(encoder, succ.reshape(-1, *succ.shape[2:]), mask)
            feats[split] = flat.reshape(n, a, -1)
        width = feats["dev"].shape[-1] // 2
        for name, sl in (("cls", slice(0, width)), ("patch_mean", slice(width, 2 * width))):
            r, _ = score_both(feats["train"][..., sl], y["train"], feats["dev"][..., sl], y["dev"],
                              spec, device)
            rows.append({"mask": mask, "rung": name, **r})
            print(json.dumps({"stage": "masked", "mask": mask, "rung": name,
                              "safe": r["mean_safe"], "auc": r["mean_auc"]}), flush=True)
        del feats
    del bundle, encoder
    torch.cuda.empty_cache()
    payload = {"schema": "d4mj_phase6_masked_v1",
               "what": "re-encodes ACTUAL masked successor images; token subsetting of contextualized "
                       "ViT outputs cannot establish that HUD pixels carry the advantage",
               "masks": {"hud_only": "pixel rows 49-62 kept, 0-48 zeroed",
                         "map_only": "pixel rows 0-48 kept, 49-62 zeroed"}, "rows": rows}
    publish(out / "masked_reencode.json", payload, inputs)
    return payload


def build_u_bundle(device):
    """The matched run's own construction: the frozen encoder wrapped in its PatchPCAEncoder
    shim, with the u->u world loaded into a real ModelBundle.

    My first attempt drove `scan_pairs` from the root latent with no memory and failed parity by
    53.9 -- because the published features come through the gate's `_encode_lewm`, which prefills
    over `lewm_context` frames first. Reusing the gate's own path is the only way the taps are
    the tensors that were scored.
    """
    sys.path.insert(0, str(ROOT / "artifacts/experiments/20260918_matched_10k"))
    from matched_10k import PatchPCAEncoder
    cache = torch.load(ROOT / "artifacts/experiments/20260917_state_transition/evidence/state_cache.pt",
                       map_location="cpu", weights_only=False)
    # matched_10k.py's own ENCODER constant. Note the trap: the `paired_window/raw` SLOT holds
    # the variant=tc, centering=consecutive arm -- the "frozen consecutive-TC encoder". Its hash
    # is asserted against the state cache below rather than trusted from the path.
    encoder_path = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
    if _sha256(encoder_path) != cache["checkpoint_sha256"]:
        raise RuntimeError("u-world encoder does not match the state cache it was built with")
    bundle, _, _ = load_m03_bundle(encoder_path, device=device, dataset_sha256=_sha256(DATASET))
    saved = torch.load(WORLDS / "world_u_u.pt", map_location="cpu", weights_only=False)
    if saved.get("source") != "u" or saved.get("target") != "u":
        raise ValueError("expected the u->u world")
    bundle.world.load_state_dict(saved["state_dict"])
    bundle.world.eval()
    bundle.encoder = PatchPCAEncoder(bundle.encoder, cache["pca"]).to(device).eval()
    return bundle


@torch.inference_mode()
def u_world_taps(bundle, split, settings, batch=16):
    """The gate's own 17-action branch fan through the u->u world, every stage hooked."""
    world = bundle.world
    captured, collected, generated = {}, {}, []

    def grab(name):
        def hook(_m, _i, output):
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
    try:
        for start in range(0, len(context), batch):
            end = min(len(context), start + batch)
            n = end - start
            frame = context[start:end, -settings.lewm_context:].to(device)
            past = actions[start:end, -settings.lewm_context + 1:].to(device)
            u, _ = bundle.encoder.projected_and_cls(frame)
            state = bundle.prefill(u, past)
            branches = bundle.repeat_state(state, bundle.n_actions)
            action = all_actions.repeat(n)[:, None]
            captured.clear()
            predicted, _ = bundle.advance(branches, action)
            generated.append(predicted.latent[:, 0].reshape(n, bundle.n_actions, -1).cpu())
            for name, value in captured.items():
                collected.setdefault(name, []).append(
                    value.reshape(n * bundle.n_actions, -1).reshape(n, bundle.n_actions, -1))
    finally:
        for handle in handles:
            handle.remove()
    taps = {k: torch.cat(v) for k, v in collected.items()}
    taps["generated_u"] = torch.cat(generated)
    return taps


def stage_u_internals(side, y, spec, out, device):
    inputs = {"world": sha(WORLDS / "world_u_u.pt"), "path": "gate_encode_lewm_v2"}
    got = cached(out / "u_world_internals.json", inputs)
    if got:
        return got
    settings = M03Settings()
    bundle = build_u_bundle(device)
    taps = {s: u_world_taps(bundle, side[s], settings) for s in ("train", "dev")}
    published = resolve_payload(torch.load(MATCHED / "features/tc.dev.pt", map_location="cpu",
                                           weights_only=False))["features"]["generated_successor"].float()
    delta = (taps["dev"]["generated_u"] - published).abs().max().item()
    print(json.dumps({"stage": "u_parity", "generated_u_max_abs": delta}), flush=True)
    if delta > 1e-3:
        raise RuntimeError(f"u-world parity failed ({delta}); refusing to report taps that are "
                           "not the tensors the gate scored")

    gen = torch.Generator().manual_seed(909)
    order = derangement(17, gen)
    names = (["pair_projection"] + [f"mamba_block_{i}" for i in range(len(bundle.world.layers))]
             + ["final_norm_h"] + sorted(n for n in taps["dev"] if n.startswith("projector_"))
             + ["generated_u"])
    rows = []
    for name in names:
        if name not in taps["dev"]:
            continue
        r, _ = score_both(taps["train"][name], y["train"], taps["dev"][name], y["dev"],
                          spec, device, order=order)
        rows.append({"tap": name, **r})
        print(json.dumps({"stage": "u_tap", "tap": name, "safe": r["mean_safe"],
                          "auc": r["mean_auc"], "frozen_auc_cost": r["frozen_auc_cost"]}), flush=True)
    feats = {s: resolve_payload(torch.load(MATCHED / f"features/tc.{s}.pt", map_location="cpu",
                                           weights_only=False))["features"] for s in ("train", "dev")}
    real = {s: feats[s]["observed_successor"].float() for s in ("train", "dev")}
    rr, _ = score_both(real["train"], y["train"], real["dev"], y["dev"], spec, device, order=order)
    rows.append({"tap": "REFERENCE_real_u", **rr})
    print(json.dumps({"stage": "u_tap", "tap": "REFERENCE_real_u", "safe": rr["mean_safe"],
                      "auc": rr["mean_auc"], "frozen_auc_cost": rr["frozen_auc_cost"]}), flush=True)
    publish(out / "u_world_internals.json",
            {"schema": "d4mj_phase6_u_internals_v2", "generated_u_parity_max_abs": delta,
             "role": "u->u is a DIAGNOSTIC for Raw/TC, not a canonical architecture",
             "question": "is u consequence information weak throughout the transition, or "
                         "specifically discarded by the final export?", "rows": rows}, inputs)
    del bundle, taps
    torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase6")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stage", default="all")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    if args.stage in ("all", "masked"):
        stage_masked(side, y, spec, args.out, args.device)
    if args.stage in ("all", "u"):
        stage_u_internals(side, y, spec, args.out, args.device)
    print(json.dumps({"status": "phase6_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
