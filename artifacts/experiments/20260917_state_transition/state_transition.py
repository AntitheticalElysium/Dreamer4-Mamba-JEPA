"""Can the world preserve and predict a richer 192-D state, or only the one it was trained on?

The learned-export control showed that a label-free patch summary carries more decodable
successor-state information than the current export, at identical width (binary AUC 0.618 ->
0.717 mlp, 0.619 -> 0.678 linear; continuous R^2 gets *worse*, -0.086 -> -0.177, so it is not
uniformly better state). That was a probe result. It says nothing about whether a world model
can transition that representation.

Four matched one-step arms, encoder frozen throughout, same Mamba size everywhere:

    z -> z   matched retraining baseline
    u -> z   does a richer input help, keeping the old target?
    z -> u   can the old input predict the richer target?
    u -> u   can richer information become a transitionable world state?

`u` is the **fixed, TRAIN-fitted, label-free PCA** of the frozen 4x4 pooled patch grid; its
mean and basis are persisted so every arm and every later re-scoring uses the same map.

The encoder is forwarded **once**: z and u are cached for a fixed window pool, and all four
arms train off those tensors. Nothing here retrains an encoder, touches a gate artifact, or
authorizes anything.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe
from d4mj.data import _sha256, atomic_manifest, load_joint_corpus, JointSampler
from d4mj.m03.gate import load_m03_bundle

CHECKPOINT = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
LATENT = 192


def fit_pca(samples: torch.Tensor, components: int = LATENT) -> dict:
    """Label-free basis over pooled patch vectors, with the null tail dropped.

    The probe standardizes every input dimension with a clamp_min, so a near-zero principal
    direction would be stretched to unit variance and inject noise. Columns beyond the
    retained rank stay exactly zero and survive that standardization as zeros.
    """
    mean = samples.mean(0, keepdim=True)
    centred = (samples - mean).double()
    _, singular, v = torch.linalg.svd(centred, full_matrices=False)
    scale = singular[:components] / max(len(centred) - 1, 1) ** 0.5
    keep = int((scale > scale[0] * 1e-3).sum())
    return {"mean": mean.float(), "basis": v[:components].T[:, :keep].float(),
            "rank": keep, "components": components, "samples": len(samples)}


def apply_pca(pca: dict, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(*x.shape[:-1], pca["components"], device=x.device, dtype=torch.float32)
    out[..., :pca["rank"]] = (x - pca["mean"].to(x.device)) @ pca["basis"].to(x.device)
    return out


@torch.no_grad()
def build_cache(args) -> dict:
    """Forward the frozen encoder once over a fixed window pool; keep z and u only."""
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    dataset_sha = stored["dataset"]["sha256"]
    config = load_recipe(ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json")
    del stored
    bundle, _, _ = load_m03_bundle(CHECKPOINT, device=args.device, dataset_sha256=dataset_sha)
    bundle.encoder.eval()
    episodes, contract = load_joint_corpus(args.dataset, config)
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(args.seed))

    # The consecutive recipe encodes seven frames but rolls out four: the window is the
    # union of the prediction and centering sets. Keep the prediction frames alone, exactly
    # as `joint_loss` does, or the actions will not match the sequence length.
    from d4mj.lewm_config import window_layout
    predicted_at = list(window_layout(config.joint)[1])
    z_rows, u_rows, a_rows, patch_pool = [], [], [], []
    started = time.perf_counter()
    for index in range(args.windows // config.joint.batch):
        batch = sampler.sample()
        frames = batch.frames.to(bundle.device)
        z, _, patch = bundle.encoder.export(frames, grid=4)
        z, patch = z[:, predicted_at], patch[:, predicted_at]
        flat = patch.flatten(2)
        z_rows.append(z[:, :, 0].cpu())
        a_rows.append(batch.actions.cpu())
        if len(patch_pool) * config.joint.batch < args.pca_samples:
            patch_pool.append(flat[:, 0].cpu())
        u_rows.append(flat.cpu())
        if index == 0:
            print(json.dumps({"stage": "first_batch_seconds",
                              "seconds": round(time.perf_counter() - started, 3),
                              "batch": config.joint.batch, "frames": int(frames.shape[1])}), flush=True)
    elapsed = time.perf_counter() - started
    pca = fit_pca(torch.cat(patch_pool)[:args.pca_samples])
    u = torch.cat([apply_pca(pca, chunk) for chunk in u_rows])
    cache = {"z": torch.cat(z_rows), "u": u, "actions": torch.cat(a_rows),
             "pca": {k: v for k, v in pca.items()},
             "dataset_id": contract["sha256"] if "sha256" in contract else None,
             "seed": args.seed, "encode_seconds": elapsed, "prediction_frames": predicted_at,
             "checkpoint_sha256": _sha256(CHECKPOINT)}
    del bundle
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return cache


def train_arm(cache, config, source, target, args, *, device):
    """One matched arm. Same init seed, same batch order, same budget for all four."""
    from d4mj.lewm import LeWMWorld
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device == "cuda" else []):
        torch.manual_seed(args.init_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.init_seed)
        world = LeWMWorld(config).to(device)
    world.train()
    parameters = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.joint.learning_rate,
                                  betas=tuple(config.joint.betas), eps=config.joint.optimizer_eps,
                                  weight_decay=config.joint.weight_decay)
    order = torch.Generator().manual_seed(args.batch_seed)
    n, batch = len(cache["z"]), config.joint.batch
    history = []
    for step in range(args.steps):
        index = torch.randint(n, (batch,), generator=order)
        src = cache[source][index].to(device).unsqueeze(2)
        tgt = cache[target][index].to(device).unsqueeze(2)
        actions = cache["actions"][index].to(device)
        predicted = world.teacher(src, actions).predicted
        loss = (predicted.float() - tgt[:, 1:].float()).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.joint.grad_clip)
        optimizer.step()
        if (step + 1) % max(1, args.steps // 20) == 0:
            history.append({"step": step + 1, "loss": float(loss)})
            print(json.dumps({"stage": "train", "arm": f"{source}->{target}",
                              "step": step + 1, "loss": round(float(loss), 6)}), flush=True)
    return world.eval(), history


@torch.no_grad()
def encode_sidecar(bundle, sidecar, pca, settings, batch=32):
    """Context and observed successors for the M03 roots, in both z and u space."""
    out = {}
    for split in ("train", "dev"):
        rows = sidecar["splits"][split]
        context, successors = rows["context"], rows["successors"]
        cz, cu, sz, su = [], [], [], []
        for start in range(0, len(context), batch):
            end = min(len(context), start + batch)
            frame = context[start:end, -settings.lewm_context:].to(bundle.device)
            z, _, patch = bundle.encoder.export(frame, grid=4)
            cz.append(z[:, :, 0].cpu()); cu.append(apply_pca(pca, patch.flatten(2)).cpu())
            flat = successors[start:end].reshape(-1, 1, *successors.shape[-3:]).to(bundle.device)
            z2, _, patch2 = bundle.encoder.export(flat, grid=4)
            sz.append(z2[:, 0, 0].reshape(end - start, 17, -1).cpu())
            su.append(apply_pca(pca, patch2.flatten(2))[:, 0].reshape(end - start, 17, -1).cpu())
        out[split] = {"context_z": torch.cat(cz), "context_u": torch.cat(cu),
                      "observed_z": torch.cat(sz), "observed_u": torch.cat(su),
                      "past_actions": rows["past_actions"][:, -settings.lewm_context + 1:]}
    return out


@torch.no_grad()
def roll(bundle, encoded, split, source, batch=32):
    """Prefill on the arm's own input space, fan over all actions, keep latent and history."""
    rows = encoded[split]
    context = rows[f"context_{source}"]
    generated, history = [], []
    all_actions = torch.arange(bundle.n_actions, device=bundle.device, dtype=torch.long)
    for start in range(0, len(context), batch):
        end = min(len(context), start + batch)
        z = context[start:end].unsqueeze(2).to(bundle.device)
        past = rows["past_actions"][start:end].to(bundle.device)
        state = bundle.prefill(z, past)
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(end - start)[:, None]
        predicted, _ = bundle.advance(branches, action)
        generated.append(predicted.latent[:, 0].reshape(end - start, 17, -1).cpu())
        history.append(predicted.history[:, 0].reshape(end - start, 17, -1).cpu())
    return torch.cat(generated), torch.cat(history)


def score_arm(encoded, rolled, sidecar, settings, device, target):
    """Native TRAIN-fitted readouts *and* observed-to-generated transfer, both metrics.

    The decoder artifact found earlier makes the distinction essential: a transfer number
    alone cannot separate absent information from a decoder fitted elsewhere.
    """
    from d4mj.m03.gate import (STATIC_BINARY, STATIC_CONTINUOUS, _binary_metrics,
                               _fit_probe_many, _one_hot_actions, _regression_metrics)
    splits = sidecar["splits"]
    tb = {s: splits[s]["next_binary"].flatten(0, 1).float().to(device) for s in ("train", "dev")}
    tc = {s: splits[s]["next_continuous"].flatten(0, 1).float() for s in ("train", "dev")}
    act = {s: _one_hot_actions(len(splits[s]["next_binary"]), device=device) for s in ("train", "dev")}
    roots = splits["dev"]["episode"]
    fork = roots.reshape(-1, 1).expand(len(roots), 17).reshape(-1)

    def pack(t, s):
        return torch.cat((t.flatten(0, 1).float().to(device), act[s]), 1)

    inputs = {
        "generated": {s: pack(rolled[s]["generated"], s) for s in ("train", "dev")},
        "hidden": {s: pack(rolled[s]["history"], s) for s in ("train", "dev")},
        "observed": {s: pack(encoded[s][f"observed_{target}"], s) for s in ("train", "dev")},
    }
    out = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        block = {}
        for name in ("generated", "hidden"):
            native = _fit_probe_many(inputs[name]["train"], tb["train"], {"dev": inputs[name]["dev"]},
                                     settings, hidden=hidden, binary=True)
            native_c = _fit_probe_many(inputs[name]["train"], tc["train"].to(device),
                                       {"dev": inputs[name]["dev"]}, settings, hidden=hidden, binary=False)
            block[f"{name}_native_fit"] = {
                "binary": _binary_metrics(native["dev"], splits["dev"]["next_binary"], roots,
                                          STATIC_BINARY, settings),
                "continuous": _regression_metrics(native_c["dev"], tc["dev"], fork, STATIC_CONTINUOUS, settings)}
        transfer = _fit_probe_many(inputs["observed"]["train"], tb["train"],
                                   {"dev": inputs["generated"]["dev"]}, settings, hidden=hidden, binary=True)
        transfer_c = _fit_probe_many(inputs["observed"]["train"], tc["train"].to(device),
                                     {"dev": inputs["generated"]["dev"]}, settings, hidden=hidden, binary=False)
        block["generated_observed_fit_transfer"] = {
            "binary": _binary_metrics(transfer["dev"], splits["dev"]["next_binary"], roots,
                                      STATIC_BINARY, settings),
            "continuous": _regression_metrics(transfer_c["dev"], tc["dev"], fork, STATIC_CONTINUOUS, settings)}
        action_only = _fit_probe_many(act["train"], tb["train"], {"dev": act["dev"]},
                                      settings, hidden=hidden, binary=True)
        block["action_only"] = {"binary": _binary_metrics(action_only["dev"], splits["dev"]["next_binary"],
                                                          roots, STATIC_BINARY, settings)}
        out[family] = block
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/craftax_support_v2")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", type=int, default=25600)
    parser.add_argument("--pca-samples", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--init-seed", type=int, default=7)
    parser.add_argument("--batch-seed", type=int, default=11)
    parser.add_argument("--cache-only", action="store_true", help="build and benchmark the cache, then stop")
    parser.add_argument("--evaluate", action="store_true", help="score already-trained arms")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        return evaluate(args)
    cache_path = args.out / "state_cache.pt"

    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(json.dumps({"stage": "cache_loaded", "windows": len(cache["z"])}), flush=True)
    else:
        cache = build_cache(args)
        torch.save(cache, cache_path)
        print(json.dumps({"stage": "cache_built", "windows": len(cache["z"]),
                          "encode_seconds": round(cache["encode_seconds"], 1),
                          "pca_rank": cache["pca"]["rank"]}), flush=True)
    if args.cache_only:
        return 0

    config = load_recipe(ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json")
    report = {"schema": "d4mj_state_transition_v1",
              "question": "can the world preserve and predict a richer 192-D state?",
              "scope": "frozen encoder; four matched one-step arms; diagnostic, not a gate run",
              "u": "label-free TRAIN-fitted PCA of the frozen 4x4 pooled patch grid",
              "checkpoint_sha256": cache["checkpoint_sha256"],
              "budget": {"windows": len(cache["z"]), "steps": args.steps,
                         "batch": config.joint.batch, "init_seed": args.init_seed,
                         "batch_seed": args.batch_seed},
              "pca": {k: v for k, v in cache["pca"].items() if k in ("rank", "components", "samples")},
              "arms": {}, "m4_authorized": False}
    for source in ("z", "u"):
        for target in ("z", "u"):
            name = f"{source}->{target}"
            world, history = train_arm(cache, config, source, target, args, device=args.device)
            torch.save({"state_dict": world.state_dict(), "source": source, "target": target},
                       args.out / f"world_{source}_{target}.pt")
            report["arms"][name] = {"history": history, "final_loss": history[-1]["loss"] if history else None}
            del world
            if args.device == "cuda":
                torch.cuda.empty_cache()
    atomic_manifest(args.out / "training.json", report)
    print(json.dumps({"status": "trained", "report": str(args.out / "training.json")}))
    return 0


def evaluate(args) -> int:
    from d4mj.m03.gate import M03Settings
    cache = torch.load(args.out / "state_cache.pt", map_location="cpu", weights_only=False)
    settings = M03Settings()
    run = ROOT / "artifacts/lewm_gates_20260916/m03_window"
    sidecar = torch.load(run / "sidecar/sidecar.probe_only.pt", map_location="cpu", weights_only=False)
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    bundle, _, _ = load_m03_bundle(CHECKPOINT, device=args.device,
                                   dataset_sha256=stored["dataset"]["sha256"])
    del stored
    encoded = encode_sidecar(bundle, sidecar, cache["pca"], settings)
    print(json.dumps({"stage": "sidecar_encoded"}), flush=True)
    report = {"schema": "d4mj_state_transition_eval_v1",
              "scope": "frozen encoder; four matched one-step arms scored on the M03 roots",
              "u": "label-free TRAIN-fitted PCA of the frozen 4x4 pooled patch grid",
              "pca": {k: cache["pca"][k] for k in ("rank", "components", "samples")},
              "checkpoint_sha256": cache["checkpoint_sha256"],
              "sidecar_sha256": _sha256(run / "sidecar/sidecar.probe_only.pt"),
              "script_sha256": _sha256(Path(__file__)), "arms": {}, "m4_authorized": False}
    for source in ("z", "u"):
        for target in ("z", "u"):
            path = args.out / f"world_{source}_{target}.pt"
            if not path.exists():
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            bundle.world.load_state_dict(saved["state_dict"])
            bundle.world.eval()
            rolled = {}
            for split in ("train", "dev"):
                generated, history = roll(bundle, encoded, split, source)
                rolled[split] = {"generated": generated, "history": history}
            report["arms"][f"{source}->{target}"] = {
                "world_sha256": _sha256(path),
                "scores": score_arm(encoded, rolled, sidecar, settings, args.device, target)}
            print(json.dumps({"stage": "scored", "arm": f"{source}->{target}"}), flush=True)
    atomic_manifest(args.out / "evaluation.json", report)
    print(json.dumps({"status": "complete", "report": str(args.out / "evaluation.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
