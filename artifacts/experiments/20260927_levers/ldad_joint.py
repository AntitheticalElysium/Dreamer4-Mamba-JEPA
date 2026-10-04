"""E4. Delta-JEPA's Latent Difference Action Decoder (LDAD) added to canonical joint training, paired with the
canonical Raw / TC runs.

Delta-JEPA (arXiv 2606.31232, read in full): an inverse-dynamics decoder D reconstructs the executed action from the
ENCODER's latent displacement dz_t = z_{t+1} - z_t, trained end to end: L = L_pred + lambda L_action (their
lambda = 10 main; best 50 on Push-T; lambda = 0 nearly collapses because they have no SIGReg). They replace
distribution-matching regularizers; here LDAD is ADDED to the canonical objective (next-latent MSE + 0.09 SIGReg,
raw or temporally centered), the only change. Adaptations, stated: Craftax actions are discrete (17), so
L_action is cross-entropy, not MSE; the decoder is single-step (their eq. 3-5) as a 2-layer MLP 192 -> 256 -> 17,
not their 5-query Transformer multi-step extension; lambda = 10.
--no-sigreg is Delta-JEPA as published: L = L_pred + lambda L_action, SIGReg still computed (logged; its projection
RNG consumed, so batches stay paired) but out of the objective. Their N = 5 action queries decode the 5 actions of
one latent step (le-wm frameskip 5 in all four of their environments, config/train/data/*.yaml); Craftax has one
action per step, so single-step decoding is the faithful analogue.

Pairing: the canonical loop itself (`_joint_components`, `JointSampler`, `joint_loss`, `learning_rate`,
`optimizer_step`) with the canonical run's own config. ModelBundle.create seeds from config.seed, the sampler from
seed + 1 and SIGReg's projections from seed + 1001, so batches, projections and initial weights equal the canonical
run's; the initial-weight identity is checked against the canonical checkpoint's recorded `initial_identity`.
The LDAD head is created afterwards (seed + 7777) and added as a parameter group with the joint weight decay.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/eda/levers_ldad_v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("raw", "tc"))
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--no-sigreg", action="store_true", help="Delta-JEPA as published: no SIGReg in the objective")
    args = parser.parse_args(argv)
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.data import load_joint_corpus
    from d4mj.lewm import SIGReg, joint_loss
    from d4mj.train import _joint_components, autocast_context, learning_rate, optimizer_step
    os.chdir(ROOT)
    run = ROOT / f"artifacts/lewm_m4_canonical/{args.variant}"
    canonical = read_lewm_bundle(run / "joint/step-010000.pt")
    config = config_from_dict(canonical["config"])
    record = json.loads((run / "dataset.json").read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("corpus differs from the canonical run's dataset contract")
    bundle, opt, sampler, projection_rng, identity = _joint_components(episodes, config)
    if identity != canonical["initial_identity"]:
        raise SystemExit("initial weights differ from the canonical run: not a paired arm")
    torch.manual_seed(config.seed + 7777)
    head = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, config.dynamics.n_actions)).to(config.runtime.device)
    opt.add_param_group({"params": list(head.parameters()), "weight_decay": config.joint.weight_decay})
    params = [p for g in opt.param_groups for p in g["params"]]
    regularizer = SIGReg(config.joint.knots, config.joint.projections).to(config.runtime.device)
    # A run shorter than the schedule is a SCREEN: the canonical recipe's first `steps` updates (the cosine schedule
    # keeps config.joint.steps), paired with the canonical run's own step checkpoint.
    name = f"{args.variant}_lam{args.lam:g}" + ("_nosig" if args.no_sigreg else "") + ("" if args.steps == config.joint.steps else f"_screen{args.steps}")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "metrics.jsonl"
    if log.exists():
        raise SystemExit("output exists; this script does not resume")
    started = time.time()
    for update in range(args.steps):
        batch = sampler.sample().to(config.runtime.device)
        with autocast_context(config):
            loss = joint_loss(bundle.encoder, bundle.world, batch.frames, batch.actions, regularizer, projection_rng, config)
            z = loss.latent[:, :, 0].float()
            logits = head(z[:, 1:] - z[:, :-1]).float()
            ce = F.cross_entropy(logits.flatten(0, 1), batch.actions.flatten())
            total = (loss.prediction if args.no_sigreg else loss.total) + args.lam * ce
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"nonfinite objective at update {update}")
        norm = optimizer_step(opt, total, params, learning_rate=learning_rate(config, update),
                              grad_clip=config.joint.grad_clip, strict=True, zero_grad=True)
        row = {"update": update + 1, "prediction": float(loss.prediction.detach()), "regularization": float(loss.regularization.detach()),
               "ldad_ce": float(ce), "ldad_acc": float((logits.argmax(-1) == batch.actions).float().mean()),
               "gradient_norm": float(norm), "seconds": round(time.time() - started, 1)}
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if (update + 1) % 500 == 0:
            print(json.dumps(row), flush=True)
        if (update + 1) in (2000, args.steps):
            torch.save({"config": canonical["config"], "variant": args.variant, "lam": args.lam, "no_sigreg": args.no_sigreg, "step": update + 1,
                        "modules": {"encoder": bundle.encoder.state_dict(), "world": bundle.world.state_dict(),
                                    "ldad": head.state_dict()}}, out / f"step-{update + 1:06d}.pt")


if __name__ == "__main__":
    main()
