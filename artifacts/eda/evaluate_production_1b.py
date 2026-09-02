"""Do the repairs survive the production objective and the generated path?

The diagnostic fork objective never exercised the two-step generated-prefix rollout
that production `_direct_loss` trains, so every result so far is about a path Direct
does not actually take during generation. This scores the production-trained world on
four things:

  teacher_forced    one-step MSE with the true prefix, the easy path
  generated_first   first `advance` step from a generated prefix
  generated_second  second step, where errors compound -- the path imagination uses
  fork              the held-out all-17 metrics, unchanged from the diagnostic work

The fork set is directly comparable: it was encoded with the same 64x16 tokenizer the
production run uses, so no re-encoding is involved. The comparison against abm0 is
deliberately unfair in abm0's favour -- abm0 was trained *on* those roots while the
production world never saw them -- so abm0's numbers are an upper bound and the
question is how much of the repair transfers, not whether production wins.

Read-only. No training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts"))
sys.path.insert(0, str(HERE))
from run_stage_a import corpus

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.data import sample_batch
from d4mj.train import _to, cache_latents
from d4mj.transition import World, WorldState, advance, commit_inputs

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


@torch.no_grad()
def rollout_errors(world, episodes, config, batches=64, seed=7):
    """Teacher-forced and two-step generated-prefix error, exactly as _direct_loss."""
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 99)
    sampler = torch.Generator().manual_seed(seed)   # draw tables live on CPU
    teacher, first_step, second_step = [], [], []
    for index in range(batches):
        batch = _to(sample_batch(episodes, sampler, config, index, batches), DEVICE)
        committed, conditioning = commit_inputs(batch.latents, rng, config)
        features, _, memory = world(None, batch.led_to_action, committed, conditioning)
        predicted = world.predict(features[:, :-1], batch.led_to_action[:, 1:])
        teacher.append(float((predicted - batch.latents[:, 1:]).pow(2).mean()))

        length = batch.latents.shape[1]
        if length < 3:
            continue
        prefix, _, memory = world(None, batch.led_to_action[:, :-2],
                                  committed[:, :-2], conditioning[:, :-2])
        state = WorldState(batch.latents[:, -3:-2], memory, length - 2, prefix[:, -1:])
        one, _ = advance(world, state, batch.led_to_action[:, -2:-1], rng, config)
        two, _ = advance(world, one, batch.led_to_action[:, -1:], rng, config)
        first_step.append(float((one.latent - batch.latents[:, -2:-1]).pow(2).mean()))
        second_step.append(float((two.latent - batch.latents[:, -1:]).pow(2).mean()))
    return {"teacher_forced": float(np.mean(teacher)),
            "generated_first": float(np.mean(first_step)),
            "generated_second": float(np.mean(second_step)),
            "compounding": float(np.mean(second_step) / np.mean(first_step))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=HERE / "production_1b" / "world.pt")
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--tag", default="production",
                        help="names the output file, so scoring an arm never overwrites "
                             "another arm's report")
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    # the arm records the mixer it trained with; hardcoding attention here silently
    # fails to load a mamba checkpoint, and the T-versus-M comparison cannot be run at all
    mixer = "attention"
    report_path = args.world.parent / "training_report.json"
    if report_path is not None and report_path.exists():
        mixer = json.loads(report_path.read_text()).get("time_mixer", "attention")
    config = replace(base, transition="direct", time_mixer=mixer)
    world = World(config).to(DEVICE)
    world.load_state_dict(torch.load(args.world, weights_only=False)["world"])
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)

    result = {"world": str(args.world)}

    # ---- production path, on held-out DEV episodes
    os.chdir(ROOT)
    _, dev_set = corpus(base, args.expert, print, support=ROOT / "artifacts/craftax_support_v2")
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()
    started = time.time()
    cached_dev = cache_latents(encoder, dev_set, base)
    encoder.cpu()
    torch.cuda.empty_cache()
    print(f"dev cached in {time.time()-started:.0f}s", flush=True)
    result["dev"] = rollout_errors(world, cached_dev, config, batches=args.batches)

    # the fork-trained diagnostic arm on the same batches: it never saw the generated
    # path during training, so the gap is what production training buys on its own path
    sys.path.insert(0, str(HERE))
    from legacy import open_checkpoint
    # the abm0 reference is an attention checkpoint whatever the arm under test is, so
    # it needs its own config -- reusing the arm's mamba one fails to load it
    attention = replace(config, time_mixer="attention")
    reference = open_checkpoint(HERE / "phase1b_abm0_n64" / "world_020000.pt",
                                attention, "promoted")
    result["dev_fork_trained"] = rollout_errors(reference, cached_dev, attention,
                                                batches=args.batches)
    del reference
    torch.cuda.empty_cache()

    print(f"\nDEV, production path       production-trained   fork-trained")
    for key in ("teacher_forced", "generated_first", "generated_second"):
        print(f"  {key:<24}{result['dev'][key]:14.6f}{result['dev_fork_trained'][key]:15.6f}")
    print(f"  {'compounding':<24}{result['dev']['compounding']:13.2f}x"
          f"{result['dev_fork_trained']['compounding']:14.2f}x", flush=True)

    # ---- held-out all-17 forks, same tokenizer, never seen by this world
    os.chdir(HERE)
    from evaluate_coverage_ab import measures
    from evaluate_phase1b_fork import predict_branches
    from reevaluate_phase1b_delta import fit_probe, within_state
    from train_phase1b_fork import fork_actions, load_forkset, seed_split

    rows = load_forkset(HERE / "forkset_s1_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    test = torch.from_numpy(splits == "test")
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1].float()
    led = fork_actions(rows)

    pred = predict_branches(world, config, history[test], led[test]).float()
    m = measures(pred, branch[test])
    # per-root values travel, not only their means: every arm-minus-control difference
    # needs a paired interval, and a scalar cannot be bootstrapped against another arm
    result["fork"] = {"action_mse": float(m["action_mse"].mean()),
                      "nse": float(m["nse"].mean()), "cosine": float(m["cosine"].mean()),
                      "retrieval": m["retrieval"], "geometry": m["geometry"],
                      "per_root": {k: m[k].tolist() for k in ("action_mse", "nse", "cosine")}}

    width = branch.shape[-1]
    delta_true = branch - root[:, None]
    fm, tm = torch.from_numpy(splits == "fit"), torch.from_numpy(splits == "tune")
    probe, _ = fit_probe(delta_true[fm].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "fit"].reshape(-1)).float().to(DEVICE),
                         delta_true[tm].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "tune"].reshape(-1)).float().to(DEVICE),
                         seed=11)

    def read(mix):
        with torch.no_grad():
            s = torch.cat([probe(mix[lo:lo+128].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(mix), 128)]).numpy().reshape(-1, 17)
        return within_state(s, labels[test.numpy()])

    v_true, v_pred = read(delta_true[test]), read(pred - root[test][:, None])
    auc_true, auc_pred = float(v_true.mean()), float(v_pred.mean())
    result["fork"]["R_delta"] = (auc_pred - 0.5) / (auc_true - 0.5)
    result["fork"]["per_root"]["damage_true"] = v_true.tolist()
    result["fork"]["per_root"]["damage_pred"] = v_pred.tolist()
    f = result["fork"]
    print(f"\nheld-out all-17 forks (never seen by this world)")
    print(f"  action MSE {f['action_mse']:.5f}  NSE {f['nse']:.4f}  cosine {f['cosine']:.4f}")
    print(f"  retrieval {f['retrieval']:.4f}  geometry {f['geometry']:.4f}  "
          f"R_delta {f['R_delta']:.3f}")
    print(f"\n  reference, abm0 trained ON these roots: action MSE 0.01414  NSE 0.2858  "
          f"cosine 0.8768  retrieval 0.8936  geometry 0.9901  R_delta 0.364", flush=True)

    (HERE / f"production_1b_evaluation_{args.tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
