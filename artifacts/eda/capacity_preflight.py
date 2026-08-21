"""VRAM and step-time for candidate bottleneck geometries, Phase-1A settings.

Nothing but `n_latents` and `d_bottleneck` move. Everything else -- encoder/decoder
width and depth, patching, batch, gradient checkpointing, optimizer, masking, LPIPS,
seeds -- is the production Phase-1A configuration, so the numbers are comparable and
the smoke arms fail here rather than four hours into a run.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus

from d4mj.config import Config
from d4mj.data import Episode, EpisodeCorpus, sample_batch
from d4mj.representation import Decoder, Encoder, reconstruction_loss
from d4mj.train import _generators, _to, optimizer

ARMS = [(32, 16), (64, 16), (32, 32), (128, 16), (256, 16)]
STEPS = 12

# S44: in-process trials do not release memory and gave non-monotonic garbage, so
# each geometry is measured in its own process. Re-invoked as `python file.py N D`.
BATCH = None
if len(sys.argv) >= 3:
    ARMS = [(int(sys.argv[1]), int(sys.argv[2]))]
if len(sys.argv) == 4:
    BATCH = int(sys.argv[3])

rows = corpus.train_rows()[:64]
manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
episodes = []
for row in rows:
    frames = corpus.frames_of(row)
    n = len(frames) - 1
    episodes.append(Episode(
        observations=torch.from_numpy(frames.copy()),
        actions_taken=torch.zeros(n, dtype=torch.long), rewards=torch.zeros(n),
        terminated=torch.zeros(n, dtype=torch.bool), truncated=torch.zeros(n, dtype=torch.bool),
        events=torch.zeros(n, dtype=torch.bool)))
pool = EpisodeCorpus(episodes)
print(f"{len(episodes)} episodes held; ARMS={ARMS} BATCH={BATCH}", flush=True)

import lpips

perceptual = lpips.LPIPS(net="alex", verbose=False).to(Config().device).eval()
for parameter in perceptual.parameters():
    parameter.requires_grad_(False)

report = {}
print(f"\n{'arm':<16}{'scalars/frame':>14}{'n_spatial':>11}{'peak VRAM':>12}"
      f"{'s/step short':>14}{'s/step long':>13}")
for n_latents, d_bottleneck in ARMS:
    config = replace(Config(), n_latents=n_latents, d_bottleneck=d_bottleneck,
                     **({"batch": BATCH} if BATCH else {}))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(config.seed)
        encoder, decoder = Encoder(config).to(config.device), Decoder(config).to(config.device)
        opt = optimizer([encoder, decoder], config)
        sampler, rng = _generators(config, 0)
        timings = {}
        for label, long in (("short", False), ("long", True)):
            step_index = 3 if long else 0
            torch.cuda.synchronize(); started = time.time()
            for step in range(STEPS):
                batch = _to(sample_batch(pool, sampler, config, step_index, 0), config.device)
                z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
                predicted, _ = decoder(z)
                losses = reconstruction_loss(predicted, batch.patches, masked,
                                             batch.scored, perceptual, config)
                loss = losses["mse"] + config.lpips_weight * losses["lpips"]
                opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            timings[label] = (time.time() - started) / STEPS
        peak = torch.cuda.max_memory_allocated() / 2**30
        report[f"{n_latents}x{d_bottleneck}"] = {
            "scalars_per_frame": n_latents * d_bottleneck, "n_spatial": config.n_spatial,
            "peak_vram_gib": peak, "s_per_step_short": timings["short"],
            "s_per_step_long": timings["long"],
            "hours_for_3000_steps": (0.75 * timings["short"] + 0.25 * timings["long"]) * 3000 / 3600,
        }
        print(f"{f'{n_latents}x{d_bottleneck}':<16}{n_latents*d_bottleneck:>14}"
              f"{config.n_spatial:>11}{peak:>11.2f}G{timings['short']:>14.3f}"
              f"{timings['long']:>13.3f}")
        del encoder, decoder, opt
    except torch.OutOfMemoryError:
        report[f"{n_latents}x{d_bottleneck}"] = {"peak_vram_gib": None, "oom": True}
        print(f"{f'{n_latents}x{d_bottleneck}':<16}{n_latents*d_bottleneck:>14}"
              f"{config.n_spatial:>11}{'OOM':>12}")
    torch.cuda.empty_cache()

print()
for name, row in report.items():
    if row.get("oom"):
        continue
    print(f"  {name:<10} 3000 Phase-1A steps ~ {row['hours_for_3000_steps']:.2f} h; "
          f"latent cache for 3.11M frames {row['scalars_per_frame']*3_111_438*4/1e9:.1f} GB")
name = (f"capacity_preflight_{ARMS[0][0]}x{ARMS[0][1]}_b{BATCH or Config().batch}.json"
        if len(sys.argv) >= 3 else "capacity_preflight.json")
(HERE / name).write_text(json.dumps(report, indent=2))
