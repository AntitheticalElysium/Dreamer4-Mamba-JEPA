"""Stage 2 control (SPATIAL_ENCODER_PREREG.md): the SAME patch predictor trained on the FROZEN Raw encoder's patch states.

Patch state: LN(last-layer patch tokens) of artifacts/lewm_m4_canonical/raw/joint/step-010000.pt (what D / IC / PI use). Predictor:
spatial_encoder.PatchPredictor (K 4 + router), L_patch = mean over patches of min_k MSE + 0.1 router CE, the 2 transitions of each
4-frame window with a previous frame. Same corpus, d4mj JointSampler, batch 128, AdamW (d4mj grouping), Raw's 10,000-step
schedule, clip 1.0, bf16. Only the predictor trains.
Frozen-state control (FROZEN_STATE_CONTROL_PREREG.md): --encoder spatial:<ckpt> --stage deployed|backbone_ln trains the same
predictor on a frozen patch-trained encoder's deployed (projector, eval-mode BN) or backbone_ln tokens. Defaults = the Raw-predictor.
Usage: raw_predictor.py --out DIR [--stop-at N] [--seed S] [--encoder SPEC] [--stage deployed|backbone_ln]
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_encoder as SE                                                   # noqa: E402  (sets ROOT, sys.path, cwd)
from d4mj.config import load_recipe                                           # noqa: E402
from d4mj.data import load_joint_corpus, JointSampler                         # noqa: E402
from d4mj.train import optimizer, learning_rate, autocast_context             # noqa: E402
from spatial_eval import PatchState                                           # noqa: E402

RAW = 'raw@10000'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--stop-at', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260731)
    ap.add_argument('--encoder', default=RAW); ap.add_argument('--stage', choices=('deployed', 'backbone_ln'), default='deployed')
    a = ap.parse_args()
    a.out = (SE.CALLER_CWD / a.out).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    base = load_recipe(SE.RECIPE)
    config = dataclasses.replace(base, seed=a.seed)
    (a.out / 'arm.json').write_text(json.dumps({'arm': 'raw_predictor' if a.encoder == RAW else 'frozen_state', 'encoder': a.encoder,
                                                'stage': a.stage, 'K': 4, 'seed': a.seed, 'batch': config.joint.batch}, indent=1))
    t0 = time.time()
    episodes, contract = load_joint_corpus(SE.CORPUS, config)
    print(json.dumps({'corpus_loaded_s': round(time.time() - t0, 1), 'dataset_sha256': contract['sha256']}), flush=True)
    enc = PatchState(a.encoder)
    if a.stage == 'backbone_ln':
        enc.proj = None                                                            # PatchState then returns LN(last-layer patch tokens)
    for p in list(enc.enc.parameters()) + (list(enc.proj.parameters()) if enc.proj is not None else []):
        p.requires_grad_(False)
    torch.manual_seed(a.seed)
    predictor = SE.PatchPredictor(4).cuda()
    opt = optimizer([predictor], config.joint, exclude_vectors=True)
    params = [p for g in opt.param_groups for p in g['params']]
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    log = (a.out / 'metrics.jsonl').open('a')
    for update in range(a.stop_at):
        batch = sampler.sample()
        B, T = batch.frames.shape[:2]
        with torch.no_grad():
            zp = enc(batch.frames.flatten(0, 1)).view(B, T, SE.P, SE.D)
        acts = batch.actions.cuda()
        with autocast_context(config):
            cand, router = predictor(zp[:, 0:2].flatten(0, 1), zp[:, 1:3].flatten(0, 1), acts[:, 1:3].flatten(0, 1))
        tgt = zp[:, 2:4].flatten(0, 1)
        err = (cand.float() - tgt[:, :, None]).square().mean(-1)
        best = err.argmin(-1)
        patch_pred = err.gather(-1, best[..., None]).mean()
        router_ce = F.cross_entropy(router.float().flatten(0, 1), best.flatten())
        loss = patch_pred + 0.1 * router_ce
        lr = learning_rate(config, update)
        for gp in opt.param_groups:
            gp['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, config.joint.grad_clip, error_if_nonfinite=True)
        opt.step()
        if (update + 1) % 50 == 0 or update == 0:
            row = {'update': update + 1, 'patch_pred': round(float(patch_pred), 5), 'router_ce': round(float(router_ce), 4),
                   'grad_norm': round(float(norm), 4), 'lr': lr, 'elapsed_s': round(time.time() - t0, 1),
                   'gpu_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.write(json.dumps(row) + '\n'); log.flush()
            if (update + 1) % 500 == 0 or update == 0:
                print(json.dumps(row), flush=True)
        if (update + 1) % 1000 == 0 or update + 1 == a.stop_at:
            torch.save({'update': update + 1, 'encoder_spec': a.encoder, 'stage': a.stage, 'predictor': predictor.state_dict(), 'optimizer': opt.state_dict()},
                       a.out / f'step-{update + 1:06d}.pt')


if __name__ == '__main__':
    main()
