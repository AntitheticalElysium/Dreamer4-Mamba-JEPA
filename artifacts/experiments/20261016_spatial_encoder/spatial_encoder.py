"""Spatial JEPA encoder training (SPATIAL_ENCODER_PREREG.md governs; 2026-10-16).

The encoder's objective is Raw's (projector(CLS), next-z MSE + SIGReg 0.09, 6-layer Mamba world; d4mj.lewm.joint_loss logic,
reproduced line for line) PLUS a patch path:
  patch state   zp = LeWMProjector(last-layer patch tokens), per patch (BatchNorm over every patch of the batch)
  predictor     4-layer pre-LN transformer, width 192, over [zp_{t-1} (81) + frame / position embeddings, zp_t (81), action];
                from the zp_t slots: K candidates for zp_{t+1} per patch + router logits; transitions t = 1, 2 of each 4-frame window
  L_patch       mean over patches of min_k mean-squared error to zp_{t+1} (no stop-gradient, LeWM's choice) + 0.1 router CE
  anti-collapse on patches (all frames): SIGReg (512 random patches per frame index, weight 0.09 x 128 / 512) or VICReg variance hinge at 1 (weight 1)
                + covariance (weight 0.04)
  L_inv (full)  each frame also encoded under a random per-pixel colour map on the map rows (+ Gaussian pixel noise, p 0.5,
                sigma ~ U(0, 0.05)); MSE between augmented and clean map-patch states (weight 1)
  L_idm (full)  17-way CE of the action from [zp_t, zp_{t+1}] (per-patch linear 384 -> 64, mean over patches, MLP), weight 0.1
Data / schedule: Raw's corpus (craftax_expert_store_v1 + craftax_support_v2, train split, d4mj JointSampler, 4-frame windows),
batch 128 (Raw's), AdamW (d4mj grouping, vectors exempt), Raw's 10,000-step warmup + cosine schedule (lr 5e-5 -> 5e-6), clip 1.0, bf16.
Usage: spatial_encoder.py <arm> --out DIR [--stop-at N] [--seed S]
  arms: naive (K 1, SIGReg) | mop_sig (K 4, SIGReg) | mop_vic (K 4, VICReg) | full_sig | full_vic (+ L_inv + L_idm)
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
CALLER_CWD = Path.cwd()                                                     # --out is resolved against the caller's directory
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from d4mj.config import load_recipe                                     # noqa: E402
from d4mj.data import load_joint_corpus, JointSampler                   # noqa: E402
from d4mj.world_api import ModelBundle                                  # noqa: E402
from d4mj.lewm import SIGReg, LeWMProjector                             # noqa: E402
from d4mj.train import set_phase_mode, optimizer, learning_rate, autocast_context  # noqa: E402

RECIPE = ROOT / 'd4mj/recipes/lewm_mamba_raw.json'
CORPUS = [ROOT / 'artifacts/craftax_expert_store_v1', ROOT / 'artifacts/craftax_support_v2']
ARMS = {'naive': dict(K=1, ac='sigreg', inv=False, idm=False), 'mop_sig': dict(K=4, ac='sigreg', inv=False, idm=False),
        'mop_vic': dict(K=4, ac='vicreg', inv=False, idm=False), 'full_sig': dict(K=4, ac='sigreg', inv=True, idm=True),
        'full_vic': dict(K=4, ac='vicreg', inv=True, idm=True)}
SIG_N = 512
BATCH, MAP_PX, D, P = 128, 49, 192, 81                                   # Raw's actual batch (the recipe validator requires it)
LUM = torch.tensor([0.299, 0.587, 0.114])


class PatchPredictor(nn.Module):
    def __init__(self, K, layers=4, n_actions=17):
        super().__init__()
        self.K = K
        self.pos = nn.Parameter(torch.zeros(P, D)); self.frame = nn.Parameter(torch.zeros(2, D))
        self.act = nn.Embedding(n_actions, D)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, K * D); self.router = nn.Linear(D, K)
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.frame, std=0.02)

    def forward(self, prev, cur, a):
        """prev, cur [N,81,D], a [N] -> candidates [N,81,K,D], router logits [N,81,K]"""
        x = torch.cat([prev + self.pos + self.frame[0], cur + self.pos + self.frame[1], self.act(a)[:, None]], 1)
        h = self.norm(self.body(x)[:, P:2 * P])
        return self.out(h).view(*h.shape[:2], self.K, D), self.router(h)


class IDMHead(nn.Module):
    def __init__(self, n_actions=17):
        super().__init__()
        self.local = nn.Linear(2 * D, 64)
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, n_actions))

    def forward(self, a, b):
        return self.mlp(self.local(torch.cat([a, b], -1)).mean(-2))


def vicreg(x):
    """x [N,D] -> variance hinge (target std 1) + 0.04 x covariance, VICReg (2105.04906) eqs 1-4"""
    x = x.float()
    std = torch.sqrt(x.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    xc = x - x.mean(0)
    cov = (xc.T @ xc) / (len(x) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + 0.04 * off.pow(2).sum() / x.shape[1], float(var), float(off.pow(2).sum() / x.shape[1])


def augment(frames, g):
    """uint8 [N,63,63,3] -> uint8 with the map rows under a random per-pixel colour map (R49 family) and, p 0.5, pixel noise"""
    n = len(frames)
    s = torch.rand(n, generator=g) * 1.2
    s = torch.where(torch.rand(n, generator=g) < 0.3, torch.zeros_like(s), s)
    b = 0.4 + 0.9 * torch.rand(n, generator=g)
    t = -0.1 + 0.25 * torch.rand(n, 3, generator=g)
    L = LUM[None, :].expand(3, 3)
    M = b[:, None, None] * (s[:, None, None] * torch.eye(3) + (1 - s)[:, None, None] * L)
    sigma = torch.where(torch.rand(n, generator=g) < 0.5, 0.05 * torch.rand(n, generator=g), torch.zeros(n))
    noise = torch.randn(n, MAP_PX, 63, 3, generator=g) * sigma[:, None, None, None]
    M, t, noise = M.to(frames.device), t.to(frames.device), noise.to(frames.device)
    x = frames.float() / 255.0
    m = torch.einsum('nhwc,ndc->nhwd', x[:, :MAP_PX], M) + t[:, None, None, :] + noise
    out = frames.clone()
    out[:, :MAP_PX] = (m * 255).round().clamp(0, 255).to(torch.uint8)
    return out


def effective_rank(x):
    s = torch.linalg.svdvals(x.float() - x.float().mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=sorted(ARMS)); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stop-at', type=int, default=10000); ap.add_argument('--seed', type=int, default=20260731)
    a = ap.parse_args()
    a.out = (CALLER_CWD / a.out).resolve()
    arm = ARMS[a.arm]
    base = load_recipe(RECIPE)
    config = dataclasses.replace(base, seed=a.seed)
    assert config.joint.batch == BATCH
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'arm.json').write_text(json.dumps({'arm': a.arm, **arm, 'seed': a.seed, 'batch': BATCH, 'recipe': str(RECIPE)}, indent=1))
    t0 = time.time()
    episodes, contract = load_joint_corpus(CORPUS, config)
    print(json.dumps({'corpus_loaded_s': round(time.time() - t0, 1), 'dataset_sha256': contract['sha256']}), flush=True)
    torch.manual_seed(a.seed)
    bundle = ModelBundle.create(config)
    set_phase_mode(bundle, 'joint')
    dev = config.runtime.device
    patch_proj = LeWMProjector(D, config.encoder).to(dev)
    predictor = PatchPredictor(arm['K']).to(dev)
    idm = IDMHead().to(dev)
    mods = [bundle.encoder, bundle.world, patch_proj, predictor, idm]
    opt = optimizer(mods, config.joint, exclude_vectors=True)
    params = [p for g in opt.param_groups for p in g['params']]
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    proj_rng = torch.Generator(device=dev).manual_seed(config.seed + 1001)
    patch_rng = torch.Generator(device=dev).manual_seed(config.seed + 2001)
    aug_g = torch.Generator().manual_seed(config.seed + 3001)
    sig_cls = SIGReg(config.joint.knots, config.joint.projections).to(dev)
    sig_patch = SIGReg(config.joint.knots, config.joint.projections).to(dev)
    log = (a.out / 'metrics.jsonl').open('a')
    for update in range(a.stop_at):
        batch = sampler.sample().to(dev)
        frames, acts = batch.frames, batch.actions                                 # [B,4,63,63,3] uint8, [B,3]
        B, T = frames.shape[:2]
        with autocast_context(config):
            enc_in = frames
            if arm['inv']:
                aug = augment(frames.flatten(0, 1), aug_g).view_as(frames)
                enc_in = torch.cat([frames, aug], 0)
            zc, _, tokens, _, _ = bundle.encoder._hidden(enc_in)                    # [n*T, D], [n*T, 81, D]
            zp_all = patch_proj(tokens.reshape(-1, D)).view(-1, T, P, D)
            zc = zc.view(-1, T, 1, D)
            if arm['inv']:
                zp, zp_aug, z = zp_all[:B], zp_all[B:], zc[:B]
            else:
                zp, zp_aug, z = zp_all, None, zc
            # CLS path: d4mj.lewm.joint_loss, raw variant, consecutive 4-frame layout
            pred_cls = bundle.world.teacher(z, acts).predicted
            # patch path
            prev = zp[:, 0:2].flatten(0, 1); cur = zp[:, 1:3].flatten(0, 1); tgt = zp[:, 2:4].flatten(0, 1)
            cand, router = predictor(prev, cur, acts[:, 1:3].flatten(0, 1))
        with torch.autocast(device_type='cuda', enabled=False):
            cls_pred = (pred_cls.float() - z[:, 1:].float()).square().mean()
            cls_reg = sig_cls(z[:, :, 0].float().transpose(0, 1), proj_rng)
            err = (cand.float() - tgt.float()[:, :, None]).square().mean(-1)        # [N,81,K]
            best = err.argmin(-1)
            patch_pred = err.gather(-1, best[..., None]).mean()
            router_ce = F.cross_entropy(router.float().flatten(0, 1), best.flatten())
            if arm['ac'] == 'sigreg':
                # amendment (pre-result): N = SIG_N random patches per frame index (memory); the Epps-Pulley statistic is
                # multiplied by N (d4mj/lewm.py:132), so the weight is 0.09 * 128 / N, LeWM's per-discrepancy strength at N = 128
                flat = zp.float().permute(1, 0, 2, 3).reshape(T, B * P, D)
                pick = torch.randperm(B * P, device=dev, generator=patch_rng)[:SIG_N]
                ac = sig_patch(flat[:, pick], patch_rng)
                ac_w, ac_parts = 0.09 * 128 / SIG_N, {}
            else:
                ac, v_, c_ = vicreg(zp.reshape(-1, D))
                ac_w, ac_parts = 1.0, {'vic_var': round(v_, 5), 'vic_cov': round(c_, 5)}
            loss = cls_pred + 0.09 * cls_reg + patch_pred + 0.1 * router_ce + ac_w * ac
            row_extra = {}
            if arm['inv']:
                inv = (zp_aug[:, :, :63].float() - zp[:, :, :63].float()).square().mean()
                loss = loss + inv; row_extra['inv'] = round(float(inv), 5)
            if arm['idm']:
                logits = idm(zp[:, :-1].float(), zp[:, 1:].float())                  # [B,3,17]
                ce = F.cross_entropy(logits.flatten(0, 1), acts.flatten())
                loss = loss + 0.1 * ce
                row_extra['idm_ce'] = round(float(ce), 4); row_extra['idm_acc'] = round(float((logits.argmax(-1) == acts).float().mean()), 4)
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss at update {update}')
        lr = learning_rate(config, update)
        for gp in opt.param_groups:
            gp['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, config.joint.grad_clip, error_if_nonfinite=True)
        opt.step()
        if (update + 1) % 50 == 0 or update == 0:
            with torch.no_grad():
                sub = zp[:8].reshape(-1, D)
                row = {'update': update + 1, 'loss': round(float(loss), 5), 'cls_pred': round(float(cls_pred), 5),
                       'cls_reg': round(float(cls_reg), 4), 'patch_pred': round(float(patch_pred), 5),
                       'router_ce': round(float(router_ce), 4), 'ac': round(float(ac), 5), **ac_parts, **row_extra,
                       'patch_eff_rank': round(effective_rank(sub), 2), 'patch_std': round(float(zp.float().std(0).mean()), 4),
                       'grad_norm': round(float(norm), 4), 'lr': lr, 'elapsed_s': round(time.time() - t0, 1),
                       'gpu_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.write(json.dumps(row) + '\n'); log.flush()
            if (update + 1) % 250 == 0 or update == 0:
                print(json.dumps(row), flush=True)
        if (update + 1) % 500 == 0 or update + 1 == a.stop_at:
            torch.save({'update': update + 1, 'arm': a.arm, 'seed': a.seed, 'encoder': bundle.encoder.state_dict(),
                        'world': bundle.world.state_dict(), 'patch_proj': patch_proj.state_dict(),
                        'predictor': predictor.state_dict(), 'idm': idm.state_dict(), 'optimizer': opt.state_dict()},
                       a.out / f'step-{update + 1:06d}.pt')


if __name__ == '__main__':
    main()
