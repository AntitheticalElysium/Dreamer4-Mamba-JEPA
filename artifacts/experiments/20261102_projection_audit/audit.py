"""Authorized frozen Test 2; PLAN.md. No native-model parameter updates."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from saccade import model as M, train as TR

F1 = importlib.import_module('artifacts.experiments.20261102_foveal_prediction.predict')
Q = importlib.import_module('artifacts.experiments.20261102_direct_chooser.direct')
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '20261102_foveal_prediction'
CK_SHA = '95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128'
MODES = ('base', 'fine', 'native', 'retained', 'null', 'random_retained', 'mixed', 'recovered_bf16')


def load(split):
    current = (HERE.parent / '20261102_foveal_information' / 'train_400.pt') if split == 'train' else OLD / 'dev_400.pt'
    target = OLD / f'{split}_400_future.pt'
    z = torch.load(current, map_location='cpu', weights_only=False)
    t = torch.load(target, map_location='cpu', weights_only=False)
    assert t['checkpoint_sha256'] == CK_SHA
    return z, t


def linear_maps(m):
    w = m.belief.percept_in.weight.detach().double().cpu()
    centres = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))
    cells = M.window(centres)
    wg = w[:, 1296:6480].reshape(256, 81, 64)
    a = torch.stack([wg[:, c].flatten(1) for c in cells])
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    pinv = vh.transpose(-1, -2) @ (u.transpose(-1, -2) / s[..., None])
    project = pinv @ a
    ident = torch.eye(256, dtype=torch.float64)
    generator = torch.Generator().manual_seed(611804)
    q, _ = torch.linalg.qr(torch.randn(576, 256, dtype=torch.float64, generator=generator))
    g = torch.randn(49, 576, dtype=torch.float64, generator=generator)
    native = torch.einsum('bij,bj->bi', a, g)
    retained = torch.einsum('bij,bj->bi', pinv, native)
    null = g - retained
    checks = dict(rank_per_fixation=(s > s[:, :1] * 1e-12).sum(-1).tolist(),
        condition_per_fixation=(s[:, 0] / s[:, -1]).tolist(),
        singular_min=float(s.min()), singular_max=float(s.max()),
        right_inverse_max=float((a @ pinv - ident).abs().max()),
        symmetry_max=float((project - project.transpose(-1, -2)).abs().max()),
        idempotence_max=float((project @ project - project).abs().max()),
        null_projection_max=float(torch.einsum('bij,bj->bi', a, null).abs().max()),
        retained_projection_max=float((torch.einsum('bij,bj->bi', a, retained) - native).abs().max()),
        retained_orthogonal_null_max=float((retained * null).sum(-1).abs().max()))
    assert min(checks['rank_per_fixation']) == 256
    assert checks['right_inverse_max'] < 1e-10 and checks['null_projection_max'] < 1e-10
    aq = a.bfloat16().float().double()
    uq, sq, vq = torch.linalg.svd(aq, full_matrices=False)
    pinvq = vq.transpose(-1, -2) @ (uq.transpose(-1, -2) / sq[..., None])
    out = dict(a=a.float(), pinv=pinv.float(), pinvq=pinvq.float(), random_q=q.float(),
               centres=centres, cells=cells, singular_values=s)
    torch.save(out, HERE / 'maps.pt')
    return out, checks


@torch.no_grad()
def prepare(m, z, maps, device, cfg):
    n = len(z['g'])
    outputs = {k: [] for k in MODES if k not in ('base', 'fine')}
    metrics = {k: [] for k in ('raw_energy', 'retained_energy', 'null_energy', 'full_fp32_subtraction_error',
        'fine_energy', 'bf16_total_error', 'bf16_accum_output_error', 'quantization_error',
        'bf16_recovered_error', 'cached_fine_error', 'mixed_energy')}
    a, pinv, pinvq, qr = (maps[k].to(device) for k in ('a', 'pinv', 'pinvq', 'random_q'))
    w, b = m.belief.percept_in.weight.float(), m.belief.percept_in.bias.float()
    wq, bq = w.bfloat16().float(), b.bfloat16().float()
    for st in range(0, n, 512):
        ix = slice(st, min(st + 512, n))
        p, g, v, fix = (z[k][ix].to(device) for k in ('p', 'g', 'v', 'fix'))
        cells = M.window(fix)
        fid = (fix[:, 0] - 1) * 7 + fix[:, 1] - 1
        gf = g.flatten(1).float()
        native = torch.bmm(a[fid], gf[..., None]).squeeze(-1)
        retained = torch.bmm(pinv[fid], native[..., None]).squeeze(-1)
        null = gf - retained
        random = (gf @ qr) @ qr.T
        canvas = torch.zeros(len(g), 81, 64, device=device).scatter(1, cells[..., None].expand_as(g), g.float())
        mask = torch.zeros(len(g), 81, device=device).scatter(1, cells, 1.)
        x = torch.cat([p.flatten(1).float(), canvas.flatten(1), mask, v.float()], -1)
        xc = x.clone(); xc[:, 1296:6480] = 0
        full = F.linear(x, w, b); offset = F.linear(xc, w, b)
        xq = x.bfloat16().float(); xcq = xc.bfloat16().float()
        quantized_full = F.linear(xq, wq, bq)
        quantized_offset = F.linear(xcq, wq, bq)
        with TR.autocast(device, cfg):
            mixed = m.belief.percept_embed(M.Percept(p, g, v, fix)).float()
        recovered = torch.bmm(pinvq[fid], (mixed - quantized_offset)[..., None]).squeeze(-1)
        for k, t in dict(native=native, retained=retained, null=null,
                         random_retained=random, mixed=mixed, recovered_bf16=recovered).items():
            assert torch.isfinite(t).all()
            outputs[k].append(t.cpu())
        for k, t in dict(raw_energy=gf.square().mean(-1), retained_energy=retained.square().mean(-1),
            null_energy=null.square().mean(-1), full_fp32_subtraction_error=(full - offset - native).square().mean(-1),
            fine_energy=native.square().mean(-1), bf16_total_error=(mixed - full).square().mean(-1),
            bf16_accum_output_error=(mixed - quantized_full).square().mean(-1),
            quantization_error=(quantized_full - full).square().mean(-1),
            bf16_recovered_error=(recovered - retained).square().mean(-1),
            cached_fine_error=(native - z['input'][ix].to(device).float()).square().mean(-1),
            mixed_energy=mixed.square().mean(-1)).items():
            metrics[k].append(t.cpu())
    return {k: torch.cat(v) for k, v in outputs.items()}, {k: torch.cat(v) for k, v in metrics.items()}


class Fine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        original = F1.Predictor(dim)  # exact F1 fine-path initialization/RNG consumption
        self.body, self.head = original.fine_body, original.fine
    def forward(self, b, s):
        return self.head(self.body(torch.cat([b, s], -1))).reshape(-1, 9, 64)


def supplement(feat, extras, mode):
    if mode == 'base': return None
    return feat.z['g'].flatten(1).float() if mode == 'fine' else extras[mode].to(feat.device)


def inputs(feat, ix, extra):
    b, s, _ = feat.make(ix, 'base')
    if extra is not None:
        s[:, :extra.shape[-1]] = extra[ix]
    return b, s


@torch.no_grad()
def evaluate(net, feat, extra, ids):
    errs = []
    for ix in ids.split(512):
        yp = net(*inputs(feat, ix, extra))
        errs.append((yp - feat.target['g'][ix].float()).square().mean(-1).cpu())
    return torch.cat(errs)


def fit(feat, extras, mode, seed, steps):
    torch.manual_seed(seed)
    net = Fine(feat.dim).to(feat.device)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    ids = torch.where((feat.z['episode'] < 360) & ~feat.target['monitor'])[0]
    val = torch.where((feat.z['episode'] >= 360) & ~feat.target['monitor'])[0]
    extra = supplement(feat, extras, mode)
    gen = torch.Generator(device=feat.device).manual_seed(seed + 10)
    start = time.time(); curve = []
    for step in range(steps):
        ix = ids[torch.randint(len(ids), (512,), device=feat.device, generator=gen)]
        yp = net(*inputs(feat, ix, extra))
        loss = (yp - feat.target['g'][ix].float()).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        if step % 600 == 0 or step == steps - 1:
            r = dict(step=step + 1, train=float(loss.detach()),
                val=float(evaluate(net, feat, extra, val[:2048]).mean()),
                grad_norm=float(gn), seconds=time.time() - start)
            curve.append(r); print('fit', seed, mode, r, flush=True)
    return net, dict(seed=seed, mode=mode, steps=steps, curve=curve, seconds=time.time()-start,
        parameters=sum(p.numel() for p in net.parameters()), fit_rows=len(ids), validation_rows=len(val))


def masks(z, t):
    cells = M.window(z['fix'])
    source = torch.zeros(len(cells), 81)
    source[:, :63] = (z['truth'] > 0).float()
    in_crop = torch.zeros_like(source).scatter(1, cells, 1.)
    source *= in_crop
    around = F.max_pool2d(source.reshape(-1, 1, 9, 9), 3, stride=1, padding=1).reshape(-1, 81)
    ent = source.any(-1)
    return dict(episode=z['episode'], night=z['night'], entity=ent, scroll=t['scroll'],
        source=source.gather(1, cells), neighborhood=around.gather(1, cells))


def mechanical_checks(feat, extras, maps):
    ix = torch.tensor([2, 20, 40], device=feat.device)
    b, s = inputs(feat, ix, supplement(feat, extras, 'fine'))
    assert torch.equal(s[:, :576], feat.z['g'][ix].flatten(1).float())
    assert torch.count_nonzero(s[:, 576:]) == 0
    before = [t.clone() for t in (b, s)]
    old = feat.target['g'][ix].clone(); feat.target['g'][ix] = 900
    after = inputs(feat, ix, supplement(feat, extras, 'fine'))
    assert all(torch.equal(x, y) for x, y in zip(before, after))
    feat.target['g'][ix] = old
    # All transformations are current-crop-only and raw = retained + null.
    assert torch.allclose(extras['retained'] + extras['null'], feat.z['g'].flatten(1).float().cpu(), atol=2e-6)
    torch.manual_seed(611701); new = Fine(feat.dim)
    torch.manual_seed(611701); oldnet = F1.Predictor(feat.dim)
    assert all(torch.equal(v, oldnet.fine_body.state_dict()[k]) for k,v in new.body.state_dict().items())
    assert all(torch.equal(v, oldnet.fine.state_dict()[k]) for k,v in new.head.state_dict().items())
    assert maps['a'].shape == (49, 256, 576)
    return dict(whole_crop=True, targets_not_inputs=True, raw_equals_retained_plus_null=True,
                fine_initialization_matches_F1=True, fixations=49, transforms_from_weights_only=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('smoke', 'run')); args = ap.parse_args()
    torch.set_num_threads(4); device = torch.device('cuda')
    start = time.time()
    assert TR.file_sha256(Q.CK) == CK_SHA
    ck = torch.load(Q.CK, map_location='cpu', weights_only=False)
    CFG = ck['cfg']
    m = M.M0(CFG).to(device); m.load_state_dict(ck['model']); m.eval(); m.requires_grad_(False); del ck
    assert CFG['p1']['precision'] == 'bf16_autocast'
    # Numeric mechanics must not silently use TF32 as their FP32 reference.
    torch.backends.cuda.matmul.allow_tf32 = False
    maps, algebra = linear_maps(m)
    z, t = load('train'); extras, numeric = prepare(m, z, maps, device, CFG)
    torch.save(numeric, HERE / 'numeric_train.pt')
    feat = F1.Features(z, t, m, device)
    checks = mechanical_checks(feat, extras, maps)
    (HERE / 'checks.json').write_text(json.dumps(dict(algebra=algebra, instruments=checks), indent=2))
    print('mechanical checks passed', algebra['singular_min'], algebra['singular_max'], flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # identical to original F1 fitted heads
    if args.stage == 'smoke':
        net, r = fit(feat, extras, 'retained', 611799, 3)
        (HERE / 'smoke.json').write_text(json.dumps(r, indent=2)); return
    nets = {}; stats = {}
    for seed in (611701, 611702):
        for mode in MODES:
            net, r = fit(feat, extras, mode, seed, 2400)
            key = f'{mode}_{seed}'
            nets[key] = {k:v.detach().cpu() for k,v in net.state_dict().items()}; stats[key] = r
            torch.save(nets, HERE / 'predictors.pt')
            (HERE / 'fit.json').write_text(json.dumps(stats, indent=2))
            del net
    del feat, extras, z, t, numeric
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    z, t = load('dev'); extras, numeric = prepare(m, z, maps, device, CFG)
    torch.save(numeric, HERE / 'numeric_dev.pt'); torch.save(masks(z,t), HERE / 'metadata.pt')
    feat = F1.Features(z, t, m, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    ids = torch.arange(feat.n, device=device)
    for key, state in nets.items():
        mode = key.rsplit('_', 1)[0]
        net = Fine(feat.dim).to(device); net.load_state_dict(state); net.eval()
        extra = supplement(feat, extras, mode)
        out = evaluate(net, feat, extra, ids); torch.save(out, HERE / f'errors_{key}.pt')
        print('read', key, float(out.mean()), flush=True)
        del net, extra
    assert TR.file_sha256(Q.CK) == CK_SHA
    assert all(not p.requires_grad for p in m.parameters())
    identity = dict(checkpoint_sha256=CK_SHA, script_sha256=TR.file_sha256(__file__),
        plan_sha256=TR.file_sha256(HERE / 'PLAN.md'), seconds=time.time()-start,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        files={str(p.relative_to(TR.ROOT)):TR.file_sha256(p) for p in [
            Q.CK, OLD/'predict.py', OLD/'train_400_future.pt', OLD/'dev_400_future.pt',
            OLD/'dev_400.pt', HERE.parent/'20261102_foveal_information/train_400.pt',
            TR.ROOT/'saccade/model.py', TR.ROOT/'saccade/train.py']})
    (HERE / 'identity.json').write_text(json.dumps(identity, indent=2))
    print('complete', identity['seconds'], identity['peak_GiB'], flush=True)


if __name__ == '__main__': main()
