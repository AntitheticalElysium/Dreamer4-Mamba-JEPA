"""Frozen F1.6 projection decomposition; PLAN.md. World parameters never train."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import argparse
import importlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

R = importlib.import_module('artifacts.experiments.20261103_fusion_retention.audit')
A = R.A
P = R.P
M = R.M
TR = R.TR
HERE = Path(__file__).resolve().parent
OLD = R.HERE / 'p1f16'
MODES = ('joint', 'native', 'retained', 'null', 'random_retained', 'mixed', 'recovered_bf16')
SEEDS = (611701, 611702)
DEVICE = torch.device('cuda')


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(split):
    x = torch.load(OLD / f'{split}.pt', map_location='cpu', weights_only=False)
    assert x['checkpoint_sha256'] == R.SHAS['p1f16']
    R.parity(x['current'], x['future'], split)
    return x['current'], x['future']


def maps(m):
    # Reuse the mechanically qualified algebra, but write only into this campaign.
    original = A.HERE
    A.HERE = HERE
    try:
        return A.linear_maps(m)
    finally:
        A.HERE = original


@torch.no_grad()
def prepare(m, cfg, z, linear):
    output = {k: [] for k in MODES}
    metrics = {k: [] for k in ('raw_energy', 'retained_energy', 'null_energy',
        'full_fp32_subtraction_error', 'fine_energy', 'bf16_total_error',
        'bf16_accum_output_error', 'quantization_error', 'bf16_recovered_error',
        'mixed_energy')}
    a, pinv, pinvq, qr = (linear[k].to(DEVICE) for k in ('a', 'pinv', 'pinvq', 'random_q'))
    w, b = m.belief.percept_in.weight.float(), m.belief.percept_in.bias.float()
    wq, bq = w.bfloat16().float(), b.bfloat16().float()
    max_native = 0.
    for st in range(0, len(z['g']), 512):
        sl = slice(st, st + 512)
        pp = M.Percept(*(z[k][sl].to(DEVICE).float() for k in ('p', 'g', 'v')), z['fix'][sl].to(DEVICE))
        cells = M.window(pp.fix)
        fid = (pp.fix[:, 0] - 1) * 7 + pp.fix[:, 1] - 1
        with TR.autocast(DEVICE, cfg):
            joint = m.belief.spatial_fusion(pp, cells)
            mixed = m.belief.percept_embed(pp).float()
        u = joint.flatten(1).float()
        native = torch.bmm(a[fid], u[..., None]).squeeze(-1)
        retained = torch.bmm(pinv[fid], native[..., None]).squeeze(-1)
        null = u - retained
        random = (u @ qr) @ qr.T
        canvas = torch.zeros(len(u), 81, 64, device=DEVICE).scatter(1, cells[..., None].expand_as(joint), joint.float())
        mask = torch.zeros(len(u), 81, device=DEVICE).scatter(1, cells, 1.)
        x = torch.cat([pp.p.flatten(1), canvas.flatten(1), mask, pp.v], -1)
        xc = x.clone(); xc[:, 1296:6480] = 0
        full, offset = F.linear(x, w, b), F.linear(xc, w, b)
        xq, xcq = x.bfloat16().float(), xc.bfloat16().float()
        quantized_full, quantized_offset = F.linear(xq, wq, bq), F.linear(xcq, wq, bq)
        recovered = torch.bmm(pinvq[fid], (mixed - quantized_offset)[..., None]).squeeze(-1)
        max_native = max(max_native, float((full - offset - native).abs().max()))
        for k, v in dict(joint=u, native=native, retained=retained, null=null,
                         random_retained=random, mixed=mixed, recovered_bf16=recovered).items():
            assert torch.isfinite(v).all(), k
            output[k].append(v.cpu())
        for k, v in dict(raw_energy=u.square().mean(-1), retained_energy=retained.square().mean(-1),
            null_energy=null.square().mean(-1), full_fp32_subtraction_error=(full-offset-native).square().mean(-1),
            fine_energy=native.square().mean(-1), bf16_total_error=(mixed-full).square().mean(-1),
            bf16_accum_output_error=(mixed-quantized_full).square().mean(-1),
            quantization_error=(quantized_full-full).square().mean(-1),
            bf16_recovered_error=(recovered-retained).square().mean(-1), mixed_energy=mixed.square().mean(-1)).items():
            metrics[k].append(v.cpu())
    out = {k: torch.cat(v) for k, v in output.items()}
    assert torch.allclose(out['retained'] + out['null'], out['joint'], atol=2e-6)
    # Independent check against the preceding campaign's unmodified native transform.
    old = R.transforms(m, cfg, {k: v[:2048] if isinstance(v, torch.Tensor) else v for k, v in z.items()})
    assert torch.equal(out['joint'][:2048], old['joint'])
    assert torch.equal(out['mixed'][:2048], old['mixed'])
    assert max_native < 1e-4, max_native
    return out, {k: torch.cat(v) for k, v in metrics.items()}


def validate_pins():
    pins = json.loads((HERE / 'SOURCE_PINS.json').read_text())
    for path, expected in pins.items():
        assert TR.file_sha256(TR.ROOT / path) == expected, path


def checks(m, feat, ex, linear):
    ix = torch.tensor([2, 20, 40], device=DEVICE)
    extra = ex['retained'].to(DEVICE)
    before = [x.clone() for x in A.inputs(feat, ix, extra)]
    old = feat.target['g'][ix].clone(); feat.target['g'][ix] = 900
    assert all(torch.equal(a, b) for a, b in zip(before, A.inputs(feat, ix, extra)))
    feat.target['g'][ix] = old
    net = A.Fine(feat.dim).to(DEVICE)
    b, s = A.inputs(feat, ix, extra); s = s.detach().requires_grad_()
    grad = torch.autograd.grad(net(b, s).square().mean(), s)[0]
    assert grad[:, :576].abs().sum() > 0
    assert all(not p.requires_grad for p in m.parameters())
    assert linear['a'].shape == (49, 256, 576)
    del net, extra
    return dict(native_joint_and_mixed_match=True, future_targets_excluded=True,
                retained_plus_null_equals_joint=True, predictor_gradient_nonzero=True,
                world_frozen=True, current_crop_only=True, baseline_dimension=feat.dim)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('smoke', 'run')); args = ap.parse_args()
    torch.set_num_threads(4); start = time.time(); validate_pins()
    m, cfg = R.load_world('p1f16')
    torch.backends.cuda.matmul.allow_tf32 = False
    linear, algebra = maps(m)
    z, t = load('train'); ex, numeric = prepare(m, cfg, z, linear)
    feat = P.Features(z, t, m, DEVICE)
    write('checks.json', dict(algebra=algebra, instrument=checks(m, feat, ex, linear)))
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == 'smoke':
        net, info = A.fit(feat, ex, 'retained', 611799, 3)
        write('smoke.json', info); validate_pins(); return
    nets, fitlog = {}, {}
    for seed in SEEDS:
        for mode in MODES:
            net, info = A.fit(feat, ex, mode, seed, 2400)
            key = f'{mode}_{seed}'
            nets[key] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            fitlog[key] = info
            torch.save(nets, HERE / 'predictors.pt'); write('fit.json', fitlog); del net
    del feat, ex, z, t, numeric
    torch.cuda.empty_cache(); torch.backends.cuda.matmul.allow_tf32 = False
    z, t = load('dev'); ex, numeric = prepare(m, cfg, z, linear)
    torch.save(numeric, HERE / 'numeric_dev.pt'); torch.save(A.masks(z, t), HERE / 'metadata.pt')
    feat = P.Features(z, t, m, DEVICE); ids = torch.arange(feat.n, device=DEVICE)
    torch.backends.cuda.matmul.allow_tf32 = True
    replication = {}
    for key, state in nets.items():
        mode = key.rsplit('_', 1)[0]
        net = A.Fine(feat.dim).to(DEVICE); net.load_state_dict(state); net.eval()
        extra = ex[mode].to(DEVICE); err = A.evaluate(net, feat, extra, ids)
        torch.save(err, HERE / f'errors_{key}.pt')
        if mode in ('joint', 'mixed'):
            old = torch.load(OLD / f'errors_{key}.pt', map_location='cpu', weights_only=False)
            difference = float((err - old).abs().max()); assert difference < 1e-6, (key, difference)
            replication[key] = difference
        print('read', key, float(err.mean()), flush=True); del net, extra
    validate_pins()
    assert TR.file_sha256(R.W.WORLDS['p1f16']) == R.SHAS['p1f16']
    assert all(not p.requires_grad for p in m.parameters())
    write('identity.json', dict(seconds=time.time()-start, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
          checkpoint_sha256=R.SHAS['p1f16'], replication_max_error=replication, phase='fits_complete'))
    print('complete', time.time()-start, flush=True)


if __name__ == '__main__':
    main()
