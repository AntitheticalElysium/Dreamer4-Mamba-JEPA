"""P1-F3 preflight (PLAN.md "Preflight"): F2.1's arm checks unchanged (../20261104_p1f21/checks.py: run) with the
weighted loss installed, plus F3's own: the loss algebra, update 1 against F2's logged update 1, the monitor with the
patch installed vs removed, G1's magnitude. No efficacy reading, no launch."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
import json
from pathlib import Path
import torch
from saccade import data as D, model as M, train as TR
L = importlib.import_module('artifacts.experiments.20261105_p1f3.train_p1f3')
F21C = importlib.import_module('artifacts.experiments.20261104_p1f21.checks')
HERE = Path(__file__).resolve().parent
F2_LOG = HERE.parent / '20261103_p1f2/run/log.jsonl'


def algebra(cfg):
    """Synthetic batch through a fresh denoiser: (a) inactive == original, (b) unit weights == original, bitwise with
    the same noise; (c) active == 3.425 c_out^2 x physical squared error + glimpse squared error (per-part calls)."""
    torch.manual_seed(5); dev = torch.device('cuda')
    den = M.M0(cfg).to(dev).denoiser
    B, T = 2, 9
    x0 = (torch.randn(B, T, 81, 16, device=dev), torch.randn(B, T, 9, 64, device=dev), torch.randn(B, T, 8, device=dev))
    g = torch.Generator(device=dev).manual_seed(7)
    sigma = (TR.sample_sigma((B, T), cfg, dev, g), TR.sample_sigma((B, T), cfg, dev, g))
    noise = tuple(torch.randn(t.shape, device=dev, generator=g) for t in x0)
    pa = torch.randint(0, 17, (B, T), device=dev); mem = torch.randn(B, T, 4, 256, device=dev)
    fix = torch.randint(1, 8, (B, T, 2), device=dev); ep = torch.zeros(B, T, dtype=torch.long, device=dev)
    valid = torch.ones(B, T, dtype=torch.bool, device=dev); sd = cfg['edm']['sigma_data']
    args = (den, x0, sigma, pa, mem, fix, ep, valid, sd, noise)
    with torch.no_grad(), TR.autocast(dev, cfg):
        ref = L.ORIG_LOSS(*args)
        L.ACTIVE[0] = False; inactive = L.denoise_loss(*args)
        unit = L.denoise_loss(*args, weights=[1, 1, 1])
        parts = [L.denoise_loss(*args, weights=[float(i == k) for i in range(3)]) for k in range(3)]
        L.ACTIVE[0] = True; active = L.denoise_loss(*args); L.ACTIVE[0] = False
        co2 = M.precondition(sigma[0], sd)[1].float() ** 2
        manual = L.WEIGHTING['scale'] * co2 * (parts[0] + parts[2]) + parts[1]
    out = {'inactive_bitwise': bool(torch.equal(inactive, ref)), 'unit_weights_bitwise': bool(torch.equal(unit, ref)),
           'parts_sum_vs_original_max_rel': float(((parts[0] + parts[1] + parts[2]) - ref).abs().max() / ref.abs().max()),
           'active_vs_manual_max_rel': float((active - manual).abs().max() / manual.abs().max())}
    assert out['inactive_bitwise'] and out['unit_weights_bitwise'], out
    assert out['parts_sum_vs_original_max_rel'] < 1e-5 and out['active_vs_manual_max_rel'] < 1e-5, out
    return out


def extra(cfg, out):
    res = {'loss_algebra': algebra(cfg)}
    # (2) update 1 of the checks' straight run (small cfg, log_every=1) vs F2's production update 1: before any step,
    # only G1 (and the gradient norm it changes) may differ.
    f2 = json.loads(F2_LOG.read_text().splitlines()[0]); f3 = json.loads((out.parent / 'straight/log.jsonl').read_text().splitlines()[0])
    assert f2['update'] == f3['update'] == 1
    same = {k: f2[k] == f3[k] for k in f2 if k not in ('g1', 'grad_norm', 'sec_per_update', 'peak_gib')}
    res['update1_vs_f2'] = {'identical_except_g1': same, 'g1': [f2['g1'], f3['g1']], 'grad_norm': [f2['grad_norm'], f3['grad_norm']]}
    assert all(same.values()) and f2['g1'] != f3['g1'], res['update1_vs_f2']
    # (3) monitor: the patch is inactive outside p1_losses / encoder_diagnostics -> bitwise identical to the original
    model = M.M0(cfg).cuda(); eps = D.load_split(cfg)['monitor']
    a = TR.monitor(model, cfg, eps, n_windows=16)
    saved = M.denoise_loss; M.denoise_loss = L.ORIG_LOSS
    try:
        b = TR.monitor(model, cfg, eps, n_windows=16)
    finally:
        M.denoise_loss = saved
    res['monitor_patch_invariant'] = a == b
    assert res['monitor_patch_invariant'], (a, b)
    # (4) G1 magnitude, descriptive: the sustained run's g1 trajectory vs F2's at the same updates
    rows = [json.loads(x) for x in (out / 'log.jsonl').read_text().splitlines()]
    f2rows = {r['update']: r for r in map(json.loads, F2_LOG.read_text().splitlines()) if 'g1' in r}
    res['g1_vs_f2_same_update'] = {r['update']: [f2rows[r['update']]['g1'], r['g1']] for r in rows if r['update'] in f2rows}
    return res


def main():
    L.install()
    F21C.run(L, HERE, 'artifacts.experiments.20261105_p1f3.train_p1f3', prepare=lambda cfg: None, extra=extra)


if __name__ == '__main__':
    main()
