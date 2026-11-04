"""Bounded E1/E2/E3 head-only experiments; never trains the world or encoder."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import argparse
from copy import deepcopy
import gc
import importlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

C = importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
Q = importlib.import_module('artifacts.experiments.20261102_direct_chooser.direct')
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S
SEEDS = (611601, 611602)


def lr_at(step, total):
    warm = min(1., (step+1)/100)
    frac = max(0., (step-100)/max(1, total-100))
    return warm*(1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*frac)))


class AuxHead(Q.Head):
    def __init__(self):
        super().__init__()
        self.aux = nn.Linear(256, 64)

    def forward(self, inputs):
        p, ctx, act, aux = inputs
        x = self.local(p.transpose(1, 2).flatten(2))
        x = x+self.context(ctx)[:, None]+self.action(act)[:, None]+self.position+self.aux(aux)[:, None]
        return p[:, -1]+self.output(self.blocks(x))


class AuxData(Q.Data):
    def __init__(self, z, device, aux, mean, std):
        super().__init__(z, device)
        self.aux = aux.to(device)
        self.mean, self.std = mean.to(device), std.to(device)

    def inputs(self, ix, action=None):
        return (*super().inputs(ix, action), (self.aux[ix].float()-self.mean)/self.std)


def init_fit(seed, device, aux=False):
    torch.manual_seed(seed)
    h = (AuxHead() if aux else Q.Head()).to(device)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(seed+10)
    return h, opt, rng


def split_rows(data):
    z = data.z
    return (torch.where((z['episode'] < 360) & ~z['monitor'] & (z['t'] > 0))[0],
            torch.where((z['episode'] >= 360) & ~z['monitor'] & (z['t'] > 0))[0])


def advance(h, opt, rng, data, start, stop, total, hook=None):
    train, val = split_rows(data)
    records = []
    for step in range(start, stop):
        opt.param_groups[0]['lr'] = lr_at(step, total)
        ix = train[torch.randint(len(train), (256,), generator=rng, device=data.device)]
        loss = (h(data.inputs(ix))-data.z['p'][ix].float()).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(h.parameters(), 1.); opt.step()
        if step % 500 == 0 or step == stop-1:
            with torch.no_grad():
                vv = [(h(data.inputs(v))-data.z['p'][v].float()).square().mean((-1, -2)) for v in val[:4096].split(256)]
            rec = dict(step=step+1, loss=float(loss.detach()), val_mse=float(torch.cat(vv).mean()),
                       lr=opt.param_groups[0]['lr'], gradient_norm=float(gn))
            records.append(rec); print('fit', rec, flush=True)
        if hook:
            hook(step+1)
    return records


@torch.no_grad()
def predict(h, data, ids):
    taken, null = [], []
    for ix in ids.split(256):
        x = data.inputs(ix)
        taken.append(h(x))
        null.append(h((*x[:2], torch.zeros_like(x[2]), *x[3:])))
    taken, null = torch.cat(taken), torch.cat(null)
    actual = data.z['p'][ids].float()
    c = (taken-null).square().sum(-1)[:, Q.WIN.to(data.device)].sum(-1)
    s = (actual-taken).square().sum(-1)[:, Q.WIN.to(data.device)].sum(-1)
    pr = dict(C=c.cpu().numpy(), S=s.cpu().numpy(), prob=Q.policy(c,s).cpu().numpy(),
              mse=(taken-actual).square().mean((-1,-2)).cpu().numpy(),
              copy_mse=(data.z['p'][ids-1].float()-actual).square().mean((-1,-2)).cpu().numpy(),
              prediction=taken.half().cpu().numpy(), noop=null.half().cpu().numpy())
    return pr, taken, null


@torch.no_grad()
def snapshot(h, data, stem):
    meta = C.cohorts()
    for cohort, m in meta.items():
        ids = data.row_ids(m['ep'], m['t'])
        pred, taken, no = predict(h, data, ids)
        actual = data.z['p'][ids].float()
        previous = data.z['p'][ids-1].float()
        ar = torch.arange(len(ids), device=data.device)
        cell = torch.tensor(np.maximum(m['counter'], 0), device=data.device)
        target_cell = actual[ar, cell]
        prediction, null = taken[ar, cell].float(), no[ar, cell].float()
        errors = dict(counter_mse=(prediction-target_cell).square().mean(-1).cpu().numpy(),
                      counter_copy_mse=(previous[ar, cell]-target_cell).square().mean(-1).cpu().numpy(),
                      counter_C=(prediction-null).square().sum(-1).cpu().numpy(),
                      counter_target_step=(previous[ar, cell]-target_cell).square().sum(-1).cpu().numpy(),
                      factual_mse=pred['mse'], copy_mse=pred['copy_mse'])
        # Forecast archives retain original fp16 storage. Errors above use the full
        # float32 head outputs; no semantic counter decoder is assumed.
        np.savez_compressed(C.HERE/(stem+'_'+cohort+'.npz'), **{**pred, **m, **errors})
        C.write(C.HERE/(stem+'_'+cohort+'.json'), C.summarize(m, pred['C'], pred['S'], errors))


def load_cache(checkpoint, split, device):
    n = 400 if split == 'train' else 600
    if checkpoint == 10:
        return torch.load(C.OLD/f'{split}_{n}_coarse.pt', map_location='cpu', weights_only=False)
    folder = C.HERE/f'cache_ck{checkpoint:02d}'; folder.mkdir(exist_ok=True)
    path = folder/f'{split}_{n}_coarse.pt'
    ckpath = C.CKDIR/f'ckpt_{checkpoint:02d}.pt'
    if path.exists():
        z = torch.load(path, map_location='cpu', weights_only=False)
        assert z['checkpoint_sha256'] == C.sha(ckpath)
        return z
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    model = M.M0(ck['cfg']).to(device)
    model.load_state_dict(ck['model']); model.eval().requires_grad_(False)
    old_here, old_ck = Q.HERE, Q.CK
    try:
        Q.HERE, Q.CK = folder, ckpath
        z = Q.collect(model, ck['cfg'], split, n, device)
    finally:
        Q.HERE, Q.CK = old_here, old_ck
    del model, ck
    gc.collect(); torch.cuda.empty_cache()
    return z


def e1(device):
    train = Q.Data(load_cache(10, 'train', device), device)
    dev = Q.Data(load_cache(10, 'dev', device), device)
    results = {}
    for seed in SEEDS:
        h, opt, rng = init_fit(seed, device)
        snapshot(h, dev, f'e1_{seed}_u0000')
        def hook(step):
            if step in (100, 500, 1000, 2000, 3000, 4000):
                before = rng.get_state().clone()
                snapshot(h, dev, f'e1_{seed}_u{step:04d}')
                assert torch.equal(before, rng.get_state())
        curve = advance(h, opt, rng, train, 0, 4000, 4000, hook)
        old = torch.load(C.OLD/f'head_{seed}.pt', weights_only=False, map_location='cpu')['state']
        diff = max(float((v.detach().cpu()-old[k]).abs().max()) for k, v in h.state_dict().items())
        results[str(seed)] = dict(final_weight_max_abs_difference=diff, curve=curve)
        torch.save(dict(state=h.cpu().state_dict(), optimizer=opt.state_dict(), rng=rng.get_state()),
                   C.HERE/f'e1_{seed}_final.pt')
        del h, opt; gc.collect(); torch.cuda.empty_cache()
    C.write(C.HERE/'e1_identity.json', results)
    del train, dev


def e2(device):
    identity = {}
    # Caches are immutable snapshots; memory stays bounded to one representation.
    for seed in SEEDS:
        h, opt, rng = init_fit(seed, device)
        identity[str(seed)] = {}
        for stage, checkpoint in enumerate((1, 5, 10)):
            train = Q.Data(load_cache(checkpoint, 'train', device), device)
            dev = Q.Data(load_cache(checkpoint, 'dev', device), device)
            snapshot(h, dev, f'e2_carry_{seed}_ck{checkpoint:02d}_s0000')
            def hook(step):
                local = step-4000*stage
                if local in (100, 1000, 4000):
                    snapshot(h, dev, f'e2_carry_{seed}_ck{checkpoint:02d}_s{local:04d}')
            curve = advance(h, opt, rng, train, stage*4000, (stage+1)*4000, 12000, hook)
            torch.save(dict(state={k:v.detach().cpu() for k,v in h.state_dict().items()},
                            optimizer=opt.state_dict(), rng=rng.get_state(), checkpoint=checkpoint),
                       C.HERE/f'e2_carry_{seed}_ck{checkpoint:02d}.pt')
            identity[str(seed)][str(checkpoint)] = dict(carry_curve=curve, cumulative_updates=(stage+1)*4000)
            if stage:
                fresh, fo, fr = init_fit(seed, device)
                freshcurve = advance(fresh, fo, fr, train, 0, (stage+1)*4000, 12000)
                snapshot(fresh, dev, f'e2_fresh_{seed}_ck{checkpoint:02d}_endpoint')
                identity[str(seed)][str(checkpoint)]['fresh_curve'] = freshcurve
                del fresh, fo
            # Target scale is descriptive only; no inference from raw loss drift.
            z = train.z; ix = split_rows(train)[0]
            identity[str(seed)][str(checkpoint)]['target_variance'] = float(z['p'][ix].float().var(0).mean())
            del train, dev; gc.collect(); torch.cuda.empty_cache()
        del h, opt
    C.write(C.HERE/'e2_identity.json', identity)


def sketch(z):
    gen = torch.Generator().manual_seed(611730)
    w = torch.randn(81*16, 256, generator=gen)/math.sqrt(81*16)
    aux = torch.zeros(len(z['p']), 256, dtype=torch.float16)
    # Frame t gets only p[t-1]; episode starts receive zero.
    for st in range(0, len(aux), 1024):
        ids = torch.arange(st, min(st+1024, len(aux)))
        previous = (ids-1).clamp(min=0)
        x = z['p'][previous].float().flatten(1) @ w
        x[z['t'][ids] == 0] = 0
        aux[ids] = x.half()
    return aux


@torch.no_grad()
def native_history(model, cfg, split, z, device, limit=None):
    n = 400 if split == 'train' else 600
    path = C.HERE/f'{split}_preaction_belief.pt'
    if path.exists() and limit is None:
        out = torch.load(path, map_location='cpu', weights_only=False)
        assert out['checkpoint_sha256'] == z['checkpoint_sha256']
        return out['aux']
    labels = S.first_episodes(dict(np.load(S.LABELS/f's0_{split}_labels.npz')), n)
    eps = S.episodes(labels)
    result = []; maxerr = 0.
    for i, ep in enumerate(eps[:limit] if limit else eps):
        rr = np.random.default_rng([cfg['p1']['monitor']['fixed_fixation_seed'], ep.shard, ep.slot])
        fx = [D.fixation_prior(rr, ()) for _ in range(ep.T+1)]
        recs = [[(0, 1, t, *fx[t]) for t in range(ep.T+1)]]
        c = TR.tensors(D.gather([ep], recs), device)
        with TR.autocast(device, cfg):
            parts = [model.perception(c['frames'][:, a:a+512], c['fixation'][:, a:a+512]) for a in range(0, ep.T+1, 512)]
            P = M.Percept(*(torch.cat(v, 1) for v in zip(*[p.parts() for p in parts])), c['fixation'])
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(P), c['reward'],
                                c['terminated'], c['action']), model.belief.initial(1, device))
        b = y[0, 0::2].float().half().cpu()
        before = torch.cat([torch.zeros_like(b[:1]), b[:-1]])
        result.append(before)
        a, end = z['offsets'][i:i+2]
        err = float((P.p[0].half().cpu()-z['p'][a:end]).abs().max())
        maxerr = max(maxerr, err)
        assert err == 0., (split, i, err)
        assert torch.equal(c['prev_action'][0].cpu(), z['prev_action'][a:end])
        if i == 0:
            # Prefix causality test: changing the queried action and later evidence
            # cannot change corrected b[t-1]. A short replay tests the actual Mamba.
            t = min(12, ep.T)
            cut = 2*t-1  # tokens through percept at t-1, before queried action
            with TR.autocast(device, cfg):
                tok = model.belief.tokens(model.belief.percept_embed(P), c['reward'], c['terminated'], c['action'])
                prefix, _ = model.belief(tok[:, :cut], model.belief.initial(1, device))
                corrupted = tok.clone(); corrupted[:, cut:] += 7
                alternative, _ = model.belief(corrupted, model.belief.initial(1, device))
            rel = float((prefix[0, -1].float()-b[t-1].to(device).float()).norm()/b[t-1].float().norm().clamp(min=1e-8))
            assert rel < .02, rel
            assert torch.equal(alternative[0, cut-1], y[0, cut-1]), 'future/action leakage into belief'
        if i % 100 == 0:
            print('native history', split, i, n, flush=True)
    aux = torch.cat(result)
    if limit is None:
        assert len(aux) == len(z['p']) and (aux[z['t'] == 0] == 0).all()
        torch.save(dict(aux=aux, checkpoint_sha256=z['checkpoint_sha256'], max_coarse_error=maxerr), path)
    return aux


def aux_stats(aux, z):
    ids = torch.where((z['episode'] < 360) & ~z['monitor'] & (z['t'] > 0))[0]
    x = aux[ids].float()
    return x.mean(0), x.std(0).clamp(min=1e-3)


def e3(device):
    tr, dv = load_cache(10, 'train', device), load_cache(10, 'dev', device)
    ck = torch.load(C.CKDIR/'ckpt_10.pt', map_location='cpu', weights_only=False)
    model = M.M0(ck['cfg']).to(device)
    model.load_state_dict(ck['model']); model.eval().requires_grad_(False)
    rich = [native_history(model, ck['cfg'], split, z, device) for split, z in (('train', tr), ('dev', dv))]
    del model, ck; gc.collect(); torch.cuda.empty_cache()
    sketches = [sketch(z) for z in (tr, dv)]
    identity = {}
    for name, inputs in (('coarse', sketches), ('rich', rich)):
        mean, std = aux_stats(inputs[0], tr)
        train = AuxData(tr, device, inputs[0], mean, std)
        dev = AuxData(dv, device, inputs[1], mean, std)
        identity[name] = dict(aux_mean=mean.tolist(), aux_std=std.tolist(),
                              std_floor_channels=int((std == 1e-3).sum()), seeds={})
        for seed in SEEDS:
            h, opt, rng = init_fit(seed, device, aux=True)
            curve = advance(h, opt, rng, train, 0, 4000, 4000)
            snapshot(h, dev, f'e3_{name}_{seed}_endpoint')
            identity[name]['seeds'][str(seed)] = dict(curve=curve, parameters=sum(p.numel() for p in h.parameters()))
            torch.save(dict(state=h.cpu().state_dict(), mean=mean, std=std), C.HERE/f'e3_{name}_{seed}.pt')
            del h, opt; gc.collect(); torch.cuda.empty_cache()
        del train, dev
    C.write(C.HERE/'e3_identity.json', identity)


def smoke(device):
    z = load_cache(10, 'train', device)
    # Use complete episode data for causal boundary tests; optimization is bounded.
    data = Q.Data(z, device)
    h, opt, rng = init_fit(611799, device)
    advance(h, opt, rng, data, 0, 2, 20)
    saved = deepcopy((h.state_dict(), opt.state_dict(), rng.get_state()))
    advance(h, opt, rng, data, 2, 4, 20)
    h2, o2, r2 = init_fit(611799, device)
    h2.load_state_dict(saved[0]); o2.load_state_dict(saved[1]); r2.set_state(saved[2])
    advance(h2, o2, r2, data, 2, 4, 20)
    assert all(torch.equal(v, h2.state_dict()[k]) for k,v in h.state_dict().items())
    ck = torch.load(C.CKDIR/'ckpt_10.pt', map_location='cpu', weights_only=False)
    model = M.M0(ck['cfg']).to(device); model.load_state_dict(ck['model']); model.eval().requires_grad_(False)
    b = native_history(model, ck['cfg'], 'train', z, device, limit=1)
    sk = sketch({**z, 'p':z['p'][:len(b)], 't':z['t'][:len(b)]})
    assert b.shape == sk.shape and bool((b[0] == 0).all())
    del model, ck
    au = AuxData(z, device, sketch(z), torch.zeros(256), torch.ones(256))
    hh, oo, rr = init_fit(611799, device, aux=True)
    advance(hh, oo, rr, au, 0, 2, 20)
    assert hh.aux.weight.grad is not None and bool(torch.isfinite(hh.aux.weight.grad).all())
    assert float(hh.aux.weight.grad.norm()) > 0
    return dict(exact_optimizer_continuation=True, actual_native_prefix_capture=True,
                auxiliary_training=True, peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('smoke', 'e1', 'e2', 'e3')); args = p.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda'); start = time.time()
    if args.phase == 'smoke':
        C.write(C.HERE/'gpu_smoke.json', smoke(device))
    else:
        globals()[args.phase](device)
    C.write(C.HERE/(args.phase+'_complete.json'), dict(seconds=time.time()-start,
             peak_GiB=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
