"""Matched-batch LDAD gradient decomposition; diagnostic, not an outcome gate.

Evaluate the canonical prediction + SIGReg objective and the unweighted LDAD CE
at the common initialization and the λ=1/10 2k and 10k checkpoints. Report
separate encoder gradients and their weighted direction before global clipping.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from d4mj.checkpoint import read_lewm_bundle
from d4mj.config import config_from_dict
from d4mj.data import load_joint_corpus, JointSampler
from d4mj.lewm import SIGReg, joint_loss
from d4mj.train import _joint_components, autocast_context

OUT = Path(__file__).with_name('ldad_grad.json')
RUN = ROOT / 'artifacts/lewm_m4_canonical/raw'
CKPTS = {'lambda1': ROOT / 'artifacts/eda/levers_ldad_v1/raw_lam1',
         'lambda10': ROOT / 'artifacts/eda/levers_ldad_v1/raw_lam10'}


def geometry(p, s, c, lam):
    def dot(a, b):
        return sum((x.float() * y.float()).sum().item() for x, y in zip(a, b))
    base = [a + b for a, b in zip(p, s)]
    total = [a + lam * b for a, b in zip(base, c)]
    nb, nc, nt = dot(base, base) ** .5, dot(c, c) ** .5, dot(total, total) ** .5
    return {'base_norm': nb, 'ce_unweighted_norm': nc, 'ce_weighted_over_base': lam * nc / nb,
            'cos_base_ce': dot(base, c) / (nb * nc),
            'cos_total_base': dot(total, base) / (nt * nb),
            'cos_total_ce': dot(total, c) / (nt * nc)}


def main():
    canonical = read_lewm_bundle(RUN / 'joint/step-010000.pt')
    config = config_from_dict(canonical['config'])
    record = json.loads((RUN / 'dataset.json').read_text())
    episodes, contract = load_joint_corpus(record['paths'], config)
    assert contract == record['contract']
    bundle, _, _, _, identity = _joint_components(episodes, config)
    assert identity == canonical['initial_identity']
    initial = {name: {k: v.detach().clone() for k, v in getattr(bundle, name).state_dict().items()}
               for name in ('encoder', 'world')}
    torch.manual_seed(config.seed + 7777)
    head = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, config.dynamics.n_actions)).to(config.runtime.device)
    initial_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    batches = [sampler.sample().to(config.runtime.device) for _ in range(2)]
    regularizer = SIGReg(config.joint.knots, config.joint.projections).to(config.runtime.device)
    params = list(bundle.encoder.parameters())
    assert all(p.requires_grad for p in params)
    out = {'checkpoint_sha256': {f'{name}@{step}': hashlib.sha256((path / f'step-{step:06d}.pt').read_bytes()).hexdigest()
                                 for name, path in CKPTS.items() for step in (2000, 10000)},
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'batch_count': len(batches), 'batch_size': config.joint.batch,
           'sigreg_weight': config.joint.sigreg_weight, 'cases': {}}
    for case in ('initial', 'lambda1@2000', 'lambda10@2000', 'lambda1@10000', 'lambda10@10000'):
        if case == 'initial':
            for name in ('encoder', 'world'):
                getattr(bundle, name).load_state_dict(initial[name])
            head.load_state_dict(initial_head)
            lam = 1.0
        else:
            name, step = case.split('@')
            st = torch.load(CKPTS[name] / f'step-{int(step):06d}.pt', map_location='cpu', weights_only=False)
            assert st['variant'] == 'raw' and st['step'] == int(step)
            lam = float(st['lam'])
            bundle.encoder.load_state_dict(st['modules']['encoder'])
            bundle.world.load_state_dict(st['modules']['world'])
            head.load_state_dict(st['modules']['ldad'])
        bundle.encoder.train(); bundle.world.train(); head.train()
        rows = []
        for bi, batch in enumerate(batches):
            rng = torch.Generator(device=config.runtime.device).manual_seed(config.seed + 1001 + bi)
            with autocast_context(config):
                loss = joint_loss(bundle.encoder, bundle.world, batch.frames, batch.actions,
                                  regularizer, rng, config)
                z = loss.latent[:, :, 0].float()
                logits = head(z[:, 1:] - z[:, :-1]).float()
                ce = F.cross_entropy(logits.flatten(0, 1), batch.actions.flatten())
            pp = torch.autograd.grad(loss.prediction, params, retain_graph=True, allow_unused=True)
            ss = torch.autograd.grad(config.joint.sigreg_weight * loss.regularization,
                                     params, retain_graph=True, allow_unused=True)
            cc = torch.autograd.grad(ce, params, allow_unused=True)
            def dense(gs):
                return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)]
            row = geometry(dense(pp), dense(ss), dense(cc), lam)
            row.update({'prediction': float(loss.prediction.detach()),
                        'sigreg_weighted': float((config.joint.sigreg_weight * loss.regularization).detach()),
                        'ce': float(ce.detach()),
                        'ce_accuracy': float((logits.argmax(-1) == batch.actions).float().mean().detach())})
            rows.append(row)
        out['cases'][case] = {'per_batch': rows,
                              'mean': {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}}
        OUT.write_text(json.dumps(out, indent=2) + '\n')
        print(case, json.dumps(out['cases'][case]['mean']), flush=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')

if __name__ == '__main__':
    main()
