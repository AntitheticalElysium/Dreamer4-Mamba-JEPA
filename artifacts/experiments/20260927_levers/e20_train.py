"""Declared E20 Long-Mamba A/B/C continuations, with immutable full-state resume.

Endpoint classes have identical batch counts at every observed context length.
No future pixels/actions enter prediction. C alone supervises scalar health change
through a frozen factual-TRAIN health reader of generated token63. No fork labels.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e20_data as D
import e19 as E
import h16_resume as R

COUNTS = torch.tensor([2, 10, 26, 2])
REFERENCE = torch.tensor([.0532475, .0129825, .912445, .021325])


def load_pool():
    manifest = json.loads((D.OUT / 'pool.json').read_text())
    if not manifest['fresh_support_pass']:
        raise RuntimeError('Expanded factual fresh-hit support failed; world training held')
    path = D.OUT / 'pool/labels.pt'
    if R.file_hash(path) != manifest['labels_sha256']:
        raise RuntimeError('Pool labels changed')
    labels = torch.load(path, map_location='cpu', weights_only=False)
    spec = json.loads((D.OUT / 'pool/contract.json').read_text())
    fc = R.FeatureCache(D.OUT / 'pool/tokens.f16', spec['shape'], batch=spec['batch'])
    if fc.start != spec['shape'][0]:
        raise RuntimeError('Pool encoding incomplete')
    return labels, fc.mm


def schedule(labels, arm, updates=6000):
    if updates % 12:
        raise ValueError('Complete balanced context cycles required')
    order = torch.Generator().manual_seed(11)
    clock = torch.Generator().manual_seed(19)
    lengths = torch.cat([torch.randperm(12, generator=clock)+4 for _ in range(updates//12)])
    classes = torch.repeat_interleave(torch.arange(4), COUNTS)
    ledger = torch.empty(updates, 40, dtype=torch.long)
    for u in range(updates):
        uniform = torch.rand(40, generator=order)
        ids = torch.empty(40, dtype=torch.long)
        for c, q in enumerate(labels['cohorts'][arm]):
            use = classes == c
            ids[use] = q[(uniform[use]*len(q)).long()]
        permutation = torch.randperm(40, generator=order)
        ledger[u] = ids[permutation]
    if not torch.equal(torch.bincount(lengths-4, minlength=12), torch.full((12,), updates//12)):
        raise RuntimeError('Context-length imbalance')
    actual = labels['classes'][ledger]
    for length in range(4,16):
        got = torch.bincount(actual[lengths == length].flatten(), minlength=4)
        if not torch.equal(got, COUNTS*(updates//12)):
            raise RuntimeError('Class/context position correlation')
    return ledger, lengths


def batch(mm, labels, ids, length, device):
    src = torch.from_numpy(np.array(mm[ids.numpy(), 15-length:15])).float().to(device)
    target = torch.from_numpy(np.array(mm[ids.numpy(), 15])).float().to(device)
    actions = labels['actions'][ids, 15-length:15].to(device)
    return src, target, actions, labels['classes'][ids].to(device), labels['dh'][ids,-1].float().to(device)/9


def read_health(token, reader):
    return ((token.float()-reader['mean'])/reader['std']) @ reader['weight'] + reader['bias']


def objective(world, batch_values, reader=None):
    src, target, actions, classes, dh = batch_values
    prediction = world(src, actions)[0][:,-1].float()
    error = (prediction-target).abs().mean((-1,-2))
    importance = (REFERENCE/(COUNTS/40)).to(error.device)[classes]
    teacher = (importance*error).mean()
    health = teacher.new_zeros(())
    if reader is not None:
        change = read_health(prediction[:,63], reader) - read_health(src[:,-1,63], reader).detach()
        scalar_error = (change-dh).abs()
        health = torch.stack([scalar_error[classes == c].mean() for c in range(4)]).mean()
    return teacher+health, {'teacher': float(teacher.detach()), 'health': float(health.detach())}


def make_world(seed, device):
    parent = E.T.OUT / f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    world = E.T.TWorld('corrt', backbone='fmamba', frames=16).to(device)
    world.load_state_dict(saved['world'])
    return world, parent


def source_pins():
    files = set(D.ROOT.joinpath('d4mj').rglob('*.py'))
    files.update(HERE / n for n in ('e20_train.py','e20_data.py','e19.py','tworld.py','scroll.py','h16_resume.py','E20.md'))
    files.add(HERE.parent/'20260921_readout_ladder/spatial.py')
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and name.endswith('.py'):
            p = Path(name).resolve()
            if p.is_file() and p.is_relative_to(D.ROOT) and '.venv' not in p.relative_to(D.ROOT).parts:
                files.add(p)
    return {str(p): R.file_hash(p) for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A','B','C'], required=True)
    parser.add_argument('--seed', type=int, choices=[7,8], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    labels, mm = load_pool()
    ledger, lengths = schedule(labels, args.arm)
    model, parent = make_world(args.seed, device)
    cfg = E.config()
    from d4mj.train import phase_optimizer, optimizer_step, autocast_context, _phase_lr
    opt = phase_optimizer([model], cfg)
    params = [p for g in opt.param_groups for p in g['params']]
    reader = None
    if args.arm == 'C':
        health_record = json.loads((D.OUT/'health_reader.json').read_text())
        if not health_record['passed']:
            raise RuntimeError('Factual health-reader positive control failed; C held')
        reader = torch.load(D.OUT/'health_reader.pt', map_location=device, weights_only=False)
    proof = json.loads((D.OUT/'mechanics.json').read_text())
    if not proof['passed']:
        raise RuntimeError('Mechanics proof failed')
    if any(not Path(p).exists() or R.file_hash(p)!=sha for p,sha in proof['sources'].items()):
        raise RuntimeError('Mechanics source contract changed')
    spec = {'version':'e20-long-mamba-v1', 'arm':args.arm, 'seed':args.seed, 'updates':6000,
            'ledger_sha256':R.tensor_hash(ledger), 'lengths_sha256':R.tensor_hash(lengths),
            'pool_labels_sha256':R.file_hash(D.OUT/'pool/labels.pt'),
            'pool_manifest_sha256':R.file_hash(D.OUT/'pool.json'),
            'parent_sha256':R.file_hash(parent), 'reader_sha256':R.file_hash(D.OUT/'health_reader.pt'),
            'sources':source_pins(), 'mechanics_sha256':R.file_hash(D.OUT/'mechanics.json'),
            'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(),'tf32':torch.backends.cuda.matmul.allow_tf32},
            'class_counts':COUNTS.tolist(), 'teacher_class_mass':REFERENCE.tolist()}
    if R.file_hash(D.OUT/'health_reader.pt') != json.loads((D.OUT/'health_reader.json').read_text())['reader_sha256']:
        raise RuntimeError('Health-loss reader changed')
    name=f'e20_{args.arm}_s{args.seed}_fmamba_fromM16'
    final=E.T.OUT/(name+'.pt')
    store=R.Store(D.OUT/'states'/name,spec)
    clock=time.monotonic()
    def log(**values):
        print(json.dumps({'name':name,'seconds_session':round(time.monotonic()-clock,2),**values}),flush=True)
    with store.lock():
        done=store.load('complete')
        if done is not None:
            if not final.exists() or R.file_hash(final)!=done['sha256']:
                raise RuntimeError('Completed checkpoint changed')
            log(stage='complete_exists'); return
        if final.exists():
            saved=torch.load(final,map_location='cpu',weights_only=False)
            if saved.get('contract')!=store.contract or saved.get('update')!=6000:
                raise RuntimeError('Unbound final checkpoint')
            store.save('complete',{'sha256':R.file_hash(final)},1); return
        start, history=0,[]
        saved=store.load('train')
        if saved is not None:
            model.load_state_dict(saved['world']); opt.load_state_dict(saved['optimizer'])
            R.restore_rng(saved['rng'],device)
            start,history=saved['update'],saved['history']
        log(stage='resume' if start else 'init',update=start,contract=store.contract)
        model.train()
        for u in range(start,6000):
            values=batch(mm,labels,ledger[u],int(lengths[u]),device)
            with autocast_context(cfg):
                loss,parts=objective(model,values,reader)
            norm=optimizer_step(opt,loss,params,learning_rate=_phase_lr(cfg,u),
                                grad_clip=cfg.agent.grad_clip,strict=True,zero_grad=True)
            if (u+1)%100==0 or u==start:
                row={'update':u+1,'length':int(lengths[u]),'loss':float(loss.detach()),
                     'gradient_norm':float(norm),'peak_gb':torch.cuda.max_memory_allocated()/1e9,**parts}
                history.append(row); log(stage='train',**row)
            if (u+1)%500==0 or u+1==6000:
                store.save('train',{'update':u+1,'world':model.state_dict(),'optimizer':opt.state_dict(),
                                   'rng':R.rng_state(device),'history':history},u+1)
                log(stage='checkpoint',update=u+1)
        R.atomic_torch(final,{'name':name,'args':{'head':'corrt','pool':'rawlong','loss':'teacher',
            'seed':str(args.seed),'backbone':'fmamba','regions':'all','skip':'False','frames':'16',
            'updates':'6000','e20_arm':args.arm}, 'world':model.state_dict(),'update':6000,
            'history':history,'parent':str(parent),'contract':store.contract})
        store.save('complete',{'sha256':R.file_hash(final)},1)
        log(stage='saved',path=str(final))


if __name__=='__main__': main()
