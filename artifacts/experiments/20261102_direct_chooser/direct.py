"""Frozen spatial forward-mean chooser. PLAN.md; no world training."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import argparse
import ast
import importlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
CK = TR.ROOT / 'artifacts/experiments/20261101_p1f15/run/ckpt_10.pt'
REF = TR.ROOT / 'artifacts/experiments/20261101_p1f15/readings/locus/p1f15.npz'
CENTRES = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))
WIN = M.window(CENTRES)
H = 8


def policy(c, s):
    norm = lambda x: torch.where(x.mean(-1, keepdim=True) > 1e-12,
                                x / x.mean(-1, keepdim=True).clamp(min=1e-12), torch.zeros_like(x))
    return .8 * (.5 * norm(c) + .5 * norm(s)).softmax(-1) + .2 / 49


def bootstrap(values, episodes, draws=2000):
    values, episodes = np.asarray(values, float), np.asarray(episodes)
    ids, inv = np.unique(episodes, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(611630)
    bs = rng.integers(len(ids), size=(draws, len(ids)))
    v = sums[bs].sum(1) / counts[bs].sum(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(v, .025)),
                high=float(np.quantile(v, .975)), episodes=len(ids), n=len(values))


def reference():
    """Freeze the exact mixed-policy baseline before any fit."""
    z = dict(np.load(REF))
    prob = policy(torch.tensor(z['contingency']), torch.tensor(z['surprise'])).numpy()
    win = WIN.numpy()
    report = {}
    for c in np.unique(z['cls']):
        if c == 'random':
            continue
        ix = z['cls'] == c
        remote = (win[None] == z['counter'][ix, None, None]).any(-1)
        contact = (win[None] == z['faced'][ix, None, None]).any(-1)
        q = {}
        for name, x in (('C', z['contingency'][ix]), ('S', z['surprise'][ix]), ('mixed', prob[ix])):
            choice = x.argmax(-1)
            q[name + '_remote_argmax'] = bootstrap(remote[np.arange(ix.sum()), choice], z['ep'][ix])
            q[name + '_contact_argmax'] = bootstrap(contact[np.arange(ix.sum()), choice], z['ep'][ix])
        q['mixed_remote_mass'] = bootstrap((prob[ix] * remote).sum(-1), z['ep'][ix])
        q['mixed_contact_mass'] = bootstrap((prob[ix] * contact).sum(-1), z['ep'][ix])
        q['uniform_remote_mass'] = float(remote.mean())
        report[str(c)] = q
    out = {'reference_sha256': TR.file_sha256(REF), 'plan_sha256': TR.file_sha256(HERE / 'PLAN.md'),
           'classes': report, 'margin': .05}
    (HERE / 'reference.json').write_text(json.dumps(out, indent=2))
    return out


@torch.no_grad()
def collect(model, cfg, split, n, device):
    path = HERE / f'{split}_{n}_coarse.pt'
    if path.exists():
        z = torch.load(path, map_location='cpu', weights_only=False)
        assert z['checkpoint_sha256'] == TR.file_sha256(CK)
        return z
    labels = S.first_episodes(dict(np.load(S.LABELS / f's0_{split}_labels.npz')), n)
    eps = S.episodes(labels)
    monitor = {(e.shard, e.slot) for e in D.load_split(cfg)['monitor']}
    fs = {k: [] for k in ('p', 'v', 'reward', 'terminated', 'prev_action', 'episode', 't', 'eps', 'monitor')}
    offsets = [0]
    for e, ep in enumerate(eps):
        raw = D.RP.episode_fields(ep.shard, ep.slot)
        frames = raw['observations']
        pp, vv = [], []
        for st in range(0, len(frames), 512):
            f = frames[st:st+512].to(device)
            fix = torch.ones(len(f), 2, dtype=torch.long, device=device)
            with TR.autocast(device, cfg):
                P = model.perception(f, fix)
            pp.append(P.p.half().cpu()); vv.append(P.v.half().cpu())
        T = ep.T
        vals = dict(p=torch.cat(pp), v=torch.cat(vv), reward=torch.cat([torch.zeros(1), raw['rewards'].float()]),
                    terminated=torch.cat([torch.zeros(1, dtype=torch.bool), raw['terminated'].bool()]),
                    prev_action=torch.cat([torch.tensor([-1]), raw['actions_taken'].long()]),
                    episode=torch.full((T+1,), e), t=torch.arange(T+1),
                    eps=torch.full((T+1,), float(labels['ep_eps'][e])),
                    monitor=torch.full((T+1,), (ep.shard, ep.slot) in monitor, dtype=torch.bool))
        for k, v in vals.items():
            assert len(v) == T+1, (k, len(v), T)
            fs[k].append(v)
        offsets.append(offsets[-1] + T+1)
        if e % 100 == 0:
            print('collect', split, e, n, flush=True)
    z = {k: torch.cat(v) for k, v in fs.items()}
    z.update(offsets=np.array(offsets), episode_keys=np.stack([labels['ep_shard'], labels['ep_slot']], -1),
             checkpoint_sha256=TR.file_sha256(CK))
    torch.save(z, path)
    return z


class Data:
    def __init__(self, z, device):
        self.z = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in z.items()}
        self.device = device
        self.n = len(z['p'])
        self.offsets = z['offsets']

    def inputs(self, ix, action=None):
        z = self.z
        # A target frame t sees only t-8..t-1. Clamp indices solely for safe gather;
        # validity masks zero every field before the target episode's first frame.
        h = ix[:, None] - torch.arange(H, 0, -1, device=self.device)[None]
        valid = z['t'][ix, None] >= torch.arange(H, 0, -1, device=self.device)[None]
        h = h.clamp(min=0)
        p = z['p'][h].float() * valid[..., None, None]
        v = z['v'][h].float() * valid[..., None]
        pa = F.one_hot((z['prev_action'][h]+1).clamp(0, 17), 18).float() * valid[..., None]
        r = M.SYMLOG(z['reward'][h].float()) * valid
        d = z['terminated'][h].float() * valid
        ctx = torch.cat([v, pa, r[..., None], d[..., None], valid[..., None]], -1).flatten(1)
        act = z['prev_action'][ix].long() if action is None else action
        assert bool((act >= 0).all()), 'Frame zero is not a transition target'
        return p, ctx, act

    def row_ids(self, ep, t):
        return torch.tensor(self.offsets[np.asarray(ep)] + np.asarray(t), device=self.device)


class Block(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.proj = nn.Linear(w, 3*w), nn.Linear(w, w)
        self.ff = nn.Sequential(nn.Linear(w, 2*w), nn.GELU(), nn.Linear(2*w, w))

    def forward(self, x):
        b, n, w = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b,n,3,4,w//4).permute(2,0,3,1,4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1,2).reshape(b,n,w)
        x = x + self.proj(y)
        return x + self.ff(self.n2(x))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Linear(H*16, 64)
        self.context = nn.Sequential(nn.Linear(H*29, 64), nn.GELU(), nn.Linear(64,64))
        self.action = nn.Embedding(17,64)
        self.position = nn.Parameter(torch.randn(81,64)*.02)
        self.blocks = nn.Sequential(Block(), Block())
        self.output = nn.Linear(64,16)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        p, ctx, act = inputs
        x = self.local(p.transpose(1,2).flatten(2))
        x = x + self.context(ctx)[:,None] + self.action(act)[:,None] + self.position
        return p[:,-1] + self.output(self.blocks(x))


@torch.no_grad()
def predict(head, data, ids):
    pred, noop = [], []
    for ix in ids.split(256):
        x = data.inputs(ix)
        pred.append(head(x))
        noop.append(head((x[0], x[1], torch.zeros_like(x[2]))))
    taken, no = torch.cat(pred), torch.cat(noop)
    actual = data.z['p'][ids].float()
    win = WIN.to(data.device)
    c = (taken-no).square().sum(-1)[:,win].sum(-1)
    s = (actual-taken).square().sum(-1)[:,win].sum(-1)
    return dict(C=c.cpu().numpy(), S=s.cpu().numpy(), prob=policy(c,s).cpu().numpy(),
                mse=(taken-actual).square().mean((-1,-2)).cpu().numpy(),
                copy_mse=(data.z['p'][ids-1].float()-actual).square().mean((-1,-2)).cpu().numpy(),
                prediction=taken.half().cpu().numpy(), noop=no.half().cpu().numpy())


def checks():
    torch.set_num_threads(4)
    torch.manual_seed(611600)
    z = dict(p=torch.randn(12,81,16), v=torch.randn(12,8), reward=torch.randn(12),
             terminated=torch.zeros(12,dtype=torch.bool), prev_action=torch.arange(12)%17,
             episode=torch.tensor([0]*6+[1]*6), t=torch.tensor(list(range(6))*2), offsets=np.array([0,6,12]))
    d = Data(z, torch.device('cpu')); h=Head()
    # Nonzero output tests the entire network, not a zero-output tautology.
    nn.init.normal_(h.output.weight, std=.02)
    ix=torch.tensor([2,7,9]); before=h(d.inputs(ix)).detach()
    # Changing the latest target cannot affect that target's forward prediction.
    ref=h(d.inputs(torch.tensor([9]))).detach()
    for k in ('p','v','reward','terminated'):
        z[k][9] = torch.ones_like(z[k][9]) * (True if k=='terminated' else 900)
    assert torch.equal(ref,h(d.inputs(torch.tensor([9]))).detach())
    p,ctx,_=d.inputs(torch.tensor([7])); assert torch.count_nonzero(p[:,:-1])==0
    assert torch.equal(p[0,-1],z['p'][6])
    out=predict(h,d,torch.tensor([1,7]));
    z['prev_action'][[1,7]]=0
    out=predict(h,d,torch.tensor([1,7])); assert (out['C']==0).all()
    # The historical module allocates CUDA tensors at import. Execute only its
    # unchanged priorities function so this mechanical check really runs on CPU.
    source=TR.ROOT/'artifacts/experiments/20261102_p1f15_readiness/gaze_checks.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='priorities')
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    a,b=torch.rand(11,49),torch.rand(11,49)
    assert torch.equal(policy(a,b),namespace['priorities'](a,b))
    assert torch.allclose(policy(torch.zeros_like(a),torch.zeros_like(b)),torch.full_like(a,1/49))
    lo=h(d.inputs(ix)).square().mean(); lo.backward()
    assert sum(float(p.grad.norm()) for p in h.parameters() if p.grad is not None)>0
    assert all(not v.requires_grad for v in z.values() if isinstance(v,torch.Tensor))
    return dict(history_target_exclusion=True, reset_padding=True, noop_exact_zero=True,
                policy_exact_match=True, gradients_head_only=True, parameters=sum(p.numel() for p in h.parameters()))


def fit(data, seed, steps, smoke=False):
    torch.manual_seed(seed)
    head=Head().to(data.device)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    z=data.z
    train=torch.where((z['episode']<360)&~z['monitor']&(z['t']>0))[0]
    val=torch.where((z['episode']>=360)&~z['monitor']&(z['t']>0))[0]
    if smoke: train,val=train[:2048],val[:256]
    gen=torch.Generator(device=data.device).manual_seed(seed+10)
    start=time.time(); curves=[]
    for step in range(steps):
        warm=min(1.,(step+1)/100)
        frac=max(0.,(step-100)/max(1,steps-100))
        lr=warm*(1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*frac)))
        opt.param_groups[0]['lr']=lr
        ix=train[torch.randint(len(train),(256,),generator=gen,device=data.device)]
        loss=(head(data.inputs(ix))-z['p'][ix].float()).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if step%500==0 or step==steps-1:
            with torch.no_grad():
                vi=val[:4096]; vloss=[]; copy=[]
                for vx in vi.split(256):
                    vloss.append((head(data.inputs(vx))-z['p'][vx].float()).square().mean((-1,-2)))
                    copy.append((z['p'][vx-1].float()-z['p'][vx].float()).square().mean((-1,-2)))
            rec=dict(step=step+1,loss=float(loss.detach()),val_mse=float(torch.cat(vloss).mean()),
                     copy_val_mse=float(torch.cat(copy).mean()),lr=lr,gradient_norm=float(gn),seconds=time.time()-start)
            curves.append(rec);print('fit',seed,rec,flush=True)
    return head, dict(seed=seed,steps=steps,train_rows=len(train),val_rows=len(val),curve=curves,
                      seconds=time.time()-start,parameters=sum(p.numel() for p in head.parameters()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','collect','smoke','fit','reference'])
    args=p.parse_args();torch.set_num_threads(4)
    if args.mode=='reference': reference();return
    if args.mode=='check':
        r=checks();(HERE/'checks.json').write_text(json.dumps(r,indent=2));print(r);return
    dev=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=True
    assert torch.cuda.is_available()
    if args.mode=='collect':
        ck=torch.load(CK,map_location='cpu',weights_only=False)
        m=M.M0(ck['cfg']).to(dev);m.load_state_dict(ck['model']);m.eval();m.requires_grad_(False)
        for split,n in (('train',400),('dev',600)):collect(m,ck['cfg'],split,n,dev)
        return
    tr=Data(torch.load(HERE/'train_400_coarse.pt',map_location='cpu',weights_only=False),dev)
    if args.mode=='smoke':
        start=time.time();h,r=fit(tr,611699,25,True);torch.cuda.synchronize()
        r['total_seconds']=time.time()-start;r['peak_GiB']=torch.cuda.max_memory_allocated()/2**30
        (HERE/'smoke.json').write_text(json.dumps(r,indent=2));return
    assert (HERE/'reference.json').exists() and (HERE/'checks.json').exists()
    for seed in (611601,611602):
        head,rec=fit(tr,seed,4000)
        torch.save({'state':head.cpu().state_dict(),'fit':rec,'checkpoint_sha256':tr.z['checkpoint_sha256'],
                    'script_sha256':TR.file_sha256(__file__),'plan_sha256':TR.file_sha256(HERE/'PLAN.md')},HERE/f'head_{seed}.pt')
        (HERE/f'fit_{seed}.json').write_text(json.dumps(rec,indent=2))


if __name__=='__main__':main()
