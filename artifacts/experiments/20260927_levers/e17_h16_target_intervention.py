"""Matched frozen E17 trajectory-reader target diagnosis, declared 2026-10-07.

Two frozen Mamba worlds (seeds7/8), exact original 768 pooled features and FIT
normalization, Hazard(768,16), head seeds0/1/2, AdamW1e-3/wd1e-4,4000 updates.
All four arms use identical initial weights and batches32 FIT roots x17 actions.
conditional: original alive-weighted per-step conditional-hazard BCE.
cumulative: H16 cumulative Bernoulli likelihood, same additive hazard model.
rank: H16 within-root pairwise loss on total negative log survival.
rank8: same H16 ranking target/model, but only steps1-8 contribute to the score.
DEV-A selects every200; DEV-B is never used for selection. Primary contrasts:
cumulative-conditional, rank-conditional, rank-rank8, each versus FIT-chosen DOWN.
This isolates reader supervision/access, NOT world dynamics or actor training.
Reused DEV-B panel is exploratory; rank energies are not calibrated probabilities,
and cumulative-only training does not identify per-step death timing.
Original versus grouped conditional is not a loss-only contrast (batch grouping).
Each head's own action choice is averaged, not an ensemble policy. Paired4000
episode-cluster bootstrap; intervals conditional on these two trained worlds.
Exact per-update sampler ledger, full optimizer/model/best/RNG checkpoint every200,
exclusive atomic source/input/runtime-bound resume. CPU threads2, CUDA cap12%.
Logs go to EDA; this script never edits NOTEBOOK. --smoke checks actual optimizer
resume identity and the cumulative-likelihood formula/finite gradients on CPU.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import h16_resume as R
import check_h16_traj as H
import check_h16_signal as HS
import e17_h16_diagnose as D

HERE = Path(__file__).resolve().parent
ARMS = ('conditional', 'cumulative', 'rank', 'rank8')
STEPS = 4000


def score(z, arm):
    return F.softplus(z[..., :8] if arm == 'rank8' else z).sum(-1)


def objective(z, P, arm):
    if arm == 'conditional':
        prev = F.pad(P, (1, 0))[..., :-1]
        alive = (1 - prev).clamp_min(0)
        q = ((P - prev) / alive.clamp_min(1e-6)).clamp(0, 1)
        return (alive * F.binary_cross_entropy_with_logits(z, q, reduction='none')).mean()
    s, y = score(z, arm), P[..., -1]
    if arm == 'cumulative':
        # log P(survive)=-s; log P(dead)=log(1-exp(-s)), stable at both extremes.
        return ((1 - y) * s - y * torch.log((-torch.expm1(-s)).clamp_min(1e-30))).mean()
    w = (y[:, None, :] - y[:, :, None]).clamp_min(0)
    return (w * F.softplus(s[:, :, None] - s[:, None, :])).sum() / w.sum().clamp_min(1e-9)


def advance(model, opt, generator, x, p, arm, ledger, start, stop):
    for step in range(start, stop):
        idx = torch.randint(len(x), (32,), generator=generator)
        ledger[step] = idx
        z = model(x[idx].float().flatten(0, 1)).view(32, 17, 16)
        loss = objective(z, p[idx], arm)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def smoke():
    torch.set_num_threads(2)
    torch.manual_seed(90)
    x = torch.randn(40, 17, 16, 8)
    increments = torch.rand(40, 17, 16) * .06
    p = 1 - torch.exp(-increments.cumsum(-1))
    reports = {}
    for arm in ARMS:
        torch.manual_seed(2)
        m = H.Hazard(8, 16)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(2)
        ledger = torch.full((4, 32), -1, dtype=torch.long)
        advance(m, opt, g, x, p, arm, ledger, 0, 2)
        saved = copy.deepcopy({'model': m.state_dict(), 'optimizer': opt.state_dict(),
                               'generator': g.get_state(), 'rng': torch.get_rng_state(), 'ledger': ledger})
        advance(m, opt, g, x, p, arm, ledger, 2, 4)
        m2 = H.Hazard(8, 16)
        o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=1e-4)
        m2.load_state_dict(saved['model']); o2.load_state_dict(saved['optimizer'])
        g2 = torch.Generator(); g2.set_state(saved['generator']); torch.set_rng_state(saved['rng'])
        l2 = saved['ledger']; advance(m2, o2, g2, x, p, arm, l2, 2, 4)
        assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m.state_dict().items())
        assert torch.equal(ledger, l2) and torch.equal(g.get_state(), g2.get_state())
        reports[arm] = {'resume_exact': True, 'final_loss': float(objective(m(x[:2].flatten(0, 1)).view(2,17,16),p[:2],arm).detach())}
    z = (torch.randn(3,17,16,dtype=torch.float64)-3).requires_grad_()
    y = p[:3].double()
    ref = F.binary_cross_entropy(1 - torch.exp(-score(z, 'cumulative')), y[..., -1])
    val = objective(z,y,'cumulative')
    assert torch.allclose(val, ref, atol=1e-6)
    val.backward(); assert torch.isfinite(z.grad).all()
    for scalar in (-100., 100.):
        zz = torch.full((2,17,16),scalar,requires_grad=True)
        ll = objective(zz,p[:2],'cumulative'); ll.backward()
        assert torch.isfinite(ll) and torch.isfinite(zz.grad).all()
    result = {'arms': reports, 'cumulative_probability_bce_error': float((val-ref).abs().detach()),
              'extreme_loss_gradients_finite': True, 'sources': {__file__:R.file_hash(__file__), H.__file__:R.file_hash(H.__file__)}}
    R.atomic_json(HERE/'evals/e17_h16_target_smoke.json',result)
    print(json.dumps(result),flush=True)


@torch.no_grad()
def scores(model, features, norm, indices, arm, device):
    pieces=[]
    for i in range(0,len(indices),16):
        idx=indices[i:i+16]
        x=((features[idx].float()-norm['mu'])/norm['sd']).half().float().to(device)
        z=model(x.flatten(0,1)).view(len(idx),17,16)
        pieces.append(score(z,arm).cpu())
    return torch.cat(pieces)


def selected_safe(s,p):
    return 1-p[torch.arange(len(p)),s.argmin(1),-1]


def main(device):
    torch.set_num_threads(2)
    device=torch.device(device)
    if device.type=='cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.12)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    final_dir=HERE/'evals/resume/e17_h16_final'
    fc=json.loads((final_dir/'contract.json').read_text())
    final=R.Store(final_dir,fc)
    assert final.load('result') is not None
    sources={str(Path(f).resolve()):R.file_hash(f) for f in (__file__,H.__file__,HS.__file__,D.__file__,R.__file__)}
    for f,h in fc['sources'].items(): assert R.file_hash(f)==h,f
    meta={s:torch.load(D.CACHE/f'{s}_meta.pt',mmap=True,weights_only=False) for s in ('fit','dev')}
    P,_,split=HS.load(full=True)
    labels={'fit':P[split==0],'dev':P[split==1]}
    inputs={str(D.CACHE/f'{s}_meta.pt'):R.file_hash(D.CACHE/f'{s}_meta.pt') for s in meta}
    worlds={}
    for seed in (7,8):
        name=f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000'
        ds=list(D.RESUME.glob(name+'__w15__det__*')); assert len(ds)==1
        path=ds[0]; c=json.loads((path/'contract.json').read_text())
        for kind in ('sources','checkpoints'):
            for f,h in c[kind].items(): assert R.file_hash(f)==h,f
        official=R.Store(path,c).load('result'); assert official is not None
        for sp in ('fit','dev'):
            f=path/f'features_{sp}.f16'
            inputs[str(f)]=R.file_hash(f)
            assert fc['inputs'][str(f)]==inputs[str(f)]
        nr=json.loads((final_dir/f'mamba{seed}_norm.json').read_text())
        inputs[str(final_dir/nr['file'])]=R.file_hash(final_dir/nr['file'])
        worlds[seed]=(path,c,official,final.load(f'mamba{seed}_norm'))
    for s in meta:
        assert len(labels[s])==len(meta[s]['seed'])
        for k in (1,4,16): assert torch.equal(labels[s][...,k-1],meta[s][f'p{k}'])
    spec={'scope':__doc__,'sources':sources,'inputs':inputs,
          'label_hashes':{s:R.tensor_hash(v) for s,v in labels.items()},
          'worlds':{s:v[1] for s,v in worlds.items()},'final_contract':final.contract,
          'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'threads':2,'device':str(device),
                     'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
                     'tf32':False,'deterministic_cudnn':True,'cuda_cap':.12 if device.type=='cuda' else None},
          'protocol':{'updates':STEPS,'batch_roots':32,'actions':17,'arms':ARMS,'head_seeds':[0,1,2],
                      'checkpoint_every':200,'bootstrap_draws':4000,'bootstrap_seed':20261007}}
    store=R.Store(HERE/'evals/resume/e17_h16_target_intervention',spec)
    with store.lock():
        done=store.load('result')
        if done is not None: print(json.dumps(done),flush=True);return
        ia=torch.where(meta['dev']['seed']%2==0)[0]; ib=torch.where(meta['dev']['seed']%2==1)[0]
        pa,pb=labels['dev'][ia],labels['dev'][ib]
        opa,opb=pa[...,-1].amax(1)>pa[...,-1].amin(1),pb[...,-1].amax(1)>pb[...,-1].amin(1)
        assert int(opb.sum())==1139
        ids=meta['dev']['seed'][ib][opb].numpy()
        fit_opp=labels['fit'][...,-1].amax(1)>labels['fit'][...,-1].amin(1)
        prior=int(labels['fit'][fit_opp,:,-1].mean(0).argmin())
        prior_rows=(1-pb[:,prior,-1])[opb].numpy()
        z=meta['dev']['visible'][ib,1071:1512].reshape(-1,7,9,7)[...,0]
        zombie=(z[:,2,4]+z[:,4,4]+z[:,3,3]+z[:,3,5]>0)[opb].numpy()
        summaries={}; allrows={}; started=time.monotonic()
        for worldseed,(path,c,official,norm) in worlds.items():
            features={}
            for sp in ('fit','dev'):
                prog=json.loads((path/f'features_{sp}.progress.json').read_text())
                assert prog['next']==len(meta[sp]['seed'])
                features[sp]=torch.from_numpy(np.memmap(path/f'features_{sp}.f16',dtype=np.float16,mode='c',shape=prog['layout']['shape']))
            world_results={}; ledgers={}
            for arm in ARMS:
                runs=[]; energies=[]; headrows=[]
                for seed in range(3):
                    key=f'm{worldseed}_{arm}_{seed}'
                    completed=store.load(key+'_result')
                    if completed is None:
                        torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
                        model=H.Hazard(768,16).to(device)
                        init_hash=R.digest({k:R.tensor_hash(v) for k,v in model.state_dict().items()})
                        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
                        best,best_state,start=-1.,None,0
                        ledger=torch.full((STEPS,32),-1,dtype=torch.long)
                        state=store.load(key)
                        if state is not None:
                            assert state['initial_hash']==init_hash
                            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                            best,best_state,start=state['best'],state['best_state'],state['step']
                            ledger=state['ledger'];g.set_state(state['generator']);R.restore_rng(state['rng'],device)
                        model.train()
                        for step in range(start,STEPS):
                            idx=torch.randint(len(labels['fit']),(32,),generator=g);ledger[step]=idx
                            x=((features['fit'][idx].float()-norm['mu'])/norm['sd']).half().float().to(device)
                            p=labels['fit'][idx].to(device)
                            zz=model(x.flatten(0,1)).view(32,17,16)
                            loss=objective(zz,p,arm);assert torch.isfinite(loss)
                            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                            if (step+1)%200==0:
                                model.eval();sa=scores(model,features['dev'],norm,ia,arm,device)
                                value=float(selected_safe(sa,pa)[opa].mean())
                                if value>best:
                                    best=value;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                                store.save(key,{'step':step+1,'model':model.state_dict(),'optimizer':opt.state_dict(),
                                    'best':best,'best_state':best_state,'generator':g.get_state(),'rng':R.rng_state(device),
                                    'ledger':ledger,'initial_hash':init_hash},step+1)
                                print(json.dumps({'world_seed':worldseed,'arm':arm,'head_seed':seed,'update':step+1,
                                    'loss':float(loss.detach()),'devA':value,'best':best,'seconds':round(time.monotonic()-started,1)}),flush=True)
                                model.train()
                        model.load_state_dict(best_state);model.eval()
                        energy=scores(model,features['dev'],norm,ib,arm,device)
                        completed={'devA':best,'energy':energy,'safe':selected_safe(energy,pb)[opb],
                                   'ledger_hash':R.tensor_hash(ledger),'initial_hash':init_hash}
                        store.save(key+'_result',completed,1)
                        del model,opt,x,p,zz,loss
                        if device.type=='cuda':torch.cuda.empty_cache()
                    for pair in ('ledger_hash','initial_hash'):
                        lk=(seed,pair)
                        if lk in ledgers:assert ledgers[lk]==completed[pair],(arm,seed,pair)
                        ledgers[lk]=completed[pair]
                    runs.append(completed['safe']);energies.append(completed['energy']);headrows.append(completed['devA'])
                rows=torch.stack(runs);ss=torch.stack(energies)
                item={'score':float(rows.mean()),'per_head_seed':rows.mean(1).tolist(),'devA_selected':headrows,
                      'zombie':float(rows[:,zombie].mean()),'actions':torch.bincount(ss[:,opb].argmin(-1).flatten(),minlength=17).tolist()}
                if arm in ('conditional','cumulative'):
                    prob=1-torch.exp(-ss)
                    item['p16_brier']=float((prob-pb[...,-1]).square().mean())
                world_results[arm]=item;allrows[f'm{worldseed}_{arm}']=rows.mean(0).numpy()
                store.save(f'm{worldseed}_{arm}_rows',{'safe':rows,'energy':ss},1)
            contrasts={}
            for left,right in (('cumulative','conditional'),('rank','conditional'),('rank','rank8')):
                l,r=allrows[f'm{worldseed}_{left}'],allrows[f'm{worldseed}_{right}']
                contrasts[f'{left}_minus_{right}']={'all':D.paired(l,r,ids),'zombie':D.paired(l[zombie],r[zombie],ids[zombie])}
            for arm in ARMS:
                l=allrows[f'm{worldseed}_{arm}']
                contrasts[f'{arm}_minus_prior']={'all':D.paired(l,prior_rows,ids),'zombie':D.paired(l[zombie],prior_rows[zombie],ids[zombie])}
            summaries[str(worldseed)]={'scores':world_results,'contrasts':contrasts,'matched_initialization_and_batch_ledgers':True}
            print(json.dumps({'completed_world_seed':worldseed,'summary':summaries[str(worldseed)]}),flush=True)
            del features
        result={'scope':__doc__,'contract':spec,'roots':len(ids),'clusters':len(np.unique(ids)),
                'zombie_roots':int(zombie.sum()),'prior':float(prior_rows.mean()),'worlds':summaries}
        store.save('rows',{'safe':allrows,'seed':ids,'prior':prior_rows,'zombie':zombie},1)
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_h16_target_intervention.json',result)
        print(json.dumps({'finished':True,'worlds':summaries}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    if args.smoke:smoke()
    else:main(args.device)
