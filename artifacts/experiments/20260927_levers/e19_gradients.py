"""Frozen local loss mechanism, with exact generator/copy reconstruction.

All 279 ordinary hits plus 512 deterministic unchanged controls on reused roots.
Differentiate health-token L1 through the ACTUAL mixture, holding the backbone and
candidate tokens fixed; positive generate-logit derivative means L1 descent closes
generation. This is a local endpoint mechanism, not the full training history.
Sweep that same logit and decode single-token replacements through the unchanged
true-fitted HUD readout; exact reconstruction and finite differences are controls.
Every selected batch resumes under a checkpoint/data/source-bound contract.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e19_diagnose as D
T,E,R,H=D.T,D.E,D.R,D.H


def hud_coeff(cache,meta,fit,seeds):
    va=torch.isin(meta['seed'],seeds[:len(seeds)//5])
    toks=torch.cat([cache['ctx'][:,-1:],cache['fut']],1)[:,:,63:].float().flatten(0,1).flatten(1)
    vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
    rows=lambda a:a[:,None].expand(-1,17).flatten()
    probe=T.ridge(toks,vis[...,1512:1516].flatten(0,1),rows(fit&~va),rows(va))
    closure=dict(zip(probe.__code__.co_freevars,[c.cell_contents for c in probe.__closure__]))
    scale=dict(zip(closure['xs'].__code__.co_freevars,[c.cell_contents for c in closure['xs'].__closure__]))
    return (closure['w'][:192,0]/scale['sd'][:192]*9).float()


def run(path,meta,cache,frames,actions,masks,shift,selection,coef,device):
    w,st=T.load_world(path,device)
    oldpath=HERE/'evals'/(st['name']+'__e19_router_trace.pt')
    old=torch.load(oldpath,map_location='cpu',weights_only=False)
    store=R.frozen_eval_store(path,st['name']+'__e19_local_gradients',
        {'batch':4,'unchanged':512,'seed':20261005,'sweep':[-8,-4,-2,0,2,4,8],'epsilon':.01},
        {'frames':frames,'actions':actions,'selection':selection,'coef':coef},[oldpath])
    with store.lock():
        result=store.load('result')
        if result is not None:return result
        parts=[]
        for start in range(0,len(selection),4):
            part=store.load('batch_'+str(start))
            if part is None:
                cases=selection[start:start+4];nb=len(cases)
                ss=torch.stack([frames[r,3+k-4:3+k+1]for r,k in cases]).to(device).float()
                aa=torch.stack([actions[r,3+k-4:3+k+1]for r,k in cases]).to(device)
                true=torch.stack([frames[r,4+k,63]for r,k in cases]).to(device).float()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p,h,g=w(ss,aa)
                p=p[:,-1,63].float();g=g[:,-1,63].detach().requires_grad_()
                ww=w.last_weights[:,-1,63].float()
                cur=ss[:,-1];grid=cur.reshape(nb,9,9,192);pad=F.pad(grid,(0,0,1,1,1,1))
                copies=torch.stack([grid]+[pad[:,1+dr:10+dr,1+dc:10+dc]for dr,dc in E.T.NEIGHBOURS],-2).reshape(nb,81,5,192)[:,63]
                logits=ww.clamp_min(1e-30).log().detach().requires_grad_()
                cands=torch.cat([copies,g[:,None]],1)
                def compose(z):return F.layer_norm((z.softmax(-1)[...,None]*cands).sum(1),(192,))
                pred=compose(logits);reconstruction=float((pred-p).abs().max())
                # Re-normalizing logged probabilities can change a final fp32 ulp.
                assert reconstruction<=2e-5,reconstruction
                errors=(pred-true).abs().mean(-1)
                lg,gg=torch.autograd.grad(errors.sum(),(logits,g),retain_graph=True)
                direction=torch.zeros_like(logits);direction[:,5]=.01
                fd=(((compose(logits+direction)-true).abs().mean(-1)-
                     (compose(logits-direction)-true).abs().mean(-1))/.02).detach()
                sweeps={}
                actual_hp=torch.tensor([float(old['delta']['actual'][r,k])for r,k in cases],device=device)
                for offset in (-8,-4,-2,0,2,4,8):
                    z=logits.detach().clone();z[:,5]+=offset
                    new=compose(z).detach()
                    sweeps[str(offset)]={'L1':(new-true).abs().mean(-1).cpu(),
                                        'delta_hp':(actual_hp+((new-p)*coef.to(device)).sum(-1)).cpu()}
                part={'cases':cases,'reconstruction':reconstruction,'L1':errors.detach().cpu(),
                      'generate_logit_gradient':lg[:,5].cpu(),'finite_difference':fd.cpu(),
                      'raw_gen_gradient_norm':gg.norm(dim=-1).cpu(),
                      'gen_weight':ww[:,5].cpu(),'sweeps':sweeps}
                store.save('batch_'+str(start),part,1)
            parts.append(part)
            if start%100==0:print(json.dumps({'name':st['name'],'cases':start}),flush=True)
        raw={k:torch.cat([p[k]for p in parts])for k in ('cases','L1','generate_logit_gradient','finite_difference','raw_gen_gradient_norm','gen_weight')}
        raw['sweeps']={s:{k:torch.cat([p['sweeps'][s][k]for p in parts])for k in ('L1','delta_hp')}for s in parts[0]['sweeps']}
        rr,kk=raw['cases'].T
        hit=masks['drop2'][rr,kk];fresh=masks['adjacent'][rr,kk]&~masks['win'][rr,kk]&~masks['adjwin'][rr,kk]
        groups={'all_hit':hit,'scroll_hit':hit&(shift[rr,kk]!=0),'stationary_hit':hit&(shift[rr,kk]==0),
                'fresh_hit':hit&fresh,'unchanged_control':~hit}
        result={'name':st['name'],'scope':__doc__,'groups':{},'reconstruction_max':max(p['reconstruction']for p in parts),
                'derivative_FD_max_abs':float((raw['generate_logit_gradient']-raw['finite_difference']).abs().max()),
                'source_contract':store.contract}
        for label,mask in groups.items():
            grad=raw['generate_logit_gradient'][mask]
            result['groups'][label]={'n':int(mask.sum()),'L1_mean':float(raw['L1'][mask].mean()),
                'L1_descent_closes_gen':int((grad>0).sum()),'L1_descent_opens_gen':int((grad<0).sum()),
                'gen_logit_gradient_mean':float(grad.mean()),
                'generator_gradient_norm_mean':float(raw['raw_gen_gradient_norm'][mask].mean()),
                'generator_weight_mean':float(raw['gen_weight'][mask].mean()),
                'sweeps':{s:{'L1_mean':float(x['L1'][mask].mean()),'drawn_at_1.5':int((x['delta_hp'][mask]<-1.5).sum())}
                          for s,x in raw['sweeps'].items()}}
        R.atomic_torch(HERE/'evals'/(st['name']+'__e19_local_gradients_raw.pt'),raw)
        R.atomic_json(HERE/'evals'/(st['name']+'__e19_local_gradients.json'),result)
        store.save('result',result,1)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[7,8],default=7)
    p.add_argument('--backbone',choices=['fmamba','full'],default='fmamba')
    args=p.parse_args()
    torch.set_num_threads(3);device=torch.device('cuda');torch.cuda.set_per_process_memory_fraction(.16)
    meta,fit,seeds=T.split();cache=T.build_cache('raw',device);fut,_=D.D.SD.token_cache(device)
    frames=torch.cat([cache['ctx'].cpu(),fut[:,0].cpu()],1)
    actions=torch.cat([cache['ctx_a'].cpu(),cache['fut_a'].cpu()],1)
    masks={k:v[:,0]for k,v in D.D.DR.masks(meta).items()};shift,ok=D.true_shifts(meta)
    valid=masks['valid']&masks['k3'];assert(ok|~valid).all()
    hits=(valid&masks['drop2']).nonzero();neg=(valid&(masks['dh']==0)).nonzero()
    neg=neg[torch.randperm(len(neg),generator=torch.Generator().manual_seed(20261005))[:512]]
    selection=torch.cat([hits,neg]);coef=hud_coeff(cache,meta,fit,seeds)
    for arm in ('C','B'):
        path=E.T.OUT/f'e19_{arm}_s{args.seed}_{args.backbone}_from36000.pt'
        print(json.dumps(run(path,meta,cache,frames,actions,masks,shift,selection,coef,device),indent=2),flush=True)
        torch.cuda.empty_cache()
