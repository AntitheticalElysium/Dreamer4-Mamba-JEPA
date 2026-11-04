"""Native, weighted route accounting on matched checkpoint continuation scenes.

No optimizer step. A local Adam differential is not a long-run intervention result.
"""
import os
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse, importlib, json, time
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, data as D, train as TR

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TG=importlib.import_module('artifacts.experiments.20261030_p1_guided.train_guided')
WORLDS={'p1f':ROOT/'artifacts/experiments/20261030_p1_guided/run',
        'f15':ROOT/'artifacts/experiments/20261101_p1f15/run'}
ORIGINAL=M.denoise_loss
CAPTURE={}

def instrument(den,x0,sigma,prev_action,memory,fix,episode,valid,sd=1.,noise=None):
    noise=noise or tuple(torch.randn_like(t) for t in x0)
    pre=[M.precondition(sg,sd) for sg in M._parts(sigma)]
    xs=tuple(t+M._bcast(sg,t)*n for t,n,sg in zip(x0,noise,M._parts(sigma)))
    cn=(pre[0][3],pre[1][3]) if isinstance(sigma,tuple) else pre[0][3]
    fo=den(tuple(M._bcast(c[2],t)*t for t,c in zip(xs,pre)),cn,prev_action,memory,fix,episode,valid)
    se=[((f.float()-(t-M._bcast(c[0],t)*s)/M._bcast(c[1],t))**2).flatten(2).sum(-1)
        for f,t,s,c in zip(fo,x0,xs,pre)]
    dim=sum(t[0,0].numel() for t in x0)
    CAPTURE['parts']=[v/dim for v in se]
    return sum(se)/dim

def module(name):
    if name.startswith('perception.'):
        return 'encoder_fovea' if name.startswith('perception.fovea') else 'encoder_periphery_light'
    if name.startswith('belief.'):
        return 'memory_projection' if name.startswith('belief.memory_out') else 'belief'
    if name.startswith('denoiser.'):
        if any(x in name for x in ('p_in','p_out','g_in','g_out','v_in','v_out')):return 'denoiser_io'
        return 'denoiser_shared'
    return name.split('.')[0]

def norm(g):
    return sum(float(x.detach().float().square().sum()) for x in g if x is not None)**.5

def routes(model,terms,P):
    s=model.harmoniser.log_sigma.exp()
    out={k:v[:,P:].mean()/s[0] for k,v in zip(('g1p','g1g','g1v'),CAPTURE['parts'])}
    out.update({k:terms[k]/s[1] for k in ('change','tsp','rec')})
    out.update({k:terms[k]/s[2] for k in ('reward','cont')})
    out['noise']=terms['noise']
    out['balance']=sum(torch.log1p(s))
    return out

def accounting(model,terms,ck,P):
    named=list(model.named_parameters());params=[p for _,p in named]
    idx={m:[i for i,(n,p) in enumerate(named) if module(n)==m] for m in sorted({module(n) for n,p in named})}
    total=model.harmoniser(TR.groups(terms))+terms['noise']
    get=lambda loss:torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    gt=get(total);route=routes(model,terms,P)
    assert torch.allclose(sum(route.values()),total,atol=2e-5,rtol=1e-6)
    base={m:norm([gt[i] for i in ii]) for m,ii in idx.items()}
    out={'total_loss':float(total.detach()),'total_norm':norm(gt),'module_total_norm':base,'routes':{}}
    summed=[None]*len(params)
    for key,loss in route.items():
        gg=get(loss);stats={}
        for m,ii in idx.items():
            dot=sum(float((gg[i].detach().float()*gt[i].detach().float()).sum())
                    for i in ii if gg[i] is not None and gt[i] is not None)
            stats[m]={'norm':norm([gg[i] for i in ii]),'projection_fraction':dot/max(base[m]**2,1e-30)}
        out['routes'][key]=stats
        for i,g in enumerate(gg):
            if g is not None:summed[i]=g.detach().float().clone() if summed[i] is None else summed[i]+g.detach().float()
    out['gradient_sum_relative_error']=norm([x-g.detach().float() for x,g in zip(summed,gt) if x is not None and g is not None])/max(norm(gt),1e-30)
    # Explicit remaining graph preserves None-versus-zero semantics for AdamW.
    nofine=get(sum(v for k,v in route.items() if k not in ('g1g','tsp','rec','noise')))
    clips=[min(1.,ck['cfg']['optimizer']['grad_clip']/(norm(g)+1e-6)) for g in (gt,nofine)]
    byid={}
    for live,saved in zip(model.param_groups(),ck['opt']['param_groups']):
        for p,pid in zip(live['params'],saved['params']):byid[id(p)]=(saved,ck['opt']['state'].get(pid,{}))
    sums={m:[0.,0.,0.,0.] for m in idx}
    for (name,p),g0,g1 in zip(named,gt,nofine):
        group,state=byid[id(p)]
        us=[]
        for g,clip in zip((g0,g1),clips):
            if g is None:us.append(torch.zeros_like(p,dtype=torch.float32));continue
            g=g.detach().float()*clip
            b1,b2=group['betas'];step=int(state.get('step',0))+1
            m=state.get('exp_avg',torch.zeros_like(p)).to(p.device).float()*b1+(1-b1)*g
            v=state.get('exp_avg_sq',torch.zeros_like(p)).to(p.device).float()*b2+(1-b2)*g.square()
            u=-group['lr']*(m/(1-b1**step))/(v.sqrt()/((1-b2**step)**.5)+group['eps'])
            u=u-group['lr']*group['weight_decay']*p.detach().float()
            us.append(u)
        a,b=us;z=sums[module(name)]
        for j,v in enumerate((a.square().sum(),b.square().sum(),(a-b).square().sum(),(a*b).sum())):z[j]+=float(v)
    out['adam']={'clips':clips,'modules':{m:{'full_norm':a**.5,'without_fine_norm':b**.5,
            'difference_over_full':(d/max(a,1e-30))**.5,'cosine':dot/max((a*b)**.5,1e-30)}
            for m,(a,b,d,dot) in sums.items()}}
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('--blocks',type=int,default=8);a.add_argument('--only');args=a.parse_args()
    torch.set_num_threads(4);dev=torch.device('cuda')
    dest=HERE/'gradients';dest.mkdir(exist_ok=True)
    result={'script_sha256':TR.file_sha256(__file__),'plan_sha256':TR.file_sha256(HERE/'PLAN.md'),'cases':{}}
    for k in (5,10):
        reference=None
        for world,folder in WORLDS.items():
            tag=f'{world}{k:02d}'
            if args.only and tag not in args.only.split(','):continue
            path=folder/f'ckpt_{k:02d}.pt';ck=torch.load(path,weights_only=False,map_location='cpu');cfg=ck['cfg']
            split=D.load_split(cfg);off=D.frame_offsets(split['train'])
            kw=dict(seed=ck['world_seed'],targets_offsets=off)
            streams=(TG.FT2.GuidedStreams(split['train'],16,8,place=np.load(TG.M05.PLACE),**kw)
                     if cfg['p1'].get('guided') else D.Streams(split['train'],16,8,**kw))
            streams.load_state(ck['streams']);chunk=streams.next(64)
            if reference is None:reference=chunk
            else:
                for key in chunk:
                    if key!='fixation':assert np.array_equal(reference[key],chunk[key]),key
            targets=np.load(ROOT/cfg['p1']['out_dir']/'targets/change_targets.npy',mmap_mode='r')
            model=M.M0(cfg).to(dev).train();model.load_state_dict(ck['model']);model.denoiser.checkpoint=True
            z=result['cases'][tag]={'checkpoint':str(path),'sha256':TR.file_sha256(path),'blocks':[]}
            for b in range(min(args.blocks,8)):
                start=time.time();rows=np.arange(2*b,2*b+2);c=D.select(chunk,rows)
                carry=tuple(M.MambaCarry(cc.conv[rows].to(dev),cc.ssm[rows].to(dev)) for cc in ck['carry'])
                cache=ck['cache'][rows].to(dev)
                seed=611220+k*100+b
                if b==0:
                    M.denoise_loss=ORIGINAL;torch.manual_seed(seed)
                    old,_,_=TR.p1_losses(model,c,carry,cache,targets,cfg)
                    values={q:float(v.detach()) for q,v in old.items()}
                    oldgrad=torch.autograd.grad(model.harmoniser(TR.groups(old))+old['noise'],tuple(model.parameters()),allow_unused=True)
                    oldgrad=[g.detach().cpu() if g is not None else None for g in oldgrad]
                    del old
                M.denoise_loss=instrument;torch.manual_seed(seed)
                terms,_,_=TR.p1_losses(model,c,carry,cache,targets,cfg)
                if b==0:
                    assert values=={q:float(v.detach()) for q,v in terms.items()},'instrument changed native loss'
                    newgrad=torch.autograd.grad(model.harmoniser(TR.groups(terms))+terms['noise'],tuple(model.parameters()),retain_graph=True,allow_unused=True)
                    error=norm([g.detach().cpu()-o for g,o in zip(newgrad,oldgrad) if g is not None and o is not None])
                    assert error==0.,error
                    z['instrument_full_gradient_max_error']=error
                    del newgrad,oldgrad
                out=accounting(model,terms,ck,8);out.update(block=b,seconds=time.time()-start,
                    fixation_guidance_fraction=float(np.any(reference['fixation'][rows]!=chunk['fixation'][rows],axis=-1)[:,8:].mean()))
                z['blocks'].append(out)
                del terms;CAPTURE.clear()
                (dest/'summary.json').write_text(json.dumps(result,indent=1))
                print(tag,b,'seconds',round(time.time()-start,1),'grad',round(out['total_norm'],4),flush=True)
            M.denoise_loss=ORIGINAL;del model,ck;torch.cuda.empty_cache()

if __name__=='__main__':main()
