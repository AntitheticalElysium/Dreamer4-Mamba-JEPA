"""Frozen F4 readout audit. See PLAN.md. collect/evaluate use GPU; fit uses CPU. No canonical mutations."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saccade import model as M, train as TR
from saccade.eval import s1 as S
DC = importlib.import_module('artifacts.experiments.20261107_f4_followups.danger_checks')
CV, RT, IF = DC.CV, DC.RT, DC.CV.IF
HERE = Path(__file__).resolve().parent
CKPT = 'artifacts/experiments/20261106_p1f4/run/ckpt_10.pt'

def save_json(name, x):
    (HERE/name).write_text(json.dumps(x, indent=2))

def select(T):
    rng = np.random.default_rng(110714)
    rows, weights, groups = [], [], []
    for i, (sl, adj, cap) in enumerate([(False, True, 6000), (True, True, 999999),
                                      (False, False, 6000), (True, False, 4000)]):
        ix = np.flatnonzero((T['t'] >= CV.P) & (T['asleep'] == sl) & (T['adjacent'] == adj))
        keep = np.sort(rng.choice(ix, min(cap, len(ix)), replace=False))
        rows.extend(keep); weights.extend([len(ix)/len(keep)]*len(keep)); groups.extend([i]*len(keep))
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(weights)[order], np.array(groups)[order]

def stack(enc, T, rows, key, dt=0):
    return torch.stack([enc[int(T['ep'][j])][key][int(T['t'][j])+dt] for j in rows]).to(CV.DEV)

def context(model, cfg, enc, T, rows, gen):
    cache = model.denoiser.new_cache()
    for dt in range(-CV.P, 0):
        x = tuple(stack(enc,T,rows,k,dt).float()[:,None] for k in ('p','g','v'))
        cache = M.context_pass(model.denoiser,x,stack(enc,T,rows,'prev_action',dt)[:,None],
                               model.belief.memory(stack(enc,T,rows,'bp',dt).float())[:,None],
                               stack(enc,T,rows,'fix',dt)[:,None], cache,cfg,gen)
    return (stack(enc,T,rows,'prev_action')[:,None],
            model.belief.memory(stack(enc,T,rows,'bp').float())[:,None],stack(enc,T,rows,'fix')[:,None],cache)

def physical(model, cfg, args, gen, capture=None):
    den = model.denoiser
    b = len(args[0]); dev = CV.DEV
    p = torch.randn(b,1,81,16,device=dev,generator=gen)
    v = torch.randn(b,1,8,device=dev,generator=gen)
    init = (p.clone(),torch.zeros(b,1,9,64,device=dev),v.clone())
    for i,(signal,tau) in enumerate(M.shortcut_ladder(cfg)[0]):
        code = torch.full((b,1),M.shortcut_code(signal,2,8),device=dev)
        if capture is not None: capture['tag'] = f'native{i}'
        out,_ = den.step((p,init[1],v),code,*args)
        p = p + (out[0].float().clamp(-3,3)-p)/(1-tau)*.25
        v = v + (out[2].float().clamp(-3,3)-v)/(1-tau)*.25
    if capture is not None: capture['tag'] = None
    return p[:,0],v[:,0],init

def gram(x,y,w,fold):
    x,y = x.double().cpu(),y.double().cpu()
    w = torch.as_tensor(w,dtype=torch.float64)
    x = torch.cat([x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)],-1)
    result = []
    for f in range(5):
        m = fold == f
        xx,yy = x[m].flatten(0,-2),y[m].flatten(0,-2)
        ww = w[m].repeat_interleave(x.shape[1])
        result.append((xx.T@(ww[:,None]*xx),xx.T@(ww[:,None]*yy),float((ww[:,None]*yy.square()).sum()),float(ww.sum())))
    return result

def add_stats(stats,key,parts):
    if key not in stats: stats[key] = parts
    else: stats[key] = [tuple(a+b for a,b in zip(p,q)) for p,q in zip(stats[key],parts)]

@torch.no_grad()
def collect(smoke=False):
    torch.set_num_threads(2)
    model,cfg=DC.load(CKPT,4);den=model.denoiser
    L=dict(np.load(CV.LABELS/'s0_dev_labels.npz'))
    if smoke: L=S.first_episodes(L,30)
    enc=S.encode(model,L,cfg);T=CV.transitions(L)
    rows,w,groups=select(T)
    if smoke: rows,w,groups=rows[:256],w[:256],groups[:256]
    ep=T['ep'][rows];fold=RT.folds_of(ep)
    # Fit only the health reader, with the same train400 episode split/protocol.
    Lt=dict(np.load(CV.LABELS/'s0_train400_labels.npz'))
    if smoke: Lt=S.first_episodes(Lt,30)
    et=S.encode(model,Lt,cfg)
    xt=torch.cat([torch.cat([a['p'][:,63],a['v']],1) for a in et]).float().to(CV.DEV)
    yt=torch.as_tensor(Lt['s_body'][:,0].clip(0,9),device=CV.DEV).long()
    held=torch.as_tensor(np.repeat(np.arange(len(Lt['ep_T'])),Lt['ep_T']+1)%10==9,device=CV.DEV)
    xa,xv,ya,yv=xt[~held],xt[held],yt[~held],yt[held]
    with torch.enable_grad():
        reader,log=S.fit(lambda ix,val:(xv if val else xa)[ix],
                        lambda o,ix,val:F.cross_entropy(o['y'],(yv if val else ya)[ix]),len(ya),len(yv),{'y':10},0)
    del et,xt,yt
    src=torch.cat([torch.cat([a['p'][:,63],a['v']],1) for a in enc]).float()
    pr=torch.cat([reader(x.to(CV.DEV))['y'].argmax(1).cpu() for x in src.split(32768)])
    validity=float((pr==torch.as_tensor(L['s_body'][:,0].clip(0,9))).float().mean())
    print('reader validity',validity,flush=True)
    caps,stats,got={}, {}, {}
    original=den.head
    def head(h,c):
        out=original(h,c)
        if caps.get('tag') is not None:
            shift,scale=den.final_mod(F.silu(c[0])).unsqueeze(2).chunk(2,-1)
            norm=den.final_norm(h);mod=norm*(1+scale)+shift
            caps[caps['tag']]={'raw':h[:,0,63].float().cpu(),'norm':norm[:,0,63].float().cpu(),
                               'mod':mod[:,0,63].float().cpu(),'out':out[0][:,0,63].float().cpu(),
                               'all_mod':mod[:,0,:81].float().cpu(),'all_out':out[0][:,0].float().cpu()}
        return out
    den.head=head
    gen=torch.Generator(device=CV.DEV).manual_seed(110714)
    for a in range(0,len(rows),128):
        j=rows[a:a+128]; b=len(j);caps.clear()
        with TR.autocast(CV.DEV,cfg):
            args=context(model,cfg,enc,T,j,gen)
            if a==0:
                gs=gen.get_state(); pp,vv,_=physical(model,cfg,args,gen)
                gen.set_state(gs)
                full=M.sample(den,[(81,16),(9,64),(8,)],*args,cfg,gen)
                assert torch.equal(pp,full[0][:,0]) and torch.equal(vv,full[2][:,0]),'physical sham'
            p,v,init=physical(model,cfg,args,gen,caps)
            truth=stack(enc,T,j,'p').float(); copy=stack(enc,T,j,'p',-1).float()
            code=lambda s: M.shortcut_code(torch.as_tensor(s,device=CV.DEV).expand(b,1),3,8)
            caps['tag']='fine0';f1,_=den.step(init,code(0),*args)
            v1=tuple(o.float()-z for o,z in zip(f1,init))
            z1=tuple(z+vv*.125 for z,vv in zip(init,v1))
            caps['tag']='fine1';f2,_=den.step(z1,code(1),*args)
            teacher=init[0]+.5*(v1[0]+(f2[0].float()-z1[0])/.875)
            signal=torch.randint(0,8,(b,1),device=CV.DEV,generator=gen);tau=signal.float()/8
            tr=(truth[:,None],stack(enc,T,j,'g').float()[:,None],stack(enc,T,j,'v').float()[:,None])
            z=tuple((1-M._bcast(tau,x))*torch.randn(x.shape,device=CV.DEV,generator=gen)+M._bcast(tau,x)*x for x in tr)
            caps['tag']='flow';den.step(z,M.shortcut_code(signal,3,8),*args);caps['tag']=None
            drawn=reader(torch.cat([p[:,63],v],-1))['y'].argmax(1).cpu()
        fw=w[a:a+b]*(.1+.9*tau[:,0].float().cpu().numpy())
        add_stats(stats,'shared',gram(caps['flow']['all_mod'],truth,fw,fold[a:a+b]))
        add_stats(stats,'health',gram(caps['flow']['all_mod'][:,63:64],truth[:,63:64],fw,fold[a:a+b]))
        rec={f'{tag}/{key}':d[key] for tag,d in caps.items() if isinstance(d,dict)
             for key in ('raw','norm','mod','out') if tag.startswith('native') or tag=='fine0'}
        rec.update(teacher=teacher[:,0,63].float().cpu(),truth=truth[:,63].cpu(),copy=copy[:,63].cpu(),
                   drawn=drawn,flow_x=caps['flow']['mod'],flow_y=truth[:,63].cpu(),
                   flow_out=caps['flow']['out'],flow_tau=tau.cpu()[:,0])
        for k,val in rec.items():got.setdefault(k,[]).append(val)
        if a%1024==0:print('capture',a,len(rows),flush=True)
    den.head=original
    output={'states':{k:torch.cat(v) for k,v in got.items()},'stats':stats,'rows':rows,'w':w,'groups':groups,
            'ep':ep,'fold':fold,'old_weight':torch.cat([den.out[0].weight.detach().cpu(),den.out[0].bias.detach().cpu()[:,None]],1).T.double(),
            'T':T,'cfg':cfg,'reader_state':reader.state_dict(),'reader_validity':validity,'reader_log':log,
            'checkpoint_sha256':TR.file_sha256(TR.ROOT/CKPT)}
    target=HERE/('smoke.pt' if smoke else 'capture.pt');torch.save(output,target)
    if not smoke: torch.save(enc,HERE/'enc.pt')
    print('saved',target,flush=True)

def fit():
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False);old=z['old_weight'];fits={}
    for name,st in z['stats'].items():
        fits[name]=[]
        for f in range(5):
            xx,xy,yy,n=tuple(sum(a[i] for ff,a in enumerate(st) if ff!=f) for i in range(4))
            reg=torch.eye(len(xx),dtype=torch.float64)*(1e-4*xx.diag().mean());reg[-1,-1]=0
            delta=torch.linalg.solve(xx+reg,xy-xx@old)
            fits[name].append(old+delta)
    torch.save(fits,HERE/'fits.pt')
    report={}
    for name,st in z['stats'].items():
        report[name]={}
        for arm,ws in {'old':[old]*5,**fits}.items():
            err=0.;nn=0.
            for (xx,xy,yy,n),ww in zip(st,ws):
                err+=float((ww*(xx@ww)).sum()-2*(ww*xy).sum()+yy);nn+=n
            report[name][arm]={'heldout_mse':err/(nn*16),'weight_delta_rms':float(torch.stack([((w-old)**2).mean().sqrt() for w in ws]).mean())}
    save_json('FIT.json',report);print(json.dumps(report,indent=2))

@torch.no_grad()
def evaluate():
    torch.set_num_threads(2)
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False)
    fits=torch.load(HERE/'fits.pt',map_location='cpu',weights_only=False)
    enc=torch.load(HERE/'enc.pt',map_location='cpu',weights_only=False)
    model,cfg=DC.load(CKPT,4);den=model.denoiser
    reader=S.MLP(24,{'y':10}).to(CV.DEV);reader.load_state_dict(z['reader_state']);reader.eval()
    scene=S.load_reader(TR.ROOT/Path(CKPT).parent/'s1/results.reader.pt',TR.ROOT/CKPT,CV.DEV)
    rows,fold,T=z['rows'],z['fold'],z['T'];got={};ctx={'weights':None,'arm':'old'}
    original=den.out[0].forward
    def output(x):
        old=original(x)
        if ctx['weights'] is None:return old
        xx=torch.cat([x.float(),torch.ones(*x.shape[:-1],1,device=x.device)],-1)
        pred=torch.einsum('btnd,bdc->btnc',xx,ctx['weights']).to(old.dtype)
        if ctx['arm']=='health':
            old=old.clone();old[:,:,63]=pred[:,:,63];return old
        return pred
    den.out[0].forward=output
    gen=torch.Generator(device=CV.DEV).manual_seed(110715)
    for a in range(0,len(rows),64):
        jj=rows[a:a+64];j=np.repeat(jj,4);ff=np.repeat(fold[a:a+64],4)
        ctx['weights']=None
        with TR.autocast(CV.DEV,cfg):
            args=context(model,cfg,enc,T,j,gen);gs=gen.get_state()
            for arm in ('old','shared','health'):
                ctx['arm']=arm
                ctx['weights']=None if arm=='old' else torch.stack([fits[arm][f] for f in ff]).float().to(CV.DEV)
                gen.set_state(gs);p,v,_=physical(model,cfg,args,gen)
                digit=reader(torch.cat([p[:,63],v],1))['y'].argmax(1).cpu()
                mp=p[:,:63];sf=torch.cat([mp,v[:,None].expand(-1,63,-1)],-1)
                kinds=scene(sf.flatten(0,1))['kind'].argmax(1).reshape(len(p),63)
                kinds[:,31]=0
                count=(kinds>0).sum(1).cpu()
                mse=(p-stack(enc,T,j,'p').float()).square().mean((1,2)).cpu()
                got.setdefault(arm+'/digit',[]).append(digit.reshape(-1,4));got.setdefault(arm+'/sample_mse',[]).append(mse.reshape(-1,4))
                got.setdefault(arm+'/entity_count',[]).append(count.reshape(-1,4))
        if a%1024==0:print('eval',a,len(rows),flush=True)
    den.out[0].forward=original
    torch.save({k:torch.cat(v) for k,v in got.items()},HERE/'native.pt')

if __name__=='__main__':
    {'collect':lambda:collect('--smoke' in sys.argv),'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
