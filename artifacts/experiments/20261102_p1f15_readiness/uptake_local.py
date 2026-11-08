"""Frozen native useful-glimpse diagnosis; UPTAKE_PROTOCOL.md. No world training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CK=ROOT/'artifacts/experiments/20261101_p1f15/run/ckpt_10.pt'
FP=importlib.import_module('artifacts.experiments.20261029_fovea_probes.measure')
DEV=torch.device('cuda')
CENTRES=FP.CENTRES.to(DEV)
WIN=M.window(CENTRES)


def subset(L, rows):
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    return {k:v[rows] if k.startswith('ep_') else np.concatenate([
        v[(so if k.startswith('s_') else ao)[e]:(so if k.startswith('s_') else ao)[e+1]] for e in rows]) for k,v in L.items()}


def load():
    c=torch.load(CK,map_location='cpu',weights_only=False)
    m=M.M0(c['cfg']).to(DEV);m.load_state_dict(c['model']);m.eval();m.requires_grad_(False)
    return m,c['cfg']


@torch.no_grad()
def reader_data(m,cfg):
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_train_labels.npz')),400)
    en=S.encode(m,L,cfg);grids=G.episode_grids(L)
    fields={k:[] for k in ('p','g','v','emb','b','bp','cell','y','episode')}
    for e,q in enumerate(en):
        n=len(q['p'])-1
        P=M.Percept(*(q[k][:n].float().to(DEV) for k in ('p','g','v')),q['fix'][:n].to(DEV))
        with TR.autocast(DEV,cfg):emb=m.belief.percept_embed(P).float().cpu()
        cells=M.window(q['fix'][:n]);valid=cells<63
        fr=torch.arange(n)[:,None].expand(-1,9)[valid]
        cc=cells[valid]
        for k,val in (('p',q['p'][:n][fr,cc]),('g',q['g'][:n][valid]),('v',q['v'][:n][fr]),
                      ('emb',emb[fr]),('b',q['b'][:n][fr]),('bp',q['bp'][1:n+1][fr]),
                      ('cell',F.one_hot(cc,81).half()),('y',torch.as_tensor(grids[e]['grid'][:n].reshape(n,63))[fr,cc]),
                      ('episode',torch.full((len(fr),),e))):
            fields[k].append(val.half() if k not in ('y','episode') else val)
    return {k:torch.cat(v).to(DEV) for k,v in fields.items()}


def features(z,idx,mode,m):
    pv=[z['p'][idx].float(),z['v'][idx].float()]
    if mode=='g':return torch.cat([z['g'][idx].float(),z['v'][idx].float()],-1)
    if mode=='p':return torch.cat(pv+[z['cell'][idx].float()],-1)
    a=z[mode][idx].float() if mode!='memory' else m.belief.memory(z['bp'][idx].float()).flatten(-2)
    return torch.cat(pv+[z['cell'][idx].float(),a],-1)


def fit_readers(m,cfg,out):
    z=reader_data(m,cfg);train=z['episode']<360;val=~train
    inds=[torch.where(train & (z['y']==k))[0] for k in range(5)]
    assert all(len(i)>0 for i in inds)
    print('reader rows',len(z['y']),[len(i) for i in inds],flush=True)
    stats={};saved={}
    for mode in ('g','p','emb','b','bp','memory'):
        for seed in ((0,1) if mode=='g' else (0,)):
            torch.manual_seed(611301+seed)
            reader=S.MLP(features(z,torch.arange(2,device=DEV),mode,m).shape[-1],{'kind':5}).to(DEV)
            opt=torch.optim.AdamW(reader.parameters(),lr=.001,weight_decay=.0001)
            gen=torch.Generator(device=DEV).manual_seed(611310+seed)
            for step in range(1200):
                ix=torch.cat([a[torch.randint(len(a),(256,),generator=gen,device=DEV)] for a in inds])
                loss=F.cross_entropy(reader(features(z,ix,mode,m))['kind'],z['y'][ix].long())
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            ix=torch.where(val)[0];pred=[]
            with torch.no_grad():
                for a in ix.split(4096):pred.append(reader(features(z,a,mode,m))['kind'].argmax(-1))
            yh=torch.cat(pred);y=z['y'][ix].long();conf=torch.bincount(y*5+yh,minlength=25).reshape(5,5)
            stats[f'{mode}_{seed}']={'confusion':conf.cpu().tolist(),
                'kind_accuracy_by_true_kind':(conf.diag()/conf.sum(1).clamp(min=1)).cpu().tolist(),
                'val_rows':len(ix),'train_kind_rows':[len(i) for i in inds]}
            saved[f'{mode}_{seed}']={'dim':reader.body[0].in_features,'state':{k:v.cpu() for k,v in reader.state_dict().items()}}
            print('reader',mode,seed,stats[f'{mode}_{seed}']['kind_accuracy_by_true_kind'],flush=True)
    torch.save(saved,out/'readers.pt')
    (out/'readers.json').write_text(json.dumps(stats,indent=1))


def expanded(carry,n):
    return tuple(M.MambaCarry(c.conv.expand(n,*c.conv.shape[1:]).contiguous(),
                              c.ssm.expand(n,*c.ssm.shape[1:]).contiguous()) for c in carry)


def expandcache(cache,n):
    return {'t':cache['t'],'kv':[(k.expand(n,*k.shape[1:]).contiguous(),v.expand(n,*v.shape[1:]).contiguous()) for k,v in cache['kv']]}


@torch.no_grad()
def enter(m,cfg,parts,pa,bp,fx,cache,gen):
    # One context noise shared across all valid foveations of a physical frame.
    sc=cfg['edm']['inference_context_sigma'];sd=cfg['edm']['sigma_data']
    xs=tuple(x+sc*torch.randn((1,*x.shape[1:]),device=DEV,generator=gen) for x in parts)
    _,_,cin,cn=M.precondition(torch.full((len(xs[0]),1),sc,device=DEV),sd)
    _,new=m.denoiser.step(tuple(M._bcast(cin,x)*x for x in xs),cn,pa,m.belief.memory(bp),fx,cache)
    return m.denoiser.commit(cache,new)


@torch.no_grad()
def measure(m,cfg,out):
    old=np.load(ROOT/'artifacts/experiments/20261101_p1f15/readings/uptake/uptake.npz')
    use=np.flatnonzero(old['night_set']).tolist()+np.flatnonzero(~old['night_set'])[:256].tolist()
    rows=sorted(set((int(old['e'][i]),int(old['t'][i])) for i in use))
    if os.environ.get('UPTAKE_FINE_ONLY'):
        prev=dict(np.load(out/'raw.npz'));good=prev['useful'].any(1)
        rows=list(zip(prev['episode'][good].tolist(),prev['t'][good].tolist()))
    if os.environ.get('UPTAKE_SMOKE'):rows=rows[:2]
    (out/('future_tsp_rows.json' if os.environ.get('UPTAKE_FINE_ONLY') else 'rows.json')).write_text(json.dumps(rows))
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250)
    es=sorted({e for e,t in rows});sub=subset(L,es);en=S.encode(m,sub,cfg);en=dict(zip(es,en))
    eps=S.episodes(L);grids=G.episode_grids(L);so=np.r_[0,np.cumsum(L['ep_T']+1)]
    pr=S.load_reader(CK.parent/'s1/results.reader.pt',CK,DEV)
    saved=torch.load(out/'readers.pt',weights_only=False,map_location=DEV);fine=[]
    for seed in (0,1):
        z=saved[f'g_{seed}'];r=S.MLP(z['dim'],{'kind':5}).to(DEV);r.load_state_dict(z['state']);r.eval();fine.append(r)
    stage_readers={}
    for mode in ('p','emb','b','bp','memory'):
        z=saved[f'{mode}_0'];r=S.MLP(z['dim'],{'kind':5}).to(DEV);r.load_state_dict(z['state']);r.eval();stage_readers[mode]=r
    sig=FP.levels(cfg);recs=[];checks=[]
    rng=torch.Generator(device=DEV).manual_seed(611330)
    for e in es:
        q=en[e];ts=[t for ee,t in rows if ee==e]
        with TR.autocast(DEV,cfg):
            Pc=M.Percept(*(q[k].to(DEV).float()[None] for k in ('p','g','v')),q['fix'].to(DEV)[None])
            tok=m.belief.tokens(m.belief.percept_embed(Pc),q['reward'].to(DEV)[None],q['terminated'].to(DEV)[None],q['action'].to(DEV)[None])
            carry=m.belief.initial(1,DEV);at=0
            for t in ts:
                _,carry=m.belief(tok[:,at:2*t],carry);at=2*t  # strictly before percept t
                obs=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'][t].to(DEV)
                Pt=m.perception(obs[None].expand(49,-1,-1,-1),CENTRES)
                pv=torch.cat([Pt.p[0,:63].float(),Pt.v[:1].float().expand(63,-1)],-1)
                pk=pr(pv)['kind'].argmax(-1)
                fg=torch.cat([Pt.g.float(),Pt.v[:,None].expand(-1,9,-1).float()],-1)
                gk=torch.stack([r(fg)['kind'].argmax(-1) for r in fine])
                truth=torch.as_tensor(grids[e]['grid'][t].reshape(63),device=DEV).long()
                alltruth=F.pad(truth,(0,18));wg=alltruth[WIN];wp=F.pad(pk,(0,18))[WIN]
                # Fine readability is a positive control for EACH actual candidate crop.
                useful_cell=(wg>0)&(wp!=wg)&(gk[0]==wg)&(gk[1]==wg)
                useful=useful_cell.any(-1);none=(wg==0).all(-1)
                emb=m.belief.percept_embed(Pt)
                tk=m.belief.tokens(emb[:,None],q['reward'][t].to(DEV).expand(49,1),q['terminated'][t].to(DEV).expand(49,1),q['action'][t].to(DEV).expand(49,1))
                yy,_=m.belief(tk,expanded(carry,49));bc=yy[:,0].float();bp=yy[:,1].float();mem=m.belief.memory(bp).float()
                recidx=int(((CENTRES==q['fix'][t].to(DEV)).all(-1)).nonzero()[0])
                checks.append(float((bp[recidx]-q['bp'][t+1].to(DEV).float()).square().mean().sqrt()))
                assert checks[-1]<.03,('prefix replay',checks[-1])
                # Input algebra: value block + gaze mask, with p/v identical.
                Np=81*Pt.p.shape[-1];Ng=81*Pt.g.shape[-1]
                canvas=torch.zeros(49,81,Pt.g.shape[-1],device=DEV,dtype=Pt.g.dtype).scatter(1,WIN[:,:,None].expand_as(Pt.g),Pt.g)
                mask=F.one_hot(WIN,81).sum(1).to(Pt.g.dtype)
                W=m.belief.percept_in.weight.float()
                eg=F.linear(canvas.flatten(1).float(),W[:,Np:Np+Ng]);em=F.linear(mask.float(),W[:,Np+Ng:Np+Ng+81])
                base=torch.cat([Pt.p[:1,:63].float().expand(49,-1,-1),
                    Pt.v[:,None].float().expand(-1,63,-1),torch.eye(81,device=DEV)[:63][None].expand(49,-1,-1)],-1)
                stage_ps=[];stage_correct=[]
                for mode,rr in stage_readers.items():
                    extra={'emb':emb.float(),'b':bc,'bp':bp,'memory':mem.flatten(1)}.get(mode)
                    xx=base if mode=='p' else torch.cat([base,extra[:,None].expand(-1,63,-1)],-1)
                    logits=rr(xx)['kind'].float()
                    stage_ps.append(logits.softmax(-1).gather(-1,truth[None,:,None].expand(49,-1,-1)).squeeze(-1).cpu().numpy())
                    stage_correct.append((logits.argmax(-1)==truth[None]).cpu().numpy())
                # Source-defined neighborhood, BEFORE seeing the next realization.
                src=torch.unique(WIN[useful_cell]);local=torch.zeros(63,device=DEV,dtype=torch.bool)
                for c in src:
                    rr=torch.arange(63,device=DEV)//9;cc=torch.arange(63,device=DEV)%9
                    local|=((rr-c//9).abs()<=2)&((cc-c%9).abs()<=2)
                if not local.any():local[:]=True  # marked unqualified, cannot enter primary estimate
                if os.environ.get('UPTAKE_FINE_ONLY'):
                    assert len(src)>0
                    center=torch.stack([(src[0]//9).clamp(1,7),(src[0]%9).clamp(1,7)])[None].expand(49,-1)
                    obs1=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'][t+1].to(DEV)[None].expand(49,-1,-1,-1)
                    future=m.perception(obs1,center);cells=M.window(center)
                    mu,_=m.pixels(m.tsp.predict(bp,future.p,future.v,cells),future.v,cells)
                    pixels=M.cell_pixels(M.crop(obs1,center)).float()/127.5-1
                    mse=(mu.float()-pixels).square().mean(-1)
                    recs.append({'episode':e,'t':t,'useful':useful.cpu().numpy(),'held_mob':(~none).cpu().numpy(),
                                 'pixel_mse':mse.cpu().numpy(),'query_fix':center[0].cpu().numpy()})
                    continue
                cache=m.denoiser.new_cache()
                for s in range(t-8,t):
                    cache=enter(m,cfg,tuple(q[k][s:s+1].to(DEV).float()[None] for k in ('p','g','v')),
                        q['prev_action'][s:s+1].to(DEV)[None],q['bp'][s:s+1].to(DEV).float()[None],q['fix'][s:s+1].to(DEV)[None],cache,rng)
                cache=expandcache(cache,49)
                cache=enter(m,cfg,tuple(x[:,None] for x in Pt.parts()),q['prev_action'][t].to(DEV).expand(49,1),
                    q['bp'][t].to(DEV).float().expand(49,1,-1),CENTRES[:,None],cache,rng)
                x0=tuple(q[k][t+1:t+2].to(DEV).float()[None].expand(49,-1,*q[k].shape[1:]) for k in ('p','g','v'))
                pa=q['prev_action'][t+1].to(DEV).expand(49,1);fx=q['fix'][t+1].to(DEV).expand(49,1,2)
                losses=[];preds=[]
                for draw in range(2):
                    ls=[]
                    for sigma in sig:
                        xs=tuple(x+sigma*torch.randn((1,*x.shape[1:]),generator=rng,device=DEV) for x in x0)
                        cs,co,ci,cn=M.precondition(torch.full((49,1),sigma,device=DEV),cfg['edm']['sigma_data'])
                        fo,_=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in xs),cn,pa,mem[:,None],fx,cache)
                        se=(fo[0].float()-(x0[0]-M._bcast(cs,x0[0])*xs[0])/M._bcast(co,x0[0])).square()[:,0]
                        ls.append(torch.stack([se[:,:63].mean((1,2)),se[:,:63][:,local].mean((1,2))],-1).cpu().numpy())
                    losses.append(np.stack(ls))
                    hat=cfg['edm']['sampler']['sigma_max']*(1+min(cfg['edm']['sampler']['s_churn']/cfg['edm']['sampler']['steps'],2**.5-1))
                    xs=tuple(hat*torch.randn((1,*x.shape[1:]),generator=rng,device=DEV).expand_as(x) for x in x0)
                    dd,_=M.denoised(m.denoiser,xs,torch.full((49,1),hat,device=DEV),pa,mem[:,None],fx,cache,3.,cfg['edm']['sigma_data'])
                    preds.append(dd[0][:,0,:63].float().cpu().numpy())
                stage={k:v.detach().float().cpu().numpy() for k,v in (('emb',emb),('g_input',eg),('mask_input',em),('corrected',bc),('predicted',bp),('memory',mem.flatten(1)))}
                recs.append({'episode':e,'t':t,'night':float(L['s_light'][so[e]+t])<.5,'useful':useful.cpu().numpy(),
                    'held_mob':(~none).cpu().numpy(),'useful_cells':useful_cell.cpu().numpy(),'loss':np.stack(losses),
                    'prediction':np.stack(preds),'local':local.cpu().numpy(),'stage_prob':np.stack(stage_ps),
                    'stage_correct':np.stack(stage_correct),'truth':truth.cpu().numpy(),**stage})
        if es.index(e)%10==0 or e==es[-1]:
            name='future_tsp.npz' if os.environ.get('UPTAKE_FINE_ONLY') else 'smoke.npz' if os.environ.get('UPTAKE_SMOKE') else 'raw.npz'
            np.savez_compressed(out/name,
                **{k:np.stack([r[k] for r in recs]) for k in recs[0]},sigmas=np.array(sig))
        print('uptake episodes',e,'frames',len(recs),'qualified',sum(r['useful'].any() for r in recs),flush=True)
    (out/('future_tsp_identity.json' if os.environ.get('UPTAKE_FINE_ONLY') else 'identity.json')).write_text(json.dumps({'checkpoint':str(CK),'sha256':TR.file_sha256(CK),
        'script_sha256':TR.file_sha256(__file__),'protocol_sha256':TR.file_sha256(HERE/'UPTAKE_PROTOCOL.md'),
        'prefix_rms_error_max':max(checks),'frames':len(recs)},indent=1))


def main():
    torch.set_num_threads(4);out=HERE/'uptake_local';out.mkdir(exist_ok=True)
    m,cfg=load();start=time.time()
    if not (out/'readers.pt').exists():fit_readers(m,cfg,out)
    measure(m,cfg,out)
    print('seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
