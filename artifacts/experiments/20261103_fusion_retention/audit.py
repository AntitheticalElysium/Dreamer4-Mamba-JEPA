"""Approved frozen retention audit; PLAN.md. Only diagnostic predictors train."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import argparse
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from saccade import model as M, train as TR, data as D
from saccade.eval import s1 as S, gates as G

P=importlib.import_module('artifacts.experiments.20261102_foveal_prediction.predict')
A=importlib.import_module('artifacts.experiments.20261102_projection_audit.audit')
C=importlib.import_module('artifacts.experiments.20261102_foveal_information.conditional')
U=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.uptake_local')
W=importlib.import_module('artifacts.experiments.20261103_p1f16_review.local_uptake')
HERE=Path(__file__).resolve().parent
DEV=torch.device('cuda')
MODES=('base','fine','joint','mixed','predicted','memory')
SEEDS=(611701,611702)
SHAS={'p1f15':'95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128',
      'p1f16':'34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'}


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def load_world(name):
    ck=W.WORLDS[name];assert TR.file_sha256(ck)==SHAS[name]
    x=torch.load(ck,map_location='cpu',weights_only=False)
    m=M.M0(x['cfg']).to(DEV);m.load_state_dict(x['model']);m.eval().requires_grad_(False)
    return m,x['cfg']


@torch.no_grad()
def collect(m,cfg,name,split):
    out=HERE/name;out.mkdir(exist_ok=True);path=out/f'{split}.pt'
    if path.exists():
        x=torch.load(path,map_location='cpu',weights_only=False);assert x['checkpoint_sha256']==SHAS[name]
        return x['current'],x['future']
    if name=='p1f15':
        z,t=A.load(split)
        assert t['checkpoint_sha256']==SHAS[name]
        return z,t
    L=S.first_episodes(dict(np.load(S.LABELS/f's0_{split}_labels.npz')),400)
    enc=S.encode(m,L,cfg);grids=G.episode_grids(L);eps=S.episodes(L)
    fields={k:[] for k in ('p','g','v','prior','corrected','predicted','action','prev_action','reward','terminated','fix','truth','episode','t','hist','localg','night')}
    nxt={k:[] for k in ('p','g','scroll','monitor','eps')};off=0;so=np.r_[0,np.cumsum(L['ep_T']+1)]
    monitor={(ep.shard,ep.slot) for ep in D.load_split(cfg)['monitor']}
    for e,q in enumerate(enc):
        T=eps[e].T;assert len(q['p'])==T+1
        cells=M.window(q['fix'][:T]);lg=torch.full((T,63),-1,dtype=torch.int16)
        for j in range(9):
            ok=cells[:,j]<63;lg[torch.arange(T)[ok],cells[ok,j]]=j
        hist=torch.arange(T)[:,None]-torch.arange(8,0,-1)[None]
        hist=torch.where(hist>=0,hist+off,-1)
        vals={k:q[k][:T] for k in ('p','g','v','action','prev_action','reward','terminated','fix')}
        vals.update(prior=q['bp'][:T],corrected=q['b'][:T],predicted=q['bp'][1:T+1],
            truth=torch.as_tensor(grids[e]['grid'][:T].reshape(T,63)).long(),episode=torch.full((T,),e),t=torch.arange(T),hist=hist,localg=lg,
            night=torch.as_tensor(L['s_light'][so[e]:so[e]+T]<.5))
        for k,v in vals.items():fields[k].append(v)
        raw=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'];ng=[]
        for st in range(0,T,512):
            fr=raw[st+1:min(T,st+512)+1].to(DEV);fix=q['fix'][st:st+len(fr)].to(DEV)
            with TR.autocast(DEV,cfg):pp=m.perception(fr,fix)
            ng.append(pp.g.half().cpu())
        pos=L['s_pos'][so[e]:so[e]+T+1]
        for k,v in dict(p=q['p'][1:T+1],g=torch.cat(ng),scroll=torch.as_tensor(np.any(pos[1:]!=pos[:-1],axis=1)),
            monitor=torch.full((T,),(eps[e].shard,eps[e].slot) in monitor,dtype=torch.bool),eps=torch.full((T,),float(L['ep_eps'][e]))).items():nxt[k].append(v)
        off+=T
    z={k:torch.cat(v) for k,v in fields.items()};z['episode_keys']=np.stack([L['ep_shard'],L['ep_slot']],1)
    t={k:torch.cat(v) for k,v in nxt.items()};t['checkpoint_sha256']=SHAS[name]
    torch.save(dict(current=z,future=t,checkpoint_sha256=SHAS[name]),path)
    return z,t


def parity(z,t,split):
    old,ot=A.load(split)
    for k in ('episode','t','fix','action','prev_action','reward','terminated','truth','hist','night','localg'):
        assert torch.equal(z[k],old[k]),('physical/cache parity',split,k)
    for k in ('monitor','scroll','eps'):assert torch.equal(t[k],ot[k]),('target metadata',split,k)
    assert np.array_equal(z['episode_keys'],old['episode_keys'])


@torch.no_grad()
def transforms(m,cfg,z):
    fields={k:[] for k in ('joint','mixed')}
    for st in range(0,len(z['p']),512):
        sl=slice(st,st+512);pp=M.Percept(*(z[k][sl].to(DEV).float() for k in ('p','g','v')),z['fix'][sl].to(DEV))
        with TR.autocast(DEV,cfg):
            joint=pp.g if m.belief.spatial_fusion is None else m.belief.spatial_fusion(pp,M.window(pp.fix))
            mixed=m.belief.percept_embed(pp)
        fields['joint'].append(joint.flatten(1).float().cpu());fields['mixed'].append(mixed.float().cpu())
    return {k:torch.cat(v) for k,v in fields.items()}


def supplement(feat,extra,mode):
    if mode=='base':return None
    if mode=='fine':return feat.z['g'].flatten(1).float()
    if mode=='memory':return feat.memory.float()
    if mode=='predicted':return feat.z['predicted'].float()
    return extra[mode].to(DEV)


def predictive_fit(feat,extra,mode,seed,steps):
    torch.manual_seed(seed);net=A.Fine(feat.dim).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    ids=torch.where((feat.z['episode']<360)&~feat.target['monitor'])[0]
    val=torch.where((feat.z['episode']>=360)&~feat.target['monitor'])[0]
    s=supplement(feat,extra,mode);gen=torch.Generator(device=DEV).manual_seed(seed+10);curve=[]
    for step in range(steps):
        ix=ids[torch.randint(len(ids),(512,),device=DEV,generator=gen)]
        pred=net(*A.inputs(feat,ix,s));loss=(pred-feat.target['g'][ix].float()).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%600==0 or step==steps-1:
            curve.append(dict(step=step+1,loss=float(loss.detach()),validation=float(A.evaluate(net,feat,s,val[:2048]).mean()),gradient_norm=float(gn)))
            print('predictive',seed,mode,curve[-1],flush=True)
    return net,curve


class SemanticFeatures(C.Features):
    def __init__(self,z,m,extra):
        self.extra=extra
        super().__init__(z,m)
    @classmethod
    def share(cls,feat,extra):
        # Both instruments use the identical physical history. Share immutable
        # tensors rather than duplicate the entire observation bank on the GPU.
        obj=object.__new__(cls);obj.extra={k:v.to(DEV) for k,v in extra.items()}
        obj.z,obj.n,obj.history,obj.hist,obj.memory=feat.z,feat.n,feat.history,feat.hidx,feat.memory
        obj.identity=torch.eye(81,device=DEV)
        obj.dim=obj.make(torch.arange(2,device=DEV),'base')[0].shape[-1]
        return obj
    def make(self,ix,mode):
        base,side,cell=super().make(ix,'base');f=ix//63;lg=self.z['localg'][f,cell].long();valid=lg>=0
        if mode in ('fine','joint'):
            g=self.z['g'] if mode=='fine' else self.extra['joint'].view(-1,9,64)
            side[:,:64]=g[f,lg.clamp(min=0)].float()*valid[:,None]
        elif mode=='mixed':side[:,:256]=self.extra['mixed'][f]
        elif mode=='predicted':side[:,:256]=self.z['predicted'][f].float()
        elif mode=='memory':side=self.memory[f].float()
        return base,side,cell


def semantic_fit(feat,monitor,mode,seed,steps):
    torch.manual_seed(seed);net=C.Reader(feat.dim,'gated').to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    z=feat.z;ok=((z['episode']<360)&~monitor)[:,None].expand(-1,63);inside=z['localg']>=0
    pops=[torch.where((ok&(z['truth']==k)&(inside==a)).flatten())[0] for k in range(5) for a in (True,False)]
    assert min(map(len,pops))>0
    gen=torch.Generator(device=DEV).manual_seed(seed+10);curve=[]
    for step in range(steps):
        ix=torch.cat([p[torch.randint(len(p),(128,),device=DEV,generator=gen)] for p in pops])
        loss=F.cross_entropy(net(feat.make(ix,mode)),z['truth'].flatten()[ix])
        opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step%600==0 or step==steps-1:
            curve.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(gn)));print('semantic',seed,mode,curve[-1],flush=True)
    net.eval()
    # Flattened indices, preserving the independent validation episodes.
    val=torch.where((((z['episode']>=360)&~monitor)[:,None].expand(-1,63)&inside).flatten())[0]
    conf=torch.zeros(5,5,device=DEV,dtype=torch.long)
    with torch.no_grad():
        for ix in val.split(2048):
            yh=net(feat.make(ix,mode)).argmax(-1);y=z['truth'].flatten()[ix]
            conf+=torch.bincount(y*5+yh,minlength=25).reshape(5,5)
    return net,dict(curve=curve,validation_confusion=conf.cpu().tolist(),
                    validation_recall=(conf.diag()/conf.sum(1).clamp(min=1)).cpu().tolist())


def unique_modes(name):return [k for k in MODES if not(name=='p1f15' and k=='joint')]


def checks(m,cfg,z,t,feat,extra,name):
    parity(z,t,'train');ix=torch.tensor([2,20,40],device=DEV)
    before=[q.clone() for q in feat.make(ix,'base')];g=feat.target['g'][ix].clone();feat.target['g'][ix]=900
    assert all(torch.equal(a,b) for a,b in zip(before,feat.make(ix,'base')));feat.target['g'][ix]=g
    assert all(not p.requires_grad for p in m.parameters())
    if name=='p1f15':assert torch.equal(extra['joint'],z['g'].flatten(1).float())
    with TR.autocast(DEV,cfg):
        pp=M.Percept(*(z[k][:512].to(DEV).float() for k in ('p','g','v')),z['fix'][:512].to(DEV))
        mixed=m.belief.percept_embed(pp).float()
    assert torch.equal(mixed.cpu(),extra['mixed'][:512])
    # Query and observed fixation are separate; outside-crop evidence is zero.
    sf=SemanticFeatures.share(feat,extra);qq=torch.tensor([0,20,40],device=DEV)
    b,s,c=sf.make(qq,'fine');available=sf.z['localg'][qq//63,c]>=0
    assert s[~available].count_nonzero()==0
    # Stage input reaches a nonzero diagnostic readout, not merely a zero head.
    net=A.Fine(feat.dim).to(DEV);b,s=A.inputs(feat,ix,supplement(feat,extra,'fine'));s=s.detach().requires_grad_()
    loss=net(b,s).square().mean();grad=torch.autograd.grad(loss,s)[0];assert grad[:,:576].abs().sum()>0
    del sf,net
    return dict(physical_episode_time_gaze_parity=True,future_target_excluded=True,world_frozen=True,
                actual_complete_mixed_input=True,outside_crop_fine_zero=True,diagnostic_gradient_connected=True,
                baseline_dimension=feat.dim)


@torch.no_grad()
def paired(m,cfg,name,z,extra,states,smoke=False):
    source,weights=W.fixed_cohort();out=HERE/name
    if smoke:
        source={k:v[:2] for k,v in source.items()};out=out/'smoke_pair';out.mkdir(exist_ok=True)
    feat=SemanticFeatures(z,m,{k:v.to(DEV) for k,v in extra.items()})
    old=np.load(U.HERE/'uptake_local/raw.npz');lookup={(int(e),int(t)):i for i,(e,t) in enumerate(zip(old['episode'],old['t']))}
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250);eps=S.episodes(L)
    nets={}
    for key,state in states.items():
        r=C.Reader(feat.dim,'gated').to(DEV);r.load_state_dict(state);r.eval();nets[key]=r
    records=[];logps={k:[] for k in nets};gaze=[];max_replay=0.
    offsets=np.r_[0,np.cumsum(L['ep_T'])]
    for e in sorted(set(source['episode'].tolist())):
        ids=torch.where(feat.z['episode']==e)[0];at=0;carry=m.belief.initial(1,DEV)
        with TR.autocast(DEV,cfg):
            pp=M.Percept(*(feat.z[k][ids].float()[None] for k in ('p','g','v')),feat.z['fix'][ids][None])
            toks=m.belief.tokens(m.belief.percept_embed(pp),feat.z['reward'][ids][None],feat.z['terminated'][ids][None],feat.z['action'][ids][None])
            for si in np.where(source['episode']==e)[0]:
                t0=int(source['t'][si]);f=int(offsets[e]+t0)
                _,carry=m.belief(toks[:,at:2*t0],carry);at=2*t0
                obs=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'][t0].to(DEV)
                pc=m.perception(obs[None].expand(49,-1,-1,-1),U.CENTRES)
                joint=pc.g if m.belief.spatial_fusion is None else m.belief.spatial_fusion(pc,U.WIN)
                emb=m.belief.percept_embed(pc)
                tok=m.belief.tokens(emb[:,None],feat.z['reward'][f].expand(49,1),feat.z['terminated'][f].expand(49,1),feat.z['action'][f].expand(49,1))
                yy,_=m.belief(tok,U.expanded(carry,49));pred=yy[:,1].float();mem=m.belief.memory(pred).float()
                rec=int(((U.CENTRES==feat.z['fix'][f]).all(-1)).nonzero()[0]);err=float((pred[rec]-feat.z['predicted'][f].float()).square().mean().sqrt())
                max_replay=max(max_replay,err);assert err<.03,('paired prefix replay',name,e,t0,err)
                src=np.unique(W.A.WIN[source['useful_cells'][si]])
                for cell in src:
                    # Same source-specific row-matched controls as original study.
                    raw_i=lookup[(int(e),t0)];info=(old['useful_cells'][raw_i]&(W.A.WIN==cell)).any(-1)
                    pair=W.A.weights(info,~old['held_mob'][raw_i])
                    if pair is None:continue
                    avail=(U.WIN==int(cell)).any(-1);lg=(U.WIN==int(cell)).long().argmax(-1)
                    ix=torch.full((49,),f*63+int(cell),device=DEV,dtype=torch.long)
                    base,_,query=feat.make(ix,'base');base[:,-3:-1]=U.CENTRES.float()/8;base[:,-1]=avail.float()
                    for key,net in nets.items():
                        mode=key.rsplit('_',1)[0];s=torch.zeros(49,1024,device=DEV)
                        if mode in ('fine','joint'):
                            x=pc.g if mode=='fine' else joint;s[:,:64]=x[torch.arange(49,device=DEV),lg].float()*avail[:,None]
                        elif mode=='mixed':s[:,:256]=emb.float()
                        elif mode=='predicted':s[:,:256]=pred
                        elif mode=='memory':s=mem.flatten(1)
                        with torch.autocast('cuda',enabled=False):
                            logps[key].append(net((base.float(),s.float(),query)).float().log_softmax(-1).cpu())
                    records.append(dict(episode=int(e),t=t0,cell=int(cell),truth=int(source['truth'][si,cell]),a=pair[0],b=pair[1]))
        print('paired',name,'episode',e,'queries',len(records),'replay',max_replay,flush=True)
    meta={k:np.stack([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(out/'paired_meta.npz',**meta)
    torch.save({k:torch.stack(v) for k,v in logps.items()},out/'paired_logp.pt')
    return dict(queries=len(records),scenes=len(source['t']),prefix_rms_max=max_replay)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('smoke','run'));args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.time();(HERE/'STATUS.json').write_text(json.dumps(dict(phase=args.stage,started=time.time())))
    for name in SHAS:
        m,cfg=load_world(name);out=HERE/name;out.mkdir(exist_ok=True)
        z,t=collect(m,cfg,name,'train');parity(z,t,'train');extra=transforms(m,cfg,z)
        feat=P.Features(z,t,m,DEV);write(out/'checks.json',checks(m,cfg,z,t,feat,extra,name))
        if args.stage=='smoke':
            net,curve=predictive_fit(feat,extra,'joint',611799,3);del net,feat
            sf=SemanticFeatures(z,m,{k:v.to(DEV) for k,v in extra.items()})
            net,sc=semantic_fit(sf,t['monitor'].to(DEV),'joint',611799,3)
            dummy={f'{mode}_611799':{k:v.detach().cpu() for k,v in net.state_dict().items()} for mode in ('base','fine','joint','mixed','predicted','memory')}
            del sf,net,extra,z,t;torch.cuda.empty_cache()
            dz,dt=collect(m,cfg,name,'dev');parity(dz,dt,'dev');ex=transforms(m,cfg,dz)
            pr=paired(m,cfg,name,dz,ex,dummy,smoke=True)
            write(out/'smoke.json',dict(predictive=curve,semantic=sc,paired=pr));del ex,dz,dt,m,dummy;torch.cuda.empty_cache();continue
        predicted={};semantic={};fitlog={}
        for seed in SEEDS:
            for mode in unique_modes(name):
                key=f'{mode}_{seed}';net,curve=predictive_fit(feat,extra,mode,seed,2400)
                predicted[key]={k:v.detach().cpu() for k,v in net.state_dict().items()};fitlog['predictive_'+key]=curve;del net
                torch.save(predicted,out/'predictors.pt');write(out/'fit.json',fitlog)
        del feat;torch.cuda.empty_cache()
        sf=SemanticFeatures(z,m,{k:v.to(DEV) for k,v in extra.items()})
        for seed in SEEDS:
            for mode in unique_modes(name):
                key=f'{mode}_{seed}';net,curve=semantic_fit(sf,t['monitor'].to(DEV),mode,seed,2400)
                semantic[key]={k:v.detach().cpu() for k,v in net.state_dict().items()};fitlog['semantic_'+key]=curve;del net
                torch.save(semantic,out/'semantic.pt');write(out/'fit.json',fitlog)
        del sf,extra,z,t;torch.cuda.empty_cache()
        z,t=collect(m,cfg,name,'dev');parity(z,t,'dev');extra=transforms(m,cfg,z)
        feat=P.Features(z,t,m,DEV);torch.save(A.masks(z,t),out/'metadata.pt')
        torch.save((z['g'].float()-t['g'].float()).square().mean(-1),out/'copy_errors.pt')
        for key,state in predicted.items():
            mode=key.rsplit('_',1)[0];net=A.Fine(feat.dim).to(DEV);net.load_state_dict(state);net.eval()
            s=supplement(feat,extra,mode);err=A.evaluate(net,feat,s,torch.arange(feat.n,device=DEV))
            torch.save(err,out/f'errors_{key}.pt');del net,s
        if name=='p1f15':
            for seed in SEEDS:
                for mode in ('base','fine'):
                    new=torch.load(out/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False)
                    old=torch.load(A.HERE/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False)
                    value=float((new-old).abs().max());print('REPLICATION',mode,seed,value,flush=True)
                    assert value<1e-6,(mode,seed,value)
        del feat;torch.cuda.empty_cache()
        pr=paired(m,cfg,name,z,extra,semantic)
        assert all(not p.requires_grad for p in m.parameters());assert TR.file_sha256(W.WORLDS[name])==SHAS[name]
        write(out/'identity.json',dict(checkpoint_sha256=SHAS[name],paired=pr,seconds=time.time()-start))
        del m,z,t,extra,predicted,semantic;torch.cuda.empty_cache()
    write(HERE/'STATUS.json',dict(phase='smoke_complete' if args.stage=='smoke' else 'fits_complete',seconds=time.time()-start))


if __name__=='__main__':main()
