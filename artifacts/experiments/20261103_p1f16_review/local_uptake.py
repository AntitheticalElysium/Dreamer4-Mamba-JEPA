"""Frozen paired native uptake, fixed pre-fusion source cohort; PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import importlib
import json
import sys
from pathlib import Path
import time
import numpy as np
import torch
from saccade import model as M, train as TR, data as D
from saccade.eval import s1 as S, gates as G

U=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.uptake_local')
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')
HERE=Path(__file__).resolve().parent
ROOT=TR.ROOT
WORLDS={'p1f15':ROOT/'artifacts/experiments/20261101_p1f15/run/ckpt_10.pt',
        'p1f16':ROOT/'artifacts/experiments/20261102_p1f16/run/ckpt_10.pt'}


def fixed_cohort():
    z=dict(np.load(U.HERE/'uptake_local/raw.npz'))
    ids=[i for i in range(len(z['episode'])) if A.weights(z['useful'][i],~z['held_mob'][i]) is not None]
    assert len(ids)==58 and z['night'][ids].all()
    w=np.array([A.weights(z['useful'][i],~z['held_mob'][i]) for i in ids])
    return {k:z[k][ids] for k in ('episode','t','local','useful_cells','truth')},w


@torch.no_grad()
def run_world(name,ck,source,weights):
    U.CK=ck
    m,cfg=U.load()
    out=HERE/name;out.mkdir(exist_ok=True)
    if name=='p1f15':readers=U.HERE/'uptake_local/readers.pt'
    else:
        readers=out/'readers.pt'
        if not readers.exists():
            # Disposable supervised instruments; the world is frozen. Explicitly
            # enable only instrument gradients inside this no-grad native read.
            with torch.enable_grad():U.fit_readers(m,cfg,out)
    saved=torch.load(readers,map_location=U.DEV,weights_only=False)
    fine=[]
    for seed in (0,1):
        q=saved[f'g_{seed}'];r=S.MLP(q['dim'],{'kind':5}).to(U.DEV)
        r.load_state_dict(q['state']);r.eval();fine.append(r)
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250)
    es=sorted(set(source['episode'].tolist()));eps=S.episodes(L)
    enc=dict(zip(es,S.encode(m,U.subset(L,es),cfg)))
    levels=U.FP.levels(cfg);records=[];max_replay=0.;checks=[]
    for e in es:
        q=enc[e]
        with TR.autocast(U.DEV,cfg):
            pc=M.Percept(*(q[k].to(U.DEV).float()[None] for k in ('p','g','v')),q['fix'].to(U.DEV)[None])
            tok=m.belief.tokens(m.belief.percept_embed(pc),q['reward'].to(U.DEV)[None],q['terminated'].to(U.DEV)[None],q['action'].to(U.DEV)[None])
            carry=m.belief.initial(1,U.DEV);at=0
            for i in np.where(source['episode']==e)[0]:
                t=int(source['t'][i]);a,b=weights[i]
                _,carry=m.belief(tok[:,at:2*t],carry);at=2*t
                frames=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations']
                p=m.perception(frames[t].to(U.DEV)[None].expand(49,-1,-1,-1),U.CENTRES)
                emb=m.belief.percept_embed(p)
                tt=m.belief.tokens(emb[:,None],q['reward'][t].to(U.DEV).expand(49,1),q['terminated'][t].to(U.DEV).expand(49,1),q['action'][t].to(U.DEV).expand(49,1))
                yy,_=m.belief(tt,U.expanded(carry,49));bc,bp=yy[:,0].float(),yy[:,1].float();mem=m.belief.memory(bp)
                rec=int(((U.CENTRES==q['fix'][t].to(U.DEV)).all(-1)).nonzero()[0])
                err=float((bp[rec]-q['bp'][t+1].to(U.DEV).float()).square().mean().sqrt())
                assert err<.03,('prefix replay',name,e,t,err)
                max_replay=max(max_replay,err)
                cells=U.WIN;truth=torch.as_tensor(source['truth'][i],device=U.DEV)
                alltruth=torch.nn.functional.pad(truth,(0,18));wg=alltruth[cells]
                fg=torch.cat([p.g.float(),p.v[:,None].expand(-1,9,-1).float()],-1)
                readable=torch.stack([r(fg)['kind'].argmax(-1)==wg for r in fine]).all(0)
                # Check the SAME source cells and crops used in the original arm.
                source_ok=readable.cpu().numpy()[source['useful_cells'][i]].mean()
                # Per-scene seed makes both worlds and all genuine looks share
                # the same standard-normal/context draws, independent of batching.
                gen=torch.Generator(device=U.DEV).manual_seed(612000+int(i))
                cache=m.denoiser.new_cache()
                for s in range(t-8,t):
                    cache=U.enter(m,cfg,tuple(q[k][s:s+1].to(U.DEV).float()[None] for k in ('p','g','v')),
                        q['prev_action'][s:s+1].to(U.DEV)[None],q['bp'][s:s+1].to(U.DEV).float()[None],q['fix'][s:s+1].to(U.DEV)[None],cache,gen)
                cache=U.expandcache(cache,49)
                cache=U.enter(m,cfg,tuple(x[:,None] for x in p.parts()),q['prev_action'][t].to(U.DEV).expand(49,1),
                    q['bp'][t].to(U.DEV).float().expand(49,1,-1),U.CENTRES[:,None],cache,gen)
                target=tuple(q[k][t+1:t+2].to(U.DEV).float()[None].expand(49,-1,*q[k].shape[1:]) for k in ('p','g','v'))
                pa=q['prev_action'][t+1].to(U.DEV).expand(49,1);fx=q['fix'][t+1].to(U.DEV).expand(49,1,2)
                local=torch.as_tensor(source['local'][i],device=U.DEV);loss=[]
                for draw in range(2):
                    bylevel=[]
                    for sig in levels:
                        xs=tuple(x+sig*torch.randn((1,*x.shape[1:]),device=U.DEV,generator=gen) for x in target)
                        cs,co,ci,cn=M.precondition(torch.full((49,1),sig,device=U.DEV),cfg['edm']['sigma_data'])
                        fo,_=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in xs),cn,pa,mem[:,None],fx,cache)
                        se=(fo[0].float()-(target[0]-M._bcast(cs,target[0])*xs[0])/M._bcast(co,target[0])).square()[:,0,:63]
                        bylevel.append(torch.stack([se.mean((1,2)),se[:,local].mean((1,2))],-1).cpu().numpy())
                    loss.append(bylevel)
                src=np.unique(A.WIN[source['useful_cells'][i]])
                center=torch.tensor([[int(np.clip(src[0]//9,1,7)),int(np.clip(src[0]%9,1,7))]],device=U.DEV).expand(49,-1)
                obs1=frames[t+1].to(U.DEV)[None].expand(49,-1,-1,-1)
                nxt=m.perception(obs1,center);cc=M.window(center)
                mu,_=m.pixels(m.tsp.predict(bp,nxt.p,nxt.v,cc),nxt.v,cc)
                pix=M.cell_pixels(M.crop(obs1,center)).float()/127.5-1
                tsp=(mu.float()-pix).square().mean((-1,-2)).cpu().numpy()
                stages={}
                for label,x in (('embedding',emb),('corrected',bc),('predicted',bp),('memory',mem.flatten(1))):
                    delta=np.einsum('wd,w->d',x.float().cpu().numpy(),a-b)
                    stages[label]=float(np.sqrt(np.mean(delta*delta)))
                # Repeated recorded crop is a self-consistency control; source
                # inputs never include future observations or outcomes.
                assert torch.equal(p.p,p.p[:1].expand_as(p.p))
                records.append(dict(index=int(i),episode=int(e),t=t,source_readable=float(source_ok),loss=np.asarray(loss),tsp=tsp,**stages))
                checks.append(err)
        print(name,'episode',e,'scenes',len(records),'prefix_max',max_replay,flush=True)
    records.sort(key=lambda x:x['index'])
    z={k:np.stack([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(out/'local.npz',**z,weights=weights,levels=levels)
    return dict(checkpoint_sha256=TR.file_sha256(ck),reader_sha256=TR.file_sha256(readers),prefix_rms_max=max_replay,scenes=len(records))


def summarize():
    worlds={};raw={}
    both_readable=np.load(HERE/'p1f16/local.npz')['source_readable']==1
    for name in WORLDS:
        z=dict(np.load(HERE/name/'local.npz'));a,b=z['weights'][:,0],z['weights'][:,1]
        diff=np.einsum('isnwg,iw->isng',z['loss'],a-b);base=np.einsum('isnwg,iw->isng',z['loss'],b)
        tspd=np.einsum('iw,iw->i',z['tsp'],a-b);tspb=np.einsum('iw,iw->i',z['tsp'],b)
        out={'source_readable_mean':float(z['source_readable'].mean()),'scenes_all_old_source_crops_readable':int((z['source_readable']==1).sum())}
        raw[name]={}
        for region,j in (('map',0),('local',1)):
            for band,sl in (('all',slice(None)),('high',slice(-2,None))):
                key=region+'_'+band;d=diff[:, :, sl,j].mean((1,2));v=base[:,:,sl,j].mean((1,2));fraction=d/v
                out[key]={'difference':A.estimate(d,z['episode']),'control_loss':A.estimate(v,z['episode']),'fraction':A.estimate(fraction,z['episode'])}
                raw[name][key]=fraction
        out['tsp_fraction']=A.estimate(tspd/tspb,z['episode']);raw[name]['tsp']=tspd/tspb
        out['stage_rms']={key:A.estimate(z[key],z['episode']) for key in ('embedding','corrected','predicted','memory')}
        out['both_worlds_all_source_crops_readable']={k:A.estimate(v[both_readable],z['episode'][both_readable]) for k,v in raw[name].items()}
        worlds[name]=out
    delta={k:A.estimate(raw['p1f16'][k]-raw['p1f15'][k],z['episode']) for k in raw['p1f15']}
    (HERE/'local_summary.json').write_text(json.dumps(dict(worlds=worlds,paired_fraction_change=delta),indent=2))
    print(json.dumps(dict(worlds=worlds,paired_fraction_change=delta),indent=2),flush=True)


def main():
    torch.set_num_threads(4);source,weights=fixed_cohort();start=time.time();ident={}
    if '--smoke' in sys.argv:
        print(run_world('p1f15',WORLDS['p1f15'],{k:v[:2] for k,v in source.items()},weights[:2]))
        return
    for name,ck in WORLDS.items():ident[name]=run_world(name,ck,source,weights)
    summarize()
    (HERE/'local_identity.json').write_text(json.dumps(dict(worlds=ident,seconds=time.time()-start,
        source_sha256=TR.file_sha256(U.HERE/'uptake_local/raw.npz'),script_sha256=TR.file_sha256(__file__),
        plan_sha256=TR.file_sha256(HERE/'PLAN.md')),indent=2))


if __name__=='__main__':main()
