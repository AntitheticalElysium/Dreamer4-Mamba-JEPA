"""Exact-recent-history timer contrasts. CPU simulator build, native F4 GPU evaluation; PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent
CKPT='artifacts/experiments/20261106_p1f4/run/ckpt_10.pt'

@torch.no_grad()
def run(arm, checkpoint):
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    torch.set_num_threads(2)
    IF=importlib.import_module('artifacts.experiments.20261030_actor_interface.interface')
    model,cfg=IF.load(checkpoint);dev=torch.device('cuda')
    bank=torch.load(HERE.parent/'20261108_full_frame_change/phase_bank.pt',weights_only=False)
    # Refit on real frames from a separate fixed TRAIN bank; validate by episode and on both real paired endpoints.
    L=dict(np.load(D.ROOT/'artifacts/eda/s0/s0_train400_labels.npz'))
    off=np.r_[0,np.cumsum(L['ep_T']+1)]
    xs=[];ys=[];held=[]
    for e,ep in enumerate(S.episodes(L)[:100]):
        f=D.RP.episode_fields(ep.shard,ep.slot)['observations'].numpy()
        ix=np.arange(0,len(f),2)
        for ids in np.array_split(ix,max(1,(len(ix)+511)//512)):
            z=torch.as_tensor(f[ids],device=dev)
            with TR.autocast(dev,cfg):
                P=model.perception(z,torch.full((len(z),2),4,device=dev,dtype=torch.long))
            xs.append(torch.cat([P.p[:,64],P.v],-1).float())
            ys.append(L['s_body'][off[e]+ids,1])
            held.extend([e%5==4]*len(ids))
    X=torch.cat(xs);Y=torch.as_tensor(np.concatenate(ys),device=dev).long()
    valid=torch.as_tensor(held,device=dev)
    xt,xv,yt,yv=X[~valid],X[valid],Y[~valid],Y[valid]
    with torch.enable_grad():
        reader,log=S.fit(lambda idx,val:(xv if val else xt)[idx],lambda o,idx,val:torch.nn.functional.cross_entropy(o['y'],(yv if val else yt)[idx]),len(yt),len(yv),{'y':10},110812)
    reader.eval();vp=torch.cat([reader(x)['y'].argmax(-1) for x in xv.split(8192)])
    qualification={'accuracy':float((vp==yv).float().mean()),'per_value':{str(k):float((vp[yv==k]==k).float().mean()) for k in yv.unique().tolist()}}
    assert qualification['accuracy']>=.99,'food reader failed holdout qualification'
    del X,Y,xt,xv,yt,yv
    K=32;results=[];saved=[]
    for i,r in enumerate(bank):
        npre=len(r['prefix_frames']);end=r['end'];T=npre+end+1
        f=np.concatenate([np.broadcast_to(r['prefix_frames'],(2,*r['prefix_frames'].shape)),r['frames'][:,:end+1]],1)
        actions=np.concatenate([np.broadcast_to(r['prefix_actions'],(2,npre)),np.zeros((2,end+1),np.int64)],1)
        rw=np.concatenate([np.broadcast_to(np.r_[0.,r['prefix_reward']],(2,npre+1)),r['reward'][:,:end]],1)
        td=np.concatenate([np.broadcast_to(np.r_[False,r['prefix_term']],(2,npre+1)),r['term'][:,:end]],1)
        # A common fixation history; the final recent full images coincide so all acquired crops coincide too.
        fx=D.fixation_prior(np.random.default_rng(12000+i),(T,));fix=torch.as_tensor(np.broadcast_to(fx,(2,*fx.shape)).copy(),device=dev)
        rw=torch.as_tensor(rw,device=dev,dtype=torch.float32);td=torch.as_tensor(td,device=dev)
        act=torch.as_tensor(actions,device=dev);prev=torch.cat([act.new_full((2,1),-1),act[:,:-1]],1)
        with TR.autocast(dev,cfg):
            P=model.perception(torch.as_tensor(f,device=dev),fix)
            y,_=model.belief(model.belief.tokens(model.belief.percept_embed(P),rw,td,act),model.belief.initial(2,dev))
            bp=torch.cat([torch.zeros_like(y[:,:1]),y[:,1::2][:,:-1]],1)
            next_bp=y[:,-1]
            assert all(torch.equal(z[0,-8:],z[1,-8:]) for z in P.parts())
            rg=torch.Generator(device=dev).manual_seed(14000+i)
            # Identical context-noise draws per branch. The only difference may be Mamba-derived information.
            for t in range(T-8,T):
                rng_state=rg.get_state();parts=[]
                # context_pass makes draws with a B=1 call per branch, then caches are merged below.
                if t==T-8:bc=[model.denoiser.new_cache(),model.denoiser.new_cache()]
                for br in (0,1):
                    rg.set_state(rng_state)
                    bc[br]=M.context_pass(model.denoiser,tuple(x[br:br+1,t:t+1] for x in P.parts()),prev[br:br+1,t:t+1],model.belief.memory(bp[br:br+1,t:t+1]),fix[br:br+1,t:t+1],bc[br],cfg,rg)
            query=D.fixation_prior(np.random.default_rng(15000+i),())
            gen=[]
            shapes=[tuple(x.shape[2:]) for x in P.parts()]
            for br in (0,1):
                cache={'t':bc[br]['t'],'kv':[tuple(z.expand(K,*z.shape[1:]) for z in kv) for kv in bc[br]['kv']]}
                xs=M.sample(model.denoiser,shapes,act.new_zeros(K,1),model.belief.memory(next_bp[br:br+1])[:,None].expand(K,-1,-1,-1),torch.as_tensor(query,device=dev)[None,None].expand(K,1,-1),cache,cfg,torch.Generator(device=dev).manual_seed(16000+i))
                gen.append((xs[0][:,0].float(),xs[2][:,0].float()))
            truth=model.perception(torch.as_tensor(r['frames'][:,end+1],device=dev),torch.as_tensor(query,device=dev)[None].expand(2,-1))
        observed=reader(torch.cat([truth.p[:,64],truth.v],-1).float())['y'].argmax(-1).cpu().numpy()
        assert np.array_equal(observed,r['food'][:,end+1]),'successor reader failed on the exact pair'
        pr=[reader(torch.cat([p[:,64],v],-1))['y'].float().softmax(-1).cpu().numpy() for p,v in gen]
        digit=[p.argmax(-1) for p in pr];current=int(r['food'][0,end])
        drops=[float((d<current).mean()) for d in digit]
        results.append({'e':r['e'],'current':current,'true_next':r['food'][:,end+1].tolist(),'generated_drop':drops,'contrast':drops[0]-drops[1],
                        'generated_digits':[np.bincount(d,minlength=10).tolist() for d in digit],'mean_reader_confidence':[float(p.max(-1).mean()) for p in pr]})
        with torch.no_grad():
            center=P.p[0,-1,64].float()
            endpoints=truth.p[:,64].float()
            assert (endpoints[0]-center).norm()>.1
            geom=[]
            for pp,vv in gen:
                x=pp[:,64];d0=(x-endpoints[0]).norm(dim=-1);d1=(x-endpoints[1]).norm(dim=-1)
                geom.append(float((d0<d1).float().mean()))
            results[-1]['nearest_real_decrement']=geom
        saved.append({'e':r['e'],'bp':next_bp.float().cpu(),'gen':[(p.cpu(),v.cpu()) for p,v in gen]})
        print('generated',i+1,drops,flush=True)
        (HERE/f'PHASE_{arm}.json').write_text(json.dumps({'reader':qualification,'K':K,'checkpoint_sha256':TR.file_sha256(D.ROOT/checkpoint),'pairs':results},indent=2))
    torch.save(saved,HERE/f'phase_{arm}.pt')


if __name__=="__main__": run(sys.argv[1],sys.argv[2])
