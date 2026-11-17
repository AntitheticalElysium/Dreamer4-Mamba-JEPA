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

def build(n=24):
    import jax,jax.numpy as jnp
    assert jax.default_backend()=='cpu'
    L=dict(np.load(D.ROOT/'artifacts/eda/s0/s0_dev_labels.npz'))
    off=np.r_[0,np.cumsum(L['ep_T']+1)]
    _,_,_,step,render=D.RP.env_and_render()
    bank=[];excluded=[]
    for e,T in enumerate(L['ep_T']):
        if e%5!=4:continue # reader fitting uses other physical episodes
        lo,hi=off[e:e+2]
        b=L['s_body'][lo:hi]
        ok=(np.arange(T+1)>=12)&(np.arange(T+1)<=80)&(b.min(1)>=5)&~L['s_sleep'][lo:hi]&(L['s_light'][lo:hi]>=.9)
        idx=np.flatnonzero(ok)
        if not len(idx):continue
        t=int(idx[0]);sh,sl=int(L['ep_shard'][e]),int(L['ep_slot'][e])
        state=D.RP.advance_to(sh,sl,t)
        orig=D.RP.episode_fields(sh,sl)
        assert np.array_equal(np.asarray(render(state)),orig['observations'][t].numpy())
        states=[state.replace(player_hunger=jnp.asarray(v,state.player_hunger.dtype)) for v in (20.,12.)]
        frames=[[],[]];rewards=[[],[]];term=[[],[]];food=[[],[]];hunger=[[],[]]
        keys=jax.random.split(jax.random.PRNGKey(110812+e),50)
        for s in range(42):
            for branch in range(2):
                st=states[branch]
                frames[branch].append(np.asarray(render(st)));food[branch].append(int(st.player_food));hunger[branch].append(float(st.player_hunger))
                _,nxt,r,d,_=step(keys[s],st,jnp.asarray(0,jnp.int32))
                states[branch]=nxt;rewards[branch].append(float(r));term[branch].append(bool(d))
        frames=np.asarray(frames);food=np.asarray(food);rewards=np.asarray(rewards);term=np.asarray(term);hunger=np.asarray(hunger)
        found=[]
        for s in range(16,40):
            event=food[:,s+1]<food[:,s]
            if event.tolist()!=[True,False]:continue
            if not np.array_equal(frames[0,s-7:s+1],frames[1,s-7:s+1]):continue
            if not np.array_equal(rewards[0,s-8:s],rewards[1,s-8:s]) or not np.array_equal(term[0,s-8:s],term[1,s-8:s]):continue
            if term[:,:s+1].any():continue
            drops=[np.flatnonzero(np.diff(food[k,:s+1])<0)+1 for k in (0,1)]
            if not all(len(x)>0 for x in drops):continue
            found.append((s,[int(x[-1]) for x in drops]))
        if not found:excluded.append({'e':e,'reason':'no qualified pair'});continue
        s,resets=found[0]
        bank.append({'e':e,'t0':t,'end':s,'frames':frames[:,:s+2],'reward':rewards[:,:s+1],'term':term[:,:s+1],
                     'food':food[:,:s+2],'hunger':hunger[:,:s+2],'last_drop':resets,
                     'prefix_frames':orig['observations'][:t].numpy(),'prefix_actions':orig['actions_taken'][:t].numpy(),
                     'prefix_reward':orig['rewards'][:t].numpy(),'prefix_term':orig['terminated'][:t].numpy()})
        print('pair',len(bank),'episode',e,'step',s,'last drops',resets,flush=True)
        if len(bank)>=n:break
    torch.save(bank,HERE/'phase_bank.pt')
    (HERE/'PHASE_BUILD.json').write_text(json.dumps({'pairs':len(bank),'excluded':excluded,'rows':[{'e':r['e'],'t0':r['t0'],'end':r['end'],'last_drop':r['last_drop'],'current_food':r['food'][:,r['end']].tolist(),'next_food':r['food'][:,r['end']+1].tolist()} for r in bank]},indent=2))

@torch.no_grad()
def run():
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    torch.set_num_threads(2)
    IF=importlib.import_module('artifacts.experiments.20261030_actor_interface.interface')
    model,cfg=IF.load(CKPT);dev=torch.device('cuda')
    bank=torch.load(HERE/'phase_bank.pt',weights_only=False)
    E=torch.load(HERE.parent/'20261107_health_readout_audit/enc.pt',weights_only=False,map_location='cpu',mmap=True)
    L=dict(np.load(D.ROOT/'artifacts/eda/s0/s0_dev_labels.npz'));off=np.r_[0,np.cumsum(L['ep_T']+1)]
    # Fixed protocol from S1; evaluation scenes are never reader training episodes.
    X=torch.cat([torch.cat([a['p'][:,64],a['v']],-1) for a in E]).float().to(dev)
    Y=torch.as_tensor(L['s_body'][:,1].clip(0,9),device=dev).long()
    valid=torch.as_tensor(np.repeat(np.arange(len(E))%5==4,L['ep_T']+1),device=dev)
    xt,xv,yt,yv=X[~valid],X[valid],Y[~valid],Y[valid]
    with torch.enable_grad():
        reader,log=S.fit(lambda idx,val:(xv if val else xt)[idx],lambda o,idx,val:torch.nn.functional.cross_entropy(o['y'],(yv if val else yt)[idx]),len(yt),len(yv),{'y':10},110812)
    reader.eval();vp=torch.cat([reader(x)['y'].argmax(-1) for x in xv.split(8192)])
    qualification={'accuracy':float((vp==yv).float().mean()),'per_value':{str(k):float((vp[yv==k]==k).float().mean()) for k in yv.unique().tolist()}}
    assert qualification['accuracy']>=.99,'food reader failed holdout qualification'
    del X,Y,xt,xv,yt,yv,E
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
            cache=model.denoiser.new_cache();rg=torch.Generator(device=dev).manual_seed(14000+i)
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
        saved.append({'e':r['e'],'bp':next_bp.float().cpu(),'gen':[(p.cpu(),v.cpu()) for p,v in gen]})
        print('generated',i+1,drops,flush=True)
        (HERE/'PHASE_RESULTS.json').write_text(json.dumps({'reader':qualification,'K':K,'checkpoint_sha256':TR.file_sha256(D.ROOT/CKPT),'pairs':results},indent=2))
    torch.save(saved,HERE/'phase_generated.pt')

if __name__=='__main__':
    if sys.argv[1]=='build':build(int(sys.argv[2]) if len(sys.argv)>2 else 24)
    else:run()
