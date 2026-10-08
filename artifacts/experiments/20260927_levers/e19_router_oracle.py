"""Frozen dense generate-logit family: latent-L1 oracle versus depicted health.

Both C seeds, all279 damage cases and the existing512 unchanged controls.
193 offsets[-12,12] plus pure-copy-proportions/pure-generator endpoints. Future
health token chooses the diagnostic latent-L1 optimum only. This is a finite
one-dimensional routing family, not an information ceiling or trained repair.
GPU forward capped16%; all routing sweeps CPU. Historical seven-offset parity,
atomic source/input-bound per-case journals, no automatic notebook updates.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import e19_gradients as G

D,T,E,R=G.D,G.T,G.E,G.R
HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(3);device=torch.device('cuda');torch.cuda.set_per_process_memory_fraction(.16)
    cp=Path(str(T.CACHE).format('raw'));assert cp.exists()and(D.D.SD.CACHE/'DONE').exists()
    cache=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
    meta,fit,seeds=T.split();n=len(meta['seed']);assert int((D.D.SD.CACHE/'DONE').read_text())==n
    future=torch.from_numpy(D.D.SD.memmap(D.D.SD.CACHE/'fut5.f16',(n,5,16,81,192)))[:,0]
    frames=torch.cat([cache['ctx'],future],1);actions=torch.cat([cache['ctx_a'],cache['fut_a']],1)
    masks={k:v[:,0]for k,v in D.D.DR.masks(meta).items()};shift,ok=D.true_shifts(meta)
    valid=masks['valid']&masks['k3'];assert (ok|~valid).all()
    coef=G.hud_coeff(cache,meta,fit,seeds)
    offsets=torch.linspace(-12,12,193);cfg=E.config()
    for seed in (7,8):
        name=f'e19_C_s{seed}_fmamba_from36000';path=E.T.OUT/(name+'.pt')
        rp=HERE/'evals'/(name+'__e19_local_gradients_raw.pt');tp=HERE/'evals'/(name+'__e19_router_trace.pt')
        prior=torch.load(rp,map_location='cpu',weights_only=False);trace=torch.load(tp,map_location='cpu',mmap=True,weights_only=False)
        selection=prior['cases'];assert len(selection)==791
        spec={'scope':__doc__,'checkpoint':R.file_hash(path),'inputs':{str(p):R.file_hash(p)for p in (cp,T.META,rp,tp)},
              'frames':R.tensor_hash(frames),'actions':R.tensor_hash(actions),'selection':R.tensor_hash(selection),
              'coefficient':R.tensor_hash(coef),'offsets':offsets.tolist(),'endpoints':['copy','generator'],
              'sources':{str(p.resolve()):R.file_hash(p)for p in [Path(__file__),Path(G.__file__),Path(D.__file__),
                  Path(T.__file__),Path(E.__file__),HERE/'tworld.py',HERE/'h16_resume.py']},
              'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU_precision':'canonical BF16 autocast',
                         'sweep_precision':'CPU FP32','threads':3},'batch':4,'allocator_fraction':.16}
        store=R.Store(HERE/'evals/resume'/(name+'__e19_router_oracle'),spec)
        with store.lock():
            done=store.load('result')
            if done is not None:print(json.dumps(done,indent=2),flush=True);continue
            w,_=T.load_world(path,device);parts=[]
            for start in range(0,len(selection),4):
                part=store.load('batch_'+str(start))
                if part is None:
                    cases=selection[start:start+4];b=len(cases)
                    s=torch.stack([frames[r,3+k-4:3+k+1]for r,k in cases]).to(device).float()
                    a=torch.stack([actions[r,3+k-4:3+k+1]for r,k in cases]).to(device)
                    target=torch.stack([frames[r,4+k,63]for r,k in cases]).float()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred,_,gen=w(s,a)
                    actual=pred[:,-1,63].float().cpu();rawgen=gen[:,-1,63].float().cpu();ww=w.last_weights[:,-1,63].float().cpu()
                    current=s[:,-1].cpu();grid=current.reshape(b,9,9,192);pad=F.pad(grid,(0,0,1,1,1,1))
                    copies=torch.stack([grid]+[pad[:,1+dr:10+dr,1+dc:10+dc]for dr,dc in E.T.NEIGHBOURS],-2).reshape(b,81,5,192)[:,63]
                    cands=torch.cat([copies,rawgen[:,None]],1);logits=ww.clamp_min(1e-30).log()
                    shifted=logits[:,None,:].expand(-1,len(offsets),-1).clone();shifted[:,:,5]+=offsets
                    probs=shifted.softmax(-1)
                    copy=ww[:,:5]/ww[:,:5].sum(-1,keepdim=True).clamp_min(1e-30)
                    endpoints=torch.stack([F.pad(copy,(0,1)),F.one_hot(torch.full((b,),5),6).float()],1)
                    probs=torch.cat([probs,endpoints],1)
                    values=F.layer_norm((probs[...,None]*cands[:,None]).sum(-2),(192,))
                    loss=(values-target[:,None]).abs().mean(-1)
                    oldhp=trace['delta']['actual'][cases[:,0],cases[:,1]]
                    hp=oldhp[:,None]+((values-actual[:,None])*coef).sum(-1)
                    zero=int((offsets==0).nonzero()[0]);recon=float((values[:,zero]-actual).abs().max());assert recon<=2e-5,recon
                    ld=hd=0.
                    for off in (-8,-4,-2,0,2,4,8):
                        ii=int((offsets==off).nonzero()[0]);old=prior['sweeps'][str(off)]
                        ld=max(ld,float((loss[:,ii]-old['L1'][start:start+b]).abs().max()))
                        hd=max(hd,float((hp[:,ii]-old['delta_hp'][start:start+b]).abs().max()))
                    assert ld<=2e-5 and hd<=2e-4,(ld,hd)
                    best=loss.argmin(1);br=torch.arange(b)
                    part={'cases':cases,'loss':loss,'health_change':hp,'best_index':best,
                          'best_loss':loss[br,best],'best_hp':hp[br,best],
                          'actual_loss':loss[:,zero],'actual_hp':hp[:,zero],
                          'reconstruction':recon,'historical_loss_max_delta':ld,'historical_health_max_delta':hd,
                          'candidates':cands,'original_weights':ww,'target':target}
                    store.save('batch_'+str(start),part,1)
                parts.append(part)
                if start%100==0:print(json.dumps({'name':name,'cases_done':start+len(part['cases'])}),flush=True)
            raw={k:torch.cat([p[k]for p in parts])for k in ('cases','loss','health_change','best_index','best_loss',
                 'best_hp','actual_loss','actual_hp')}
            rr,kk=raw['cases'].T;hit=masks['drop2'][rr,kk];fresh=masks['adjacent'][rr,kk]&~masks['win'][rr,kk]&~masks['adjwin'][rr,kk]
            groups={'all_hit':hit,'fresh_hit':hit&fresh,'scroll_hit':hit&(shift[rr,kk]!=0),
                    'stationary_hit':hit&(shift[rr,kk]==0),'unchanged_control':~hit}
            summary={}
            for label,m in groups.items():
                any_drawn=raw['health_change'][m].min(1).values<-1.5
                best_drawn=raw['best_hp'][m]<-1.5
                summary[label]={'n':int(m.sum()),'actual_drawn':int((raw['actual_hp'][m]<-1.5).sum()),
                    'latent_L1_oracle_drawn':int(best_drawn.sum()),'any_family_member_drawn':int(any_drawn.sum()),
                    'possible_but_not_L1_optimum':int((any_drawn&~best_drawn).sum()),
                    'actual_L1_mean':float(raw['actual_loss'][m].mean()),
                    'oracle_L1_mean':float(raw['best_loss'][m].mean())}
            result={'scope':__doc__,'name':name,'contract':spec,'groups':summary,
                    'reconstruction_max':max(p['reconstruction']for p in parts),
                    'historical_loss_max_delta':max(p['historical_loss_max_delta']for p in parts),
                    'historical_health_max_delta':max(p['historical_health_max_delta']for p in parts)}
            store.save('rows',raw,1);store.save('result',result,1)
            R.atomic_json(HERE/'evals'/(name+'__e19_router_oracle.json'),result)
            print(json.dumps(result,indent=2),flush=True);del w;torch.cuda.empty_cache()


if __name__=='__main__':main()
