"""E17 current-time-embedding transplant, declared Oct7 before execution.

Frozen Mambas7/8, same851 cases, same repeated current images/past NOOP/final
actual action as the completed batch16 health-clock lesion. Reproduce its intact
length5 and length15 outputs. Change ONLY the last input's time embedding: at
length15 replace row14 with row4; at length5 replace row4 with row14. All other
weights, prior embeddings, pixels, actions, carry initialization, scan length and
batch dimensions stay fixed within each contrast. This identifies an explicit
clock input's causal effect on frozen outputs, not a trained repair, IID control
or proof that terminal alignment caused historical weights. HUD ridge coefficients
must match the completed lesion. Full forward source closure matches the original
teacher evaluator. Atomic16-case chunks with its tail10 batch dimensions; CUDA30%.
"""
import json
from pathlib import Path
import torch
import teval as T
import stochdiag as SD
import spatial as S
import h16_resume as R
from d4mj.config import config_from_dict

HERE=Path(__file__).resolve().parent
MODES={'intact5':(5,4),'intact15':(15,14),'len15_last4':(15,4),'len5_last14':(5,14)}


def main():
    torch.set_num_threads(2);device=torch.device('cuda');assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.30)
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    base=HERE/'evals/resume/e17_health_clock_b16';bc=json.loads((base/'contract.json').read_text())
    for kind in ('sources','inputs','checkpoints'):
        for f,h in bc[kind].items():assert R.file_hash(f)==h,f
    prior=R.Store(base,bc);assert prior.load('result')['original_teacher_counts_reproduced']
    rows=prior.load('rows');pairs=rows['pairs'];rr,kk=pairs.T
    meta,tr,seeds=T.split();cp=Path(str(T.CACHE).format('raw'))
    cache=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
    fut,_=SD.token_cache(torch.device('cpu'))
    current=torch.empty(len(pairs),81,192,dtype=torch.float16)
    current[kk==0]=cache['ctx'][rr[kk==0],-1]
    current[kk>0]=fut[rr[kk>0],0,kk[kk>0]-1]
    actions=cache['fut_a'][rr,kk]
    val=torch.isin(meta['seed'],seeds[:len(seeds)//5]);fit=tr&~val
    toks=torch.cat([cache['ctx'][:,-1:],cache['fut']],1)
    vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
    expand=lambda m:m[:,None].expand(-1,17).flatten()
    hud=T.ridge(toks[:,:,63:81].flatten(0,1).flatten(1).float(),T.facts_of(vis)['hud'].flatten(0,1),expand(fit),expand(val))
    o=dict(zip(hud.__code__.co_freevars,(c.cell_contents for c in hud.__closure__)))
    xs=o['xs'];i=dict(zip(xs.__code__.co_freevars,(c.cell_contents for c in xs.__closure__)))
    coeff={'coefficients':o['w'],'input_mean':i['mu'],'input_std':i['sd']}
    for k,v in coeff.items():assert R.tensor_hash(v)==bc['tensors']['probe_'+k],k
    health=lambda tok:hud(tok.float().reshape(-1,18*192))[:,0]*9
    originals={};closure={}
    for seed in (7,8):
        p=next(Path('artifacts/eda/frozen_eval_resume_v1').glob(f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000__damage_w15__*'))
        originals[seed]=p;oc=json.loads((p/'contract.json').read_text())
        for f,h in oc['sources'].items():assert R.file_hash(f)==h,f
        closure.update(oc['sources'])
    closure.update({f:R.file_hash(f) for f in (__file__,T.__file__,SD.__file__,S.__file__,R.__file__)})
    spec={'scope':__doc__,'sources':closure,'reference_contract':bc,'inputs':{str(base/'contract.json'):R.file_hash(base/'contract.json')},
          'cases':{k:R.tensor_hash(v) for k,v in {'pairs':pairs,'current':current,'action':actions,**coeff}.items()},
          'modes':MODES,'cuda_fraction':.30,'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
    store=R.Store(HERE/'evals/resume/e17_health_time_swap',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        summaries={};allrows={}
        batch=torch.where(rr>=992,10,16)
        for seed in (7,8):
            path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt'
            world,_=T.load_world(path,device);original=world.time.detach().clone();preds={};ss={}
            for mode,(length,clock) in MODES.items():
                with torch.no_grad():world.time.copy_(original);world.time[length-1].copy_(original[clock])
                out=torch.empty(len(pairs),18,192)
                for bs in (16,10):
                    indices=torch.where(batch==bs)[0]
                    for lo in range(0,len(indices),bs):
                        key=f's{seed}_{mode}_B{bs}_{lo}';saved=store.load(key);ids=indices[lo:lo+bs]
                        if saved is None:
                            padded=torch.cat([ids,ids[:1].expand(bs-len(ids))])
                            x=current[padded].float()[:,None].expand(-1,length,-1,-1).contiguous()
                            a=torch.zeros(bs,length,dtype=torch.long);a[:,-1]=actions[padded]
                            pred=T.step(world,x,a,device,cfg)
                            saved={'ids':ids,'hud':pred[:len(ids),63:81]};store.save(key,saved,lo+len(ids))
                        assert torch.equal(ids,saved['ids']);out[ids]=saved['hud']
                dh=health(out)-rows['current_health'];preds[mode]=dh
                if mode.startswith('intact'):
                    ref=rows['worlds'][seed]['repeat'+str(length)]['drawn_health_change']
                    assert torch.equal(dh<-.5,ref<-.5),f'intact reproduction failed seed{seed} {mode}'
                    assert (dh-ref).abs().max()<1e-4
                ss[mode]={name:{'n':int(mask.sum()),'drops':int((dh[mask]<-.5).sum()),'severe_drops':int((dh[mask]<-1.5).sum()),
                               'mean_health_change':float(dh[mask].mean())} for name,mask in rows['masks'].items()}
                print(json.dumps({'world_seed':seed,'mode':mode,'counts':ss[mode]}),flush=True)
            summaries[str(seed)]=ss;allrows[seed]=preds
            del world;torch.cuda.empty_cache()
        store.save('rows',{'worlds':allrows,'pairs':pairs,'masks':rows['masks'],'seed':rows['episode_seeds']},1)
        result={'scope':__doc__,'contract':spec,'worlds':summaries,'intact_reproduction_passed':True}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_health_time_swap.json',result)
        print(json.dumps({'finished':True,'worlds':summaries}),flush=True)


if __name__=='__main__':main()
