"""Confirm last-frame time-embedding effect on E17's real teacher histories.

Declared Oct7 after the repeated-current transplant. Same sealed851 cases and
worlds. Evaluate only histories whose existing context length is15; intact baseline
must reproduce completed batch16 HUD predictions. Compare intact row14 with row4
on ONLY the current frame. All actual images, past actions, carry initialization,
earlier time embeddings, scan length and other weights are unchanged. A deliberate
internal intervention, not a trained repair or proof about the cause of training.
Atomic16-case chunks with original tail10 shapes, source closure checked, CUDA30%.
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


def main():
    torch.set_num_threads(2);dev=torch.device('cuda');assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.30)
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    base=HERE/'evals/resume/e17_health_clock_b16';bc=json.loads((base/'contract.json').read_text())
    old=R.Store(base,bc);assert old.load('result')['original_teacher_counts_reproduced'];raw=old.load('rows')
    closure=json.loads((HERE/'evals/e17_health_time_swap.json').read_text())['contract']['sources']
    for f,h in closure.items():assert R.file_hash(f)==h,f
    sources={**closure,__file__:R.file_hash(__file__)}
    meta,tr,seeds=T.split();cp=Path(str(T.CACHE).format('raw'))
    cache=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
    fut,_=SD.token_cache(torch.device('cpu'));seq=torch.cat([cache['ctx'],fut[:,0]],1)
    acts=torch.cat([cache['ctx_a'],cache['fut_a']],1)
    val=torch.isin(meta['seed'],seeds[:len(seeds)//5]);fit=tr&~val
    toks=torch.cat([cache['ctx'][:,-1:],cache['fut']],1)
    vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
    expand=lambda m:m[:,None].expand(-1,17).flatten()
    hud=T.ridge(toks[:,:,63:81].flatten(0,1).flatten(1).float(),T.facts_of(vis)['hud'].flatten(0,1),expand(fit),expand(val))
    outer=dict(zip(hud.__code__.co_freevars,(c.cell_contents for c in hud.__closure__)));xs=outer['xs']
    inner=dict(zip(xs.__code__.co_freevars,(c.cell_contents for c in xs.__closure__)))
    coeff={'coefficients':outer['w'],'input_mean':inner['mu'],'input_std':inner['sd']}
    for k,v in coeff.items():assert R.tensor_hash(v)==bc['tensors']['probe_'+k]
    health=lambda tok:hud(tok.float().reshape(-1,18*192))[:,0]*9
    ids=torch.where(raw['lengths']==15)[0];pairs=raw['pairs'][ids];rr,kk=pairs.T
    masks={k:v[ids] for k,v in raw['masks'].items()};hcur=raw['current_health'][ids]
    spec={'scope':__doc__,'sources':sources,'reference_contract':bc,'case_indices':R.tensor_hash(ids),
          'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                     'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
    store=R.Store(HERE/'evals/resume/e17_health_time_true',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        allrows={};out={};batch=torch.where(rr>=992,10,16)
        for seed in (7,8):
            path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt'
            world,_=T.load_world(path,dev);clock=world.time.detach().clone();per={};summary={}
            for mode,last in (('intact',14),('last4',4)):
                with torch.no_grad():world.time.copy_(clock);world.time[14].copy_(clock[last])
                predhud=torch.empty(len(ids),18,192)
                for bs in (16,10):
                    ii=torch.where(batch==bs)[0]
                    for lo in range(0,len(ii),bs):
                        js=ii[lo:lo+bs];key=f's{seed}_{mode}_B{bs}_{lo}';saved=store.load(key)
                        if saved is None:
                            x=[];a=[]
                            for r,k in pairs[js].tolist():
                                end=4+k;x.append(seq[r,end-15:end].float());a.append(acts[r,end-15:end])
                            while len(x)<bs:x.append(x[0]);a.append(a[0])
                            pred=T.step(world,torch.stack(x),torch.stack(a),dev,cfg)
                            saved={'indices':js,'hud':pred[:len(js),63:81]};store.save(key,saved,lo+len(js))
                        assert torch.equal(js,saved['indices']);predhud[js]=saved['hud']
                dh=health(predhud)-hcur;per[mode]=dh
                if mode=='intact':
                    ref=raw['worlds'][seed]['true_history']['drawn_health_change'][ids]
                    assert torch.equal(dh<-.5,ref<-.5)
                    assert (dh-ref).abs().max()<1e-4
                summary[mode]={k:{'n':int(m.sum()),'drops':int((dh[m]<-.5).sum()),'mean_health_change':float(dh[m].mean())} for k,m in masks.items()}
                print(json.dumps({'world_seed':seed,'mode':mode,'counts':summary[mode]}),flush=True)
            out[str(seed)]=summary;allrows[seed]=per
            del world;torch.cuda.empty_cache()
        store.save('rows',{'worlds':allrows,'indices':ids,'pairs':pairs,'masks':masks,'seed':raw['episode_seeds'][ids]},1)
        result={'scope':__doc__,'contract':spec,'worlds':out,'intact_reproduction_passed':True}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_health_time_true.json',result)
        print(json.dumps({'finished':True,'worlds':out}),flush=True)


if __name__=='__main__':main()
