"""Frozen long-Mamba health-clock intervention, batch16 numerical control, Oct7.

The batch4 trial failed exact count reproduction (28 versus29). Preserve it.
This version changes only inference batch shape, matching the original batch16
(and its last ten-root batch); numerical padding never supplies another case
with a different input. Original counts and half-unit thresholds remain binding.

Reuse check_damage's1002 roots/sample0, exact real-fit HUD probe and health labels.
Cases: all337 living drops plus every fresh>=2 encounter from its declared mask,
plus512 unchanged cases fixed by RNG20261007 before model scoring. For seeds7/8:
true up-to15-frame history; repeated current frame at same history length, with
past NOOP and actual final action; repeated current at fixed lengths5,9,15.
Constant-history variants are deliberately inconsistent/OOD lesions, NOT an
information ceiling or a repair. They test scan-depth/position sensitivity and
whether history removal changes damage/false-drop detection. Original teacher
catches/fresh counts must reproduce29/337,46/337 and0/31,1/31 before interpretation.
Root true health and real-successor decoder are positive controls. Thresholds
unchanged: half-unit for both original panels;1.5 units is a separate severe-drop
metric, not the original fresh-arrival detection threshold.
World/probe/inputs/source bound, atomic16-case chunks, exclusive resume, GPU cap30%.
No fitting, encoder runs, new roots or notebook writes; logs only EDA.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import check_damage as CD
import h16_resume as R
from d4mj.config import config_from_dict
import spatial as S

HERE=Path(__file__).resolve().parent
T,SD,DR=CD.T,CD.SD,CD.DR
MODES=('true_history','repeat_same_length','repeat5','repeat9','repeat15')


def main():
    torch.set_num_threads(2)
    prepare_only='--prepare-only' in sys.argv
    config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    meta,train_roots,train_seeds=T.split()
    cache_path=Path(str(T.CACHE).format('raw'));assert cache_path.exists() and (SD.CACHE/'DONE').exists()
    cache=torch.load(cache_path,map_location='cpu',mmap=True,weights_only=False)
    fut,_=SD.token_cache(torch.device('cpu'))
    seqs=torch.cat([cache['ctx'],fut[:,0]],1)
    assert len(seqs)==1002,'Original batch16/tail10 grouping is fixed to these roots'
    acts=torch.cat([cache['ctx_a'],cache['fut_a']],1)
    alive=~meta['future_dead'].cumsum(2).bool().any(1)
    allv=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1).float()
    hp=allv[:,:,1512]*9
    drop=(hp[:,1:]<hp[:,:-1]-.5)&alive;same=(hp[:,1:]==hp[:,:-1])&alive
    assert int(drop.sum())==337
    masks=DR.masks(meta);m={k:v[:,0] for k,v in masks.items()}
    fresh=m['valid']&m['k3']&m['adjacent']&~m['win']&~m['adjwin']&m['drop2']
    assert int(fresh.sum())==31
    selected=drop|fresh
    si=torch.where(same.flatten())[0]
    choose=si[torch.randperm(len(si),generator=torch.Generator().manual_seed(20261007))[:512]]
    selected.view(-1)[choose]=True
    pairs=torch.nonzero(selected);rr,kk=pairs.T
    lengths=(4+kk).clamp_max(15)
    probes=T.Probes(cache,meta,train_roots,train_seeds)
    def health(tok):
        assert tok.shape[-2:]==(18,192)
        return (probes.hud(tok.float().reshape(-1,18*192))[:,0]*9).reshape(tok.shape[:-2])
    current=seqs[rr,3+kk,63:81]
    successor=seqs[rr,4+kk,63:81]
    hcur=health(current);hnext=health(successor)
    truth_masks={'living_drop':drop[rr,kk],'fresh_drop':fresh[rr,kk],'unchanged':same[rr,kk]}
    sources={p:R.file_hash(p) for p in (__file__,T.__file__,CD.__file__,DR.__file__,SD.__file__,R.__file__,S.__file__)}
    inputs={str(cache_path):R.file_hash(cache_path),str(S.CHECKPOINT):R.file_hash(S.CHECKPOINT)}
    # Bind concrete cases and probe coefficients rather than relying on sizes/cache names.
    tensors={'pairs':pairs,'lengths':lengths,'seqs':seqs[rr],'acts':acts[rr],'current_health':hcur,
             'next_health':hnext,'sim_current_health':hp[rr,kk],'sim_next_health':hp[rr,kk+1],
             **{k:v for k,v in truth_masks.items()}}
    # Bind the exact ridge closure: coefficient matrix, input mean and input std.
    outer=dict(zip(probes.hud.__code__.co_freevars,(c.cell_contents for c in probes.hud.__closure__)))
    xs=outer['xs'];inner=dict(zip(xs.__code__.co_freevars,(c.cell_contents for c in xs.__closure__)))
    probe={'coefficients':outer['w'],'input_mean':inner['mu'],'input_std':inner['sd']}
    tensors.update({f'probe_{k}':v for k,v in probe.items()})
    probe_outputs=health(seqs[rr,:,63:81]);tensors['all_true_hud_probe_outputs']=probe_outputs
    control={name:{'n':int(use.sum()),'true_successor_detected_or_false':int(((hnext-hcur)[use]<-.5).sum()),
                   'true_successor_severe_drop':int(((hnext-hcur)[use]<-1.5).sum())} for name,use in truth_masks.items()}
    assert control['living_drop']['true_successor_detected_or_false']==337
    assert control['fresh_drop']['true_successor_detected_or_false']==31
    assert control['unchanged']['true_successor_detected_or_false']==0
    if prepare_only:
        print(json.dumps({'prepare_only':True,'cases':len(pairs),'control':control,
                          'probe':{k:{'shape':list(v.shape),'sha256':R.tensor_hash(v)} for k,v in probe.items()}},sort_keys=True),flush=True)
        return
    device=torch.device('cuda');assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.30)
    spec={'scope':__doc__,'sources':sources,'inputs':inputs,'tensors':{k:R.tensor_hash(v) for k,v in tensors.items()},
          'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                     'threads':2,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32},
          'batch':16,'modes':MODES,'cuda_cap':.30}
    paths={seed:T.ROOT/'artifacts/eda/levers_tworlds_v1'/f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt' for seed in (7,8)}
    spec['checkpoints']={str(p):R.file_hash(p) for p in paths.values()}
    store=R.Store(HERE/'evals/resume/e17_health_clock_b16',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        out={};raw={}
        for seed,path in paths.items():
            world,_=T.load_world(path,device);per={}
            for mode in MODES:
                lens=lengths if mode in ('true_history','repeat_same_length') else torch.full_like(lengths,int(mode[6:]))
                hud=torch.empty(len(pairs),18,192)
                # Match original check_damage inference batch dimensions. The final
                # original root batch has10 entries; all preceding ones have16.
                original_batch=torch.where(rr>=992,10,16)
                for length in lens.unique().tolist():
                    for batch_size in (16,10):
                        ix=torch.where((lens==length)&(original_batch==batch_size))[0]
                        for lo in range(0,len(ix),batch_size):
                            key=f's{seed}_{mode}_L{length}_B{batch_size}_{lo}'
                            saved=store.load(key)
                            ids=ix[lo:lo+batch_size]
                            if saved is None:
                                xs=[];aa=[]
                                for j in ids.tolist():
                                    r,k=map(int,pairs[j]);end=4+k
                                    if mode=='true_history':
                                        x=seqs[r,end-length:end].float();a=acts[r,end-length:end]
                                    else:
                                        x=seqs[r,end-1].float()[None].expand(length,-1,-1)
                                        a=torch.zeros(length,dtype=torch.long);a[-1]=acts[r,end-1]
                                    xs.append(x);aa.append(a)
                                while len(xs)<batch_size:
                                    xs.append(xs[0]);aa.append(aa[0])
                                pred=T.step(world,torch.stack(xs),torch.stack(aa),device,config)
                                saved={'ids':ids,'hud':pred[:len(ids),63:81]};store.save(key,saved,lo+len(ids))
                            assert torch.equal(ids,saved['ids']);hud[ids]=saved['hud']
                hc=health(hud);dh=hc-hcur
                per[mode]={'drawn_health_change':dh,'hud':hud}
                counts={name:{'n':int(use.sum()),'catches_or_false_drops':int((dh[use]<-.5).sum()),
                    'severe_drop_count':int((dh[use]<-1.5).sum()),
                    'mean_drawn_health_change':float(dh[use].mean())} for name,use in truth_masks.items()}
                out.setdefault(str(seed),{})[mode]=counts
                print(json.dumps({'world_seed':seed,'mode':mode,'counts':counts}),flush=True)
            assert out[str(seed)]['true_history']['living_drop']['catches_or_false_drops']=={7:29,8:46}[seed]
            assert out[str(seed)]['true_history']['fresh_drop']['catches_or_false_drops']=={7:0,8:1}[seed]
            raw[seed]=per
            del world;torch.cuda.empty_cache()
        store.save('rows',{'worlds':raw,'pairs':pairs,'lengths':lengths,'masks':truth_masks,'current_health':hcur,'true_next_health':hnext,'episode_seeds':meta['seed'][rr]},1)
        result={'scope':__doc__,'contract':spec,'worlds':out,'control':control,'original_teacher_counts_reproduced':True}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_health_clock_b16.json',result)
        print(json.dumps({'finished':True,'worlds':out,'control':control}),flush=True)


if __name__=='__main__':main()
