"""Frozen TRAIN loss and shared output-head gradients on 30 actual recorded batches.

Batch updates 0,200,...,5800 are fixed before execution. C and B use identical
recorded windows/splits. Reconstruct ALL output tokens from detached backbone
features through the live proj/choose heads; move logits remain fixed. Require
<=2e-5 parity. Measure actual extra health loss by death/living-damage/unchanged,
and class/teacher gradients on proj+choose weights. This measures endpoint head
gradient conflict, not encoder/backbone gradients or the entire optimization path.
Atomic source/input-bound per-batch records support interruption and resumption.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e19 as E
R,T=E.R,E.T


def run(arm,pool,mask,training,device,seed=7,backbone='fmamba'):
    path=T.OUT/f'e19_{arm}_s{seed}_{backbone}_from36000.pt'
    import teval
    w,st=teval.load_world(path,device)
    assert not w.skip
    store=R.frozen_eval_store(path,st['name']+'__e19_train_mechanism',
        {'batch_updates':list(range(0,6000,200)),'gradient_scope':'proj.weight,choose.weight; frozen backbone/gates'},
        {'ledger':training['ledger'],'layout':training['layout'],'mask':mask,'dh':pool['dh'],'alive':pool['alive']},
        [T.POOLS['raw']/'pool.pt'])
    with store.lock():
        result=store.load('result')
        if result is not None:return result
        records=[]
        classes=('death','ordinary_ge2','unchanged','other')
        for update in range(0,6000,200):
            record=store.load('batch_'+str(update))
            if record is None:
                idx=training['ledger'][update].long();offsets=training['layout'][update].long()
                s=pool['tokens'][idx].to(device).float();a=pool['actions'][idx].to(device)
                m=mask[idx].to(device);alive=pool['alive'][idx,1:].to(device);dh=pool['dh'][idx].to(device)
                labels={'death':~alive,'ordinary_ge2':alive&(dh<=-2),'unchanged':alive&(dh==0)}
                labels['other']=~(labels['death']|labels['ordinary_ge2']|labels['unchanged'])
                count=m.sum().clamp_min(1)
                vectors={x:torch.zeros(w.proj.weight.numel()+w.choose.weight.numel())for x in (*classes,'teacher')}
                losses={x:0. for x in (*classes,'teacher')};selected={x:int((m&labels[x]).sum())for x in classes}
                reconstruction=0.
                for offset in range(5):
                    rows=(offsets==offset).nonzero().flatten()
                    if not len(rows):continue
                    for lo,hi in E.segments(offset):
                        ss=s[rows,lo:hi+1];aa=F.pad(a[rows,lo:hi],(0,1))
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):original,h,_=w(ss,aa)
                        original=original[:,:hi-lo];weights=w.last_weights[:,:hi-lo].float()
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            gen=w.proj(h.detach()).float()[:,:hi-lo]
                            base=w.choose(h.detach()).float()[:,:hi-lo]
                        # The captured softmax includes frame+target gates. Preserve their
                        # numerical values while gradients flow only through choose/proj.
                        logits=base+(weights.clamp_min(1e-30).log()-base.detach())
                        grid=ss[:,:hi-lo].reshape(len(rows),hi-lo,9,9,192)
                        pad=F.pad(grid,(0,0,1,1,1,1))
                        candidates=torch.stack([grid]+[pad[:,:,1+dr:10+dr,1+dc:10+dc]for dr,dc in T.NEIGHBOURS],-2).reshape(len(rows),hi-lo,81,5,192)
                        candidates=torch.cat([candidates,gen[...,None,:]],-2)
                        pred=F.layer_norm((logits.softmax(-1)[...,None]*candidates).sum(-2),(192,))
                        reconstruction=max(reconstruction,float((pred-original).abs().max()))
                        assert reconstruction<=2e-5,reconstruction
                        error=(pred-s[rows,lo+1:hi+1]).abs()
                        terms={'teacher':error.sum()/s[:,1:].numel()}
                        for c in classes:
                            terms[c]=(error[:,:,63]*((m&labels[c])[rows,lo:hi])[...,None]).sum()/(count*192)
                        for c,value in terms.items():
                            losses[c]+=float(value.detach())
                            if c!='teacher' and not selected[c]:continue
                            grads=torch.autograd.grad(value,(w.proj.weight,w.choose.weight),retain_graph=True)
                            vectors[c]+=torch.cat([x.detach().float().cpu().flatten()for x in grads])
                record={'update':update,'selected':selected,'losses':losses,'gradients':vectors,'reconstruction':reconstruction}
                store.save('batch_'+str(update),record,1)
            records.append(record)
            print(json.dumps({'arm':arm,'batch_update':update,'loss':record['losses']}),flush=True)
        mean={c:sum(x['losses'][c]for x in records)/len(records)for c in (*classes,'teacher')}
        gradients={c:sum(x['gradients'][c]for x in records)/len(records)for c in (*classes,'teacher')}
        health=sum(gradients[c]for c in classes)
        cosine=lambda x,y:float(F.cosine_similarity(x[None],y[None]))
        result={'arm':arm,'scope':__doc__,'batches':len(records),'selected_counts':{c:sum(x['selected'][c]for x in records)for c in classes},
                'mean_loss':mean,'extra_loss_share':{c:mean[c]/sum(mean[x]for x in classes)for c in classes},
                'mean_head_gradient_norm':{c:float(g.norm())for c,g in gradients.items()},
                'gradient_cosine':{'ordinary_vs_death':cosine(gradients['ordinary_ge2'],gradients['death']),
                                   'ordinary_vs_unchanged':cosine(gradients['ordinary_ge2'],gradients['unchanged']),
                                   'health_vs_teacher':cosine(health,gradients['teacher'])},
                'reconstruction_max':max(x['reconstruction']for x in records),'resume_contract':store.contract}
        R.atomic_json(HERE/'evals'/(st['name']+'__e19_train_mechanism.json'),result)
        store.save('result',result,1)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[7,8],default=7)
    p.add_argument('--backbone',choices=['fmamba','full'],default='fmamba')
    args=p.parse_args()
    torch.set_num_threads(3);device=torch.device('cuda');torch.cuda.set_per_process_memory_fraction(.16)
    pool=torch.load(T.POOLS['raw']/'pool.pt',mmap=True,map_location='cpu',weights_only=False)
    mask=torch.load(E.MASK,weights_only=False)['beside'].bool()
    root=T.OUT/f'state/e19_C_s{args.seed}_{args.backbone}_from36000'
    training=R.Store(root,json.loads((root/'contract.json').read_text())).load('train')
    assert training['update']==6000
    other=T.OUT/f'state/e19_B_s{args.seed}_{args.backbone}_from36000'
    control=R.Store(other,json.loads((other/'contract.json').read_text())).load('train')
    assert control['update']==6000
    assert torch.equal(training['ledger'],control['ledger']) and torch.equal(training['layout'],control['layout'])
    for arm in ('C','B'):
        print(json.dumps(run(arm,pool,mask,training,device,args.seed,args.backbone),indent=2),flush=True)
        torch.cuda.empty_cache()
