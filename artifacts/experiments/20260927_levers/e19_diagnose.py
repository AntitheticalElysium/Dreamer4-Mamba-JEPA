"""Frozen E19 mechanism trace: generator vs router, moving vs stationary, true63 substitution.

Uses the existing diagnosis futures (sample0,k>=3), no new training or sealed claim.
Every root batch is journalled with source/input/checkpoint hashes. No experiment
writes NOTEBOOK. Reports and per-transition traces remain in existing levers/EDA.
GPU allocator is capped to1GiB so this bounded forward-only diagnostic can share
the device with the booked evaluator. A reconstruction control binds the actual
copy/generator mixing equations; oracle substitutions are labelled as diagnostics.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'20260926_diagnosis'))
sys.path.insert(0,str(HERE.parent/'20260921_readout_ladder'))
import e19_eval as D
import e19 as E
T,R,H=D.T,E.R,D.H


def true_shifts(meta):
    vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
    tile=vis[...,:1071].reshape(len(vis),17,7,9,17).argmax(-1)
    a,b=tile[:,:-1],tile[:,1:]; scores=[]
    for dr,dc in D.DR.SHIFTS:
        r0,r1,c0,c1=max(0,-dr),7-max(0,dr),max(0,-dc),9-max(0,dc)
        scores.append((b[...,r0:r1,c0:c1]==a[...,r0+dr:r1+dr,c0+dc:c1+dc]).float().mean((-1,-2)))
    scores=torch.stack(scores,-1)
    return scores.argmax(-1),scores.max(-1).values>=.9


def summary(trace, meta, masks, shift, test):
    use=masks['valid']&masks['k3'];hit=masks['drop2'];same=masks['dh']==0
    fresh=masks['adjacent']&~masks['win']&~masks['adjwin']
    groups={'ordinary_hit':use&hit,'fresh_hit':use&hit&fresh,
            'scroll_hit':use&hit&(shift!=0),'stationary_hit':use&hit&(shift==0),
            'unchanged':use&same,'scroll_unchanged':use&same&(shift!=0),
            'stationary_unchanged':use&same&(shift==0)}
    hp=(torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)[... ,1512]*9).round()[:,:H]
    groups.update({'health9_hit':use&hit&(hp==9),'below9_hit':use&hit&(hp<9)})
    out={}
    for label,mask in groups.items():
        n=int(mask.sum());row={'n':n}
        if not n:
            out[label]=row;continue
        row['predicted_health_change']={name:{'caught_at_1.5':int((value[mask]<-1.5).sum()),
                                             'caught_at_0.5':int((value[mask]<-.5).sum()),
                                             'median':float(value[mask].median()),
                                             'mean':float(value[mask].mean())}
                                      for name,value in trace['delta'].items()}
        row['health63_L1_over_self_copy']={name:float(value[mask].sum()/trace['error']['self'][mask].sum().clamp_min(1e-12))
                                          for name,value in trace['error'].items()}
        row['weight_mean']={name:float(value[mask].mean()) for name,value in trace['weight'].items()}
        row['generator_closer_than_best_copy']=int((trace['error']['gen'][mask]<trace['error']['best_copy'][mask]).sum())
        row['gen_weight_quantiles']=trace['weight']['generate'][mask].quantile(torch.tensor([0.,.1,.5,.9,1.])).tolist()
        out[label]=row
    out['test_seed_metrics']=D.metrics(trace['delta']['actual']<-1.5,masks,test)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',type=Path,nargs='+')
    args=parser.parse_args()
    torch.set_num_threads(3)
    device=torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.16)
    from d4mj.train import autocast_context
    meta,fit,seeds=T.split();cache=T.build_cache('raw',device)
    fut5,_=D.SD.token_cache(device)
    frames=torch.cat([cache['ctx'].cpu(),fut5[:,0].cpu()],1)
    actions=torch.cat([cache['ctx_a'].cpu(),cache['fut_a'].cpu()],1)
    probes=T.Probes(cache,meta,fit,seeds)
    health=lambda tok:probes.hud(tok[...,63:81,:].float().cpu().flatten(-2))[...,0]*9
    masks={k:v[:,0]for k,v in D.DR.masks(meta).items()}
    shift,ok=true_shifts(meta)
    assert (ok|~(masks['valid']&masks['k3'])).all()
    cfg=E.config();N=len(meta['seed']);bs=4
    for path in args.paths:
        w,st=T.load_world(path,device)
        store=R.frozen_eval_store(path,st['name']+'__e19_router_diagnosis',
                {'window':5,'k_min':3,'batch':bs,'allocator_fraction':.16},
                {'frames':frames,'actions':actions,**{'mask_'+k:v for k,v in masks.items()}},[E.T.S.CHECKPOINT])
        with store.lock():
            trace=store.load('result')
            if trace is None:
                pieces=[];max_reconstruction=0.
                with torch.no_grad():
                    for i in range(0,N,bs):
                        piece=store.load('batch_'+str(i));nb=min(bs,N-i)
                        if piece is None:
                            delta={name:torch.zeros(nb,H) for name in
                                    ('actual','no_global_HUD','self_gen_HUD','gen63','genHUD','true63','trueHUD')}
                            error={name:torch.zeros(nb,H) for name in ('actual','self','best_copy','gen')}
                            weight={name:torch.zeros(nb,H)for name in ('self','neighbours','generate')}
                            features={name:torch.zeros(nb,H,d)for name,d in
                                      (('input63',192),('output63',192),('gen63',192),('h63',256))}
                            reconstruction=0.
                            for k in range(3,H):
                                j=3+k;s=frames[i:i+nb,j-4:j+1].to(device).float()
                                aa=actions[i:i+nb,j-4:j+1].to(device)
                                with autocast_context(cfg):pred,h,gen=w(s,aa)
                                pred,h,gen=pred[:,-1],h[:,-1],gen[:,-1]
                                current=s[:,-1];true=frames[i:i+nb,j+1].to(device).float()
                                grid=current.reshape(nb,9,9,192);pad=F.pad(grid,(0,0,1,1,1,1))
                                candidates=[grid]+[pad[:,1+dr:10+dr,1+dc:10+dc]for dr,dc in E.T.NEIGHBOURS]
                                candidates=torch.stack([x.reshape(nb,81,192)for x in candidates]+[gen],-2)
                                ww=w.last_weights[:,-1].float()
                                reconstructed=F.layer_norm((ww[...,None]*candidates).sum(-2),(192,))
                                reconstruction=max(reconstruction,float((reconstructed-pred).abs().max()))
                                assert reconstruction<=1e-5
                                with autocast_context(cfg):base=w.choose(h).float()
                                no_global=pred.clone()
                                no_global[:,63:]=F.layer_norm((base[:,63:].softmax(-1)[...,None]*candidates[:,63:]).sum(-2),(192,))
                                sg=pred.clone();wg=ww[:,63:,[0,5]];wg=wg/wg.sum(-1,keepdim=True).clamp_min(1e-30)
                                sg[:,63:]=F.layer_norm((wg[...,None]*candidates[:,63:,[0,5]]).sum(-2),(192,))
                                normalized_gen=F.layer_norm(gen,(192,))
                                g63=pred.clone();g63[:,63]=normalized_gen[:,63]
                                ghud=pred.clone();ghud[:,63:]=normalized_gen[:,63:]
                                t63=pred.clone();t63[:,63]=true[:,63]
                                thud=pred.clone();thud[:,63:]=true[:,63:]
                                baseline=health(current)
                                for name,tok in [('actual',pred),('no_global_HUD',no_global),('self_gen_HUD',sg),
                                                 ('gen63',g63),('genHUD',ghud),('true63',t63),('trueHUD',thud)]:
                                    delta[name][:,k]=health(tok)-baseline
                                target=true[:,63]
                                error['actual'][:,k]=(pred[:,63]-target).abs().mean(-1).cpu()
                                error['self'][:,k]=(current[:,63]-target).abs().mean(-1).cpu()
                                error['gen'][:,k]=(normalized_gen[:,63]-target).abs().mean(-1).cpu()
                                copies=F.layer_norm(candidates[:,63,:5],(192,))
                                error['best_copy'][:,k]=(copies-target[:,None]).abs().mean(-1).min(-1).values.cpu()
                                weight['self'][:,k]=ww[:,63,0].cpu()
                                weight['neighbours'][:,k]=ww[:,63,1:5].sum(-1).cpu()
                                weight['generate'][:,k]=ww[:,63,5].cpu()
                                for name,x in [('input63',current[:,63]),('output63',pred[:,63]),('gen63',normalized_gen[:,63]),('h63',h[:,63])]:
                                    features[name][:,k]=x.cpu()
                            piece={'delta':delta,'error':error,'weight':weight,'features':features,'reconstruction':reconstruction}
                            store.save('batch_'+str(i),piece,1)
                        pieces.append(piece);max_reconstruction=max(max_reconstruction,piece['reconstruction'])
                        if i%100==0:print(json.dumps({'name':st['name'],'root':i,'max_reconstruction':max_reconstruction}),flush=True)
                trace={key:{name:torch.cat([p[key][name]for p in pieces])for name in pieces[0][key]}
                       for key in ('delta','error','weight','features')}
                trace.update({'reconstruction_max_abs':max_reconstruction,'seed':meta['seed'],'resume_contract':store.contract})
                store.save('result',trace,1)
            out={'name':st['name'],'scope':__doc__,'reconstruction_max_abs':trace['reconstruction_max_abs'],
                 'groups':summary(trace,meta,masks,shift,~fit),'peak_gb':torch.cuda.max_memory_allocated()/1e9,
                 'resume_contract':store.contract}
            R.atomic_torch(HERE/'evals'/(st['name']+'__e19_router_trace.pt'),trace)
            R.atomic_json(HERE/'evals'/(st['name']+'__e19_router_diagnosis.json'),out)
            print(json.dumps({'name':st['name'],'fresh':out['groups']['fresh_hit'],'scroll':out['groups']['scroll_hit']},indent=2),flush=True)
        del w;torch.cuda.empty_cache()


if __name__=='__main__':main()
