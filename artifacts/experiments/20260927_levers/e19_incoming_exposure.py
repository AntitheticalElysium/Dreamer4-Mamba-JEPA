"""CPU TRAIN coverage of the gate's incoming-hazard/fresh-arrival condition.

Fresh: pre-move zombie beside post-move player, no adjacency in current/prior3
frames, no >=2 damage in prior3 transitions. Four frames exist at original target
positions3/4. Zombie is a frozen TRAIN-fitted detector; displacement comes from
true-transition token correspondence, both labelled approximations. Count sampled
window-targets and deduplicated episode/step targets, plus actual boundary lengths.
No simulator oracle, GPU or model training. Atomic source/input-bound chunks.
"""
import json
from pathlib import Path
import torch
import teval as T
import e19 as E
import h16_resume as R
from scroll import estimate,SHIFTS

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(3);pp=E.T.POOLS['raw']/'pool.pt';cp=Path(str(T.CACHE).format('raw'));assert cp.exists()
    pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
    mask=torch.load(E.MASK,map_location='cpu',weights_only=False)['beside'].bool()
    root=E.T.OUT/'state/e19_C_s7_fmamba_from36000';oldspec=json.loads((root/'contract.json').read_text())
    train=R.Store(root,oldspec).load('train');ledger=train['ledger'].long();layout=train['layout'].long()
    assert train['update']==6000 and ledger.shape==layout.shape==(6000,40)
    rows=E.train_rows(pool);assert torch.isin(ledger,rows).all()
    by_boundary=torch.bincount((ledger*5+layout).flatten(),minlength=len(mask)*5).reshape(len(mask),5)
    multiplicity=by_boundary.sum(1);ids=(multiplicity>0).nonzero().flatten()
    spec={'scope':__doc__,'ledger':R.tensor_hash(ledger),'layout':R.tensor_hash(layout),
          'inputs':{str(p):R.file_hash(p)for p in (pp,cp,E.MASK,T.META,root/'contract.json',root/'train.json')},
          'sources':{str(p.resolve()):R.file_hash(p)for p in (Path(__file__),Path(T.__file__),Path(E.__file__),
                   HERE/'scroll.py',HERE/'h16_resume.py',T.ROOT/'artifacts/experiments/20260921_readout_ladder/spatial.py')},
          'runtime':{'torch':str(torch.__version__),'precision':'CPU FP32','threads':3},'batch':128}
    store=R.Store(HERE/'evals/resume/e19_fmamba__incoming_exposure',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done,indent=2),flush=True);return
        cache=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
        meta,fit,seeds=T.split();probes=T.Probes(cache,meta,fit,seeds)
        # Only cells at Manhattan distance <=2 from the current player can
        # neighbor the post-move player under a single cardinal displacement.
        cells=torch.tensor([r*9+c for r in range(7)for c in range(9)if abs(r-3)+abs(c-4)<=2])
        coords=torch.stack([cells//9,cells%9],1)
        parts=[]
        for start in range(0,len(ids),128):
            part=store.load('batch_'+str(start))
            if part is None:
                rr=ids[start:start+128];s=pool['tokens'][rr].float();b=len(rr)
                zombie=(probes.zombie(s[:,:,cells].flatten(0,2)).reshape(b,6,len(cells))>.3)
                nowdist=(coords-torch.tensor([3,4])).abs().sum(1)
                adjacent_now=(zombie[:,:,nowdist==1]).any(-1)
                assert torch.equal(adjacent_now[:,1:],mask[rr])
                shift=estimate(s[:,:-1],s[:,1:]);disp=torch.tensor(SHIFTS)[shift]
                dist=(coords[None,None]-torch.tensor([3,4])[None,None,None]-disp[:,:,None]).abs().sum(-1)
                adjacent=(zombie[:,:-1]&(dist==1)).any(-1)
                dh=pool['dh'][rr];alive=pool['alive'][rr,1:]
                valid=torch.zeros_like(alive);valid[:,3:]=alive[:,3:]
                adjwin=torch.zeros_like(alive);damagewin=torch.zeros_like(alive)
                for t in (3,4):
                    adjwin[:,t]=adjacent_now[:,t-3:t+1].any(1)
                    damagewin[:,t]=(dh[:,t-3:t]<=-2).any(1)
                fresh=valid&adjacent&~adjwin&~damagewin
                groups={'four_frame_alive':valid,'post_move_adjacent':valid&adjacent,
                        'fresh_incoming':fresh,'scroll_fresh_incoming':fresh&(shift!=0)}
                weights=multiplicity[rr,None].expand(-1,5)
                available=torch.zeros(b,5,dtype=torch.long)
                for boundary in range(5):
                    cut=4-boundary
                    for t in range(5):
                        length=t+1 if t<cut else t-cut+1
                        if length>=4:available[:,t]+=by_boundary[rr,boundary]
                categories={'ordinary_ge2':dh<=-2,'unchanged':dh==0,'other':(dh>-2)&(dh!=0)}
                counts={};events={}
                for g,gm in groups.items():
                    counts[g]={};events[g]={}
                    for c,cm in categories.items():
                        q=gm&cm;selected=q&mask[rr]
                        counts[g][c]={'window_target_pairs':int(q.sum()),'sampled_targets':int(weights[q].sum()),
                                     'selected_sampled':int(weights[selected].sum()),
                                     'boundary_four_frames_sampled':int(available[q].sum())}
                        events[g][c]=[(pool['ids'][int(rr[j])][0],int(pool['ids'][int(rr[j])][1])+int(t))
                                      for j,t in q.nonzero().tolist()]
                part={'counts':counts,'events':events};store.save('batch_'+str(start),part,1)
            parts.append(part)
            if start%1024==0:print(json.dumps({'rows_done':start+len(ids[start:start+128])}),flush=True)
        counts={g:{c:{k:sum(p['counts'][g][c][k]for p in parts)for k in v}for c,v in cs.items()}
                for g,cs in parts[0]['counts'].items()}
        for g,cs in counts.items():
            for c,v in cs.items():v['unique_episode_step_targets']=len({x for p in parts for x in p['events'][g][c]})
        result={'scope':__doc__,'contract':spec,'sampled_windows':ledger.numel(),'groups':counts,
                'stored_mask_reproduction_disagreements':0}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e19_fmamba__incoming_exposure.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
