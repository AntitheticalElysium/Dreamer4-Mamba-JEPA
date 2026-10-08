"""CPU census of every actually sampled E19 TRAIN target, including mask coverage.

Use original pool dh/alive labels and the saved 6000x40 sampled-row ledger.
Current/next adjacency and health bins are frozen TRAIN-fitted image readouts,
not simulator annotations. Scroll is the same token correspondence estimator.
First encounter means first detected adjacency inside the original six-frame
window, not the full episode. Hash-bound atomic chunk records, no GPU/training.
"""
import json
from pathlib import Path
import torch
import teval as T
import e19 as E
import h16_resume as R
from scroll import estimate

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(3)
    pp=E.T.POOLS['raw']/'pool.pt';cp=Path(str(T.CACHE).format('raw'))
    assert cp.exists()
    pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
    mask=torch.load(E.MASK,map_location='cpu',weights_only=False)['beside'].bool()
    states={}
    bound=[]
    for seed in (7,8):
        for arm in 'ABC':
            root=E.T.OUT/f'state/e19_{arm}_s{seed}_fmamba_from36000'
            spec=json.loads((root/'contract.json').read_text())
            st=R.Store(root,spec).load('train');assert st['update']==6000
            states[(seed,arm)]=(st['ledger'],st['layout'])
            bound.append(root/'contract.json')
            j=json.loads((root/'train.json').read_text());bound.extend([root/'train.json',root/j['file']])
    ledger,layout=states[(7,'C')]
    assert ledger.shape==(6000,40)
    assert all(torch.equal(ledger,v[0])and torch.equal(layout,v[1])for v in states.values())
    assert torch.isin(ledger.long(),E.train_rows(pool)).all()
    multiplicity=torch.bincount(ledger.flatten().long(),minlength=len(mask))
    spec={'scope':__doc__,'ledger':R.tensor_hash(ledger),'layout':R.tensor_hash(layout),
          'inputs':{str(p):R.file_hash(p)for p in [pp,E.MASK,cp,T.META,*bound]},
          'sources':{str(p.resolve()):R.file_hash(p)for p in [Path(__file__),Path(T.__file__),
             Path(E.__file__),HERE/'scroll.py',HERE/'h16_resume.py']},
          'runtime':{'torch':str(torch.__version__),'precision':'CPU FP32','threads':3},'batch':128}
    name='e19_fmamba__exposure_census';store=R.Store(HERE/'evals/resume'/name,spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done,indent=2),flush=True);return
        cache=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
        meta,fit,seeds=T.split();probes=T.Probes(cache,meta,fit,seeds)
        ids=(multiplicity>0).nonzero().flatten();pieces=[]
        for start in range(0,len(ids),128):
            part=store.load('batch_'+str(start))
            if part is None:
                rr=ids[start:start+128];s=pool['tokens'][rr].float()
                adj=probes.zombie(s[:,:, [22,30,32,40]].flatten(0,2)).reshape(len(rr),6,4).amax(-1)>.3
                assert torch.equal(adj[:,1:],mask[rr]),'Exact next-frame mask did not reproduce'
                health=probes.hud(s[:,:,63:81].flatten(0,1).flatten(1))[:,0].reshape(len(rr),6)*9
                hp=health[:,:-1].round().clamp(0,9).long()
                shift=estimate(s[:,:-1],s[:,1:]);dh=pool['dh'][rr];alive=pool['alive'][rr,1:]
                classes={'death':~alive,'ordinary_ge2':alive&(dh<=-2),'unchanged':alive&(dh==0)}
                classes['other']=~(classes['death']|classes['ordinary_ge2']|classes['unchanged'])
                ca=adj[:,:-1];prioradj=ca.long().cumsum(1)-ca.long()
                damage=dh<0;priordamage=damage.long().cumsum(1)-damage.long()
                fresh=ca&(prioradj==0)&(priordamage==0)
                groups={'all':torch.ones_like(ca),'scroll':shift!=0,'stationary':shift==0,
                        'current_adjacent':ca,'first_encounter_in_window':fresh,
                        'scroll_first_encounter':fresh&(shift!=0),'health9':hp==9,'below9':hp<9}
                counts={}
                for label,g in groups.items():
                    counts[label]={}
                    for c,m in classes.items():
                        q=g&m;w=multiplicity[rr,None].expand_as(q)
                        counts[label][c]={'unique_targets':int(q.sum()),'sampled_targets':int(w[q].sum()),
                            'selected_unique':int((q&mask[rr]).sum()),'selected_sampled':int(w[q&mask[rr]].sum())}
                part={'counts':counts,'rows':rr,'mask_disagreements':0,
                      'health_bins_sampled':{str(h):int((multiplicity[rr,None]*(hp==h)).sum())for h in range(10)}}
                store.save('batch_'+str(start),part,1)
            pieces.append(part)
            if start%1024==0:print(json.dumps({'rows_done':start+len(part['rows'])}),flush=True)
        total={g:{c:{k:sum(p['counts'][g][c][k]for p in pieces)for k in v}for c,v in cs.items()}
               for g,cs in pieces[0]['counts'].items()}
        assert sum(v['sampled_targets']for v in total['all'].values())==1200000
        counts={h:sum(p['health_bins_sampled'][h]for p in pieces)for h in pieces[0]['health_bins_sampled']}
        result={'scope':__doc__,'contract':spec,'sampled_windows':ledger.numel(),'unique_sampled_windows':len(ids),
                'targets':total,'health_bins_sampled':counts,'mask_reproduction_disagreements':0,
                'ledgers_equal_all_six_arms':True}
        store.save('result',result,1);R.atomic_json(HERE/'evals'/(name+'.json'),result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
