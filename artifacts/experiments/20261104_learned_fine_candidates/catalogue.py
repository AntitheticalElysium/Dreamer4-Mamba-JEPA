"""CPU source catalogue; fixed existing F1.6 readers only. No fitted weights."""
from pathlib import Path
import json
import numpy as np
import torch
from saccade.eval import s1 as S

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CK=ROOT/'artifacts/experiments/20261102_p1f16/run/ckpt_10.pt'
DIR=np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0]])

def main():
    torch.set_num_threads(2)
    L=dict(np.load(S.LABELS/'s0_dev_labels.npz'))
    cache=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',weights_only=False,map_location='cpu',mmap=True)
    assert cache['checkpoint_sha256']=='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'
    z=cache['current'];reader=S.load_reader(CK.parent/'s1/results.reader.pt',CK,'cpu')
    so=np.r_[0,np.cumsum(L['ep_T']+1)];rows=[]
    for e,T in enumerate(L['ep_T'][:400]):
        ix=torch.where(z['episode']==e)[0];assert len(ix)==T
        for t in range(8,int(T)):
            si=int(so[e]+t);f=int(ix[t]);act=int(z['action'][f]);pos=L['s_pos'][si]
            if L['s_sleep'][si] or L['s_body'][si,0]<=0:continue
            if act in [1,2,3,4]:d=DIR[act]
            elif act==5:d=DIR[L['s_dir'][si]]
            else:continue
            cell=int((3+d[0])*9+4+d[1]);wrld=pos+d;terrain=int(L['s_view'][si].reshape(-1)[cell]);entities=[]
            for kind,ky in [('zombies',1),('cows',2),('skeletons',3),('arrows',4)]:
                jj=np.flatnonzero(L[f's_{kind}_m'][si]&(L[f's_{kind}_p'][si]==wrld).all(-1))
                entities.extend((kind,ky,int(j)) for j in jj)
            r=dict(e=e,t=t,f=f,shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e]),action=act,cell=cell,
                terrain=terrain,light=float(L['s_light'][si]),direction=d.tolist(),entities=entities,
                actually_moved=bool(np.any(L['s_pos'][si+1]!=pos)))
            # BlockType values read from the installed constants below.
            if act in [1,2,3,4] and entities:
                if len(entities)==1 and entities[0][0]!='arrows' and terrain in WALKABLE:
                    r.update(family='mob_blockage',kind=entities[0][0],ky=entities[0][1],j=entities[0][2]);rows.append(r)
            elif not entities and act==5 and terrain in RESOURCE:
                r.update(family='resource',kind=RESOURCE[terrain]);rows.append(r)
            elif not entities and act in [1,2,3,4] and terrain in BLOCKERS:
                r.update(family='terrain_blockage',kind=BLOCKERS[terrain]);rows.append(r)
    feats=torch.stack([torch.cat([z['p'][r['f'],r['cell']].float(),z['v'][r['f']].float()]) for r in rows])
    preds={k:[] for k in ['kind','terr']}
    with torch.no_grad():
        for a in feats.split(4096):
            o=reader(a)
            for k in preds:preds[k].extend(o[k].argmax(-1).tolist())
    for i,r in enumerate(rows):
        r.update(coarse_kind=preds['kind'][i],coarse_terrain=preds['terr'][i])
        if r['family']=='mob_blockage':r['coarse_missed_task']=preds['kind'][i]==0
        elif r['family']=='resource':r['coarse_missed_task']=preds['terr'][i]!=r['terrain']
        else:r['coarse_missed_task']=preds['terr'][i] in WALKABLE
    counts={}
    for r in rows:
        k=r['family']+'/'+r['kind'];a=counts.setdefault(k,dict(rows=0,missed=0,night_rows=0,night_missed=0))
        a['rows']+=1;a['missed']+=r['coarse_missed_task'];a['night_rows']+=r['light']<.5;a['night_missed']+=r['coarse_missed_task'] and r['light']<.5
    selected=[];used={}
    # Prefer night for clear references, but preserve corpus order within stratum.
    for r in sorted(rows,key=lambda r:r['light']>=.5):
        k=(r['family'],r['kind'],r['coarse_missed_task']);seen=used.setdefault(k,set())
        if len(seen)<12 and r['e'] not in seen:seen.add(r['e']);selected.append(r)
    (HERE/'catalogue.json').write_text(json.dumps(dict(counts=counts,rows=rows,selected=selected),indent=1)+'\n')
    print(json.dumps(counts,indent=2));print('selected',len(selected),flush=True)

if __name__=='__main__':
    from craftax.craftax_classic.constants import BlockType as B
    WALKABLE={B.GRASS.value,B.PATH.value,B.SAND.value}
    RESOURCE={B.WATER.value:'water',B.TREE.value:'tree',B.STONE.value:'stone'}
    BLOCKERS={b.value:b.name.lower() for b in [B.WATER,B.TREE,B.STONE,B.COAL,B.IRON,B.CRAFTING_TABLE,B.FURNACE,B.PLANT,B.RIPE_PLANT]}
    main()
