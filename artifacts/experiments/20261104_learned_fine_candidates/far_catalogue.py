"""Source-only distant semantic-cue catalogue, using existing coarse reader."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import json
import numpy as np
import torch
from saccade.eval import s1 as S,gates as G
from craftax.craftax_classic.constants import BlockType as B
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CK=ROOT/'artifacts/experiments/20261102_p1f16/run/ckpt_10.pt'

def main():
    torch.set_num_threads(2);L=dict(np.load(S.LABELS/'s0_dev_labels.npz'))
    z=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',weights_only=False,mmap=True)['current']
    reader=S.load_reader(CK.parent/'s1/results.reader.pt',CK,'cpu');so=np.r_[0,np.cumsum(L['ep_T']+1)]
    classes={b.value:b.name.lower() for b in [B.TREE,B.STONE,B.COAL,B.IRON,B.CRAFTING_TABLE,B.FURNACE]};rows=[];at=0
    rr,cc=np.divmod(np.arange(63),9);far=abs(rr-3)+abs(cc-4)>=3
    for e,T in enumerate(L['ep_T'][:400]):
        assert int(z['episode'][at])==e
        for t in range(8,int(T)):
            s=int(so[e]+t);f=at+t;action=int(z['action'][f])
            if L['s_sleep'][s] or L['s_body'][s,0]<=2 or L['s_light'][s]>=.5 or action not in [0,1,2,3,4]:continue
            position=L['s_pos'][s];visible={};occupied=set()
            for kind,ky in [('zombies',1),('cows',2),('skeletons',3),('arrows',4)]:
                ps=L[f's_{kind}_p'][s]-position;mask=L[f's_{kind}_m'][s]&(abs(ps[:,0])<=3)&(abs(ps[:,1])<=4)
                for j in np.flatnonzero(mask):
                    cell=int((ps[j,0]+3)*9+ps[j,1]+4);visible[cell]=(kind,ky,int(j));occupied.add(cell)
            common=dict(e=e,t=t,f=f,shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e]),action=action,light=float(L['s_light'][s]))
            for cell,(kind,ky,j) in visible.items():
                if kind=='arrows' or not far[cell]:continue
                row,col=divmod(cell,9);d=np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0]])[action]
                # All one-cell mob moves remain visible whether the requested
                # player move succeeds or is blocked. Thus existence, rather
                # than an exact stochastic destination, is the named event.
                if not all(1<=row-v[0]<=5 and 1<=col-v[1]<=7 for v in [np.zeros(2,int),d]):continue
                here=L[f's_{kind}_p'][s,j];others=L[f's_{kind}_m'][s].copy();others[j]=False
                if np.any(others&(abs(L[f's_{kind}_p'][s]-here).sum(-1)<=3)):continue
                prev=L[f's_{kind}_p'][s-8:s,j]-L['s_pos'][s-8:s]
                seen=L[f's_{kind}_m'][s-8:s,j]&(abs(prev[:,0])<=3)&(abs(prev[:,1])<=4)
                rows.append(dict(**common,cell=cell,family='mob',kind=kind,ky=ky,j=j,previous8_visible=int(seen.sum())))
            view=L['s_view'][s].reshape(-1)
            for cell in np.flatnonzero(far&np.isin(view,list(classes))):
                if int(cell) in occupied:continue
                row,col=divmod(int(cell),9);d=np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0]])[action]
                if not (0<=row-d[0]<=6 and 0<=col-d[1]<=8):continue
                world=position+np.array(divmod(int(cell),9))-np.array([3,4]);prev=world-L['s_pos'][s-8:s]
                seen=(abs(prev[:,0])<=3)&(abs(prev[:,1])<=4)
                rows.append(dict(**common,cell=int(cell),family='terrain',kind=classes[int(view[cell])],ky=int(view[cell]),previous8_visible=int(seen.sum())))
        at+=int(T)
    preds=[]
    with torch.no_grad():
        for a in range(0,len(rows),8192):
            block=rows[a:a+8192];ii=torch.tensor([r['f'] for r in block]);cells=torch.tensor([r['cell'] for r in block]);o=reader(torch.cat([z['p'][ii,cells].float(),z['v'][ii].float()],-1))
            p=o['kind'].argmax(-1).tolist();q=o['terr'].argmax(-1).tolist()
            for r,k,terr in zip(block,p,q):
                r['coarse_kind']=k;r['coarse_terrain']=terr;r['missed']=k==0 if r['family']=='mob' else terr!=r['ky']
    counts={}
    for r in rows:
        k=r['family']+'/'+r['kind'];v=counts.setdefault(k,dict(rows=0,missed=0,new_missed=0,missed_episodes=set()))
        v['rows']+=1;v['missed']+=r['missed'];v['new_missed']+=r['missed'] and r['previous8_visible']==0
        if r['missed']:v['missed_episodes'].add(r['e'])
    for v in counts.values():v['missed_episodes']=len(v['missed_episodes'])
    terrain=sorted([k for k in counts if k.startswith('terrain/')],key=lambda k:(-counts[k]['missed_episodes'],k))[:2]
    chosen={k for k in counts if k.startswith('mob/')}|set(terrain);selected=[];used={}
    for r in sorted(rows,key=lambda r:(r['previous8_visible']>0,r['e'],r['t'],r['cell'])):
        key=r['family']+'/'+r['kind'];u=used.setdefault((key,r['missed']),set())
        if key in chosen and len(u)<12 and r['e'] not in u:
            selected.append(r);u.add(r['e'])
    (HERE/'far_catalogue.json').write_text(json.dumps(dict(counts=counts,selected=selected,terrain_families=terrain),indent=2)+'\n')
    print(json.dumps(counts,indent=2));print('selected',len(selected),'terrain',terrain)

if __name__=='__main__':main()
