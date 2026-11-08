"""CPU census of danger contexts, not inferred causal attribution of damage."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FIELDS=['s_body','s_inv','s_pos','s_light','s_sleep','s_zombies_p','s_zombies_m',
        's_zombies_cd','s_arrows_p','s_arrows_m','s_skeletons_p','s_skeletons_m',
        't_terminated','t_action','ep_T']


def add(a,b):
    for k,v in b.items():
        if isinstance(v,dict):add(a.setdefault(k,{}),v)
        else:a[k]=a.get(k,0)+int(v)


def census(a,ids):
    so=np.r_[0,np.cumsum(a['ep_T']+1)];ao=np.r_[0,np.cumsum(a['ep_T'])];out={}
    for i,eid in enumerate(ids):
        if int(hashlib.sha256(eid.encode()).hexdigest(),16)%10000<500:continue
        g=np.arange(so[i],so[i+1]-1);tr=np.arange(ao[i],ao[i+1]);h=a['s_body'][g,0]
        hp=a['s_body'][g+1,0];dead=a['t_terminated'][tr].astype(bool)
        rel=a['s_zombies_p'][g]-a['s_pos'][g,None];zadj=(np.abs(rel).sum(-1)==1)&a['s_zombies_m'][g]
        ready=(zadj&(a['s_zombies_cd'][g]<=0)).any(-1)
        def visible(kind):
            d=a[f's_{kind}_p'][g]-a['s_pos'][g,None]
            return ((np.abs(d[:,:,0])<=3)&(np.abs(d[:,:,1])<=4)&a[f's_{kind}_m'][g]).any(-1)
        conditions={'all':np.ones(len(g),bool),'night':a['s_light'][g]<.5,
            'day':a['s_light'][g]>=.5,'health_1':h==1,'health_2':h==2,'health_above_2':h>2,
            'food_empty':a['s_body'][g,1]==0,'drink_empty':a['s_body'][g,2]==0,
            'sleeping':a['s_sleep'][g].astype(bool),'zombie_adjacent':zadj.any(-1),
            'adjacent_zombie_ready':ready,'adjacent_zombie_not_ready':zadj.any(-1)&~ready,
            'arrow_visible':visible('arrows'),'skeleton_visible':visible('skeletons'),
            'iron_equipment':(a['s_inv'][g,8]>0)|(a['s_inv'][g,11]>0)}
        events={'damage':hp<h,'death':dead,'risk':(h<=2)&zadj.any(-1),
                'risk_death':(h<=2)&zadj.any(-1)&dead}
        result={c:{'transitions':int(cond.sum()),'episodes':int(cond.any()),
                    **{name:dict(events=int((hit&cond).sum()),episodes=int((hit&cond).any()))
                       for name,hit in events.items()}} for c,cond in conditions.items()}
        add(out,result)
    return out


def source(store):
    m=json.loads((store/'manifest.json').read_text());out={};by_component={}
    for sh in m['shards']:
        p=store/sh['audit_file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sh['audit_sha256']
        with np.load(p) as z:
            a={k:z[k] for k in FIELDS}
            ids=[f"coverage-v1:{m['contract']['seed']}:{int(r)}:{int(s)}" for r,s in zip(z['ep_round'],z['ep_slot'])]
        assert len(ids)==sh['episodes'] and int(a['ep_T'].sum())==sh['transitions']
        q=census(a,ids);add(out,q);add(by_component.setdefault(sh['policy']['component'],{}),q)
    train=m['coverage']['train']
    for k,expected in [('damage',train['damage']),('death',train['deaths']),('risk',train['risk']),('risk_death',train['risk_deaths'])]:
        assert out['all'][k]['events']==expected,(store,k)
    assert out['all']['transitions']==train['transitions']
    return dict(complete=m['complete'],snapshot_rounds=m['rounds'],counts=out,components=by_component)


def main():
    stores={'advanced':ROOT/'artifacts/craftax_coverage_v1','danger':ROOT/'artifacts/craftax_risk_v1'}
    out={k:source(p) for k,p in stores.items() if (p/'manifest.json').exists() and
         json.loads((p/'manifest.json').read_text())['rounds']}
    out['interpretation']='Pre-transition contexts overlap. Empty hunger/thirst, visible projectiles or nearby zombies are not claims of the actual cause of death. Episode counts are independent source episodes within each context, not sum-disjoint categories. Danger store is a committed-batch snapshot if still running.'
    (HERE/'DANGER_CONTEXTS.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,q in out.items():
        if name=='interpretation':continue
        print(name,q['snapshot_rounds'],{k:{s:v[s]['events'] for s in ['damage','death','risk_death']}
            for k,v in q['counts'].items() if k in ['all','night','food_empty','drink_empty','sleeping','arrow_visible','iron_equipment']})


if __name__=='__main__':main()
