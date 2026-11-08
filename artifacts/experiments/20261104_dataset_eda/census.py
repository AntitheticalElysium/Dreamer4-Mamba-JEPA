"""Full TRAIN-only CPU census. No model changes or fitted scientific proxies."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
ACH=['COLLECT_WOOD','PLACE_TABLE','EAT_COW','COLLECT_SAPLING','COLLECT_DRINK','MAKE_WOOD_PICKAXE',
     'MAKE_WOOD_SWORD','PLACE_PLANT','DEFEAT_ZOMBIE','COLLECT_STONE','PLACE_STONE','EAT_PLANT',
     'DEFEAT_SKELETON','MAKE_STONE_PICKAXE','MAKE_STONE_SWORD','WAKE_UP','PLACE_FURNACE',
     'COLLECT_COAL','COLLECT_IRON','COLLECT_DIAMOND','MAKE_IRON_PICKAXE','MAKE_IRON_SWORD']
INV=['wood','stone','coal','iron','diamond','sapling','wood_pickaxe','stone_pickaxe','iron_pickaxe',
     'wood_sword','stone_sword','iron_sword']
BLOCKS=['INVALID','OUT_OF_BOUNDS','GRASS','WATER','STONE','TREE','WOOD','PATH','COAL','IRON',
        'DIAMOND','CRAFTING_TABLE','FURNACE','SAND','LAVA','PLANT','RIPE_PLANT']
DIR=np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0]])
FIELDS=['s_body','s_inv','s_pos','s_dir','s_light','s_sleep','s_view','s_ach',
        *[f's_{k}_{v}' for k in ['zombies','cows','skeletons','arrows'] for v in ['p','m','cd']],
        't_terminated','t_exogenous','t_action','t_reward','ep_T']
RESOURCES={'drink':(3,'s_body',2,None),'wood':(5,'s_inv',0,None),'sapling':(2,'s_inv',5,None),
           'stone':(4,'s_inv',1,6),'coal':(8,'s_inv',2,6),'iron':(9,'s_inv',3,7),
           'diamond':(10,'s_inv',4,8)}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def merge(a,b):
    for k,v in b.items():
        if isinstance(v,dict):merge(a.setdefault(k,{}),v)
        else:a[k]=a.get(k,0)+v


def exogenous_windows(exogenous,terminal,H):
    T=len(exogenous);start=np.arange(T);end=np.minimum(start+H,T)
    bad=np.r_[0,np.cumsum(~exogenous.astype(bool))]
    valid=(bad[end]==bad[start])&((start+H<=T)|terminal)
    start=start[valid];end=end[valid]
    d=np.zeros(T+1,np.int32);np.add.at(d,start,1);np.add.at(d,end,-1)
    return start,end,np.cumsum(d)[:-1]>0


def episode(a,g,tr):
    T=len(g);o={}
    def bump(k,v):
        n=int(np.count_nonzero(v)) if np.ndim(v) else int(v)
        o[k]={'n':n,'episodes':int(n>0)}
    action=a['t_action'][tr];dead=a['t_terminated'][tr].astype(bool);rnd=a['t_exogenous'][tr].astype(bool)
    hp=a['s_body'][g,0];next_hp=a['s_body'][g+1,0];night=a['s_light'][g]<.5
    sleep=a['s_sleep'][g].astype(bool);awake=~sleep
    zdist=np.abs(a['s_zombies_p'][g]-a['s_pos'][g,None]).sum(-1)
    zadj=(zdist==1)&a['s_zombies_m'][g].astype(bool)
    ready=(zadj&(a['s_zombies_cd'][g]<=0)).any(-1)
    risk=(hp<=2)&zadj.any(-1)
    events={'damage':next_hp<hp,'death':dead,'heal':next_hp>hp,'risk':risk,'risk_death':risk&dead,
            'risk_damage':risk&(next_hp<hp),'risk_survival':risk&~dead,'risk_ready':risk&ready,
            'sleep_enter':~sleep&a['s_sleep'][g+1].astype(bool),
            'wake':sleep&~a['s_sleep'][g+1].astype(bool),'positive_reward':a['t_reward'][tr]>0}
    for k,v in events.items():bump('event/'+k,v)
    bump('total/transitions',T);bump('total/frames',T+1);bump('total/episodes',1)
    bump('total/truncated',not dead[-1]);bump('total/night',night);bump('total/random_actions',rnd)
    for act,n in zip(*np.unique(action,return_counts=True)):bump(f'action/{int(act)}',int(n))
    age=np.arange(1,T+1);bands={'1_7':age<8,'8_63':(age>=8)&(age<64),'64_255':(age>=64)&(age<256),
                             '256_1023':(age>=256)&(age<1024),'1024_plus':age>=1024}
    for k,v in bands.items():bump('age/'+k,v)
    unlock=a['s_ach'][g+1].astype(bool)&~a['s_ach'][g].astype(bool)
    assert (unlock.sum(0)<=1).all(),'Achievement should unlock at most once per episode'
    for col,name in enumerate(ACH):
        v=unlock[:,col];bump('ach/'+name,v);bump('ach_night/'+name,v&night)
        for band,mask in bands.items():bump('ach_age/'+name+'/'+band,v&mask)
    for col,name in enumerate(INV):
        before=a['s_inv'][g,col];after=a['s_inv'][g+1,col];inc=after>before
        bump('inc/'+name,inc);bump('inc_night/'+name,inc&night);bump('dec/'+name,after<before)
        for value in range(10):bump('inc_value/'+name+'/'+str(value),inc&(before==value))
    for col,name in enumerate(['health','food','drink','energy']):
        before=a['s_body'][g,col];after=a['s_body'][g+1,col]
        bump('body_inc/'+name,after>before);bump('body_dec/'+name,after<before)
        for value in range(10):bump('body_inc_value/'+name+'/'+str(value),(after>before)&(before==value))
    visible={}
    for kind in ['zombies','cows','skeletons','arrows']:
        rel=a[f's_{kind}_p'][g]-a['s_pos'][g,None];on=a[f's_{kind}_m'][g].astype(bool)
        vis=on&(np.abs(rel[:,:,0])<=3)&(np.abs(rel[:,:,1])<=4)
        nxt=a[f's_{kind}_p'][g+1]-a['s_pos'][g+1,None]
        nv=a[f's_{kind}_m'][g+1]&(np.abs(nxt[:,:,0])<=3)&(np.abs(nxt[:,:,1])<=4)
        visible[kind]=vis.any(-1)
        bump('entity/'+kind+'/instances',vis);bump('entity/'+kind+'/night_instances',vis&night[:,None])
        bump('entity/'+kind+'/frames',visible[kind]);bump('entity/'+kind+'/near2_frames',(vis&(np.abs(rel).max(-1)<=2)).any(-1))
        bump('entity/'+kind+'/enter_visible',nv&~vis)
    conditions={'day':~night,'night':night,'sleeping':sleep,'awake':awake,
        'health1':hp==1,'health2':hp==2,'health_above2':hp>2,
        'food_empty':a['s_body'][g,1]==0,'drink_empty':a['s_body'][g,2]==0,
        'zombie_adjacent':zadj.any(-1),'zombie_ready':ready,'zombie_unready':zadj.any(-1)&~ready,
        'arrow_visible':visible['arrows'],'skeleton_visible':visible['skeletons'],
        'iron_equipment':(a['s_inv'][g,8]>0)|(a['s_inv'][g,11]>0),
        'risk_night':risk&night,'risk_day':risk&~night,'risk_ready':risk&ready,
        'risk_unready':risk&~ready,'risk_sleeping':risk&sleep,'risk_awake':risk&awake}
    for k,v in conditions.items():
        bump('condition/'+k+'/frames',v)
        for ev in ['damage','death','risk','risk_death']:bump('condition/'+k+'/'+ev,v&events[ev])
    for act in range(17):
        for ev in ['risk','risk_death','damage','death']:bump('action_condition/'+str(act)+'/'+ev,(action==act)&events[ev])
    for value in range(10):
        bump('health_before/'+str(value),hp==value)
        bump('health_death/'+str(value),(hp==value)&dead)
    lava=a['s_view'][g+1,3,4]==14
    bump('death_context/lava',dead&lava)
    bump('death_context/health_zero',dead&(next_hp<=0))
    bump('death_context/lava_and_health_zero',dead&lava&(next_hp<=0))
    bump('death_context/nonlava_health_zero',dead&~lava&(next_hp<=0))
    bump('death_context/other',dead&~lava&(next_hp>0))
    for band,mask in bands.items():
        for name in ['death','damage','risk','risk_death']:
            bump('event_age/'+name+'/'+band,events[name]&mask)
    dd=DIR[a['s_dir'][g]];face=a['s_view'][g,3+dd[:,0],4+dd[:,1]]
    for name,(block,key,col,tool) in RESOURCES.items():
        op=awake&(face==block)&(a[key][g,col]<9)
        tool_ok=np.ones(T,bool) if tool is None else a['s_inv'][g,tool]>0
        inc=a[key][g+1,col]>a[key][g,col]
        for tag,mask in [('all',op&tool_ok),('DO',op&tool_ok&(action==5)),('NOOP',op&tool_ok&(action==0)),
                         ('other',op&tool_ok&(action!=0)&(action!=5)),('DO_no_tool',op&~tool_ok&(action==5))]:
            bump('resource/'+name+'/'+tag,mask)
            bump('resource/'+name+'/'+tag+'_increment',mask&inc)
            bump('resource/'+name+'/'+tag+'_random',mask&rnd)
            bump('resource/'+name+'/'+tag+'_random_increment',mask&rnd&inc)
    for col,name in enumerate(BLOCKS):bump('terrain/'+name,a['s_view'][g]==col)
    pos_delta=a['s_pos'][g+1]-a['s_pos'][g]
    assert (np.abs(pos_delta).sum(-1)<=1).all(),'Unexpected teleport/reset within a retained episode'
    for dx,dy in [(0,0),(0,1),(0,-1),(1,0),(-1,0)]:
        ix=np.flatnonzero((pos_delta==[dx,dy]).all(-1))
        if not len(ix):continue
        lo,hi=max(0,-dx),min(7,7-dx);cl,ch=max(0,-dy),min(9,9-dy)
        before=a['s_view'][g[ix],lo+dx:hi+dx,cl+dy:ch+dy]
        after=a['s_view'][g[ix]+1,lo:hi,cl:ch]
        changed=(before!=after)&(before>=2)&(after>=2)
        pairs=np.bincount(before[changed].astype(int)*17+after[changed].astype(int),minlength=289)
        for pair in np.flatnonzero(pairs):
            k=f'edit/{BLOCKS[pair//17]}_to_{BLOCKS[pair%17]}'
            n=int(pairs[pair]);old=o.setdefault(k,{'n':0,'episodes':0});old['n']+=n
            # One episode can have several scroll directions; set support once.
            old['episodes']=1
    for H in [2,4,8,16]:
        starts,ends,covered=exogenous_windows(rnd,bool(dead[-1]),H)
        bump(f'exog/{H}/windows',len(starts));bump(f'exog/{H}/covered_transitions',covered)
        bump(f'exog/{H}/full_length_windows',starts+H<=T)
        bump(f'exog/{H}/absorbing_suffix_windows',starts+H>T)
        for k,v in events.items():
            cp=np.r_[0,np.cumsum(v)];bump(f'exog/{H}/event_windows/'+k,cp[ends]>cp[starts])
            bump(f'exog/{H}/unique_event/'+k,v&covered)
        for j,name in enumerate(ACH):
            cp=np.r_[0,np.cumsum(unlock[:,j])]
            bump(f'exog/{H}/ach_windows/'+name,cp[ends]>cp[starts])
            bump(f'exog/{H}/unique_ach/'+name,unlock[:,j]&covered)
        for j,name in enumerate(INV):
            inc=a['s_inv'][g+1,j]>a['s_inv'][g,j]
            bump(f'exog/{H}/unique_inc/'+name,inc&covered)
        inc=a['s_body'][g+1,2]>a['s_body'][g,2]
        bump(f'exog/{H}/unique_inc/drink',inc&covered)
    return o,{'T':T,'death':bool(dead[-1]),'achievements':int(unlock.sum()),'damage':int(events['damage'].sum()),
              'risk':int(risk.sum()),'risk_death':int((risk&dead).sum())}


def add_source(a,ids,groups,out,rows):
    so=np.r_[0,np.cumsum(a['ep_T']+1)];ao=np.r_[0,np.cumsum(a['ep_T'])]
    for i,eid in enumerate(ids):
        if int(hashlib.sha256(eid.encode()).hexdigest(),16)%10000<500:continue
        g=np.arange(so[i],so[i+1]-1);tr=np.arange(ao[i],ao[i+1]);stats,row=episode(a,g,tr)
        merge(out.setdefault(groups[i],{}),stats)
        row.update(id=eid,group=groups[i]);rows.append(row)


def main():
    started=time.time();out={};rows=[];pins={};snapshots={}
    oldpath=ROOT/'artifacts/eda/s0/s0_train_labels.npz';meta=json.loads((ROOT/'artifacts/craftax_support_v2/manifest.json').read_text())
    with np.load(oldpath) as z:
        a={k:z[k] for k in FIELDS};ids=[f"support-v2:{meta['seed']}:{int(s)}:{int(t)}" for s,t in zip(z['ep_shard'],z['ep_slot'])]
        groups=['old_eps_'+str(round(float(v),2)) for v in z['ep_eps']]
    pins[str(oldpath)]=sha(oldpath);add_source(a,ids,groups,out,rows)
    print('old complete',len(rows),time.time()-started,flush=True);del a
    for pool,dirname in [('advanced','craftax_coverage_v1'),('danger','craftax_risk_v1')]:
        store=ROOT/'artifacts'/dirname;raw=(store/'manifest.json').read_bytes();m=json.loads(raw)
        snapshots[pool]=m; (HERE/(pool+'_manifest_snapshot.json')).write_bytes(raw)
        pins[str(store/'manifest.json')]=hashlib.sha256(raw).hexdigest()
        for sh in m['shards']:
            side=store/sh['audit_file'];assert sha(side)==sh['audit_sha256']
            with np.load(side) as z:
                a={k:z[k] for k in FIELDS}
                ids=[f"coverage-v1:{m['contract']['seed']}:{int(r)}:{int(s)}" for r,s in zip(z['ep_round'],z['ep_slot'])]
            pol=sh['policy'];group=pool+'_'+pol['component']+'_'+str(pol['base_epsilon'])
            if pol['prefix'] is not None:group+='_'+str(pol['prefix'])
            assert len(ids)==sh['episodes'] and int(a['ep_T'].sum())==sh['transitions']
            add_source(a,ids,[group]*len(ids),out,rows)
            if len(out)%50==0:print(pool,sh['file'],flush=True)
        print(pool,'complete',len(rows),time.time()-started,flush=True)
    pools={}
    for group,q in out.items():merge(pools.setdefault(group.split('_')[0],{}),q)
    for pool,q in pools.items():
        if pool=='old':
            ref=json.loads((HERE.parent/'20261104_coverage_and_gaze/COVERAGE.json').read_text())['summary']
            assert q['total/transitions']['n']==ref['transitions'] and q['total/episodes']['n']==ref['episodes']
            for name in ACH:assert q['ach/'+name]['n']==ref['events']['unlock_'+name]['unique']
        else:
            ref=snapshots[pool]['coverage']['train']
            for k,v in [('damage','damage'),('death','deaths'),('risk','risk'),('risk_death','risk_deaths')]:
                assert q['event/'+k]['n']==ref[v],(pool,k)
            assert q['total/transitions']['n']==ref['transitions']
            for name in ACH:assert q['ach/'+name]['n']==ref['unlocks'][name]
    combined={}
    for q in pools.values():merge(combined,q)
    assert len(set(r['id'] for r in rows))==len(rows)
    result=dict(source_pins=pins,script_sha256=sha(Path(__file__)),groups=out,pools=pools,combined=combined,
                elapsed_seconds=time.time()-started,complete=all(m['complete'] for m in snapshots.values()),
                snapshot_rounds={k:m['rounds'] for k,m in snapshots.items()})
    (HERE/'CENSUS.json').write_text(json.dumps(result,indent=1)+'\n')
    with gzip.open(HERE/'episodes.json.gz','wt') as f:json.dump(rows,f,separators=(',',':'))
    print('DONE',result['complete'],result['snapshot_rounds'],len(rows),time.time()-started,flush=True)


if __name__=='__main__':main()
