"""Independent matched comparison, original outputs untouched."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent; E=P.parent
DIRS=[E/x for x in ('20261106_p1f4','20261109_p1_hud_change','20261111_p1_multistep_mamba')]
def js(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run():
    out={'arms':[x.name for x in DIRS]}
    ck=[torch.load(d/'run/ckpt_10.pt',weights_only=False,map_location='cpu',mmap=True) for d in DIRS]
    sh=[sha(d/'run/ckpt_10.pt') for d in DIRS]
    backup=DIRS[2]/'run/ckpt_10.orig_with_mask.pt'
    ck0=torch.load(backup,weights_only=False,map_location='cpu',mmap=True)
    assert set(ck0['model'])-set(ck[2]['model'])=={'belief.mask_token'}
    assert set(ck[2]['model'])<=set(ck0['model'])
    assert all(torch.equal(v,ck0['model'][k]) for k,v in ck[2]['model'].items())
    torch.set_num_threads(2)
    out['checkpoint_sha256']=sh
    out['multistep_backup']={'sha256':sha(backup),'native_model_tensors_equal':True,'removed':'belief.mask_token'}
    assert ck[1]['cfg']==ck[2]['cfg']
    for k in ('manifest_sha256','targets_sha256'):
        assert ck[1]['provenance'][k]==ck[2]['provenance'][k]
    logs=[[json.loads(x) for x in (d/'run/log.jsonl').read_text().splitlines()] for d in DIRS]
    out['update1']=[{k:r[0][k] for k in ('g1','change','tsp','rec','reward','cont','noise','grad_norm')} for r in logs]
    m=[x for x in logs[2] if 'multi' in x][-40:]
    out['last_1000_updates_median']={k:float(np.median([f(x) for x in m])) for k,f in {
        'weighted_multi':lambda x:.1*x['multi'],'weighted_g1':lambda x:x['g1']/x['sigma'][0],
        'weighted_g2':lambda x:sum(x[k] for k in ('change','tsp','rec'))/x['sigma'][1],
        'weighted_g3':lambda x:(x['reward']+x['cont'])/x['sigma'][2]}.items()}
    out['monitor']=[next(x['monitor'] for x in r if x.get('checkpoint')==10) for r in logs]

    raw=[np.load(d/'readings/consequences/curves/raw.npz') for d in DIRS]
    for k in ('asleep','mining','zombie','hit'):
        for x in raw[1:]:
            for q in ('j','stratum','rec_reward','rec_term'):
                assert np.array_equal(x[f'{k}/{q}'],raw[0][f'{k}/{q}']), (k,q)
    # Physical episodes are the unit of uncertainty; samples inside episode stay together.
    with np.load(E.parent/'eda/s0/s0_dev_labels.npz') as ll:
        T=ll['ep_T']; et=np.repeat(np.arange(len(T)),T)
    def scores(arr,ep):
        u, inv=np.unique(ep,return_inverse=True)
        n=np.bincount(inv)
        z=np.stack([np.bincount(inv,weights=x,minlength=len(u)) for x in arr])
        r=np.random.default_rng(4502).integers(0,len(u),(2000,len(u)))
        den=n[r].sum(1); b=z[:,r].sum(2)/den
        return {'n':len(ep),'episodes':len(u),'means':np.asarray(arr).mean(1).tolist(),
                'd_hud_minus_f4':np.quantile(b[1]-b[0],[.025,.975]).tolist(),
                'd_multi_minus_hud':np.quantile(b[2]-b[1],[.025,.975]).tolist()}
    out['mining']={}
    st=raw[0]['mining/stratum'].astype(str)
    n=np.array([int(x.split('_')[-1]) for x in st])
    for kind,cell in [('wood',68),('stone',69),('coal',70),('iron',71)]:
        for nn,m in [('first',n==0),('n1_3',(n>=1)&(n<=3)),('n5_8',(n>=5)&(n<=8))]:
            m=m&np.char.startswith(st,kind+'_')
            if m.any():
                out['mining'][kind+'_'+nn]=scores([(x[f'mining/gen_hud{cell}'][m]==n[m]+1) for x in raw],et[raw[0]['mining/j'][m]])
    zz=raw[0]['zombie/stratum'].astype(str);jj=raw[0]['zombie/j']
    out['zombie']={}
    for st in ('awake_ready','awake_cooling','asleep_ready','asleep_cooling'):
        m=np.char.startswith(zz,st)
        # previous health by transition index
        with np.load(E.parent/'eda/s0/s0_dev_labels.npz') as ll: h0=ll['s_body'][np.arange(len(et))+et,0]
        out['zombie'][st]={'drop':scores([x['zombie/gen_hud63'][m]<h0[jj[m]] for x in raw],et[jj[m]]),
                           'drawn_dead':scores([x['zombie/gen_hud63'][m]==0 for x in raw],et[jj[m]]),
                           'wake':scores([x['zombie/gen_asleep'][m]<.5 for x in raw],et[jj[m]])}
    out['curves']={}
    for d,s in zip(DIRS,sh):
        v=js(d/'readings/consequences/curves/curves.json');assert v['sha256']==s
        out['curves'][d.name]={k:v['readings'][k] for k in ('wake_value','wake_value_vs_change','cooldown','lethality','sleep_attack','cow_kill','zombie_kill')}
    rr=[np.load(d/'readings/recursion_steps4/raw.npz') for d in DIRS]
    for x in rr[1:]:
        for k in ('episode','t0','truth_count','truth_reward','truth_death'):
            assert np.array_equal(x[k],rr[0][k])
    out['recursion']={str(t):scores([x['free__count'][:,t-1].sum(-1) for x in rr],rr[0]['episode']) for t in (1,4,8,16)}
    out['recursion_truth']={str(t):float(rr[0]['truth_count'][:,t-1].sum(-1).mean()) for t in (1,4,8,16)}
    out['recursion_rewards']=[{k:float(x[k].mean()) for k in ('truth_reward','teacher__reward_sample','free__reward_sample')} for x in rr]
    out['interface']=[]
    out['s1']=[]
    for d,s in zip(DIRS,sh):
        v=next(iter(js(d/'readings/interface.json')['checkpoints'].values()));assert v['sha256']==s
        out['interface'].append({k:{mode:{y:o['est'] for y,o in v['classes'][k][mode].items() if y in ('term_p','unlock_p','neg_p','achievement_unlock','exp_reward')} for mode in ('recorded','gen','rec_head')} for k in ('random','risky','adjacent','ach_COLLECT_COAL','ach_COLLECT_IRON','ach_COLLECT_DRINK','ach_WAKE_UP')})
        v=js(d/'run/s1/results.json');assert v['checkpoint_sha256']==s
        vals=lambda rows,k:np.mean([[x['est'] for x in p[k]] for p in rows],axis=0).tolist()
        out['s1'].append({'tsp':v['s1_4']['tsp_nats_per_pixel']['all']['est'],
            'night_near_recall':np.mean([p['night_r1_2']['presence_recall']['est'] for p in v['s1_1']['trained']]),
            'c2st':v['s1_3']['c2st_generated'],
            'belief':{k:{'hud_body_exact':vals(b,'hud_exact')[:4],
                          'achievements_macro_ap':np.mean([x['ap'] for a in b for x in a['achievements'] if x['ap'] is not None]),
                          'departed':b[0]['departed']} for k,b in v['s1_2'].items()}})
    dirs=[E/'20261107_f4_followups/readings/danger/belief_path_f4.json',DIRS[1]/'readings/belief_path.json',DIRS[2]/'readings/belief_path_probe.json']
    out['belief_path']=[{k:{p:v for p,v in q['readers'].items() if p in ('hud_tokens/mlp','percept_embed/mlp','bp/mlp','b_prev/mlp')} for k,q in js(d)['targets'].items()} for d in dirs]
    (P/'MATCHED.json').write_text(json.dumps(out,indent=2))
    print('matched comparison complete',flush=True)
if __name__=='__main__':run()
