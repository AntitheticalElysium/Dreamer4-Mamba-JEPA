"""Whole-screen held-out acquisition and reuse of the already rendered noise panel."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json
from pathlib import Path
import numpy as np
import torch
from saccade import model as M,train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
P=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2'

def summary(meta,forecast,native=None):
    sem=A.metadata(meta);natural=meta['cls']=='random';out={};heat={};checks={}
    policies=P.policies(forecast['C'],forecast['S'])
    # These component policies are descriptive views of saved scores; neither
    # trains a head, changes the adopted policy, nor chooses a winning recipe.
    chosen={k:policies[k] for k in ['existing','uniform','C_softmax','S_softmax']}
    if native is not None:chosen['native']=P.policies(native['contingency'],native['surprise'])['existing']
    for name,prob in chosen.items():
        assert np.allclose(prob.sum(-1),1) and (prob>=.2/49-1e-8).all()
        metrics={}
        centers=P.CENTRES[:,0]*9+P.CENTRES[:,1]
        for kind,mask in sem['categories'].items():
            metrics['center_'+kind]=(prob*mask[:,centers]).sum(-1)
            metrics['cells_'+kind]=(prob*mask[:,P.WIN].mean(-1)).sum(-1)
        assert np.allclose(sum(v for k,v in metrics.items() if k.startswith('center_')),1)
        assert np.allclose(sum(v for k,v in metrics.items() if k.startswith('cells_')),1)
        metrics['touch_HUD']=(prob*(P.WIN>=63).any(-1)).sum(-1)
        out[name]={}
        for group,sel in [('all',natural),('day',natural&(sem['light']>=.5)),('night',natural&(sem['light']<.5))]:
            if not sel.any():continue
            keys=list(metrics);vals=np.stack([metrics[k][sel] for k in keys],-1)
            st=A.estimates(vals,np.ones(sel.sum()),meta['ep'][sel]);out[name][group]=dict(zip(keys,st))
            for kind,mask in sem['masks'].items():
                exposed=sel&mask.any(-1)
                if not exposed.any():continue
                hit=mask[:,P.WIN].any(-1);x=(prob*hit).sum(-1)
                uniform=(policies['uniform']*hit).sum(-1)
                vals=np.stack([x,uniform,x-uniform],-1)
                st=A.estimates(vals[exposed],np.ones(exposed.sum()),meta['ep'][exposed])
                out[name][group]['coverage_'+kind]=dict(zip(['policy','uniform','difference'],st))
        mass=np.zeros((len(prob),81))
        for j,w in enumerate(P.WIN):mass[:,w]+=prob[:,j,None]
        assert np.allclose(mass.sum(-1),9)
        heat[name]=mass[natural].mean(0).reshape(9,9)
    # Analytical evaluator power: averaging all49 circular spatial shifts of a
    # normalized priority must recover uniform opportunity for any mask.
    p=policies['existing'];shuffled=sum(np.roll(p,k,axis=1) for k in range(49))/49
    checks['all_circular_shifts_uniform_max_abs']=float(abs(shuffled-1/49).max())
    assert checks['all_circular_shifts_uniform_max_abs']<1e-12
    return dict(policies=out,checks=checks,natural_frames=int(natural.sum()),
        natural_episodes=len(np.unique(meta['ep'][natural]))),heat

@torch.no_grad()
def noise():
    path=RUN/'run/ckpt_10.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
    m=M.M0(ck['cfg']).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'])
    Q=C.Q;head=Q.Head().cuda().eval().requires_grad_(False);head.load_state_dict(ck['fixation_controller']['head'])
    data=Q.Data(torch.load(RUN/'readings/head/dev_596_coarse.pt',map_location='cpu',weights_only=False),torch.device('cuda'))
    panel=json.loads((RUN/'readings/codex/panel.json').read_text())['rows']
    frames=np.load(RUN/'readings/codex/panel.npz')['frames'];assert frames.shape==(32,8,63,63,3)
    ep=np.array([r['episode'] for r in panel]);tt=np.array([r['t'] for r in panel]);ids=data.row_ids(ep,tt)
    x=data.inputs(ids);pred=head(x);noop=head((x[0],x[1],torch.zeros_like(x[2])))
    actual=[]
    for f in torch.as_tensor(frames.reshape(-1,63,63,3),device='cuda').split(256):
        with TR.autocast(torch.device('cuda'),ck['cfg']):p,v=C.coarse(m.perception,f)
        actual.append(p.float())
    actual=torch.cat(actual).reshape(32,8,81,16)
    cellC=(pred-noop).square().sum(-1);cm=cellC[:,Q.WIN.cuda()].sum(-1)
    sm=(actual-pred[:,None]).square().sum(-1)[:,:,Q.WIN.cuda()].sum(-1)
    mean=actual.mean(1);variance=(actual-mean[:,None]).square().sum(-1).mean(1)
    bias=(mean-pred).square().sum(-1)
    var=variance[:,Q.WIN.cuda()].sum(-1);bias=bias[:,Q.WIN.cuda()].sum(-1)
    assert torch.allclose(sm.mean(1),var+bias,atol=1e-5,rtol=1e-5)
    prob=Q.policy(cm[:,None].expand_as(sm).reshape(-1,49),sm.reshape(-1,49)).reshape(32,8,49)
    onlyS=Q.policy(torch.zeros_like(cm[:,None].expand_as(sm)).reshape(-1,49),sm.reshape(-1,49)).reshape(32,8,49)
    pairs=torch.combinations(torch.arange(8,device='cuda'),2)
    tv=.5*(prob[:,pairs[:,0]]-prob[:,pairs[:,1]]).abs().sum(-1)
    stv=.5*(onlyS[:,pairs[:,0]]-onlyS[:,pairs[:,1]]).abs().sum(-1)
    hud=(Q.WIN.cuda()>=63).any(-1);hm=prob[:,:,hud].sum(-1)
    rows=[]
    for i,r in enumerate(panel):rows.append(dict(**r,variance=float(var[i].mean()),bias_squared=float(bias[i].mean()),
        variance_fraction=float(var[i].mean()/(var[i]+bias[i]).mean()),combined_TV=float(tv[i].mean()),
        half_weight_S_only_TV=float(stv[i].mean()),HUD_probability_pair_difference=float((hm[i,pairs[:,0]]-hm[i,pairs[:,1]]).abs().mean())))
    agg={}
    for light in ['bright','dark']:
        rr=[r for r in rows if r['stratum']==light]
        agg[light]={k:float(np.mean([r[k] for r in rr])) for k in ['variance_fraction','combined_TV','half_weight_S_only_TV','HUD_probability_pair_difference']}
        agg[light]['states']=len(rr)
    return dict(rows=rows,summary=agg,note='Eight EXISTING renders per fixed state. Finite-bank decomposition, not population irreducible uncertainty or wasted gaze. Histories and forecasts held fixed; no new sources collected.')

def main():
    torch.set_num_threads(4);out={}
    for cohort in ['saved','confirmation']:
        z=dict(np.load(HERE/f'ck10/{cohort}.npz'));native=None
        if cohort=='saved':
            native=dict(np.load(RUN/'readings/locus/p1f2.npz'))
            assert all(np.array_equal(z[k],native[k]) for k in ['ep','t','cls','counter','faced','light'])
        result,heat=summary(z,z,native);out[cohort]=result
        np.savez_compressed(HERE/f'HEAT_{cohort}.npz',**heat)
    (HERE/'BEHAVIOR.json').write_text(json.dumps(out,indent=2)+'\n')
    n=noise();(HERE/'NOISE_REUSE.json').write_text(json.dumps(n,indent=2)+'\n')
    print('existing-panel noise',n['summary'],flush=True)
    for cohort,r in out.items():
        print(cohort,r['natural_frames'],'natural frames',flush=True)
        for light in ['all','day','night']:
            rr=r['policies']['existing'][light]
            print(light,'entities',rr.get('coverage_any_entity'),'HUD touch',rr['touch_HUD']['mean'],flush=True)

if __name__=='__main__':main()
