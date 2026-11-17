"""M4 correction only; preserve Claude's original experiment. features=GPU, probe=CPU."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from saccade import model as M,train as TR
from saccade.eval import s1 as S
CV=importlib.import_module('artifacts.experiments.20261106_determinant_curves.curves')
RT=importlib.import_module('artifacts.experiments.20261106_determinant_routes.routes')
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'20261107_f4_followups/readings/danger/denoiser_teaching_f4.json'
CACHE=HERE.parent/'20261107_health_readout_audit/enc.pt'
CKPT='artifacts/experiments/20261106_p1f4/run/ckpt_10.pt'

@torch.no_grad()
def features():
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    torch.set_num_threads(2)
    model,cfg=CV.IF.load(CKPT)
    bound=torch.load(HERE.parent/'20261107_health_readout_audit/capture.pt',map_location='cpu',weights_only=False)
    assert TR.file_sha256(TR.ROOT/CKPT)==bound['checkpoint_sha256'];del bound
    enc=torch.load(CACHE,map_location='cpu',weights_only=False)
    L=dict(np.load(CV.LABELS/'s0_dev_labels.npz'));T=CV.transitions(L)
    rows=np.sort(np.random.default_rng(2120).choice(np.flatnonzero(T['t']>=CV.P),40000,replace=False))
    got={}
    for a in range(0,len(rows),2048):
        j=rows[a:a+2048]
        for dt in (-1,0):
            at=lambda key:torch.stack([enc[int(T['ep'][x])][key][int(T['t'][x])+dt] for x in j])
            p,g,v,fx=(at(k).to(CV.DEV) for k in ('p','g','v','fix'))
            emb=model.belief.percept_embed(M.Percept(p.float(),g.float(),v.float(),fx)).float().cpu()
            got.setdefault('e'+str(dt),[]).append(emb)
            got.setdefault('p'+str(dt),[]).append(p[:,63:67].float().cpu())
        if a%8192==0:print('features',a,len(rows),flush=True)
    data={k:torch.cat(v) for k,v in got.items()}
    data.update(rows=rows,ep=T['ep'][rows],body0=T['body'][rows],body1=T['true_body'][rows],
                checkpoint_sha256=TR.file_sha256(TR.ROOT/CKPT),cache_sha256=TR.file_sha256(CACHE))
    torch.save(data,HERE/'features.pt')

def probe():
    torch.set_num_threads(2)
    data=torch.load(HERE/'features.pt',map_location='cpu',weights_only=False)
    original=json.loads(PARENT.read_text());ep=data['ep'];fold=RT.folds_of(ep)
    def fit(X,y):
        X=X.double();s=np.zeros(len(y))
        for f in range(5):
            m=fold==f;s[m]=RT.logistic(X[~m],y[~m],X[m])
        return RT.ap_boot(y,s,ep),s
    out={'checkpoint_sha256':data['checkpoint_sha256'],'rows':len(ep),'changed':{},
         'zero_values_reused_from_original':{k:v for k,v in original['M4 visibility (AP, prevalence)'].items() if '== 0' in k}}
    deltae=data['e0'].double()-data['e-1'].double()
    scores={}
    for i,name in enumerate(('health','food','drink','energy')):
        y=(data['body1'][:,i]!=data['body0'][:,i]).astype(float)
        d=data['p0'][:,i].double()-data['p-1'][:,i].double()
        r={'prevalence':float(y.mean())}
        for tag,X in [('HUD_abs',d.abs()),('embedding_abs',deltae.abs())]:
            r[tag],ss=fit(X,y);scores[name+'/'+tag]=ss
        r['control_qualified_at_AP_095']=r['HUD_abs']['ap']>=.95
        if i==0:
            signed,_=fit(d,y)
            expected=original['M4 visibility (AP, prevalence)']['health changed']['HUD token difference']
            out['signed_health_reproduction']={'ap':signed['ap'],'original':expected,'absolute_difference':abs(signed['ap']-expected)}
            assert abs(signed['ap']-expected)<.001,'Cached representations fail original control'
        out['changed'][name]=r
        (HERE/'VISIBILITY.json').write_text(json.dumps(out,indent=2))
        print(name,json.dumps(r),flush=True)
    np.savez_compressed(HERE/'visibility_scores.npz',**scores)

if __name__=='__main__':{'features':features,'probe':probe}[sys.argv[1]]()
