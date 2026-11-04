"""Search useful genuine looks where the existing strong local coarse reader is uncertain/wrong.

No reader refit. Original random looks are not assumed to cover useful cues.
The full coarse/history/TSP observers and local S1 observer must ALL be beaten.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from . import fit as A
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    local='--local' in sys.argv
    balanced='--balanced-coarse' in sys.argv
    assert not balanced or local
    z=A.load('dev');feat=A.Features(z);m,cfg=N.load();s1=N.S.load_reader(N.CK.parent/'s1/results.reader.pt',N.CK,A.DEV)
    L=dict(np.load(N.S.LABELS/'s0_dev_labels.npz'));so=np.r_[0,np.cumsum(L['ep_T']+1)];rows=[];labels=feat.z['truth'];light=L['s_light'][so[z['episode'].numpy()]+z['t'].numpy()]
    # Scan every current map cell, not just cells covered by the original random gaze.
    for f0 in ([] if local else range(0,feat.n,512)):
        p=feat.z['p'][f0:f0+512,:63].float();v=feat.z['v'][f0:f0+512,None].expand(-1,63,-1).float();probs=s1(torch.cat([p,v],-1))['kind'].softmax(-1)
        y=labels[f0:f0+len(p)];truep=probs.gather(-1,y[...,None]).squeeze(-1);eligible=torch.isin(y,torch.tensor([1,3,4],device=A.DEV))&(truep<.8)
        eligible&=feat.z['night'][f0:f0+len(p),None]&(feat.z['t'][f0:f0+len(p),None]>=8)
        for ff,c in eligible.nonzero().cpu().tolist():
            f=f0+ff;e=int(z['episode'][f]);t=int(z['t'][f]);s=int(so[e]+t)
            if L['s_sleep'][s] or L['s_body'][s,0]<=0:continue
            ky=int(labels[f,c]);kind={1:'zombies',3:'skeletons',4:'arrows'}[ky];world=L['s_pos'][s]+np.array(divmod(c,9))-np.array([3,4])
            ids=np.flatnonzero(L[f's_{kind}_m'][s]&(L[f's_{kind}_p'][s]==world).all(-1));assert len(ids);j=int(ids[-1]);start=max(int(so[e]),s-8)
            rel=L[f's_{kind}_p'][start:s,j]-L['s_pos'][start:s];seen=int((L[f's_{kind}_m'][start:s,j]&(abs(rel[:,0])<=3)&(abs(rel[:,1])<=4)).sum())
            rows.append(dict(frame=f,episode=e,time=t,cell=c,kind=ky,j=j,previous8_visible=seen,light=float(light[f]),shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e]),local_coarse_probabilities=probs[ff,c].cpu().tolist()))
    rows.sort(key=lambda r:(r['previous8_visible']>0,r['episode'],r['time'],r['cell']))
    unique=[];used=set()
    for r in rows:
        if r['frame'] in used:continue
        used.add(r['frame']);unique.append(r)
        if len(unique)>=2000:break
    if local:unique=json.loads((HERE/'focus_sources.json').read_text())['selected']
    else:(HERE/'focus_sources.json').write_text(json.dumps(dict(eligible_queries=len(rows),selected=unique),indent=2)+'\n')
    print('uncertain source queries',len(rows),'distinct source frames',len(unique),flush=True)
    localnets=[]
    if local:
        loc=torch.load(N.ROOT/'artifacts/experiments/20261103_p1f16_review/p1f16/readers.pt',map_location=A.DEV,weights_only=False)
        for seed in [0,1]:
            r=loc[f'g_{seed}'];rd=N.S.MLP(r['dim'],{'kind':5}).to(A.DEV);rd.load_state_dict(r['state']);localnets.append(rd.eval())
        if balanced:
            r=loc['p_0'];balanced_coarse=N.S.MLP(r['dim'],{'kind':5}).to(A.DEV);balanced_coarse.load_state_dict(r['state']);balanced_coarse.eval()
    saved=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False);saved.update(torch.load(HERE/'coarse_control_readers.pt',map_location='cpu',weights_only=False));control=json.loads((HERE/'coarse_control.json').read_text());nets={}
    wanted={f'{f}_{s}_fine' for f in ['flat','split'] for s in [0,1]}|{control[f'{f}_{s}_coarse_tsp']['chosen_by_task']['kind'] for f in ['flat','split'] for s in [0,1]}
    for key in wanted:
        r=saved[key];net=A.Reader(feat.dim,r['family']).to(A.DEV);net.load_state_dict(r['state']);nets[key]=net.eval()
    # Only gaze-dependent fields are updated. Native coarse frames/history remain byte-identical.
    for st in range(0,len(unique),128):
        block=unique[st:st+128];ii=torch.tensor([r['frame'] for r in block],device=A.DEV);fx=torch.tensor([[max(1,min(7,r['cell']//9)),max(1,min(7,r['cell']%9))] for r in block],device=A.DEV)
        frames=torch.stack([N.D.RP.episode_fields(r['shard'],r['slot'])['observations'][r['time']] for r in block]).to(A.DEV)
        with N.TR.autocast(A.DEV,cfg):
            g=m.perception(frames,fx).g;mean,_=m.pixels(m.tsp.predict(feat.z['prior'][ii].float(),feat.z['p'][ii].float(),feat.z['v'][ii].float(),N.M.window(fx)),feat.z['v'][ii].float(),N.M.window(fx))
        cells=N.M.window(fx);feat.z['g'][ii]=g.half();feat.z['fix'][ii]=fx;feat.z['tsp'][ii]=mean.half();feat.z['fine_pixels'][ii]=N.M.cell_pixels(N.M.crop(frames,fx))
        feat.z['localg'][ii]=-1
        for j in range(9):
            ok=cells[:,j]<63;feat.z['localg'][ii[ok],cells[ok,j]]=j
    results=[];selected=[];used={}
    for st in range(0,len(unique),256):
        block=unique[st:st+256];q=torch.tensor([r['frame']*63+r['cell'] for r in block],device=A.DEV);prob={}
        for key,net in nets.items():
            with torch.autocast('cuda',dtype=torch.bfloat16):x=net(feat.make(q,saved[key]['mode']))[:,:5].float()
            c=saved[key]['calibration']['kind'];prob[key]=(x/np.exp(c['log_temperature'])+torch.tensor(c['bias'],device=A.DEV)).softmax(-1).cpu().numpy()
        if local:
            f,c=q//63,q%63;slot=feat.z['localg'][f,c].long();gv=torch.cat([feat.z['g'][f,slot].float(),feat.z['v'][f].float()],-1)
            for j,rd in enumerate(localnets):prob[f'local_fine_{j}']=rd(gv)['kind'].softmax(-1).cpu().numpy()
            if balanced:
                pv=torch.cat([feat.z['p'][f,c].float(),feat.z['v'][f].float(),feat.identity[c]],-1)
                prob['balanced_coarse']=balanced_coarse(pv)['kind'].softmax(-1).cpu().numpy()
        for j,r in enumerate(block):
            keys=[f'{f}_{s}' for f in ['flat','split'] for s in [0,1]];coarse=[prob[control[k+'_coarse_tsp']['chosen_by_task']['kind']][j,r['kind']] for k in keys]+[r['local_coarse_probabilities'][r['kind']]];fine=[prob[k+'_fine'][j,r['kind']] for k in keys]
            if local:
                fine=[prob[f'local_fine_{k}'][j,r['kind']] for k in [0,1]]
                all_coarse_wrong=all(int(prob[control[k+'_coarse_tsp']['chosen_by_task']['kind']][j].argmax())!=r['kind'] for k in keys) and int(np.argmax(r['local_coarse_probabilities']))!=r['kind']
                if balanced:
                    coarse.append(prob['balanced_coarse'][j,r['kind']])
                    all_coarse_wrong=all_coarse_wrong and int(prob['balanced_coarse'][j].argmax())!=r['kind']
                passed=all_coarse_wrong and max(coarse)<.5 and min(fine)>=.9
            else:passed=min(fine)>max(coarse)+.05 and np.mean(fine)>=.8
            rr=dict(**r,coarse_true_probability=list(map(float,coarse)),fine_true_probability=list(map(float,fine)),qualified=bool(passed),query=r['frame']*63+r['cell'],
                task='kind',mode='fine',cohort='discovery' if r['episode']<250 else 'confirmation',focus=[max(1,min(7,r['cell']//9)),max(1,min(7,r['cell']%9))],row=len(results))
            results.append(rr)
            if passed:
                k=rr['cohort'];s=used.setdefault(k,set())
                if r['episode'] not in s and len(s)<12:selected.append(rr);s.add(r['episode'])
    def convert(x):return float(x) if isinstance(x,np.floating) else x
    output='focus_balanced_results.json' if balanced else ('focus_local_results.json' if local else 'focus_results.json')
    (HERE/output).write_text(json.dumps(dict(rows=results,selected=selected),indent=2,default=convert)+'\n')
    print('strict fine-gain queries',sum(r['qualified'] for r in results),'selected',len(selected),flush=True)
    for r in selected:print(r['cohort'],r['episode'],r['time'],r['kind'],'best coarse',max(r['coarse_true_probability']),'worst fine',min(r['fine_true_probability']),flush=True)

if __name__=='__main__':main()
