"""Validate the final focused sources with every existing coarse control.

The factual source is natural. Removing its threat is a simulator intervention
for reader responsiveness; it is not claimed to preserve an entire feasible past.
No reader or world parameters are fitted here.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import fit as A
from .consequence import KINDS,visible_kind
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    m,cfg=N.load();feat=A.Features(A.load('dev'))
    s1=N.S.load_reader(N.CK.parent/'s1/results.reader.pt',N.CK,A.DEV)
    bundle=torch.load(N.ROOT/'artifacts/experiments/20261103_p1f16_review/p1f16/readers.pt',map_location=A.DEV,weights_only=False)
    local={}
    for key in ['g_0','g_1','p_0']:
        r=bundle[key];net=N.S.MLP(r['dim'],{'kind':5}).to(A.DEV)
        net.load_state_dict(r['state']);local[key]=net.eval()
    saved=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False)
    saved.update(torch.load(HERE/'coarse_control_readers.pt',map_location='cpu',weights_only=False))
    control=json.loads((HERE/'coarse_control.json').read_text())
    chosen=[control[f'{f}_{s}_coarse_tsp']['chosen_by_task']['kind'] for f in ['flat','split'] for s in [0,1]]
    nets={}
    for key in chosen:
        r=saved[key];net=A.Reader(feat.dim,r['family']).to(A.DEV)
        net.load_state_dict(r['state']);nets[key]=net.eval()
    source=json.loads((HERE/'focus_balanced_results.json').read_text())['selected']
    _,_,_,_,render=N.D.RP.env_and_render();rows=[]
    for meta in source:
        f,c=meta['frame'],meta['cell'];q=torch.tensor([meta['query']],device=A.DEV)
        state=N.D.RP.advance_to(meta['shard'],meta['slot'],meta['time'])
        actual=np.asarray(render(state));recorded=N.D.RP.episode_fields(meta['shard'],meta['slot'])['observations'][meta['time']].numpy()
        assert np.array_equal(actual,recorded)
        world=np.asarray(state.player_position)+np.array(divmod(c,9))-np.array([3,4])
        assert visible_kind(state,world)==meta['kind']
        name=KINDS[meta['kind']];ent=getattr(state,name)
        assert np.array_equal(np.asarray(ent.position[meta['j']]),world)
        ent=ent.replace(mask=ent.mask.at[meta['j']].set(False),health=ent.health.at[meta['j']].set(0))
        absent=state.replace(**{name:ent})
        if meta['kind']!=4:
            occupied=any(bool((np.asarray(getattr(absent,k).mask)&(np.asarray(getattr(absent,k).position)==world).all(-1)).any()) for k in ['zombies','cows','skeletons'])
            absent=absent.replace(mob_map=absent.mob_map.at[world[0],world[1]].set(occupied))
        alternate=visible_kind(absent,world)
        assert alternate!=meta['kind']
        frames=torch.as_tensor(np.stack([actual,np.asarray(render(absent))]),device=A.DEV)
        fx=torch.tensor([meta['focus']],device=A.DEV).expand(2,-1)
        cells=N.M.window(fx);slots=(cells==c).nonzero()[:,1]
        assert len(slots)==2 and slots[0]==slots[1]
        with N.TR.autocast(A.DEV,cfg):
            P=m.perception(frames,fx)
            mean,_=m.pixels(m.tsp.predict(feat.z['prior'][f].float().expand(2,-1),P.p,P.v,cells),P.v,cells)
        raw=F.adaptive_avg_pool2d(frames.reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float(),3).round().byte().reshape(2,81,27)
        fields=dict(p=P.p.half(),g=P.g.half(),v=P.v.half(),raw=raw,tsp=mean.half(),fine_pixels=N.M.cell_pixels(N.M.crop(frames,fx)))
        old={k:feat.z[k][f].clone() for k in list(fields)+['fix','localg']}
        parity={k:float((old[k].float()-fields[k][0].float()).abs().max()) for k in ['p','v','raw']}
        feat.z['fix'][f]=fx[0];feat.z['localg'][f]=-1
        for j,cell in enumerate(cells[0].tolist()):
            if cell<63:feat.z['localg'][f,cell]=j
        probabilities={};coarse_inputs=[]
        for state_index in [0,1]:
            for k,v in fields.items():feat.z[k][f]=v[state_index]
            xx=feat.make(q,'coarse');coarse_inputs.append(tuple(t.clone() for t in xx))
            # Changing CURRENT fine content must not change any coarse input.
            original=feat.z['g'][f].clone();feat.z['g'][f]=100
            assert all(torch.equal(x,y) for x,y in zip(xx,feat.make(q,'coarse')))
            feat.z['g'][f]=original
            pv=torch.cat([P.p[state_index,c].float(),P.v[state_index].float()])[None]
            gv=torch.cat([P.g[state_index,int(slots[0])].float(),P.v[state_index].float()])[None]
            localprob={'s1':s1(pv)['kind'],
                'p_0':local['p_0'](torch.cat([pv,feat.identity[c:c+1]],-1))['kind'],
                'g_0':local['g_0'](gv)['kind'],'g_1':local['g_1'](gv)['kind']}
            for key,logit in localprob.items():probabilities.setdefault(key,[]).append(logit.softmax(-1)[0].cpu().tolist())
            for key,net in nets.items():
                with torch.autocast('cuda',dtype=torch.bfloat16):x=net(feat.make(q,saved[key]['mode']))[0,:5].float()
                cal=saved[key]['calibration']['kind'];p=(x/np.exp(cal['log_temperature'])+torch.tensor(cal['bias'],device=A.DEV)).softmax(-1)
                probabilities.setdefault(key,[]).append(p.cpu().tolist())
        for k,v in old.items():feat.z[k][f]=v
        coarse=chosen+['s1','p_0'];fine=['g_0','g_1'];kind=meta['kind']
        coarse_miss=all(int(np.argmax(probabilities[k][0]))!=kind for k in coarse)
        coarse_low=max(probabilities[k][0][kind] for k in coarse)<.5
        fine_correct=all(probabilities[k][0][kind]>=.9 for k in fine)
        absent_correct=all(int(np.argmax(probabilities[k][1]))==alternate for k in fine)
        rows.append(dict(meta=meta,alternate_label=alternate,probabilities=probabilities,
            fresh_cache_maxdiff=parity,coarse_miss=coarse_miss,coarse_low=coarse_low,
            fine_correct=fine_correct,absent_correct=absent_correct,
            qualified=bool(coarse_miss and coarse_low and fine_correct and absent_correct)))
        print(meta['cohort'],meta['episode'],meta['time'],'source',fine_correct,'absence',absent_correct,
            'all coarse wrong',coarse_miss,'qualified',rows[-1]['qualified'],flush=True)
        (HERE/'focus_paired_readers.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('qualified',sum(r['qualified'] for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
