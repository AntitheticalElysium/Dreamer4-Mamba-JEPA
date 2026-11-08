"""Same fixed 58-scene genuine-look uptake audit as F1.6; no crop swaps or world fitting."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from saccade import train as TR
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')
L=importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261103_p1f16_review'


def summary(root):
    z={n:dict(np.load(root/n/'local.npz')) for n in ('p1f16','p1f2')}
    for k in ('index','episode','t','weights','levels'):assert np.array_equal(z['p1f16'][k],z['p1f2'][k]),k
    readable=(z['p1f16']['source_readable']==1)&(z['p1f2']['source_readable']==1)
    worlds={};raw={}
    for name,q in z.items():
        a,b=q['weights'][:,0],q['weights'][:,1];raw[name]={};out={}
        diff=np.einsum('isnwg,iw->isng',q['loss'],a-b);base=np.einsum('isnwg,iw->isng',q['loss'],b)
        for region,j in (('map',0),('local',1)):
            for band,sl in (('all',slice(None)),('high',slice(-2,None))):
                key=region+'_'+band;d=diff[:,:,sl,j].mean((1,2));v=base[:,:,sl,j].mean((1,2))
                raw[name][key]=d/v
                out[key]=dict(difference=A.estimate(d,q['episode']),control_loss=A.estimate(v,q['episode']),
                              fraction=A.estimate(d/v,q['episode']))
        raw[name]['tsp']=np.einsum('iw,iw->i',q['tsp'],a-b)/np.einsum('iw,iw->i',q['tsp'],b)
        out['tsp_fraction']=A.estimate(raw[name]['tsp'],q['episode'])
        out['stage_rms']={k:A.estimate(q[k],q['episode']) for k in ('embedding','corrected','predicted','memory')}
        out['source_readable_mean']=float(q['source_readable'].mean())
        out['common_readable_scenes']=int(readable.sum())
        out['common_readable']={k:A.estimate(v[readable],q['episode'][readable]) for k,v in raw[name].items()} if readable.any() else {}
        worlds[name]=out
    return dict(worlds=worlds,paired_fraction_change={k:A.estimate(raw['p1f2'][k]-raw['p1f16'][k],q['episode']) for k in raw['p1f16']})


def main():
    W=importlib.import_module('artifacts.experiments.20261103_p1f16_review.local_uptake')
    torch.set_num_threads(4);ck=Path(sys.argv[1]);out=HERE/'readings/fovea';out.mkdir(parents=True,exist_ok=True)
    old=json.loads((BASE/'local_identity.json').read_text())
    assert old['worlds']['p1f16']['checkpoint_sha256']==L.CONTROL_SHA
    assert old['source_sha256']==TR.file_sha256(W.U.HERE/'uptake_local/raw.npz')
    assert old['analysis_script_sha256']==TR.file_sha256(Path(W.__file__))
    base=out/'p1f16';base.mkdir(exist_ok=True)
    for name in ('local.npz','readers.pt'):
        dst=base/name;src=(BASE/'p1f16'/name).resolve()
        if not dst.exists():dst.symlink_to(src)
        assert dst.resolve()==src
    W.HERE=out;source,weights=W.fixed_cohort()
    ident=W.run_world('p1f2',ck,source,weights)
    (out/'summary.json').write_text(json.dumps(summary(out),indent=2)+'\n')
    (out/'identity.json').write_text(json.dumps(dict(control=old['worlds']['p1f16'],candidate=ident,
        baseline_npz_sha256=TR.file_sha256(base/'local.npz'),fixed_source_sha256=old['source_sha256']),indent=2)+'\n')


if __name__=='__main__':main()
