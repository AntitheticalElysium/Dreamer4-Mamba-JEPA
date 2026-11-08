"""Held-out combined-policy quality of the saved ONLINE head; no refit or policy search."""
import importlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
from saccade import model as M, train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
P=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
HERE=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(4);path=Path(sys.argv[1]);ck=torch.load(path,map_location='cpu',weights_only=False)
    out=HERE/'readings/head';out.mkdir(parents=True,exist_ok=True)
    sha=TR.file_sha256(path);Q=C.Q;Q.HERE=out;Q.CK=path
    model=M.M0(ck['cfg']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'])
    head=Q.Head().cuda().eval();head.load_state_dict(ck['fixation_controller']['head'])
    cohorts=P.cohorts();n=max(int(q['ep'].max())+1 for q in cohorts.values())
    data=Q.Data(Q.collect(model,ck['cfg'],'dev',n,torch.device('cuda')),torch.device('cuda'))
    result={}
    for name,meta in cohorts.items():
        forecast=Q.predict(head,data,data.row_ids(meta['ep'],meta['t']))
        np.savez_compressed(out/f'{name}.npz',**meta,**forecast)
        # Only the adopted policy is reported as operational; C/S are diagnostics.
        full=P.summarize(meta,forecast['C'],forecast['S'],dict(mse=forecast['mse'],copy_mse=forecast['copy_mse']))
        result[name]={k:dict(n=v['n'],episodes=v['episodes'],combined=v['policies']['existing'],
            uniform=v['policies']['uniform'],components=v['components'],errors=v.get('errors',{})) for k,v in full.items()}
    (out/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=sha,head_updates=ck['fixation_controller']['updates'],
        cohorts=result,note='Frozen online head on fixed dev cohorts; no refit, no best checkpoint or component-C gate.'),indent=2)+'\n')


if __name__=='__main__':main()
