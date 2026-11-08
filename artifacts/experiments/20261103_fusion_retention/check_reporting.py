"""Known-effect CPU checks of score signs, units, pairing and report dispatch."""
import importlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch

R=importlib.import_module('artifacts.experiments.20261103_fusion_retention.report')


def main():
    with tempfile.TemporaryDirectory(prefix='fusion-report-',dir='/tmp') as td:
        R.HERE=Path(td);n=400
        meta=dict(episode=torch.arange(n),night=torch.ones(n,dtype=torch.bool),entity=torch.ones(n,dtype=torch.bool),
            scroll=torch.zeros(n,dtype=torch.bool),source=torch.ones(n,9),neighborhood=torch.ones(n,9))
        for world in ('p1f15','p1f16'):
            out=R.HERE/world;out.mkdir();torch.save(meta,out/'metadata.pt');torch.save(torch.ones(n,9)*.5,out/'copy_errors.pt')
            for seed in R.SEEDS:
                for mode in R.MODES:
                    err={'base':1.,'fine':.75,'joint':.8,'mixed':.9,'predicted':.95,'memory':.95}[mode]
                    torch.save(torch.ones(n,9)*err,out/f'errors_{mode}_{seed}.pt')
            a=np.zeros((4,49));b=a.copy();a[:,0]=1;b[:,1]=1
            np.savez(out/'paired_meta.npz',episode=np.arange(4),truth=np.ones(4,dtype=int),a=a,b=b)
            logs={}
            for seed in R.SEEDS:
                for mode in R.MODES:
                    x=torch.zeros(4,49,5)
                    if mode!='base':x[:,0,1]=1
                    logs[f'{mode}_{seed}']=x.log_softmax(-1)
            torch.save(logs,out/'paired_logp.pt')
        R.main();d=json.loads((R.HERE/'RESULTS.json').read_text())
        x=d['p1f16']['611701']['confirmation']['primary']['targets']['crop']
        assert abs(x['fine']['reduction_percent']-25)<1e-6
        assert abs(d['paired_world_gain_difference']['611701']['crop']['joint']['gain_difference_pp']+5)<1e-4
        assert d['paired_semantic']['p1f16']['611701']['fine']['mean']>0
        z=np.arange(1,6,dtype=float);same=R.paired_contrast(z,.75*z,z,.75*z,np.arange(5))
        assert same['gain_difference_pp']==0 and same['ci95']==[0,0]
        print('REPORT_CHECKS_PASSED: known gain25%, paired difference-5pp, positive true-kind gain, identical worlds zero')


if __name__=='__main__':main()
