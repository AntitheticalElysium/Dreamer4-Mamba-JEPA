"""Synthetic known gains/nulls qualify the report, never the scientific model."""
import contextlib
import importlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch

R=importlib.import_module('artifacts.experiments.20261103_belief_information.report')


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='saccade-belief-report-') as tmp:
        p=Path(tmp);n=8
        ep=torch.tensor([0,1,2,3,250,251,252,253])
        q=torch.tensor([[4,4],[6,4],[4,4],[6,4]]*2)
        fh=torch.tensor([[False]*9,[False]*6+[True]*3]*4)
        m=dict(episode=ep,t=torch.ones(n),night=torch.tensor([0,0,1,1]*2).bool(),scroll=torch.zeros(n).bool(),
               query=q,observed_fix=q,query_cells=torch.zeros(n,9),source=torch.ones(n,81),
               fine_source=torch.ones(n,9),fine_hud=fh,repeat=torch.ones(n).bool())
        torch.save(m,p/'metadata.pt')
        data=[]
        for i in range(n):
            a=np.zeros(49);b=np.zeros(49);a[0]=1;b[-1]=1
            data.append(dict(frame=i,episode=int(ep[i]),a=a,b=b))
        readers={}
        for seed in R.SEEDS:
            for mode in R.MODES:
                val=2. if mode=='base' else 1.
                torch.save(dict(p=torch.full((n,81),val),g=torch.full((n,9),val)),p/f'errors_{mode}_{seed}.pt')
                pp=torch.full((n,49,81),2.);gg=torch.full((n,49,9),2.)
                if mode!='base':pp[:,0]=1.;gg[:,0]=1.
                readers[f'{mode}_{seed}']=dict(p=pp,g=gg)
        native={}
        for key,w in (('p',81),('g',9)):
            x=torch.full((n,2,8,49,w),2.);x[:,:,:,0]=1.;native[key]=x
        torch.save(dict(meta=data,readers=readers,native=native),p/'paired.pt')
        original=R.HERE;R.HERE=p
        try:
            with open(p/'log','w') as log,contextlib.redirect_stdout(log):R.main()
        finally:R.HERE=original
        r=json.loads((p/'RESULTS.json').read_text())
        assert r['predictors']['612101']['confirmation']['all']['physical_all']['fine']['reduction_percent']==50.
        assert r['paired']['612101']['physical_all']['all']['fine']['mean']==1.
        assert r['native']['physical_all']['all']['fractional_gain']['mean']==.5
        assert r['qualification']['physical_all']['all']['predictive_control_both_seeds']
        assert r['qualification']['physical_all']['all']['genuine_look_control_both_seeds']
        # The same machinery must refuse a missing positive control.
        for seed in R.SEEDS:
            torch.save(dict(p=torch.full((n,81),2.),g=torch.full((n,9),2.)),p/f'errors_fine_{seed}.pt')
            readers[f'fine_{seed}']=readers[f'base_{seed}']
        torch.save(dict(meta=data,readers=readers,native=native),p/'paired.pt');R.HERE=p
        try:
            with open(p/'log','w') as log,contextlib.redirect_stdout(log):R.main()
        finally:R.HERE=original
        r=json.loads((p/'RESULTS.json').read_text())
        assert not r['qualification']['physical_all']['all']['predictive_control_both_seeds']
        assert not r['qualification']['physical_all']['all']['genuine_look_control_both_seeds']
    print('Known50% error gain, positive-look sign, native50% gain and failed-control rejection passed')


if __name__=='__main__':main()
