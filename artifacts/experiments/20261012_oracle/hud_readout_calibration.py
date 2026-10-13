"""Reproduce the CPU affine HUD control from saved outputs, without TEST tuning."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import hud_readout_refit as H


def main():
    torch.set_num_threads(4)
    root=H.G.S.OUT/'generated_prefix';out=root/'hud_readout_refit'
    manifest=json.loads((root/'readout_refit/contract.json').read_text())
    panel=torch.load(H.G.S.OUT/'panel.pt',map_location='cpu',weights_only=False)
    visible=torch.load(H.G.S.OUT/'snap_visible.pt',map_location='cpu',weights_only=False)['visible']
    eps=[r['episode_id']for r in panel['roots']]
    ids={k:np.array([eps.index(e)for e in v])for k,v in manifest['split'].items()}
    y=visible[:,16:32,1512:1516].float()*9
    reader=H.G.S.Readers(torch.load(H.G.S.OUT/'snap_bank_readers.pt',map_location='cpu',weights_only=False))
    prior=json.loads((out/'positive_control_calibration.json').read_text())
    result={};difference=0.
    inputs=[]
    for update,arm in ((0,'teacher'),(3000,'teacher'),(3000,'prefix')):
        path=root/f'eval_u{update:04d}'/f'pred_{arm}.pt';inputs.append(path)
        pred=torch.load(path,map_location='cpu',weights_only=False)['gen']
        x=reader.hud(pred[:,:,63:].float().flatten(2).flatten(0,1))[:,:4].view(200,16,4).double()
        design=lambda selected:torch.cat([x[selected].flatten(0,1),torch.ones(len(selected)*16,1,dtype=torch.float64)],1)
        xf,xd,xt=design(ids['FIT']),design(ids['DEV']),design(ids['TEST'])
        yf,yd=y[ids['FIT']].flatten(0,1).double(),y[ids['DEV']].flatten(0,1).double()
        best=None
        for alpha in H.G.ALPHAS:
            w=torch.linalg.solve(xf.T@xf+alpha*len(xf)*torch.eye(5,dtype=torch.float64),xf.T@yf)
            error=float((xd@w-yd).square().mean())
            if best is None or error<best[0]:best=error,alpha,w
        z=(xt@best[2]).view(50,16,4)
        mse=(z[:,8:]-y[ids['TEST'],8:]).square().mean((0,1))
        old=prior['frozen_output_affine_calibration'][f'{update}/{arm}']
        assert best[1]==old['alpha']
        values={counter:float(mse[j])for j,counter in enumerate(H.NAMES)}
        difference=max(difference,max(abs(values[n]-old['calibrated_mse'][n])for n in H.NAMES))
        result[f'{update}/{arm}']={'alpha':best[1],'calibrated_mse':values}
    assert difference<1e-6,difference
    H.G.S.atomic({'sources':{str(Path(p).relative_to(H.G.ROOT)):H.G.S.sha(p)for p in (__file__,H.__file__)},
                  'inputs':{str(p):H.G.S.sha(p)for p in inputs},'split':manifest['split'],
                  'scope':'unstandardized four-counter affine ridge, regularized intercept; held-episode alpha selection; no dynamics change',
                  'maximum_difference_from_initial_CPU_calibration':difference,'readings':result},
                 out/'calibration_reproduction.json',True)
    print(json.dumps({'maximum_difference':difference,'readings':result}),flush=True)


if __name__=='__main__':main()
