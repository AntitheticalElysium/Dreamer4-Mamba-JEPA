"""W/U one-step physical effect and error spectrum in common raw-U coordinates."""
import json
from pathlib import Path

import torch

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
DATA=ROOT/'artifacts/eda/diagnosis_rollouts_v1'

def main():
    u=torch.load(DATA/'U.pt',weights_only=False,mmap=True)
    w=torch.load(DATA/'W.pt',weights_only=False,mmap=True)
    pool=torch.load(ROOT/'artifacts/eda/interface_pool_v1/pool.pt',weights_only=False,mmap=True)
    std=torch.load(ROOT/'artifacts/eda/interface_worlds_white/W.pt',weights_only=False,map_location='cpu')['std'].float()
    lam_u=pool['weights']['U'].float()
    var_w=(pool['u'][~pool['terminal']].reshape(-1,192)/std).var(0)
    lam_w=var_w.rsqrt();lam_w/=lam_w.mean()
    w_eff=lam_w/std.square()
    root=u['root'].float()[:,None];true=u['one_true'].float()[:,0]
    pred_u=u['one'].float();pred_w=w['one'].float()*std
    signal=(true-root).square().mean((0,1))
    err_u=(pred_u-true).square().mean((0,1))
    err_w=(pred_w-true).square().mean((0,1))
    out={'n_roots':len(root),'n_actions':true.shape[1],
         'root_equivalence_max_abs':float((u['root'].float()-w['root'].float()*std).abs().max()),
         'true_equivalence_max_abs':float((u['one_true'].float()-w['one_true'].float()*std).abs().max()),
         'bins':{}}
    for lo,hi in ((0,10),(10,20),(20,60),(60,100),(100,192)):
        sl=slice(lo,hi);ss=float(signal[sl].sum());eu=float(err_u[sl].sum());ew=float(err_w[sl].sum())
        out['bins'][f'{lo}:{hi}']={'share_of_true_effect':ss/float(signal.sum()),
            'U_effect_r2':1-eu/ss,'W_effect_r2':1-ew/ss,'W_over_U_error':ew/eu,
            'U_loss_share_of_true_effect':float((lam_u[sl]*signal[sl]).sum()/(lam_u*signal).sum()),
            'W_loss_share_of_true_effect':float((w_eff[sl]*signal[sl]).sum()/(w_eff*signal).sum())}
    (HERE/'error_spectrum.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
