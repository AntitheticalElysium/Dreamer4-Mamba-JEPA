"""Post-hoc numerical decomposition of completed E17 matched-reader interventions.

Same frozen outputs/DEV-B outcomes, no fitting. Split probability MSE exactly
into root-mean and action-contrast MSE; compare weighted within-root pair errors,
optimal choices and selected-action error relative to each root's mean error.
Ranking energies are assessed only by order, never as calibrated probabilities.
Report numerical mechanisms of these heads, not information absence or actors.
"""
import json
from pathlib import Path
import numpy as np
import torch
import h16_resume as R
import e17_h16_diagnose as D

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    p=HERE/'evals/resume/e17_h16_target_intervention';c=json.loads((p/'contract.json').read_text())
    store=R.Store(p,c);assert store.load('result') is not None
    m=torch.load(D.CACHE/'dev_meta.pt',mmap=True,weights_only=False)
    ib=m['seed']%2==1;y=m['p16'][ib];opp=y.amax(1)>y.amin(1)
    spec={'scope':__doc__,'reference_contract':c,'sources':{f:R.file_hash(f) for f in (__file__,R.__file__,D.__file__)},
          'outcome_sha256':R.tensor_hash(y)}
    out={};raw={}
    for seed in (7,8):
        ss={}
        for arm in ('conditional','cumulative','rank','rank8'):
            d=store.load(f'm{seed}_{arm}_rows');energy=d['energy'][:,opp].double();t=y[opp].double()
            picks=energy.argmin(-1);ordered=energy[:,:,None,:]-energy[:,:,:,None]
            gt=t[:,None,:]-t[:,:,None];weight=gt.abs()
            wrong=(ordered*gt[None]<0).double()+.5*(ordered==0).double()
            miss=(wrong*weight[None]).sum((2,3))/weight.sum((1,2))[None].clamp_min(1e-12)
            chosen_truth=t[torch.arange(len(t))[None],picks]
            best=t.amin(1)[None];optimal=(chosen_truth==best).double()
            item={'weighted_pair_misordering':float(miss.mean()),'optimal32key_choice_fraction':float(optimal.mean()),
                  'safe_choice':float((1-chosen_truth).mean()),'energy_action_sd':float(energy.std(-1).mean())}
            if arm in ('conditional','cumulative'):
                pred=1-torch.exp(-energy);err=pred-t
                shared=err.mean(-1,keepdim=True);centered=err-shared
                mse=err.square().mean();rootmse=shared.square().mean();contrast=centered.square().mean()
                assert abs(float(mse-rootmse-contrast))<1e-12
                relative=err.gather(-1,picks[...,None])[...,0]-shared[...,0]
                item.update({'p16_mse_opportunities':float(mse),'root_mean_mse':float(rootmse),
                            'action_contrast_mse':float(contrast),'prediction_action_sd':float(pred.std(-1).mean()),
                            'selected_error_relative_to_root_mean':float(relative.mean())})
                raw[f'{seed}_{arm}_selected_relative_error']=relative.mean(0).numpy()
            ss[arm]=item
        ids=m['seed'][ib][opp].numpy()
        ss['cumulative_minus_conditional_selected_relative_error']=D.paired(raw[f'{seed}_cumulative_selected_relative_error'],raw[f'{seed}_conditional_selected_relative_error'],ids)
        out[str(seed)]=ss
    result={'scope':__doc__,'contract':spec,'worlds':out,'true_action_sd':float(y[opp].std(-1).mean())}
    R.atomic_json(HERE/'evals/e17_h16_target_errors.json',result);print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
