"""E17 CPU chosen-action error diagnosis, declared Oct7 before execution.

Reuse reproduced selected/final4000 original Mamba7/8 hazards from generalization.
Same DEV-B1139 opportunities, three heads' individual decisions. Measure prediction
error at argmin versus average over actions, exact32-key optimal-action fraction,
and weighted all-pair discordance. Chosen centered error removes root calibration
offset; it is selection error, not proof of absence of latent information or a
specific historical training mechanism. Comparing final versus selected weights
uses the same worlds/features/labels. No fitting/GPU/world forward. Atomic bound
raw evidence,4000 episode-cluster paired intervals. Reused panel, exploratory.
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
    p=HERE/'evals/resume/e17_h16_generalization';c=json.loads((p/'contract.json').read_text())
    old=R.Store(p,c);assert old.load('result') is not None;rows=old.load('rows')
    mp=D.CACHE/'dev_meta.pt';meta=torch.load(mp,mmap=True,weights_only=False)
    ib=torch.where(meta['seed']%2==1)[0]
    inputs={str(mp):R.file_hash(mp)}
    for k in ['rows']+[f'm{s}_dev_{lo}' for s in (7,8) for lo in range(0,len(meta['seed']),32)]:
        rec=json.loads((p/f'{k}.json').read_text());inputs[str(p/rec['file'])]=R.file_hash(p/rec['file'])
    spec={'scope':__doc__,'sources':{f:R.file_hash(f) for f in (__file__,R.__file__,D.__file__)},
          'inputs':inputs,'previous_contract':old.contract,'runtime':str(torch.__version__)}
    store=R.Store(HERE/'evals/resume/e17_h16_choice_error',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        summaries={};raw={}
        for seed in (7,8):
            h=torch.cat([old.load(f'm{seed}_dev_{lo}')['hazard'] for lo in range(0,len(meta['seed']),32)],dim=2)[:,:,ib]
            P=rows[seed]['DEV_B']['P16'];opp=P.amax(1)>P.amin(1)
            truth=P[opp].unsqueeze(0).unsqueeze(0).expand(3,2,-1,-1)
            energy=h.sum(-1)[:,:,opp];pred=1-torch.exp(-energy);choice=energy.argmin(-1)
            assert torch.equal(choice,rows[seed]['DEV_B']['choices'][:,:,opp])
            gather=lambda x: x.gather(-1,choice[...,None]).squeeze(-1)
            chosen_true=gather(truth);chosen_pred=gather(pred)
            all_error=(pred-truth).mean(-1)
            chosen_error=chosen_pred-chosen_true
            centered=chosen_error-all_error
            optimal=P[opp].amin(-1)
            regret=chosen_true-optimal
            target_delta=(truth[...,None,:]-truth[...,:,None]).clamp_min(0)
            pred_delta=energy[...,:,None]-energy[...,None,:]
            discordant=(pred_delta>0).float()+.5*(pred_delta==0).float()
            weighted=(target_delta*discordant).sum((-1,-2))/target_delta.sum((-1,-2)).clamp_min(1e-30)
            names={}
            for j,name in enumerate(('selected','final4000')):
                names[name]={'safe':float((1-chosen_true[:,j]).mean()),'regret':float(regret[:,j].mean()),
                    'chosen_prediction_error':float(chosen_error[:,j].mean()),'all_action_prediction_error':float(all_error[:,j].mean()),
                    'chosen_centered_prediction_error':float(centered[:,j].mean()),
                    'optimal_choice_fraction':float((chosen_true[:,j]==optimal).float().mean()),
                    'weighted_all_pair_discordance':float(weighted[:,j].mean()),
                    'predicted_chosen_death':float(chosen_pred[:,j].mean()),'actual_chosen_death':float(chosen_true[:,j].mean())}
            ids=rows[seed]['DEV_B']['seeds']
            summaries[str(seed)]={'scores':names,'final_minus_selected_centered_error':D.paired(centered[:,1].mean(0).numpy(),centered[:,0].mean(0).numpy(),ids),
                'final_minus_selected_regret':D.paired(regret[:,1].mean(0).numpy(),regret[:,0].mean(0).numpy(),ids)}
            raw[seed]={'choice':choice,'centered_error':centered,'chosen_error':chosen_error,'regret':regret,'pair_discordance':weighted,'seeds':ids}
        store.save('rows',raw,1);result={'scope':__doc__,'contract':spec,'worlds':summaries}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_h16_choice_error.json',result)
        print(json.dumps({'worlds':summaries}),flush=True)


if __name__=='__main__':main()
