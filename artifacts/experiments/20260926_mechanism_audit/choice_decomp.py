"""Episode-bootstrap decomposition of W versus U native-head zombie choice."""
import json
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'artifacts/experiments/20260926_diagnosis'),
                str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
from choices import move_table
from frozen_ladder import strata
DUMP = ROOT/'artifacts/eda/diagnosis_dump_v1'

def summary(p, uw, ww, cat, ix):
    p, uw, ww, cat = p[ix], uw[ix], ww[ix], cat[ix]
    cu, cw = uw.argmin(1), ww.argmin(1)
    su = (1-p.gather(1, cu[:,None]).squeeze(1)).mean()
    sw = (1-p.gather(1, cw[:,None]).squeeze(1)).mean()
    piu = torch.bincount(cu, minlength=17).float()/len(cu)
    piw = torch.bincount(cw, minlength=17).float()/len(cw)
    marg = ((1-p)*(piw-piu)).sum(1).mean()
    total = sw-su
    def detail(score):
        ok, stay = cat==0, cat==2
        has=ok.any(1)
        minok=score.masked_fill(~ok,float('inf')).amin(1)
        minstay=score.masked_fill(~stay,float('inf')).amin(1)
        meanok=(score*ok).sum(1)/ok.sum(1).clamp_min(1)
        meanstay=(score*stay).sum(1)/stay.sum(1).clamp_min(1)
        return {'stay_chosen':float((cat.gather(1,score.argmin(1)[:,None]).squeeze(1)==2).float().mean()),
                'move_ok_chosen':float((cat.gather(1,score.argmin(1)[:,None]).squeeze(1)==0).float().mean()),
                'mean_stay_minus_move_risk':float((meanstay[has]-meanok[has]).mean()),
                'min_stay_below_min_move':float((minstay[has]<minok[has]).float().mean()),
                'has_move_ok':float(has.float().mean())}
    return {'W_minus_U':float(total),'action_marginal':float(marg),
            'state_conditional_remainder':float(total-marg),'W':detail(ww),'U':detail(uw)}

def main():
    report={}
    for block in ('55k','56k','57k','58k'):
        meta=torch.load(DUMP/f'{block}_meta.pt',weights_only=False)
        u=torch.load(DUMP/f'{block}_U.pt',weights_only=False)['p_dead']
        w=torch.load(DUMP/f'{block}_W.pt',weights_only=False)['p_dead']
        cat,_=move_table(meta['visible'])
        zom=strata(meta['visible'])['zombie_adjacent']
        indices=zom.nonzero()[:,0]
        groups={}
        for i in indices.tolist(): groups.setdefault(int(meta['seed'][i]),[]).append(i)
        groups=[torch.tensor(v) for v in groups.values()]
        result=summary(meta['p_death1'],u,w,cat,indices)
        rng=random.Random(409)
        draws=[]
        for _ in range(1000):
            ix=torch.cat([groups[rng.randrange(len(groups))] for _ in groups])
            s=summary(meta['p_death1'],u,w,cat,ix)
            draws.append([s[k] for k in ('W_minus_U','action_marginal','state_conditional_remainder')])
        vals=torch.tensor(draws)
        result['episode_cluster_95pct']={k:[float(v) for v in vals[:,i].quantile(torch.tensor([.025,.975]))]
                                          for i,k in enumerate(('W_minus_U','action_marginal','state_conditional_remainder'))}
        result['n_roots']=len(indices); result['n_episodes']=len(groups)
        report[block]=result
        print(block,json.dumps(result),flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
