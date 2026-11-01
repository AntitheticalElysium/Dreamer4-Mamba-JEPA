"""Compare item-ranked and threshold-grouped AP on the reproduced teacher."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


def main():
    out=Path(__file__).parent
    data=np.load('artifacts/eda/system_design_v1/a9/a9b_cells.npz')
    groups,night,y,x=(data[k] for k in ('real_g','real_night','real_y','real_X'))
    perm=np.random.default_rng(0).permutation(np.unique(groups))
    test=np.isin(groups,perm[int(.7*len(perm)):])
    model=lgb.Booster(model_file=str(out/'teacher_reproduction/change_teacher.txt'))
    result={}
    for key,flag in [('day',False),('night',True)]:
        ix=test&(night==flag);truth=y[ix].astype(bool);scores=model.predict(x[ix])
        order=np.argsort(-scores,kind='stable');truth=truth[order];scores=scores[order]
        tp=np.cumsum(truth,dtype=np.int64);ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
        total=int(truth.sum());at=tp[ends]
        ranked=float(np.sum(tp[truth]/(np.flatnonzero(truth)+1))/total)
        grouped=float(np.sum(at/(ends+1)*np.diff(np.r_[0,at]))/total)
        assert 0<=ranked<=1 and 0<=grouped<=1
        result[key]={'rank_by_item_AP':ranked,'threshold_grouped_AP':grouped,'difference':ranked-grouped,
                     'observations':len(scores),'positives':total,'unique_scores':len(ends)}
    (out/'teacher_AP_ties.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
