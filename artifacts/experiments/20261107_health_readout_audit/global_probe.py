"""Post-hoc check of the implicit cohort information supplied to adjacent-only probes."""
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
RT=importlib.import_module('artifacts.experiments.20261106_determinant_routes.routes')
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    C=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False)
    T,j=z['T'],z['rows'];y=(T['true_body'][j,0]<T['body'][j,0]).astype(float)
    X=z['states']['native0/mod'].double();weights=torch.as_tensor(z['w']).double();fold=z['fold'];score=np.zeros(len(y))
    for f in range(5):
        a=fold!=f;b=~a;wt=weights[a]/weights[a].mean()
        m=(X[a]*wt[:,None]).mean(0);sd=((X[a]-m).square()*wt[:,None]).mean(0).sqrt().clamp(min=1e-6)
        xx=(X[a]-m)/sd;yy=torch.as_tensor(y[a]).double();te=(X[b]-m)/sd
        w=torch.zeros(X.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros(1,dtype=torch.float64,requires_grad=True)
        opt=torch.optim.LBFGS([w,bias],max_iter=500,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad();lo=C*(F.binary_cross_entropy_with_logits(xx@w+bias,yy,reduction='none')*wt).sum()+.5*w.square().sum()
            lo.backward();return lo
        opt.step(closure);score[b]=torch.sigmoid(te@w+bias).detach().numpy()
    res={}
    for name,mask in [('natural_all',np.ones(len(j),bool)),('awake_adjacent',z['groups']==0),('asleep_adjacent',z['groups']==1),
                      ('awake_nonadjacent',z['groups']==2),('asleep_nonadjacent',z['groups']==3)]:
        yy=y[mask];ss=score[mask];wt=z['w'][mask]
        res[name]={'rows':int(mask.sum()),'ap_unweighted_within_stratum':RT.G.average_precision(yy,ss),
                   'true':float(np.average(yy,weights=wt)),'prediction':float(np.average(ss,weights=wt)),
                   'brier':float(np.average((ss-yy)**2,weights=wt))}
        for lab,mm in [('recorded_hit',mask&(y>0)),('recorded_nohit',mask&(y==0))]:
            if mm.any():res[name][lab]=float(np.average(score[mm],weights=z['w'][mm]))
    suffix='' if C==1 else '_C'+str(int(C))
    (HERE/('GLOBAL_PROBE'+suffix+'.json')).write_text(json.dumps(res,indent=2));np.save(HERE/('global_probe_scores'+suffix+'.npy'),score)
    print(json.dumps(res,indent=2))

if __name__=='__main__':main()
