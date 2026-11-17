"""Post-hoc adversarial control: can identical frozen features translate to targets nonlinearly?
Not a deployable repair: health-specific, finest-step pure-noise target only, DEV episode folds.
One fixed architecture/seed/budget, natural inverse inclusion weights, no event rebalancing.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False);s=z['states']
    X=s['fine0/mod'].float();Y=s['truth'].float();base=s['fine0/out'].float()
    weight=torch.as_tensor(z['w']).float();ep=z['ep'];fold=z['fold'];pred=torch.zeros_like(Y)
    logs=[]
    for f in range(5):
        torch.manual_seed(110717);rng=np.random.default_rng(110717)
        fitting=fold!=f;ids=np.unique(ep[fitting]);valeps=rng.choice(ids,max(1,len(ids)//10),replace=False)
        val=fitting&np.isin(ep,valeps);tr=fitting&~val;te=fold==f
        m=X[tr].mean(0);sd=X[tr].std(0).clamp(min=1e-5);xx=(X-m)/sd
        model=nn.Sequential(nn.Linear(X.shape[1],64),nn.GELU(),nn.Linear(64,Y.shape[1]))
        nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf');beststate=None;curve=[];ww=weight/weight[tr].mean();ix=torch.as_tensor(np.flatnonzero(tr))
        for epoch in range(50):
            for q in ix[torch.randperm(len(ix))].split(1024):
                loss=(((base[q]+model(xx[q])-Y[q]).square().mean(1))*ww[q]).mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                v=float(((base[val]+model(xx[val])-Y[val]).square().mean(1)*weight[val]).sum()/weight[val].sum())
            curve.append(v)
            if v<best:
                best=v;beststate={k:v.clone() for k,v in model.state_dict().items()};bestepoch=epoch
        model.load_state_dict(beststate)
        with torch.no_grad():pred[te]=base[te]+model(xx[te])
        logs.append({'fold':f,'best_epoch':bestepoch,'validation':curve})
        print('fold',f,'best',bestepoch,best,flush=True)
    T,j=z['T'],z['rows'];hit=T['true_body'][j,0]<T['body'][j,0];dd=s['truth']-s['copy'];den=dd.square().sum(1).clamp(min=1e-8)
    result={}
    for name,mask in [('all',np.ones(len(j),bool)),('awake_adjacent',z['groups']==0),('asleep_adjacent',z['groups']==1),
                      ('awake_hit',(z['groups']==0)&hit),('asleep_hit',(z['groups']==1)&hit)]:
        result[name]={}
        for arm,p in [('original_fine',base),('nonlinear',pred)]:
            mse=(p-Y).square().mean(1)
            r={'weighted_mse':float((mse[mask]*weight[mask]).sum()/weight[mask].sum())}
            if 'hit' in name:r['mean_lean']=float((((p-s['copy'])*dd).sum(1)/den)[mask].mean())
            result[name][arm]=r
    torch.save(pred,HERE/'nonlinear_predictions.pt')
    (HERE/'NONLINEAR.json').write_text(json.dumps({'metrics':result,'fitting':logs},indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
