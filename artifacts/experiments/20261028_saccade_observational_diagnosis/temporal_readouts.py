"""Time structure of already saved real-path predictions, with no model calls."""
from .common import *


if __name__=='__main__':
    L=labels('dev');cov=torch.load(OUT/'natural_covariates.pt',weights_only=False);Y=hud_values(L)[:,:4];ep=cov['ep'].numpy();depth=cov['depth'].numpy();alive=cov['alive'].numpy()
    off=np.r_[0,np.cumsum(L['ep_T']+1)];end=off[ep+1];rows=np.arange(len(ep));deep=alive&(depth>=64);out={}
    delta=np.zeros_like(Y);delta[1:]=Y[1:]-Y[:-1];delta[depth==0]=0
    for rep in ('full_percept','x','b'):
        for seed in (0,1):
            pred=torch.load(OLD/f'belief_{rep}_body_body_{seed}.pt',weights_only=False,mmap=True)['hud_prediction'].numpy()[:,:4].astype(int)
            r={}
            for j,h in enumerate(HUD[:4]):
                a={'fixed_cohort_mae_by_lag':{},'events':{}}
                for lag in (0,1,2,4,8,16,32,64):
                    target=Y[(rows-lag).clip(min=0),j];a['fixed_cohort_mae_by_lag'][lag]=G.estimate(np.abs(pred[:,j]-target),deep,ep)
                for label,mask in [('decrease',delta[:,j]<0),('increase',delta[:,j]>0)]:
                    base=np.flatnonzero(mask&(depth>=4));a['events'][label]={}
                    for lag in (-4,-1,0,1,2,4,8,16):
                        dst=base+lag;ok=(dst>=off[ep[base]])&(dst<end[base]);basevalid=base[ok];dst=dst[ok];ok2=alive[dst];dst=dst[ok2];basevalid=basevalid[ok2]
                        if not len(dst):continue
                        diff=pred[dst,j]-Y[dst,j];e=ep[basevalid];m=np.ones(len(dst),bool)
                        a['events'][label][lag]={'n':len(dst),'exact':G.estimate(diff==0,m,e),'signed_error':G.estimate(diff,m,e),
                           'mae':G.estimate(np.abs(diff),m,e),'matches_event_previous_value':G.estimate(pred[dst,j]==Y[basevalid-1,j],m,e)}
                r[h]=a
            out[f'{rep}/{seed}']=r;save('temporal_readouts.json',out);print(rep,seed,flush=True)
