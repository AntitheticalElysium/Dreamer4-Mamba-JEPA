"""Correspondence engineering control, DEV simulator shift used only to evaluate generic estimators."""
import json,numpy as np
from saccade import data as D
from .audit import HERE,A,full_cells

def costs_of(frames):
    c=full_cells(frames);prev,cur=c[:-1],c[1:];out=[];rr,cc=np.meshgrid(np.arange(9),np.arange(9),indexing='ij')
    for dr,dc in A.SHIFTS:
        r,q=rr-dr,cc-dc;valid=(r>=0)&(r<9)&(q>=0)&(q<9)
        x=np.abs(cur-prev[:,r.clip(0,8),q.clip(0,8)]).mean((-1,-2,-3))
        out.append(np.where(valid,x,np.inf))
    return np.stack(out,1)

def main(n=80):
    L=dict(np.load(D.ROOT/'artifacts/eda/s0/s0_dev_labels.npz'));off=np.r_[0,np.cumsum(L['ep_T']+1)];all={}
    for e in range(n):
        f=D.RP.episode_fields(int(L['ep_shard'][e]),int(L['ep_slot'][e]))['observations'].numpy();cost=costs_of(f);x=cost.reshape(len(cost),5,81)
        c=full_cells(f)
        w=(np.abs(np.diff(c,axis=0))>0).mean((0,3,4,5)).reshape(81)+.01
        wp=(np.abs(np.diff(c,axis=0))>0).mean(0)
        # Pixelwise temporal support avoids screen-fixed glyphs dominating camera estimation.
        pix=[]
        pair_weight=[]
        rr,cc=np.meshgrid(np.arange(9),np.arange(9),indexing='ij')
        for dr,dc in A.SHIFTS:
            sr,sq=rr-dr,cc-dc;v=(sr>=0)&(sr<9)&(sq>=0)&(sq<9)
            pair_weight.append((w.reshape(9,9)*w.reshape(9,9)[sr.clip(0,8),sq.clip(0,8)]).reshape(81)*v.reshape(81))
            z=np.abs(c[1:]-c[:-1,sr.clip(0,8),sq.clip(0,8)])
            ww=np.maximum(wp,.001)**2
            pc=(z*ww).sum((-1,-2,-3))/ww.sum((-1,-2,-3))
            pix.append(np.where(v,pc,np.inf))
        pix=np.stack(pix,1).reshape(len(cost),5,81)
        gain=np.maximum(x[:,:1]-x,0)
        models={'full_mean':np.nanmean(np.where(np.isfinite(x),x,np.nan),axis=-1).argmin(1),
                'temporal_weight':np.sum(np.where(np.isfinite(x),x,0)*w,-1).__truediv__(np.sum(np.isfinite(x)*w,-1)).argmin(1),
                'temporal_weight2':np.sum(np.where(np.isfinite(x),x,0)*w**2,-1).__truediv__(np.sum(np.isfinite(x)*w**2,-1)).argmin(1),
                'gain':gain.sum(-1).argmax(1),
                'gain_trim4':np.sort(gain,axis=-1)[:,:,:-4].sum(-1).argmax(1),
                'gain_trim8':np.sort(gain,axis=-1)[:,:,:-8].sum(-1).argmax(1),
                'old':np.asarray([A.SHIFTS.index(A.est_shift(A.cells(a),A.cells(b))) for a,b in zip(f[:-1],f[1:])])}
        models['pixel_temporal']=np.nanmean(np.where(np.isfinite(pix),pix,np.nan),-1).argmin(1)
        pair_weight=np.stack(pair_weight)
        models['pair_temporal']=(np.sum(np.where(np.isfinite(x),x,0)*pair_weight,-1)/np.maximum(pair_weight.sum(-1),1e-12)).argmin(1)
        for trim in (9,18):
            wx=np.where(np.isfinite(x),x,1e8)*w**2
            order=np.argsort(wx,axis=-1)[:,:,:81-trim]
            num=np.take_along_axis(wx,order,-1).sum(-1)
            den=np.take_along_axis(np.broadcast_to(w**2,wx.shape),order,-1).sum(-1)
            models[f'weighted_trim{trim}']=(num/den).argmin(1)
        lo,hi=off[e:e+2];true=-(L['s_pos'][lo+1:hi].astype(int)-L['s_pos'][lo:hi-1].astype(int))
        night=L['s_light'][lo+1:hi]<.5
        for name,p in models.items():
            hit=(np.asarray(A.SHIFTS)[p]==true).all(-1)
            for tag,m in [('all',np.ones(len(hit),bool)),('day',~night),('night',night),('moved',np.any(true!=0,-1)),('still',~np.any(true!=0,-1))]:
                v=all.setdefault(name,{}).setdefault(tag,[0,0]);v[0]+=int(hit[m].sum());v[1]+=int(m.sum())
        if e%20==0:print(e,flush=True)
    result={k:{tag:{'correct':v[0],'n':v[1],'accuracy':v[0]/max(v[1],1)} for tag,v in d.items()} for k,d in all.items()}
    (HERE/'MOTION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
