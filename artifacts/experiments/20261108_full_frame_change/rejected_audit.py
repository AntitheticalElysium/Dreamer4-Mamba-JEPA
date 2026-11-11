"""Bounded CPU audit; simulator labels are evaluation only. No training or stored artifact mutation."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import hashlib, importlib, json, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from saccade import data as D
from saccade.eval.gates import average_precision
from .legacy_pair_features import pair_features as legacy_pair_features

HERE = Path(__file__).resolve().parent
A = D._a9b()

def full_cells(frames):
    return frames.astype(np.float32).reshape(-1,9,7,9,7,3).transpose(0,1,3,2,4,5)

def correspondence(frames):
    """Dominant translation, then leave-center-out local surround chooses that translation or identity.
    The same rule applies at every cell. No row type, player location or simulator input."""
    c = full_cells(frames)
    prev, cur = c[:-1], c[1:]
    costs, srcs, valids = [], [], []
    rr,cc = np.meshgrid(np.arange(9),np.arange(9),indexing='ij')
    for dr,dc in A.SHIFTS:
        r,q=rr-dr,cc-dc
        valid=(r>=0)&(r<9)&(q>=0)&(q<9)
        src=prev[:,r.clip(0,8),q.clip(0,8)]
        costs.append(np.abs(cur-src).mean((-1,-2,-3)))
        srcs.append(src);valids.append(valid)
    costs=np.stack(costs,1);valids=np.stack(valids)
    glob=((costs*valids).sum((-1,-2))/valids.sum((-1,-2))).argmin(1)
    gi=np.arange(len(glob))
    gc=costs[gi,glob];gv=valids[glob]
    # Neighbour votes use residuals, excluding the center. The candidate source must also be valid there.
    def surround(x,v):
        s=np.zeros_like(x);n=np.zeros_like(x)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==dc==0:continue
                r0,r1=max(0,-dr),min(9,9-dr);c0,c1=max(0,-dc),min(9,9-dc)
                s[:,r0:r1,c0:c1]+=x[:,r0+dr:r1+dr,c0+dc:c1+dc]*v[:,r0+dr:r1+dr,c0+dc:c1+dc]
                n[:,r0:r1,c0:c1]+=v[:,r0+dr:r1+dr,c0+dc:c1+dc]
        return s/np.maximum(n,1)
    identity=np.broadcast_to(valids[0],gc.shape)
    choose_id=surround(costs[:,0],identity)<=surround(gc,gv)+1e-6
    # Exact identity is direct evidence of no visible change, irrespective of the surround's motion.
    choose_id |= costs[:,0]==0
    shift=np.asarray(A.SHIFTS)[glob]
    sr=rr[None]-shift[:,0,None,None];sq=cc[None]-shift[:,1,None,None]
    sr=np.where(choose_id,rr,sr);sq=np.where(choose_id,cc,sq)
    valid=(sr>=0)&(sr<9)&(sq>=0)&(sq<9)
    src=prev[np.arange(len(prev))[:,None,None],sr.clip(0,8),sq.clip(0,8)]
    return c,src,valid,shift,choose_id

def audit(n=160,start=0,teacher=None):
    L=dict(np.load(D.ROOT/'artifacts/eda/s0/s0_dev_labels.npz'))
    off=np.r_[0,np.cumsum(L['ep_T']+1)]
    old=lgb.Booster(model_file=str(D.ROOT/'artifacts/saccade_m0/targets/change_teacher.txt'))
    model=lgb.Booster(model_file=str(teacher)) if teacher else old
    store={k:[] for k in ('y','score_identity','score_generic','residual','night','value','newvalue','cell','e')}
    aliases={};conflicts=[];total=0;matched=0;maps=[]
    for e in range(start,min(start+n,len(L['ep_T']))):
        frames=D.RP.episode_fields(int(L['ep_shard'][e]),int(L['ep_slot'][e]))['observations'].numpy()
        c,src,valid,shift,ident=correspondence(frames)
        lo,hi=off[e:e+2];body=L['s_body'][lo:hi];T=len(frames)-1
        target=c[1:,7,0:4].reshape(-1,7,7,3)
        same=c[:-1,7,0:4].reshape(-1,7,7,3)
        generic=src[:,7,0:4].reshape(-1,7,7,3)
        sc=old.predict(A.feats(target,same),num_threads=4).reshape(T,4)
        sg=model.predict(A.feats(target,generic),num_threads=4).reshape(T,4)
        sg[np.abs(target-generic).max((1,2,3)).reshape(T,4)==0]=0
        fields={'y':body[1:]!=body[:-1],'score_identity':sc,'score_generic':sg,
                'residual':np.abs(target-same).mean((1,2,3)).reshape(T,4),
                'night':np.broadcast_to(L['s_light'][lo+1:hi,None]<.5,(T,4)),
                'value':body[:-1],'newvalue':body[1:],'cell':np.broadcast_to(np.arange(4),(T,4)),
                'e':np.full((T,4),e)}
        for k,v in fields.items():store[k].append(v.flatten())
        # Old map instrument, same rows, same world-coordinate truth; new teacher has no access to these labels.
        xo,(pi,ri,qi)=legacy_pair_features(frames)
        rr,qq=np.meshgrid(np.arange(7),np.arange(9),indexing='ij')
        pos=L['s_pos'][lo:hi].astype(int);sh=-(pos[1:]-pos[:-1])
        sr=ri-sh[pi,0];sq=qi-sh[pi,1]
        good=(sr>=0)&(sr<7)&(sq>=0)&(sq<9)&~((sr==3)&(sq==4))&valid[pi,ri,qi]
        pi,ri,qi,sr,sq=(a[good] for a in (pi,ri,qi,sr,sq))
        occ=np.zeros((T+1,7,9),np.int16)
        for ki,kind in enumerate(('zombies','cows','skeletons','arrows')):
            p=L[f's_{kind}_p'][lo:hi].astype(int)-pos[:,None]
            vis=L[f's_{kind}_m'][lo:hi]&(abs(p[:,:,0])<=3)&(abs(p[:,:,1])<=4)
            ti,mi=np.nonzero(vis);occ[ti,p[ti,mi,0]+3,p[ti,mi,1]+4]=1+ki*16+mi
        ymap=(L['s_view'][lo+pi+1,ri,qi]!=L['s_view'][lo+pi,sr,sq])|(occ[pi+1,ri,qi]!=occ[pi,sr,sq])
        po=old.predict(xo[good],num_threads=4)
        xn=A.feats(c[pi+1,ri,qi],src[pi,ri,qi])
        pn=model.predict(xn,num_threads=4)
        pc=old.predict(xn,num_threads=4)
        pt=model.predict(xo[good],num_threads=4)
        exact=np.abs(c[pi+1,ri,qi]-src[pi,ri,qi]).max((1,2,3))==0;pn[exact]=0
        pc[exact]=0
        maps.append((ymap,po,pn,L['s_light'][lo+pi+1]<.5,np.full(len(pi),e),pc,pt))
        # Full resolution recent frames are a stronger observation than the model's periphery.
        f=D.RP.episode_fields(int(L['ep_shard'][e]),int(L['ep_slot'][e]))
        acts=f['actions_taken'].numpy();rew=f['rewards'].numpy();term=f['terminated'].numpy()
        fh=[hashlib.blake2b(x.tobytes(),digest_size=16).digest() for x in frames]
        for t in range(8,T):
            # Frames t-7..t, past transition signals into those frames, and the next action at t.
            data=b''.join(fh[t-7:t+1])+acts[t-8:t+1].tobytes()+rew[t-8:t].tobytes()+term[t-8:t].tobytes()
            key=hashlib.blake2b(data,digest_size=24).digest();event=(body[t+1]-body[t]).astype(int)
            total+=1
            if key in aliases:
                matched+=1
                prior=aliases[key]
                if np.any(event!=prior['event']):
                    conflicts.append({'e':e,'t':t,'other_e':prior['e'],'other_t':prior['t'],'event':event.tolist(),'other_event':prior['event'].tolist()})
            else:aliases[key]={'e':e,'t':t,'event':event}
        if e%20==0:print('episode',e,'aliases',matched,'conflicts',len(conflicts),flush=True)
    data={k:np.concatenate(v) for k,v in store.items()}
    report={'episodes':min(n,len(L['ep_T'])-start),'start_episode':start,'teacher':str(teacher),'hud':{},'map':{},'raw_history':{'windows':total,'duplicate_histories':matched,'conflicts':conflicts}}
    for col,name in enumerate(('health','food','drink','energy')):
        m=data['cell']==col;y=data['y'][m]
        report['hud'][name]={'rows':int(m.sum()),'events':int(y.sum()),'methods':{}}
        for method in ('score_identity','score_generic'):
            p=data[method][m];report['hud'][name]['methods'][method]={'ap':average_precision(y,p),'recall':float((p[y]>.5).mean()),'fpr':float((p[~y]>.5).mean())}
    yy,oo,nn,night,ep,pc,pt=(np.concatenate([m[k] for m in maps]) for k in range(7))
    for name,m in (('day',~night),('night',night)):
        report['map'][name]={'rows':int(m.sum()),'events':int(yy[m].sum()),'old_ap':average_precision(yy[m],oo[m]),'new_ap':average_precision(yy[m],nn[m]),
                            'old_teacher_new_correspondence_ap':average_precision(yy[m],pc[m]),'new_teacher_old_correspondence_ap':average_precision(yy[m],pt[m]),
                            'old_recall':float((oo[m&yy]>.5).mean()),'new_recall':float((nn[m&yy]>.5).mean()),'new_fpr':float((nn[m&~yy]>.5).mean())}
    name='QUALIFICATION' if teacher else 'AUDIT'
    np.savez_compressed(HERE/f'{name.lower()}_arrays.npz',**data,map_y=yy,map_old=oo,map_new=nn,map_night=night,map_ep=ep)
    (HERE/f'{name}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':audit(int(sys.argv[1]) if len(sys.argv)>1 else 160,int(sys.argv[2]) if len(sys.argv)>2 else 0,Path(sys.argv[3]) if len(sys.argv)>3 else None)
