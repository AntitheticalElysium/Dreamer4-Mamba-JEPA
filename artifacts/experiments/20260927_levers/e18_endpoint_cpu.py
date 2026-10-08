"""Frozen 36k canvas versus slot-Mamba: paired recall and SSM-use diagnosis.

All original 2048 held main windows, CPU FP32, intact versus SSM-reset-each-step.
Keep convolution/history, newest input-dependent SSM update, actions and clocks.
Paired window bootstrap, not a fresh task gate, actor result or memory ceiling.
Per-batch input/source/weights-bound journals; never writes NOTEBOOK.
"""
import argparse
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import e18_access as A
import e17_recurrence as E

T,C,R=A.T,A.C,A.R
HERE=Path(__file__).resolve().parent
GROUPS=('recallable','same_slot','moved_slot','unseen')
MODES=('intact','reset_ssm')


def reset_scan(mixer,inputs,keep):
    c=mixer.core
    z,xbc,dt=torch.split(c.in_proj(inputs),[c.d_ssm,c.d_ssm+2*c.d_state,c.nheads],-1)
    filt=F.silu(F.conv1d(F.pad(xbc.transpose(1,2),(c.d_conv-1,0)),c.conv1d.weight,
                       c.conv1d.bias,groups=c.conv1d.groups)).transpose(1,2)
    x,b,cv=torch.split(filt,[c.d_ssm,c.d_state,c.d_state],-1)
    x=x.reshape(*x.shape[:2],c.nheads,c.headdim)
    delta=torch.where(keep[...,None],F.softplus(dt+c.dt_bias),0.)
    # Only this step's update contributes; convolution and direct skip remain.
    read=delta[...,None]*x*(b*cv).sum(-1)[...,None,None]+c.D[None,None,:,None]*x
    gated=read.flatten(2)*F.silu(z)
    normalized=gated*torch.rsqrt(gated.square().mean(-1,keepdim=True)+c.norm.eps)
    return c.out_proj(normalized*c.norm.weight)


@contextlib.contextmanager
def cpu_mode(world,mode,parity=None):
    old=T.masked_scan
    def scan(mixer,inputs,keep):
        if parity is not None:
            control=A.reference(mixer,inputs,torch.ones_like(keep),None,[])
            actual=mixer.scan(inputs,backend='reference')[0]
            delta=float((control-actual).abs().max());parity.append(delta)
            assert delta<=1e-5,delta
        return A.reference(mixer,inputs,keep,None,[])if mode=='intact'else reset_scan(mixer,inputs,keep)
    try:
        T.masked_scan=scan
        with E.lesion(world,mode):yield
    finally:T.masked_scan=old


def stats(rows,draws=2000):
    # [window, mode, group, count/world/sighting/neighbour]
    rng=np.random.default_rng(20261006);idx=rng.integers(len(rows),size=(draws,len(rows)))
    sums=rows.sum(0);boot=rows[idx].sum(1);out={}
    for gi,g in enumerate(GROUPS):
        n=sums[0,gi,0];entry={'n':int(n),'modes':{}}
        for mi,m in enumerate(MODES):
            v=sums[mi,gi];entry['modes'][m]={'world_mse_sum_dims':float(v[1]/max(n,1)),
                'sighting_mse_sum_dims':float(v[2]/max(n,1)),'neighbour_mse_sum_dims':float(v[3]/max(n,1))}
            if g!='unseen':
                den=v[3]-v[2];entry['modes'][m]['capture']=float((v[3]-v[1])/den)if den>0 else None
        diff=sums[1,gi,1]-sums[0,gi,1]
        bd=(boot[:,1,gi,1]-boot[:,0,gi,1])/boot[:,0,gi,0].clip(1)
        entry['reset_minus_intact_error']={'difference':float(diff/max(n,1)),
            'interval95':np.quantile(bd,[.025,.975]).tolist()}
        out[g]=entry
    return out


def paired(canvas,slot):
    rng=np.random.default_rng(20261006);idx=rng.integers(len(canvas),size=(2000,len(canvas)))
    assert np.array_equal(canvas[:,:, :,0],slot[:,:,:,0])
    out={}
    for gi,g in enumerate(GROUPS):
        diff=canvas[:,0,gi,1]-slot[:,0,gi,1];n=canvas[:,0,gi,0]
        b=diff[idx].sum(1)/n[idx].sum(1).clip(1)
        v={'canvas_minus_slot_mse':float(diff.sum()/max(n.sum(),1)),
           'interval95':np.quantile(b,[.025,.975]).tolist(),'n':int(n.sum())}
        if g!='unseen':
            den=canvas[:,0,gi,3]-canvas[:,0,gi,2]
            bb=-diff[idx].sum(1)/den[idx].sum(1)
            v['canvas_minus_slot_capture']=float(-diff.sum()/den.sum())
            v['capture_interval95']=np.quantile(bb,[.025,.975]).tolist()
        out[g]=v
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=(7,8),required=True)
    args=p.parse_args();torch.set_num_threads(3)
    pp=T.POOLS['raw']/'pool.pt';pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
    mainrows=(~pool['terminal']).nonzero().flatten()
    held=mainrows[torch.randperm(len(mainrows),generator=torch.Generator().manual_seed(1))[:2048]]
    paths={bb:T.OUT/f'corrt_raw_teacher_s{args.seed}_{bb}_u36000.pt'for bb in ('fcanvas','fmamba')}
    assert all(p.exists()for p in paths.values()),paths
    sources={str(p.resolve()):R.file_hash(p)for p in [Path(__file__),Path(A.__file__),Path(E.__file__),
        Path(T.__file__),Path(C.__file__),HERE/'h16_resume.py',HERE/'scroll.py',T.ROOT/'d4mj/mamba_recurrence.py']}
    spec={'scope':__doc__,'seed':args.seed,'sources':sources,'pool':R.file_hash(pp),
          'checkpoints':{k:R.file_hash(v)for k,v in paths.items()},'held':R.tensor_hash(held),
          'constructor_config':R.file_hash(T.S.CHECKPOINT),'runtime':{'torch':str(torch.__version__),
          'precision':'CPU FP32','threads':3},'batch':16,'draws':2000,'bootstrap_seed':20261006}
    name=f'e18_s{args.seed}_at36000__endpoint_cpu';store=R.Store(HERE/'evals/resume'/name,spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done,indent=2),flush=True);return
        records={}
        for bb,path in paths.items():
            world,_=E.T.load_world(path,torch.device('cpu'));parts=[];parity=[]
            for start in range(0,len(held),16):
                part=store.load(bb+'_'+str(start))
                if part is None:
                    ids=held[start:start+16];s=pool['tokens'][ids].float();a=F.pad(pool['actions'][ids],(0,1))
                    raw=torch.zeros(len(ids),2,len(GROUPS),4,dtype=torch.float64)
                    descriptors=[C.cells(x)for x in s]
                    with torch.no_grad():
                        for mi,mode in enumerate(MODES):
                            with cpu_mode(world,mode,parity if start==0 and mode=='intact'else None):pred=world(s,a)[0]
                            for j,(tt,ce,cls,sf,sc,nb)in enumerate(descriptors):
                                truth=s[j,tt,ce];e=(pred[j,tt-1,ce]-truth).square().sum(-1)
                                sight=(s[j,sf,sc]-truth).square().sum(-1);neigh=(s[j,tt,nb]-truth).square().sum(-1)
                                rec=cls==1;groups=(rec,rec&(sc==ce),rec&(sc!=ce),cls==0)
                                for gi,m in enumerate(groups):
                                    raw[j,mi,gi]=torch.tensor([int(m.sum()),float(e[m].sum()),float(sight[m].sum()),
                                                              float(neigh[m].sum())],dtype=torch.float64)
                    assert torch.equal(raw[:,0,:,[0,2,3]],raw[:,1,:,[0,2,3]])
                    part={'raw':raw,'parity_max':max(parity)if parity else None};store.save(bb+'_'+str(start),part,1)
                parts.append(part['raw'])
                if start%128==0:print(json.dumps({'name':name,'backbone':bb,'windows_done':start+len(held[start:start+16])}),flush=True)
            records[bb]=torch.cat(parts).numpy();del world
        result={'scope':__doc__,'contract':spec,'windows':len(held),
                'backbones':{k:stats(v)for k,v in records.items()},'paired':paired(records['fcanvas'],records['fmamba'])}
        store.save('rows',records,1);store.save('result',result,1)
        R.atomic_json(HERE/'evals'/(name+'.json'),result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
