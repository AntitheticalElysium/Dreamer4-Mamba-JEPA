"""CPU Long-Mamba mechanism lesion: recurrent SSM versus convolutional history.

First64 roots of the previously fixed200-root clock-control cohort, seeds7/8.
Keep all frames/actions/time rows, comparing original and older-visual-ablated
histories. Four frozen modes: intact, reset SSM before each step, mask older conv
taps, both. The current-step SSM update/newest conv tap and biases remain. This
is a component-use intervention, not retraining or an information ceiling.
CPU FP32 reference, per-root hash-bound resume and paired seed-cluster intervals.
"""
import argparse
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import e17_clock_control as K
import stochdiag as SD
import spatial as S
from d4mj.mamba_recurrence import FunctionalMamba2

T,C,R=K.T,K.C,K.R
HERE=Path(__file__).resolve().parent
MODES=('intact','reset_ssm','reset_conv','reset_both')


def instantaneous_ssm(self,x,b,c,dt,initial):
    core=self.core
    delta=F.softplus(dt.float()+core.dt_bias.float())
    # Reset the preceding SSM carry, preserve this step's input-dependent update.
    out=[];state=initial.float()
    for t in range(x.shape[1]):
        state=delta[:,t,:,None,None]*x[:,t,:,:,None].float()*b[:,t,None,None,:].float()
        out.append((state*c[:,t,None,None,:].float()).sum(-1)+core.D[None,:,None].float()*x[:,t].float())
    return torch.stack(out,1),state


@contextlib.contextmanager
def lesion(world,mode):
    old_forward=FunctionalMamba2.forward;old_ssm=FunctionalMamba2._reference_ssm
    saved=[]
    def cpu_forward(self,inputs,carry=None,*,backend=None):
        return self.scan(inputs,carry,backend='reference')
    FunctionalMamba2.forward=cpu_forward
    if mode in ('reset_ssm','reset_both'):FunctionalMamba2._reference_ssm=instantaneous_ssm
    if mode in ('reset_conv','reset_both'):
        for layer in world.layers:
            weight=layer.mix.core.conv1d.weight
            saved.append((weight,weight.detach().clone()))
            with torch.no_grad():weight[...,:-1].zero_()
    try:yield
    finally:
        with torch.no_grad():
            for weight,prior in saved:weight.copy_(prior)
        FunctionalMamba2.forward=old_forward;FunctionalMamba2._reference_ssm=old_ssm


def statistics(raw,seeds):
    groups=[np.flatnonzero(seeds==s)for s in np.unique(seeds)]
    rng=np.random.default_rng(20261006)
    idx=[np.concatenate([groups[i]for i in rng.integers(len(groups),size=len(groups))])for _ in range(2000)]
    ret={}
    for g in ('same','moved'):
        rows=raw[g];total=rows.sum(0)
        ret[g]={'n':int(total[0,0]),'modes':{},'history_gain_contrasts_vs_intact':{}}
        for i,mode in enumerate(MODES):
            v=rows[:,i]
            # [count, original error, visual-ablated error, old-copy error, inward-copy error]
            sums=v.sum(0);d=sums[4]-sums[3]
            if d<=0:ret[g]['modes'][mode]={'n':int(sums[0]),'invalid_denominator':float(d)};continue
            gain=(sums[2]-sums[1])/d
            boots=[]
            for j in idx:
                sj=v[j].sum(0);dj=sj[4]-sj[3]
                if dj>0:boots.append((sj[2]-sj[1])/dj)
            ret[g]['modes'][mode]={'original_capture':float((sums[4]-sums[1])/d),
                                  'ablated_capture':float((sums[4]-sums[2])/d),
                                  'history_gain':float(gain),'history_gain_interval95':np.quantile(boots,[.025,.975]).tolist()}
            if i:
                diff=[]
                for j in idx:
                    sj=rows[j].sum(0);dj=sj[0,4]-sj[0,3]
                    if dj>0:diff.append(((sj[i,2]-sj[i,1])-(sj[0,2]-sj[0,1]))/dj)
                intact=ret[g]['modes']['intact']['history_gain']
                ret[g]['history_gain_contrasts_vs_intact'][mode]={'difference':float(gain-intact),
                    'interval95':np.quantile(diff,[.025,.975]).tolist()}
    return ret


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',nargs='+',type=int,default=[7,8]);args=ap.parse_args()
    torch.set_num_threads(3);device=torch.device('cpu')
    meta,_,_=T.split();p=Path(str(T.CACHE).format('raw'));assert p.exists() and (SD.CACHE/'DONE').exists()
    cache=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    roots=torch.randperm(len(meta['seed']),generator=torch.Generator().manual_seed(20261005))[:64]
    n=int((SD.CACHE/'DONE').read_text());assert n==len(meta['seed'])
    future=torch.from_numpy(SD.memmap(SD.CACHE/'fut5.f16',(n,5,16,81,192)))[roots,0]
    seqs=torch.cat([cache['ctx'][roots],future],1)
    acts=torch.cat([cache['ctx_a'][roots],cache['fut_a'][roots]],1)
    alive=torch.cat([torch.ones(len(roots),4,dtype=torch.bool),~meta['future_dead'][roots,0].cumsum(1).bool()],1)
    seeds=meta['seed'][roots]
    sources={str(q.resolve()):R.file_hash(q)for q in (Path(__file__),Path(K.__file__),Path(C.__file__),Path(S.__file__),
              HERE/'tworld.py',HERE/'scroll.py',HERE/'h16_resume.py',T.ROOT/'d4mj/mamba_recurrence.py')}
    inputs={k:R.tensor_hash(v)for k,v in {'frames':seqs,'actions':acts,'alive':alive,'roots':roots,'seeds':seeds}.items()}
    for seed in args.seeds:
        path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt'
        world,st=T.load_world(path,device)
        spec={'scope':__doc__,'sources':sources,'inputs':inputs,'checkpoint':R.file_hash(path),
              'canonical_constructor_config':R.file_hash(S.CHECKPOINT),
              'precision':'CPU FP32','threads':3,'modes':MODES,'draws':2000,'bootstrap_seed':20261006}
        # Normalize tuples to JSON lists before sealing.
        spec=json.loads(json.dumps(spec))
        store=R.Store(HERE/'evals/resume'/(st['name']+'__e17_recurrence_cpu'),spec)
        with store.lock():
            result=store.load('result')
            if result is not None:print(json.dumps(result,indent=2),flush=True);continue
            rows=[]
            with torch.no_grad():
                for ri in range(len(roots)):
                    record=store.load('root_'+str(ri))
                    if record is None:
                        s=seqs[ri].float();a=acts[ri];tt,ce,cls,sf,sc,nb=C.cells(s)
                        use=(cls==1)&alive[ri,tt]&(tt>=4)&((tt-sf)>=6)&((tt-sf)<=15)
                        record={g:torch.zeros(len(MODES),5,dtype=torch.float64)for g in ('same','moved')}
                        for t in tt[use].unique().tolist():
                            mask=use&(tt==t);x=s[max(0,t-15):t];aa=a[max(0,t-15):t]
                            ab=x.clone();ab[:-5]=x[-5]
                            frames=torch.stack([x,ab]);actions=torch.stack([aa,aa])
                            target=s[t,ce[mask]]
                            es=(s[sf[mask],sc[mask]]-target).square().sum(-1)
                            en=(s[t,nb[mask]]-target).square().sum(-1);same=sc[mask]==ce[mask]
                            for mi,mode in enumerate(MODES):
                                with lesion(world,mode):pred=world(frames,actions)[0][:,-1]
                                e1=(pred[0,ce[mask]]-target).square().sum(-1)
                                e2=(pred[1,ce[mask]]-target).square().sum(-1)
                                for g,m in [('same',same),('moved',~same)]:
                                    record[g][mi]+=torch.tensor([int(m.sum()),float(e1[m].sum()),float(e2[m].sum()),
                                            float(es[m].sum()),float(en[m].sum())],dtype=torch.float64)
                        # Paired modes must retain the exact same cell and baseline ledger.
                        for v in record.values():assert torch.equal(v[:,[0,3,4]],v[:1,[0,3,4]].expand(len(MODES),-1))
                        store.save('root_'+str(ri),record,1)
                    rows.append(record)
                    if ri%8==0:print(json.dumps({'name':st['name'],'roots_done':ri+1}),flush=True)
            raw={g:torch.stack([x[g]for x in rows]).numpy()for g in ('same','moved')}
            result={'scope':__doc__,'name':st['name'],'contract':spec,'roots':len(roots),
                    'seed_clusters':len(seeds.unique()),'groups':statistics(raw,seeds.numpy())}
            store.save('rows',{'raw':raw,'roots':roots,'seeds':seeds},1)
            store.save('result',result,1)
            R.atomic_json(HERE/'evals'/(st['name']+'__e17_recurrence_cpu.json'),result)
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
