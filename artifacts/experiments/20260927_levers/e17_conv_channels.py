"""CPU follow-up: identify which convolution branch supplies Long-Mamba recall.

Full200-root recurrence test: s7 needs older conv taps while s8 retains recall
without them. Remove older taps separately in x, B and C, preserving newest tap
and all other weights. Reuse intact per-root records after exact input/source/
checkpoint checks and one cold intact reproduction. Same fixed-clock contrast,
paired seed-cluster intervals, CPU FP32 and per-root atomic resume. Identifies
frozen component dependence, not how SGD learned it or a trained repair.
"""
import contextlib
import json
from pathlib import Path
import torch
import e17_recurrence_full as F

E,R,T,C,SD,S=F.E,F.R,F.T,F.C,F.SD,F.S
HERE=Path(__file__).resolve().parent
MODES=('intact','conv_x','conv_B','conv_C')


@contextlib.contextmanager
def lesion(world,mode):
    saved=[]
    for layer in world.layers:
        core=layer.mix.core
        lo,hi={'conv_x':(0,core.d_ssm),'conv_B':(core.d_ssm,core.d_ssm+core.d_state),
               'conv_C':(core.d_ssm+core.d_state,core.d_ssm+2*core.d_state)}[mode]
        weight=core.conv1d.weight;saved.append((weight,weight.detach().clone()))
        with torch.no_grad():weight[lo:hi,...,:-1].zero_()
    try:
        with E.lesion(world,'intact'):yield
    finally:
        with torch.no_grad():
            for weight,value in saved:weight.copy_(value)


def measure(world,s,a,alive,base):
    tt,ce,cls,sf,sc,nb=C.cells(s)
    use=(cls==1)&alive[tt]&(tt>=4)&((tt-sf)>=6)&((tt-sf)<=15)
    record={g:torch.zeros(4,5,dtype=torch.float64)for g in ('same','moved')}
    for g in record:record[g][0]=base[g][0]
    with torch.no_grad():
        for t in tt[use].unique().tolist():
            mask=use&(tt==t);x=s[max(0,t-15):t];aa=a[max(0,t-15):t]
            ab=x.clone();ab[:-5]=x[-5]
            frames=torch.stack([x,ab]);actions=torch.stack([aa,aa]);target=s[t,ce[mask]]
            es=(s[sf[mask],sc[mask]]-target).square().sum(-1)
            en=(s[t,nb[mask]]-target).square().sum(-1);same=sc[mask]==ce[mask]
            for mi,mode in enumerate(MODES[1:],1):
                with lesion(world,mode):pred=world(frames,actions)[0][:,-1]
                e1=(pred[0,ce[mask]]-target).square().sum(-1)
                e2=(pred[1,ce[mask]]-target).square().sum(-1)
                for g,m in [('same',same),('moved',~same)]:
                    record[g][mi]+=torch.tensor([int(m.sum()),float(e1[m].sum()),float(e2[m].sum()),
                                                float(es[m].sum()),float(en[m].sum())],dtype=torch.float64)
    for v in record.values():assert torch.equal(v[:,[0,3,4]],v[:1,[0,3,4]].expand(4,-1))
    return record


def main():
    torch.set_num_threads(3)
    meta,_,_=T.split();cache=torch.load(Path(str(T.CACHE).format('raw')),map_location='cpu',mmap=True,weights_only=False)
    roots=torch.randperm(len(meta['seed']),generator=torch.Generator().manual_seed(20261005))[:200]
    n=int((SD.CACHE/'DONE').read_text());assert n==len(meta['seed'])
    future=torch.from_numpy(SD.memmap(SD.CACHE/'fut5.f16',(n,5,16,81,192)))[roots,0]
    seqs=torch.cat([cache['ctx'][roots],future],1);acts=torch.cat([cache['ctx_a'][roots],cache['fut_a'][roots]],1)
    alive=torch.cat([torch.ones(len(roots),4,dtype=torch.bool),~meta['future_dead'][roots,0].cumsum(1).bool()],1)
    seeds=meta['seed'][roots];tensors={'frames':seqs,'actions':acts,'alive':alive,'roots':roots,'seeds':seeds}
    for seed in (7,8):
        name=f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000'
        path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/(name+'.pt')
        priorroot=HERE/'evals/resume'/(name+'__e17_recurrence_full_cpu')
        oldspec=json.loads((priorroot/'contract.json').read_text());old=R.Store(priorroot,oldspec)
        assert old.load('result')is not None and R.file_hash(path)==oldspec['checkpoint']
        for p,h in oldspec['sources'].items():assert R.file_hash(p)==h,p
        for k,v in tensors.items():assert R.tensor_hash(v)==oldspec['inputs'][k],k
        spec={'scope':__doc__,'base_contract':old.contract,'sources':{**oldspec['sources'],
              str(Path(__file__).resolve()):R.file_hash(__file__)},'inputs':oldspec['inputs'],
              'checkpoint':oldspec['checkpoint'],'constructor_config':R.file_hash(S.CHECKPOINT),
              'modes':list(MODES),'precision':'CPU FP32','threads':3,'draws':2000,'bootstrap_seed':20261006}
        assert spec['constructor_config']==oldspec['constructor_config']
        store=R.Store(HERE/'evals/resume'/(name+'__e17_conv_channels_cpu'),spec)
        with store.lock():
            result=store.load('result')
            if result is not None:print(json.dumps(result,indent=2),flush=True);continue
            world,_=T.load_world(path,torch.device('cpu'))
            control=store.load('reuse_control')
            if control is None:
                ri=2;base=old.load('root_'+str(ri));new=F.measure(world,seqs[ri].float(),acts[ri],alive[ri])
                delta=max(float((base[g]-new[g]).abs().max())for g in base);assert delta<=1e-6,delta
                control={'root':ri,'intact_max_absolute_record_difference':delta};store.save('reuse_control',control,1)
                print(json.dumps({'reuse_control':control}),flush=True)
            rows=[]
            for ri in range(len(roots)):
                record=store.load('root_'+str(ri))
                if record is None:
                    record=measure(world,seqs[ri].float(),acts[ri],alive[ri],old.load('root_'+str(ri)))
                    store.save('root_'+str(ri),record,1)
                rows.append(record)
                if ri%16==0:print(json.dumps({'name':name,'roots_done':ri+1}),flush=True)
            raw={g:torch.stack([x[g]for x in rows]).numpy()for g in ('same','moved')}
            original_modes=E.MODES
            try:E.MODES=MODES;stats=E.statistics(raw,seeds.numpy())
            finally:E.MODES=original_modes
            result={'scope':__doc__,'name':name,'contract':spec,'roots':200,'seed_clusters':len(seeds.unique()),
                    'groups':stats,'reuse_control':control}
            store.save('rows',{'raw':raw,'roots':roots,'seeds':seeds},1);store.save('result',result,1)
            R.atomic_json(HERE/'evals'/(name+'__e17_conv_channels_cpu.json'),result)
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
