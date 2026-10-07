"""CPU-only canvas address/readout mechanism diagnosis; no new world training.

Count direct temporal-stream addresses on the exact 2048 held pool windows. A
next-frame entering cell is off-screen at prediction time; does fcanvas gather
its stored stream, or the stream for the current cell at the same screen slot?
Then run the frozen 36k canvas on the first16 held windows containing reentry,
with an explicit FP32 CPU implementation of its masked SSM equations. Measure
off-screen state retention, zero-input output gating, and reading the same held
state with its last observed C/z/query. Check all-kept parity against the actual
FunctionalMamba2 reference backend. This is an address/component diagnosis, not
a full-model information ceiling: space attention and action/HUD streams can
provide indirect paths. No true next-frame feature enters a prediction; future
coordinates are used only to select which stored stream the diagnostic inspects.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/experiments/20260927_levers')
sys.path[:0]=[str(HERE),str(HERE.parent/'20260926_diagnosis'),str(HERE.parent/'20260921_readout_ladder')]
import tworld as T
import check_recall as C
import h16_resume as R
from scroll import estimate,SHIFTS


def address_ledger(pool,held):
    records=[]
    for i,row in enumerate(held.tolist()):
        x=pool['tokens'][row].float();sh=estimate(x[:-1],x[1:])
        off=torch.cat([torch.zeros(1,2,dtype=torch.long),torch.tensor(SHIFTS)[sh].cumsum(0)])
        target,cell,cls,sighting,oldcell,_=C.cells(x)
        for t,c,s,sc in zip(target[cls==1],cell[cls==1],sighting[cls==1],oldcell[cls==1]):
            intended=torch.tensor([c//9,c%9])+off[t]
            actual=torch.tensor([c//9,c%9])+off[t-1]
            last=torch.tensor([sc//9,sc%9])+off[s]
            assert torch.equal(intended,last)
            local=intended-off[t-1]
            outside=not(0<=local[0]<7 and 0<=local[1]<9)
            records.append({'row':row,'target':int(t),'cell':int(c),'sighting':int(s),'oldcell':int(sc),
                            'same_slot':bool(c==sc),'actual_stream_matches_target':bool(torch.equal(actual,intended)),
                            'target_outside_current_view':outside})
        if (i+1)%256==0:print(json.dumps({'stage':'CPU_address_ledger','windows':i+1}),flush=True)
    return records


def reference(mixer,inputs,keep,queries,collector,check=False):
    c=mixer.core
    z,xbc,dt=torch.split(c.in_proj(inputs),[c.d_ssm,c.d_ssm+2*c.d_state,c.nheads],-1)
    filt=F.silu(F.conv1d(F.pad(xbc.transpose(1,2),(c.d_conv-1,0)),c.conv1d.weight,c.conv1d.bias,
                       groups=c.conv1d.groups)).transpose(1,2)
    x,b,cc=torch.split(filt,[c.d_ssm,c.d_state,c.d_state],-1)
    x=x.reshape(*x.shape[:2],c.nheads,c.headdim)
    delta=F.softplus(dt+c.dt_bias)
    delta=torch.where(keep[...,None],delta,0.)
    state=torch.zeros(len(inputs),c.nheads,c.headdim,c.d_state)
    out=[];last_c=torch.zeros_like(cc[:,0]);last_x=torch.zeros_like(x[:,0]);last_z=torch.zeros_like(z[:,0])
    last_output=torch.zeros(len(inputs),c.d_model);seen=torch.zeros(len(inputs),dtype=torch.bool)
    def read(st,cv,xv,zv):
        y=((st*cv[:,None,None,:]).sum(-1)+c.D[None,:,None]*xv).flatten(1)
        gated=y*F.silu(zv)
        norm=gated*torch.rsqrt(gated.square().mean(-1,keepdim=True)+c.norm.eps)
        return c.out_proj(norm*c.norm.weight)
    for t in range(inputs.shape[1]):
        previous=state
        decay=(delta[:,t]*(-c.A_log.exp())).exp()[:,:,None,None]
        state=decay*state+delta[:,t,:,None,None]*x[:,t,:,:,None]*b[:,t,None,None,:]
        y=read(state,cc[:,t],x[:,t],z[:,t]);out.append(y)
        if queries is not None:
            ids=queries[t]
            assert (~keep[ids,t]&seen[ids]).all()
            if len(ids):
                held=read(state[ids],last_c[ids],last_x[ids],last_z[ids])
                collector.append({'n':len(ids),'state_rms':state[ids].square().mean((1,2,3)).sqrt(),
                                  'hold_max':float((state[ids]-previous[ids]).abs().max()),
                                  'offscreen_output_max':float(y[ids].abs().max()),
                                  'read_with_last_query_rms':held.square().mean(1).sqrt(),
                                  'last_query_vs_last_output_max':float((held-last_output[ids]).abs().max())})
        seen|=keep[:,t]
        last_c=torch.where(keep[:,t,None],cc[:,t],last_c)
        last_x=torch.where(keep[:,t,None,None],x[:,t],last_x)
        last_z=torch.where(keep[:,t,None],z[:,t],last_z)
        last_output=torch.where(keep[:,t,None],y,last_output)
    return torch.stack(out,1)


def main():
    torch.set_num_threads(3);torch.manual_seed(20261006)
    path=T.OUT/'corrt_raw_teacher_s7_fcanvas_u36000.pt'
    pp=T.POOLS['raw']/'pool.pt'
    pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
    mainrows=(~pool['terminal']).nonzero().flatten()
    held=mainrows[torch.randperm(len(mainrows),generator=torch.Generator().manual_seed(1))[:2048]]
    sources={str(p):R.file_hash(p)for p in [Path(__file__).resolve(),Path(T.__file__).resolve(),Path(C.__file__).resolve(),
                                        HERE/'scroll.py',HERE/'h16_resume.py',T.ROOT/'d4mj/mamba_recurrence.py']}
    spec={'scope':__doc__,'checkpoint':R.file_hash(path),'pool':R.file_hash(pp),'held':R.tensor_hash(held),
          'sources':sources,'runtime':{'torch':str(torch.__version__),'precision':'CPU FP32','threads':3},'windows':16}
    store=R.Store(HERE/'evals/resume/e18_access_s7_at36000',spec)
    with store.lock():
        finished=store.load('result')
        if finished is not None:
            R.atomic_json(HERE/'evals/e18_access_s7_at36000.json',finished);print(json.dumps(finished),flush=True);return
        records=store.load('addresses')
        if records is None:
            records=address_ledger(pool,held);store.save('addresses',records,1)
        selected=list(dict.fromkeys(r['row']for r in records))[:16]
        w=T.TWorld('corrt',backbone='fcanvas').eval()
        w.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['world'])
        samples=[];parity=[]
        for start in range(0,len(selected),4):
            part=store.load(f'batch_{start}')
            if part is None:
                ids=selected[start:start+4];s=pool['tokens'][ids].float();a=F.pad(pool['actions'][ids],(0,1))
                sh=F.pad(estimate(s[:,:-1],s[:,1:]),(1,0));b,t=s.shape[:2];pad=t-1;cols=9+2*pad
                off=torch.tensor(SHIFTS)[sh].cumsum(1)
                r=torch.arange(7)[:,None]+off[...,0,None,None]+pad
                c=torch.arange(9)[None]+off[...,1,None,None]+pad
                raw=(r*cols+c).flatten(2)
                present=torch.zeros(b,(7+2*pad)*cols,dtype=torch.bool).scatter(1,raw.flatten(1),True)
                packed=(present.long().cumsum(1)-1).gather(1,raw.flatten(1)).view(b,t,63)
                cells=int(present.sum(1).max());streams=cells+19;queries=[[]for _ in range(t)]
                for rec in records:
                    if rec['row']in ids:
                        bi=ids.index(rec['row']);qid=int(packed[bi,rec['sighting'],rec['oldcell']])
                        queries[rec['target']-1].append(bi*streams+qid)
                queries=[torch.tensor(q,dtype=torch.long)for q in queries]
                traces=[];errors=[];original=T.masked_scan
                def scan(mixer,inputs,keep):
                    with torch.no_grad():
                        all_keep=torch.ones_like(keep)
                        control=reference(mixer,inputs,all_keep,None,[],True)
                        actual=mixer.scan(inputs,backend='reference')[0]
                        errors.append(float((control-actual).abs().max()))
                        assert errors[-1]<=1e-5,errors[-1]
                        return reference(mixer,inputs,keep,queries,traces)
                try:
                    T.masked_scan=scan
                    with torch.no_grad():w(s,a)
                finally:T.masked_scan=original
                part={'rows':ids,'traces':traces,'parity_max':max(errors)}
                store.save(f'batch_{start}',part,1)
            samples.extend(part['traces']);parity.append(part['parity_max'])
            print(json.dumps({'stage':'CPU_masked_reference','windows_done':start+len(part['rows']),
                              'all_kept_parity_max':part['parity_max']}),flush=True)
        rms=torch.cat([p['state_rms']for p in samples]);read=torch.cat([p['read_with_last_query_rms']for p in samples])
        result={'scope':__doc__,'contract':spec,'address_windows':len(held),'recallable_cells':len(records),
                'same_slot':sum(r['same_slot']for r in records),'moved_slot':sum(not r['same_slot']for r in records),
                'direct_current_stream_matches':sum(r['actual_stream_matches_target']for r in records),
                'targets_outside_current_view':sum(r['target_outside_current_view']for r in records),
                'CPU_reference_windows':selected,'layer_cell_queries':len(rms),'nonzero_stored_state':int((rms>1e-9).sum()),
                'state_rms_mean':float(rms.mean()),'nonzero_last_query_read':int((read>1e-9).sum()),
                'last_query_read_rms_mean':float(read.mean()),'all_kept_reference_max_error':max(parity),
                'held_state_max_error':max(p['hold_max']for p in samples),
                'unobserved_output_max':max(p['offscreen_output_max']for p in samples),
                'restored_last_query_max_error':max(p['last_query_vs_last_output_max']for p in samples)}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e18_access_s7_at36000.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
