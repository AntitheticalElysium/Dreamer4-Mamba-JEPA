"""CPU component check: held SSM versus direct skip, and off-screen conv decay.

Same frozen 12k canvas and 16 windows as access/reroute. An old-query read also
reuses old C/x/z: quantify the SSM contribution before calling that memory useful.
Compare full, zero-SSM and zero-D reads; report pre-normalization energies and
output changes. Compare each convolution buffer to its last observed buffer.
These are component quantities, not semantic decoding or a deployable read.
No numerical model/trainer sources change. Per-batch hash-bound CPU resume.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import e18_access as A

T,C,R=A.T,A.C,A.R
HERE=Path(__file__).resolve().parent


def reference(mixer, inputs, keep, queries, traces):
    c=mixer.core
    z,xbc,dt=torch.split(c.in_proj(inputs),[c.d_ssm,c.d_ssm+2*c.d_state,c.nheads],-1)
    filtered=F.silu(F.conv1d(F.pad(xbc.transpose(1,2),(c.d_conv-1,0)),c.conv1d.weight,
                          c.conv1d.bias,groups=c.conv1d.groups)).transpose(1,2)
    x,bb,cc=torch.split(filtered,[c.d_ssm,c.d_state,c.d_state],-1)
    x=x.reshape(*x.shape[:2],c.nheads,c.headdim)
    delta=torch.where(keep[...,None],F.softplus(dt+c.dt_bias),0.)
    state=torch.zeros(len(inputs),c.nheads,c.headdim,c.d_state)
    last_c=torch.zeros_like(cc[:,0]);last_x=torch.zeros_like(x[:,0]);last_z=torch.zeros_like(z[:,0])
    last_buffer=torch.zeros(len(inputs),xbc.shape[-1],c.d_conv)
    last_step=torch.full((len(inputs),),-1,dtype=torch.long)
    buffer=torch.zeros_like(last_buffer);out=[]

    def project(hist,skip,zv):
        gated=(hist+skip).flatten(1)*F.silu(zv)
        normalized=gated*torch.rsqrt(gated.square().mean(-1,keepdim=True)+c.norm.eps)
        return c.out_proj(normalized*c.norm.weight)

    for t in range(inputs.shape[1]):
        buffer=torch.cat((buffer[:,:,1:],xbc[:,t,:,None]),-1)
        state=(delta[:,t]*(-c.A_log.exp())).exp()[:,:,None,None]*state
        state=state+delta[:,t,:,None,None]*x[:,t,:,:,None]*bb[:,t,None,None,:]
        hist=(state*cc[:,t,None,None,:]).sum(-1);skip=c.D[None,:,None]*x[:,t]
        out.append(project(hist,skip,z[:,t]))
        ids=queries[t]
        if len(ids):
            assert (~keep[ids,t]&(last_step[ids]>=0)).all()
            hp=(state[ids]*last_c[ids,None,None,:]).sum(-1)
            sp=c.D[None,:,None]*last_x[ids]
            full=project(hp,sp,last_z[ids]);no_ssm=project(torch.zeros_like(hp),sp,last_z[ids])
            no_skip=project(hp,torch.zeros_like(sp),last_z[ids])
            he=hp.square().sum((1,2));se=sp.square().sum((1,2))
            traces.append({'gap':t-last_step[ids], 'history_energy':he,'skip_energy':se,
                           'history_energy_share':he/(he+se).clamp_min(1e-20),
                           'zero_ssm_output_relative_l2':(full-no_ssm).square().sum(1)/full.square().sum(1).clamp_min(1e-20),
                           'zero_skip_output_relative_l2':(full-no_skip).square().sum(1)/full.square().sum(1).clamp_min(1e-20),
                           'zero_ssm_output_cosine':F.cosine_similarity(full,no_ssm),
                           'conv_relative_l2':(buffer[ids]-last_buffer[ids]).square().sum((1,2))/last_buffer[ids].square().sum((1,2)).clamp_min(1e-20),
                           'conv_nonzero_fraction':(buffer[ids].abs()>1e-12).float().mean((1,2)),
                           'd_conv':c.d_conv})
        last_c=torch.where(keep[:,t,None],cc[:,t],last_c)
        last_x=torch.where(keep[:,t,None,None],x[:,t],last_x)
        last_z=torch.where(keep[:,t,None],z[:,t],last_z)
        last_buffer=torch.where(keep[:,t,None,None],buffer,last_buffer)
        last_step=torch.where(keep[:,t],t,last_step)
    return torch.stack(out,1)


def main():
    torch.set_num_threads(3)
    oldroot=HERE/'evals/resume/e18_access_s7_at12000'
    old=R.Store(oldroot,json.loads((oldroot/'contract.json').read_text()))
    prior=old.load('result');records=old.load('addresses');selected=prior['CPU_reference_windows']
    path=T.OUT/'corrt_raw_teacher_s7_fcanvas_u36000_at12000.pt';pp=T.POOLS['raw']/'pool.pt'
    spec={'scope':__doc__,'sources':{str(p):R.file_hash(p)for p in (Path(__file__).resolve(),Path(A.__file__).resolve())},
          'model_sources':prior['contract']['sources'],'inputs':{str(p):R.file_hash(p)for p in (path,pp)},
          'windows':selected,'precision':'CPU FP32','threads':3,'access_contract':old.contract}
    assert spec['inputs'][str(path)]==prior['contract']['checkpoint']
    assert spec['inputs'][str(pp)]==prior['contract']['pool']
    for p,h in prior['contract']['sources'].items():assert R.file_hash(p)==h,p
    store=R.Store(HERE/'evals/resume/e18_carry_s7_at12000',spec)
    with store.lock():
        result=store.load('result')
        if result is not None:print(json.dumps(result,indent=2));return
        pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
        world=T.TWorld('corrt',backbone='fcanvas').eval()
        world.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['world'])
        parts=[]
        for start in range(0,len(selected),4):
            part=store.load(f'batch_{start}')
            if part is None:
                ids=selected[start:start+4];s=pool['tokens'][ids].float();a=F.pad(pool['actions'][ids],(0,1))
                b,t=s.shape[:2];pad=t-1;cols=9+2*pad
                sh=F.pad(A.estimate(s[:,:-1],s[:,1:]),(1,0));off=torch.tensor(A.SHIFTS)[sh].cumsum(1)
                rr=torch.arange(7)[:,None]+off[...,0,None,None]+pad
                cc=torch.arange(9)[None]+off[...,1,None,None]+pad
                raw=(rr*cols+cc).flatten(2)
                present=torch.zeros(b,(7+2*pad)*cols,dtype=torch.bool).scatter(1,raw.flatten(1),True)
                packed=(present.long().cumsum(1)-1).gather(1,raw.flatten(1)).view(b,t,63)
                streams=int(present.sum(1).max())+19;queries=[[]for _ in range(t)]
                for r in records:
                    if r['row']in ids:
                        bi=ids.index(r['row']);queries[r['target']-1].append(bi*streams+int(packed[bi,r['sighting'],r['oldcell']]))
                queries=[torch.tensor(q,dtype=torch.long)for q in queries]
                original=T.masked_scan;traces=[];errors=[];layer=[0]
                def scan(mixer,inputs,keep):
                    local=[];out=reference(mixer,inputs,keep,queries,local)
                    check=A.reference(mixer,inputs,keep,None,[])
                    err=float((out-check).abs().max());errors.append(err);assert err<=1e-6,err
                    for p in local:p['layer']=layer[0]
                    layer[0]+=1;traces.extend(local)
                    return out
                try:
                    T.masked_scan=scan
                    with torch.no_grad():world(s,a)
                finally:T.masked_scan=original
                part={'traces':traces,'rows':ids,'parity_max':max(errors)}
                store.save(f'batch_{start}',part,1)
            parts.append(part);print(json.dumps({'windows_done':start+len(part['rows'])}),flush=True)
        traces=[p for part in parts for p in part['traces']]
        metrics=[k for k,v in traces[0].items()if isinstance(v,torch.Tensor)]
        def summarize(subset):
            ret={}
            for k in metrics:
                values=torch.cat([p[k].float()for p in subset])
                ret[k]={'n':len(values),'mean':float(values.mean()),'median':float(values.median()),
                        'min':float(values.min()),'max':float(values.max())}
            return ret
        result={'scope':__doc__,'contract':spec,'windows':len(selected),
                'layer_cell_queries':sum(len(p['gap'])for p in traces),
                'reference_parity_max':max(p['parity_max']for p in parts),
                'all_layers':summarize(traces),
                'by_layer':{str(i):summarize([p for p in traces if p['layer']==i])for i in range(6)},
                'conv_kernel':traces[0]['d_conv'],
                'interpretation':'SSM removal measured conditional on stored old query; old C/x/z are additional memory. No semantic ceiling claim.'}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e18_carry_s7_at12000.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
