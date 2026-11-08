"""Frozen semantic gaze test. Protocol: PLAN.md. No fitting or world updates."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/saccade-fine-causal-mpl')
import argparse, copy, hashlib, importlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CK=ROOT/'artifacts/experiments/20261102_p1f16/run/ckpt_10.pt'
SHA='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'
DEV=torch.device('cuda')

def write(name,x): (HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def labels(): return dict(np.load(S.LABELS/'s0_dev_labels.npz'))
def load():
    assert TR.file_sha256(CK)==SHA
    ck=torch.load(CK,map_location='cpu',weights_only=False)
    m=M.M0(ck['cfg']).to(DEV);m.load_state_dict(ck['model']);m.eval().requires_grad_(False)
    return m,ck['cfg']
def gazes(cell):
    r,c=divmod(int(cell),9);f=(max(1,min(7,r)),max(1,min(7,c)))
    off=(f[0],1 if c>=4 else 7)
    assert cell not in M.window(torch.tensor([off]))[0].tolist()
    return torch.tensor([f,off],device=DEV)
def expandcarry(c,n):
    return tuple(M.MambaCarry(x.conv.expand(n,*x.conv.shape[1:]).contiguous(),
        x.ssm.expand(n,*x.ssm.shape[1:]).contiguous()) for x in c)
def expandcache(c,n):
    return dict(t=c['t'],kv=[(a.expand(n,*a.shape[1:]).contiguous(),b.expand(n,*b.shape[1:]).contiguous()) for a,b in c['kv']])
def slices(P,s): return M.Percept(P.p[s],P.g[s],P.v[s],P.fix[s])

@torch.no_grad()
def survey():
    m,cfg=load();L=labels();so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    coarse=S.load_reader(CK.parent/'s1/results.reader.pt',CK,DEV)
    rr=torch.load(ROOT/'artifacts/experiments/20261103_p1f16_review/p1f16/readers.pt',map_location=DEV,weights_only=False)
    fine=[]
    for seed in [0,1]:
        x=rr[f'g_{seed}'];r=S.MLP(x['dim'],{'kind':5}).to(DEV);r.load_state_dict(x['state']);r.eval();fine.append(r)
    src=[]
    for cohort,kind,ky in [('near_arrows','arrows',4),('near_zombies','zombies',1),('distant_arrows','arrows',4)]:
        count=0
        for e,T in enumerate(L['ep_T'][:400]):
            a,b=int(so[e]),int(so[e+1]);p=L[f's_{kind}_p'][a:b]-L['s_pos'][a:b,None]
            visible=L[f's_{kind}_m'][a:b]&(abs(p[...,0])<=3)&(abs(p[...,1])<=4)
            for t in range(8,int(T)):
                if L['s_light'][a+t]>=.5 or L['s_sleep'][a+t]:continue
                if cohort!='distant_arrows' and L['s_body'][a+t+1,0]-L['s_body'][a+t,0]>-2:continue
                eligible=visible[t]&(abs(p[t]).sum(-1)<=2)
                if cohort=='distant_arrows':
                    eligible=visible[t]&(abs(p[t]).sum(-1)>=3)&visible[t+1]&(abs(p[t+1]).sum(-1)<abs(p[t]).sum(-1))
                for j in np.flatnonzero(eligible):
                    if count>=256:break
                    r=dict(e=e,t=t,j=int(j),cohort=cohort,kind=kind,ky=ky,shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e]),
                        cell=int((p[t,j,0]+3)*9+p[t,j,1]+4),light=float(L['s_light'][a+t]),
                        health=float(L['s_body'][a+t,0]),next_health=float(L['s_body'][a+t+1,0]),
                        action=int(L['t_action'][ao[e]+t]),first_visible=bool(t==0 or not visible[t-1,j]),
                        target_cell=int((p[t+1,j,0]+3)*9+p[t+1,j,1]+4))
                    raw=D.RP.episode_fields(r['shard'],r['slot'])['observations'][t].to(DEV)
                    fx=gazes(r['cell'])[:1]
                    with TR.autocast(DEV,cfg):P=m.perception(raw[None],fx)
                    slot=int((M.window(fx)[0]==r['cell']).nonzero()[0])
                    pc=int(coarse(torch.cat([P.p[0,r['cell']].float(),P.v[0].float()])[None])['kind'].argmax(-1)[0])
                    gf=torch.cat([P.g[0,slot].float(),P.v[0].float()])[None]
                    gc=[int(rd(gf)['kind'].argmax(-1)[0]) for rd in fine]
                    with TR.autocast(DEV,cfg):future=m.perception(D.RP.episode_fields(r['shard'],r['slot'])['observations'][t+1].to(DEV)[None],fx)
                    future_kind=int(coarse(torch.cat([future.p[0,r['target_cell']].float(),future.v[0].float()])[None])['kind'].argmax(-1)[0])
                    r.update(coarse_kind=pc,fine_kind=gc,fine_qualified=all(k==ky for k in gc),coarse_missed=pc!=ky,target_reader_kind=future_kind)
                    src.append(r);count+=1
            if count>=256:break
    selected=[];used={}
    for r in src:
        k=(r['cohort'],r['coarse_missed'])
        if not r['fine_qualified'] or r['e'] in used.get(k,set()) or len(used.get(k,set()))>=12:continue
        if r['cohort']=='distant_arrows' and r['target_reader_kind']!=4:continue
        selected.append(r);used.setdefault(k,set()).add(r['e'])
    write('survey_expanded.json',dict(rows=src,selected=selected,checkpoint_sha256=SHA,
        qualification='Two previously fitted fine-kind readers correct; coarse kind reader split. Not proof of coarse information absence.'))
    print('survey',len(src),'selected',len(selected),{str(k):len(v) for k,v in used.items()},flush=True)

def causal_sources():
    import jax.numpy as jnp
    L=labels();so=np.r_[0,np.cumsum(L['ep_T']+1)]
    _,_,_,step,frame=D.RP.env_and_render();out=[];arrays=[];rejected=[]
    for r in json.loads((HERE/'survey_expanded.json').read_text())['selected']:
        st=D.RP.advance_to(r['shard'],r['slot'],r['t']);raw=D.RP.episode_fields(r['shard'],r['slot'])['observations']
        assert np.array_equal(np.asarray(frame(st)),raw[r['t']].numpy())
        ent=getattr(st,r['kind']);oldpos=ent.position[r['j']]
        ent=ent.replace(mask=ent.mask.at[r['j']].set(False),health=ent.health.at[r['j']].set(0))
        # Projectiles do not own entries in mob_map. Clear only a removed mob's
        # entry, otherwise an arrow over another creature changes TWO causes.
        mm=st.mob_map if r['kind']=='arrows' else st.mob_map.at[oldpos[0],oldpos[1]].set(False)
        absent=st.replace(**{r['kind']:ent},mob_map=mm)
        key=D.RP._slot_keys(r['shard'],r['slot'])[1][r['t']]
        seq=[];stats=[]
        for s in [st,absent]:
            _,nxt,reward,term,_=step(key,s,jnp.asarray(r['action'],jnp.int32))
            seq.append(np.asarray(frame(nxt)));stats.append(dict(health=float(nxt.player_health),reward=float(reward),terminated=bool(term)))
        assert np.array_equal(seq[0],raw[r['t']+1].numpy()),('actual successor',r)
        # Determine the causal future occupancy directly from simulator records.
        npos=np.asarray(nxt.arrows.position)-np.asarray(nxt.player_position)[None]
        target=np.array(divmod(r['target_cell'],9))-np.array([3,4])
        absent_at_target=bool((np.asarray(nxt.arrows.mask)&(npos==target).all(-1)).any())
        z=dict(**r,causal_health_difference=stats[1]['health']-stats[0]['health'],next=stats,absent_arrow_at_target=absent_at_target)
        if (r['cohort']=='distant_arrows' and absent_at_target) or (r['cohort']!='distant_arrows' and z['causal_health_difference']<1.99):
            rejected.append(z);continue
        out.append(z);arrays.append(dict(current=np.stack([raw[r['t']].numpy(),np.asarray(frame(absent))]),future=np.stack(seq)))
        print('causal',len(out),r['kind'],r['coarse_missed'],r['e'],r['t'],z['causal_health_difference'],flush=True)
    torch.save(dict(rows=out,arrays=arrays),HERE/'natural_pairs.pt')
    write('causal_qualification.json',dict(rows=out,rejected=rejected))

@torch.no_grad()
def prefix(m,cfg,z,e,t):
    ix=torch.where(z['episode']==e)[0][:t]
    with TR.autocast(DEV,cfg):
        P=M.Percept(*(z[k][ix].to(DEV).float()[None] for k in ['p','g','v']),z['fix'][ix].to(DEV)[None])
        tok=m.belief.tokens(m.belief.percept_embed(P),z['reward'][ix].to(DEV)[None],
            z['terminated'][ix].to(DEV)[None],z['action'][ix].to(DEV)[None])
        carry=m.belief.initial(1,DEV)
        for a in range(0,2*t,128):_,carry=m.belief(tok[:,a:a+128],carry)
    return carry

def delta(a,b):
    d=a.double()-b.double();return dict(equal=bool(torch.equal(a,b)),rms=float(d.square().mean().sqrt()),max=float(d.abs().max()))

@torch.no_grad()
def read_health(parts,rd,h0):
    # Existing periphery reader, fitted and qualified BEFORE this test.
    ph=parts[0][:,0,63].float()
    keys=sorted(rd['means']);means=torch.stack([rd['means'][k] for k in keys]).to(DEV).float()
    dist=torch.cdist(ph,means);ii=dist.argmin(-1);values=torch.tensor(keys,device=DEV)[ii]
    support=torch.zeros(len(ph),device=DEV,dtype=torch.bool)
    for v in keys:
        a=values==v
        if (0,int(v)) in rd['support']:
            mu,rad=rd['support'][(0,int(v))];support[a]=(ph[a]-mu.to(DEV).float()).norm(dim=-1)<=rad
    return dict(health=values.cpu(),supported=support.cpu(),drop=(values<h0).cpu())

@torch.no_grad()
def consequences(m,parts,bp,fx):
    P=M.Percept(*(x[:,0] for x in parts),fx[:,0]);emb=m.belief.percept_embed(P)
    h=m.outcome.features(bp,emb);p=m.outcome.bin_logp(h).exp();v=m.outcome.bin_value
    return dict(negative=(p*(v<-.05)).sum(-1).cpu(),reward=(p*v).sum(-1).cpu())

@torch.no_grad()
def run(smoke=False):
    m,cfg=load()
    L=labels();so=np.r_[0,np.cumsum(L['ep_T']+1)]
    cachedata=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',mmap=True,weights_only=False)
    assert cachedata['checkpoint_sha256']==SHA;z=cachedata['current']
    instruments=torch.load(ROOT/'artifacts/experiments/20261102_p1f16/readings/eval_m05/p1f16_1500_readers.pt',map_location='cpu',weights_only=False)
    assert instruments['checkpoint_sha256']==SHA
    rd=instruments['readers']['p']['health']
    assert rd['qualified']
    kind_reader=S.load_reader(CK.parent/'s1/results.reader.pt',CK,DEV)
    def read_arrow(parts,cell):
        logits=kind_reader(torch.cat([parts[0][:,0,cell].float(),parts[2][:,0].float()],-1))['kind'].float()
        return dict(arrow=(logits.argmax(-1)==4).cpu(),arrow_score=logits.softmax(-1)[:,4].cpu())
    pairs=[]
    for kind,file in [('heading','source_pairs.pt'),('natural','natural_pairs.pt')]:
        q=torch.load(HERE/file,map_location='cpu',weights_only=False)
        for r,a in zip(q['rows'],q['arrays']):pairs.append((kind,r,a))
    if smoke:pairs=[pairs[0]]
    results=[];stage=[]
    for case,(kind,r,aa) in enumerate(pairs):
        e,t=r['e'],r['t'];off=int(torch.where((z['episode']==e)&(z['t']==t))[0]);cell=r.get('cell',(r['delta'][0]+3)*9+r['delta'][1]+4) if kind=='heading' else r['cell']
        r=dict(r,health=float(L['s_body'][so[e]+t,0]))
        fx=gazes(cell);current=torch.as_tensor(aa['current'],device=DEV);future=torch.as_tensor(aa['future'],device=DEV)
        # Ordering: [fact A focus, fact A control, fact B focus, fact B control].
        fs=fx.repeat(2,1)
        carry=prefix(m,cfg,z,e,t)
        action=0 if kind=='heading' else r['action']
        with TR.autocast(DEV,cfg):
            P=m.perception(current.repeat_interleave(2,0),fs)
            assert torch.equal(P.p[0],P.p[1]) and torch.equal(P.v[0],P.v[1])
            joint=m.belief.spatial_fusion(P,M.window(fs));emb=m.belief.percept_embed(P)
            tok=m.belief.tokens(emb[:,None],z['reward'][off].to(DEV).expand(4,1),z['terminated'][off].to(DEV).expand(4,1),torch.full((4,1),action,device=DEV))
            yy,cc=m.belief(tok,expandcarry(carry,4));bp=yy[:,1].float();memory=m.belief.memory(bp)
            raw_original=D.RP.episode_fields(r['shard'],r['slot'])['observations'][t].to(DEV)
            original=m.perception(raw_original[None],z['fix'][off:off+1].to(DEV))
            original_tok=m.belief.tokens(m.belief.percept_embed(original)[:,None],z['reward'][off:off+1].to(DEV)[None],
                z['terminated'][off:off+1].to(DEV)[None],z['action'][off:off+1].to(DEV)[None])
            original_y,_=m.belief(original_tok,carry)
            prefix_rms=float((original_y[0,0].float()-z['corrected'][off].to(DEV).float()).square().mean().sqrt())
            assert prefix_rms<.03,('full prefix check',prefix_rms,e,t)
            # A fixed-gaze fine-only comparison is algebraic; the generated runs
            # below ALWAYS receive mutually consistent full native percepts.
            fineP=M.Percept(P.p[:1].expand(2,-1,-1),P.g[[0,2]],P.v[:1].expand(2,-1),fs[:1].expand(2,-1))
            je=m.belief.spatial_fusion(fineP,M.window(fineP.fix));ee=m.belief.percept_embed(fineP)
            fine_tokens=m.belief.tokens(ee[:,None],z['reward'][off].to(DEV).expand(2,1),
                z['terminated'][off].to(DEV).expand(2,1),torch.full((2,1),action,device=DEV))
            fine_y,fine_carry=m.belief(fine_tokens,expandcarry(carry,2))
            coords=M.window(fs[:1])[0];np_=81*16
            inds=torch.cat([torch.arange(np_+int(c)*64,np_+(int(c)+1)*64,device=DEV) for c in coords])
            exact=F.linear((je[0]-je[1]).flatten().double(),m.belief.percept_in.weight[:,inds].double())
        s=dict(case=case,kind=kind,meta=r,prefix_rms=prefix_rms,raw_fine=delta(P.g[0],P.g[2]),joint_fine=delta(je[0],je[1]),
            mixed_fine_only=delta(ee[0],ee[1]),fp64_fine_projection_rms=float(exact.square().mean().sqrt()),
            full_embedding=delta(emb[0],emb[2]),corrected=delta(yy[0,0],yy[2,0]),predicted=delta(bp[0],bp[2]),
            memory=delta(memory[0],memory[2]),carry_conv=[delta(c.conv[0],c.conv[2]) for c in cc],
            carry_ssm=[delta(c.ssm[0],c.ssm[2]) for c in cc],
            fine_only_corrected=delta(fine_y[0,0],fine_y[1,0]),fine_only_predicted=delta(fine_y[0,1],fine_y[1,1]),
            fine_only_carry_conv=[delta(c.conv[0],c.conv[1]) for c in fine_carry],
            fine_only_carry_ssm=[delta(c.ssm[0],c.ssm[1]) for c in fine_carry])
        if not smoke:
            path=HERE/'activations';path.mkdir(exist_ok=True)
            torch.save(dict(meta=r,p=P.p.cpu(),g=P.g.cpu(),v=P.v.cpu(),fix=fs.cpu(),joint=joint.cpu(),
                embedding=emb.cpu(),emission=yy.cpu(),memory=memory.cpu(),
                carry=[dict(conv=c.conv.cpu(),ssm=c.ssm.cpu()) for c in cc],
                fine_only_embedding=ee.cpu(),fine_only_emission=fine_y.cpu(),
                fine_only_carry=[dict(conv=c.conv.cpu(),ssm=c.ssm.cpu()) for c in fine_carry]),path/f'case_{case:03d}.pt')
        # Build all historical physical/fine keys from actual observations only.
        with TR.autocast(DEV,cfg):
            cache=m.denoiser.new_cache();gen=torch.Generator(device=DEV).manual_seed(614100+case)
            for f in range(off-7,off):
                cache=M.context_pass(m.denoiser,tuple(z[k][f:f+1].to(DEV).float()[None] for k in ['p','g','v']),
                    z['prev_action'][f:f+1].to(DEV)[None],m.belief.memory(z['prior'][f:f+1].to(DEV).float())[None],
                    z['fix'][f:f+1].to(DEV)[None],cache,cfg,gen)
            cache=expandcache(cache,4)
            # Identical context noise per condition; do not consume an
            # independent draw just because gaze or counterfactual differs.
            sc=cfg['edm']['inference_context_sigma'];sd=cfg['edm']['sigma_data']
            parts=tuple(x[:,None] for x in P.parts())
            xs=tuple(x+sc*torch.randn((1,*x.shape[1:]),device=DEV,generator=gen) for x in parts)
            _,_,ci,cn=M.precondition(torch.full((4,1),sc,device=DEV),sd)
            _,new=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in xs),cn,z['prev_action'][off].to(DEV).expand(4,1),
                m.belief.memory(z['prior'][off].to(DEV).float()).expand(4,1,-1,-1),fs[:,None],cache)
            physical=86 #81 map/HUD tokens + light +4 memory
            # Token axis is -2 for cached temporal keys? Check actual cache
            # layout by comparing all physical-index temporal streams.
            parity=[]
            for k,v in new:
                kd=delta(k[0,m.denoiser.phys],k[1,m.denoiser.phys]);vd=delta(v[0,m.denoiser.phys],v[1,m.denoiser.phys])
                assert kd['equal'] and vd['equal'],('physical cache depends on current gaze',kd,vd)
                parity.append(dict(shape=list(k.shape),physical_keys=kd,physical_values=vd))
            cache=m.denoiser.commit(cache,new)
            query=torch.tensor([7,1],device=DEV).expand(4,1,2);pa=torch.full((4,1),action,device=DEV)
            real=m.perception(future.repeat_interleave(2,0),query[:,0])
            realparts=tuple(x[:,None] for x in real.parts())
            control={**read_health(realparts,rd,float(r.get('health',r['next'][1]['health']))),**consequences(m,realparts,bp,query)}
            target_cell=r.get('target_cell',cell)
            control.update(read_arrow(realparts,target_cell))
            s['cache_key_deltas']=parity
            # Native pixel decoder acquisition control on the SAME cue crop.
            mean,_=m.pixels(m.tsp.reconstruct(P.g,P.v,M.window(fs)),P.v,M.window(fs));s['cue_pixel_mae']=((mean.float()+1)*127.5-M.cell_pixels(M.crop(current.repeat_interleave(2,0),fs)).float()).abs().mean((-1,-2)).cpu().tolist()
            slot=int((M.window(fs[:1])[0]==cell).nonzero()[0])
            # Compare the decoder's fine reconstruction against the two visible
            # cue alternatives, only at pixels which actually differ.
            pixels=M.cell_pixels(M.crop(current.repeat_interleave(2,0),fs)).float()
            a,b=pixels[0,slot],pixels[2,slot];mask=a!=b
            if mask.any():
                decoded=(mean[[0,2],slot].float()+1)*127.5
                d=((decoded[:,None,mask]-torch.stack([a,b])[None,:,mask])/255).square().mean(-1)
                s['fine_pixel_semantic_error']=d.cpu().tolist()
                s['fine_pixel_semantic_correct']=(d.argmin(-1)==torch.arange(2,device=DEV)).cpu().tolist()
        draws=[]
        for steps,churn in [(3,1.),(8,0.)]:
            c=copy.deepcopy(cfg);c['edm']['sampler'].update(steps=steps,s_churn=churn)
            num=4 if smoke else 64;rec=[]
            for b in range(0,num,16):
                n=min(16,num-b);seed=614200+case*1000+steps*100+b
                for condition in range(4):
                    gen=torch.Generator(device=DEV).manual_seed(seed)
                    one=dict(t=cache['t'],kv=[(k[condition:condition+1],v[condition:condition+1]) for k,v in cache['kv']])
                    with TR.autocast(DEV,cfg):
                        x=M.sample(m.denoiser,[(81,16),(9,64),(8,)],pa[condition:condition+1].expand(n,-1),
                            memory[condition:condition+1,None].expand(n,-1,-1,-1),query[condition:condition+1].expand(n,-1,-1),
                            expandcache(one,n),c,gen)
                        h0=float(r.get('health',r['next'][1]['health']))
                        vals={**read_health(x,rd,h0),**consequences(m,x,bp[condition:condition+1].expand(n,-1),query[condition:condition+1].expand(n,-1,-1))}
                        vals.update(read_arrow(x,target_cell))
                    rec.append(dict(condition=condition,batch=b,**vals))
            draws.append(dict(steps=steps,churn=churn,batches=rec))
        results.append(dict(kind=kind,meta=r,control=control,draws=draws));stage.append(s)
        torch.save(dict(results=results,stages=stage,checkpoint_sha256=SHA),HERE/('smoke.pt' if smoke else 'native_results.pt'))
        print('native case',case+1,'/',len(pairs),kind,e,t,'exact fine collision',s['mixed_fine_only']['equal'],flush=True)
    write('smoke.json' if smoke else 'stage_differences.json',stage)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['survey','sources','run']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='survey':survey()
    elif a.mode=='sources':causal_sources()
    else:run(a.smoke)

if __name__=='__main__':main()
