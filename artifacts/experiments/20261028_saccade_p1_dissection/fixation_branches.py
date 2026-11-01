"""Supported, paired real-crop interventions at identical incoming beliefs. Never zero an input."""
import time
from .common import *


@torch.no_grad()
def main():
    before=fingerprints();model,cfg=load_model();mh=model_hash(model)
    L=labels('dev');Y=hud_values(L);off=np.r_[0,np.cumsum(L['ep_T']+1)]
    enc=torch.load(OUT/'real_dev.pt',weights_only=False,mmap=True)
    results={k:[] for k in ('b','reference','cached_reference','field','episode','time','fixations','labels','subsequent_seen')}
    for e,(r,ep) in enumerate(zip(enc,s1.episodes(L))):
        n=len(r['p']);carry=model.belief.initial(1,'cuda');raw=None
        for t in range(0,n,64):
            stop=min(t+64,n)
            def percept(a,b):return M.Percept(*(r[k][None,a:b].cuda() for k in ('p','g','v')),r['fix'][None,a:b].cuda())
            P=percept(t,stop)
            rw=r['reward'][None,t:stop].cuda();term=r['terminated'][None,t:stop].cuda();act=r['action'][None,t:stop].cuda()
            with TR.autocast(torch.device('cuda'),cfg):
                xx=model.belief.tokens(model.belief.percept_embed(P),rw,term,act)
                original,newcarry=model.belief(xx,carry)
            if t>=64 and t+16<n:
                j=(e+t//64)%16;c=int(HUD_CELLS[j]);col=c%9
                rng=np.random.default_rng([728,e,t,j]);cover=[x for x in range(1,8) if abs(x-col)<=1]
                away=[x for x in range(1,8) if abs(x-col)>1]
                f=torch.tensor([r['fix'][t].tolist(),[7,int(rng.choice(cover))],[7,int(rng.choice(away))]],device='cuda')
                if raw is None:raw=D.RP.episode_fields(ep.shard,ep.slot)['observations'].numpy()
                frame=torch.as_tensor(raw[t:t+1],device='cuda').expand(3,-1,-1,-1)
                # Only the genuine crop and its location change; p/v are exactly shared from the frozen real path.
                with TR.autocast(torch.device('cuda'),cfg):
                    cp=M.crop(frame,f).permute(0,3,1,2).float()/127.5-1
                    g=M.out_map(model.perception.fovea(cp).flatten(2).transpose(1,2),3.)
                    base=percept(t,t+17)
                    pp=base.p.expand(3,-1,-1,-1).clone();gg=base.g.expand(3,-1,-1,-1).clone();vv=base.v.expand(3,-1,-1).clone()
                    ff=base.fix.expand(3,-1,-1).clone();gg[1:,0]=g[1:];ff[1:,0]=f[1:]
                    tok=model.belief.tokens(model.belief.percept_embed(M.Percept(pp,gg,vv,ff)),
                         r['reward'][None,t:t+17].cuda().expand(3,-1),r['terminated'][None,t:t+17].cuda().expand(3,-1),
                         r['action'][None,t:t+17].cuda().expand(3,-1))
                    bb,_=model.belief(tok,M.carry_rows(carry,torch.zeros(3,dtype=torch.long,device='cuda')))
                results['b'].append(bb[:,0::2].float().half().cpu())
                results['reference'].append(original[0,0::2][:17].float().half().cpu())
                results['cached_reference'].append(r['b'][t:t+17])
                results['field'].append(j);results['episode'].append(e);results['time'].append(t);results['fixations'].append(f.cpu())
                results['labels'].append(Y[off[e]+t:off[e]+t+17])
                results['subsequent_seen'].append((crop_cells(r['fix'][t:t+17])==c).any(1))
            carry=newcarry
        if e%100==0:print('branch',e,len(results['field']),flush=True)
    results={k:torch.stack(v) if isinstance(v[0],torch.Tensor) else np.stack(v) for k,v in results.items()}
    torch.save(results,OUT/'fixation_branches.pt')
    bb=results['b'];delta=(bb[:,0]-results['reference']).float();dc=(bb[:,0]-results['cached_reference']).float()
    parity={'chunk_rmse':float(delta.square().mean().sqrt()),'chunk_max_abs':float(delta.abs().max()),
            'cache_rmse':float(dc.square().mean().sqrt()),'cache_max_abs':float(dc.abs().max())}
    out={'parity':parity,'n_cases':len(bb),'readers':{}}
    ep=results['episode'];field=results['field'];truth=results['labels'];seen=results['subsequent_seen']
    for seed in (0,1):
        pred=[];refpred=[]
        for task in ('body','inventory'):
            ck=torch.load(OUT/f'belief_b_{task}_{task}_{seed}.pt',weights_only=False)
            rd=s1.MLP(ck['input_dim'],ck['heads']).cuda().eval();rd.load_state_dict(ck['state'])
            p=[];pr=[]
            for q in bb.reshape(-1,256).split(16384):p.append(rd(q.cuda())['hud'].reshape(-1,16,10).argmax(-1).cpu())
            for q in results['reference'].reshape(-1,256).split(16384):pr.append(rd(q.cuda())['hud'].reshape(-1,16,10).argmax(-1).cpu())
            pred.append(torch.cat(p).reshape(len(bb),3,17,16).numpy());refpred.append(torch.cat(pr).reshape(len(bb),17,16).numpy())
        pred=np.concatenate([pred[0][...,:4],pred[1][...,4:]],-1);refpred=np.concatenate([refpred[0][...,:4],refpred[1][...,4:]],-1)
        parity[f'reader_{seed}_agreement']=float((pred[:,0]==refpred).mean())
        res={}
        for j,name in enumerate(HUD):
            m=field==j;res[name]={}
            for lag in (0,1,4,8,16):
                for scope,mask in (('all',m),('no_subsequent_target_glimpse',m&~seen[:,1:lag+1].any(1))):
                    a=pred[:,1,lag,j];b=pred[:,2,lag,j];y=truth[:,lag,j]
                    res[name][f'{lag}/{scope}']={'covered_exact':G.estimate(a==y,mask,ep),'away_exact':G.estimate(b==y,mask,ep),
                        'paired_exact_difference':G.estimate((a==y).astype(float)-(b==y),mask,ep),
                        'covered_mae':G.estimate(np.abs(a-y),mask,ep),'away_mae':G.estimate(np.abs(b-y),mask,ep)}
        out['readers'][seed]=res
    out['parity_passed']=parity['chunk_rmse']<.002 and parity['chunk_max_abs']<.02 and min(parity[f'reader_{s}_agreement'] for s in (0,1))>=.995
    save('fixation_branches.json',out)
    assert model_hash(model)==mh and fingerprints()==before


if __name__=='__main__':main()
