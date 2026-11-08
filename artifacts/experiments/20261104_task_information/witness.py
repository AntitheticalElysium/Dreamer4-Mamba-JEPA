"""Final validity checks on the four paired survivors; no further selection."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import jax
from .consequence import KINDS
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);m,cfg=N.load()
    z=torch.load(N.ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',mmap=True,weights_only=False)['current']
    L=dict(np.load(N.S.LABELS/'s0_dev_labels.npz'));offs=np.r_[0,np.cumsum(L['ep_T']+1)]
    local=N.S.load_reader(N.CK.parent/'s1/results.reader.pt',N.CK,N.DEV)
    bundle=torch.load(N.ROOT/'artifacts/experiments/20261103_p1f16_review/p1f16/readers.pt',map_location=N.DEV,weights_only=False);nets={}
    for k in ['g_0','g_1','p_0']:
        r=bundle[k];net=N.S.MLP(r['dim'],{'kind':5}).to(N.DEV);net.load_state_dict(r['state']);nets[k]=net.eval()
    rows=[r for r in json.loads((HERE/'focus_paired_readers.json').read_text()) if r['qualified']]
    _,_,_,_,render=N.D.RP.env_and_render();out=[];arrays=[]
    for r in rows:
        meta=r['meta'];e,t,c,j=meta['episode'],meta['time'],meta['cell'],meta['j'];name=KINDS[meta['kind']]
        state=N.D.RP.advance_to(meta['shard'],meta['slot'],t);ent=getattr(state,name)
        absent=state.replace(**{name:ent.replace(mask=ent.mask.at[j].set(False),health=ent.health.at[j].set(0))})
        if meta['kind']!=4:
            world=np.asarray(ent.position[j]);occupied=any(bool((np.asarray(getattr(absent,k).mask)&(np.asarray(getattr(absent,k).position)==world).all(-1)).any()) for k in ['zombies','cows','skeletons'])
            absent=absent.replace(mob_map=absent.mob_map.at[world[0],world[1]].set(occupied))
        frames=np.stack([np.asarray(render(s)) for s in [state,absent]])
        fx=torch.tensor([meta['focus']],device=N.DEV).expand(2,-1);cells=N.M.window(fx);slot=int((cells[0]==c).nonzero()[0])
        with N.TR.autocast(N.DEV,cfg):
            P=m.perception(torch.as_tensor(frames,device=N.DEV),fx)
            prior=z['prior'][meta['frame']].to(N.DEV).float().expand(2,-1)
            tsp,_=m.pixels(m.tsp.predict(prior,P.p,P.v,cells),P.v,cells)
            rec,_=m.pixels(m.tsp.reconstruct(P.g,P.v,cells),P.v,cells)
        targets=N.M.cell_pixels(N.M.crop(torch.as_tensor(frames,device=N.DEV),fx)).float()[:,slot]
        contrast=targets[0]-targets[1];mask=contrast!=0
        assert mask.any()
        reconstruction={}
        for key,pred in [('tsp',tsp),('fine',rec)]:
            pred=(pred[:,slot].float()+1)*127.5
            mse=((pred[:,None,mask]-targets[None,:,mask])/255).square().mean(-1)
            reconstruction[key]=dict(alternative_mse=mse.cpu().tolist(),choice=mse.argmin(-1).cpu().tolist(),
                recovered_contrast=float(((pred[0]-pred[1])*contrast).sum()/contrast.square().sum()))
        a=int(offs[e]);active=L[f's_{name}_m'][a:a+t+1,j];start=int(np.flatnonzero(~active)[-1]+1) if (~active).any() else 0
        rel=L[f's_{name}_p'][a:a+t+1,j]-L['s_pos'][a:a+t+1]
        visible=active&(abs(rel[:,0])<=3)&(abs(rel[:,1])<=4)
        history=dict(active_interval_start=start,visible_before_current=np.where(visible[:t])[0].tolist(),
            visible_before_current_in_interval=(np.where(visible[start:t])[0]+start).tolist())
        rendered=np.stack([np.stack([np.asarray(render(s.replace(state_rng=jax.random.PRNGKey(619000+1000*e+k)))) for k in range(64)]) for s in [state,absent]])
        probabilities={k:[] for k in ['s1','p_0','g_0','g_1']}
        for state_index in [0,1]:
            ff=torch.as_tensor(rendered[state_index],device=N.DEV);fx64=fx[:1].expand(64,-1)
            with N.TR.autocast(N.DEV,cfg):p=m.perception(ff,fx64)
            pv=torch.cat([p.p[:,c].float(),p.v.float()],-1);gv=torch.cat([p.g[:,slot].float(),p.v.float()],-1)
            inputs={'s1':pv,'p_0':torch.cat([pv,F.one_hot(torch.full((64,),c,device=N.DEV),81).float()],-1),'g_0':gv,'g_1':gv}
            for key,x in inputs.items():
                net=local if key=='s1' else nets[key];probabilities[key].append(net(x)['kind'].softmax(-1).cpu().numpy())
        pscore={}
        for key,x in probabilities.items():
            x=np.stack(x);y=np.array([meta['kind'],r['alternate_label']]);pred=x.argmax(-1)
            pscore[key]=dict(accuracy=(pred==y[:,None]).mean(-1).tolist(),mean_true_score=[float(x[k,:,y[k]].mean()) for k in range(2)],
                confusion_predictions=[np.bincount(pred[k],minlength=5).tolist() for k in range(2)])
        out.append(dict(meta=meta,history=history,natural_reconstruction=reconstruction,independent_render=pscore))
        arrays.append(dict(meta=meta,current=frames,tsp=tsp.cpu(),reconstruction=rec.cpu(),resampled_probabilities=probabilities))
        print('witness',e,t,'TSP choices',reconstruction['tsp']['choice'],'independent render', {k:v['accuracy'] for k,v in pscore.items()},flush=True)
    (HERE/'witness_checks.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(arrays,HERE/'witness_arrays.pt')

if __name__=='__main__':main()
