"""Re-read genuine crop interventions with stronger conditional readers. PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,time
from pathlib import Path
import numpy as np
import torch
from saccade import model as M,data as D,train as TR
from saccade.eval import s1 as S
C=importlib.import_module('artifacts.experiments.20261102_foveal_information.conditional')
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')
HERE=Path(__file__).resolve().parent;DEV=torch.device('cuda')
CENTRES=torch.tensor([(r,c) for r in range(1,8) for c in range(1,8)],device=DEV);WIN=M.window(CENTRES)

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.time();m,cfg=C.U.load();z=torch.load(HERE/'dev_250.pt',weights_only=False);feat=C.Features(z,m)
    raw=dict(np.load(C.U.HERE/'uptake_local/raw.npz'));L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250);eps=S.episodes(L)
    off=np.r_[0,np.cumsum(L['ep_T'])];records=[]
    for i,(e,t) in enumerate(zip(raw['episode'],raw['t'])):
        truth=raw['truth'][i];cells=np.flatnonzero(truth>0)
        if not len(cells):continue
        obs=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'][t].to(DEV)
        with TR.autocast(DEV,cfg):P=m.perception(obs[None].expand(49,-1,-1,-1),CENTRES)
        f=int(off[e]+t)
        assert torch.equal(z['truth'][f],torch.tensor(truth,dtype=torch.long))
        for cell in cells:
            av=(WIN==int(cell)).any(-1);lg=(WIN==int(cell)).long().argmax(-1)
            fine=P.g[torch.arange(49,device=DEV),lg].float()*av[:,None]
            records.append(dict(raw_i=i,episode=int(e),t=int(t),frame=f,cell=int(cell),truth=int(truth[cell]),
                fine=fine.half().cpu(),available=av.cpu()))
    print('paired source queries',len(records),flush=True)
    keys=torch.load(HERE/'conditional/readers.pt',weights_only=False)
    lp={};f=torch.tensor([r['frame'] for r in records],device=DEV);cell=torch.tensor([r['cell'] for r in records],device=DEV)
    ri=np.array([r['raw_i'] for r in records]);fine=torch.stack([r['fine'] for r in records]).to(DEV)
    av=torch.stack([r['available'] for r in records]).to(DEV)
    for key,s in keys.items():
        reader=C.Reader(s['dim'],s['family']).to(DEV);reader.load_state_dict(s['state']);reader.eval();mode=s['mode'];log=[]
        for st in range(0,len(records),16):
            fr=f[st:st+16];cc=cell[st:st+16];B=len(fr)
            ix=(fr*63+cc)[:,None].expand(-1,49).flatten()
            base,_,cq=feat.make(ix,'base')
            base[:,-3:-1]=CENTRES[None].expand(B,-1,-1).flatten(0,1).float()/8
            base[:,-1]=av[st:st+B].flatten().float()
            extra=torch.zeros(B,49,1024,device=DEV)
            if mode=='fine':extra[:,:,:64]=fine[st:st+B].float()
            elif mode!='base':
                rk={'input':'g_input','corrected':'corrected','predicted':'predicted','memory':'memory'}[mode]
                a=torch.tensor(raw[rk][ri[st:st+B]],device=DEV);extra[:,:,:a.shape[-1]]=a
            out=reader((base,extra.flatten(0,1),cq)).log_softmax(-1)
            log.append(out.reshape(B,49,5).half().cpu())
        lp[key]=torch.cat(log)
        print('paired',key,flush=True)
    meta={k:np.array([r[k] for r in records]) for k in ('raw_i','episode','t','frame','cell','truth')}
    np.savez_compressed(HERE/'paired_meta.npz',**meta,available=av.cpu().numpy())
    torch.save(lp,HERE/'paired_logp.pt')
    (HERE/'paired_identity.json').write_text(json.dumps({'queries':len(records),'seconds':time.time()-start,
        'script_sha256':TR.file_sha256(__file__),'checkpoint_sha256':TR.file_sha256(C.U.CK),
        'reader_sha256':TR.file_sha256(HERE/'conditional/readers.pt')},indent=1))

if __name__=='__main__':main()
