"""Check stored evaluation tokens against the pinned frozen encoders, including fork successors."""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis')]
import teval as E  # noqa: E402
from rollouts import load_roots  # noqa: E402
from d4mj.checkpoint import read_lewm_bundle  # noqa: E402
from d4mj.config import config_from_dict  # noqa: E402
from d4mj.world_api import ModelBundle  # noqa: E402


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def encode(enc,frames):
    with torch.no_grad():
        _,_,t,_,_=enc._hidden(frames[:,None])
        return F.layer_norm(t.float(),(192,)).half().cpu()


def main():
    torch.set_num_threads(8)
    roots=load_roots()
    ids=sorted({0,len(roots)//2,len(roots)-1})
    report={'status':'complete','roots':len(roots),'sample_indices':ids,'accept_max_abs':0.002,'arms':{},
            'source_sha256':digest(Path(__file__))}
    for name in ('raw','ldad1','ldad10'):
        path=Path(str(E.CACHE).format(name))
        cache=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert len(cache['ctx'])==len(roots)
        ckpt=E.ENCODERS[name]
        raw=torch.load(ckpt,map_location='cpu',weights_only=False)
        payload=read_lewm_bundle(ckpt) if 'format' in raw else raw
        bundle=ModelBundle.create(config_from_dict(payload['config']))
        bundle.encoder.load_state_dict(payload['modules']['encoder'])
        enc=bundle.encoder.to('cpu').freeze()
        diffs=[]; mean_diffs=[]; exact_shares=[]
        for idx in ids:
            root=roots[idx]
            frames=torch.stack([root['context'][0],root['context'][-1],root['future_frames'][0,0],
                                root['onestep_frames'][0,0],root['onestep_frames'][0,6],
                                root['onestep_frames'][0,16]])
            t=encode(enc,frames)
            stored=torch.stack([cache['ctx'][idx,0],cache['ctx'][idx,-1],cache['fut'][idx,0],
                                cache['one'][idx,0],cache['one'][idx,6],cache['one'][idx,16]])
            delta=(t.float()-stored.float()).abs()
            diffs.append(delta.amax().item())
            mean_diffs.append(delta.mean().item())
            exact_shares.append((delta==0).float().mean().item())
        report['arms'][name]={'checkpoint_sha256':digest(ckpt),'cache_sha256':digest(path),
                              'max_abs_token_difference':max(diffs),'per_root_max_abs':diffs,
                              'mean_abs_token_difference':sum(mean_diffs)/len(mean_diffs),
                              'exact_token_share':sum(exact_shares)/len(exact_shares)}
        print(json.dumps({'arm':name,**report['arms'][name]}),flush=True)
        assert max(diffs)<=0.002,(name,diffs)
        del enc,bundle,cache,raw,payload
    HERE.joinpath('cache_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'complete','roots':len(roots),'arms':list(report['arms'])}),flush=True)

if __name__=='__main__':main()
