"""Final-source six-step mechanics check; predeclared 1e-6 numerical tolerance."""
import json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/eda/levers_mamba_integration_smoke_v2'
TRAIN=HERE/'short_train.py'
NAME='int_corrg_raw_suffix_s7_fmamba_u18000.resume.pt'
LIMIT=1e-6

def run(path,n):
    subprocess.run([sys.executable,str(TRAIN),'--pool','raw','--out',str(path),'--until',str(n)],
                   cwd=ROOT,check=True,env=os.environ|{'TRITON_F32_DEFAULT':'ieee'})

def diff(x,y):
    assert x.keys()==y.keys()
    return max(float((x[k].float()-y[k].float()).abs().max()) for k in x)

def main():
    a,b=OUT/'split',OUT/'direct'
    assert not a.exists() and not b.exists(),'smoke paths must be unused'
    run(a,3);run(a,6);run(b,6)
    A=torch.load(a/NAME,map_location='cpu',weights_only=False)
    B=torch.load(b/NAME,map_location='cpu',weights_only=False)
    assert A['contract']==B['contract'] and A['update']==B['update']==6
    assert torch.equal(A['order_rng'],B['order_rng'])
    resume=diff(A['world'],B['world'])
    sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
                  str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
    import tworld as T
    old,_,_=T.train('corrg','raw','suffix',7,6,torch.device('cuda'),lambda **kw:None,backbone='fmamba')
    legacy=diff(B['world'],{k:v.cpu() for k,v in old.state_dict().items()})
    report={'status':'pass' if max(resume,legacy)<=LIMIT else 'fail','limit':LIMIT,
            'resume_max_abs':resume,'legacy_max_abs':legacy,'sampler_rng_equal':True,
            'steps':6,'source_sha256':B['contract']['source_sha256']}
    (HERE/'verify_accept.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert report['status']=='pass'

if __name__=='__main__':main()
