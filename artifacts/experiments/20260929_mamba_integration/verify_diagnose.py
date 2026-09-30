"""Localize 3+3 resume discrepancy against independent 6-step and legacy runs."""
import json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/eda/levers_mamba_integration_smoke_v1'
TRAIN=HERE/'short_train.py'
NAME='int_corrg_raw_suffix_s7_fmamba_u18000.resume.pt'

def run(path,n):
    subprocess.run([sys.executable,str(TRAIN),'--pool','raw','--out',str(path),'--until',str(n)],
                   cwd=ROOT,check=True,env=os.environ|{'TRITON_F32_DEFAULT':'ieee'})

def measure(a,b):
    aa,bb=a['world'],b['world'];assert aa.keys()==bb.keys()
    rows=[]
    for k in aa:
        x=(aa[k].float()-bb[k].float()).abs()
        rows.append((float(x.max()),float(x.square().mean().sqrt()),k,int((x!=0).sum())))
    rows.sort(reverse=True)
    return {'max_abs':rows[0][0],'max_rms':max(r[1] for r in rows),
            'different_tensor_count':sum(r[0]>0 for r in rows),'top':rows[:8],
            'order_rng_equal':bool(torch.equal(a['order_rng'],b['order_rng'])),
            'loss_a':a['history'][-1]['objective'],'loss_b':b['history'][-1]['objective']}

def main():
    run(OUT/'direct2',6)
    run(OUT/'split2',3);run(OUT/'split2',6)
    states={n:torch.load(OUT/n/NAME,map_location='cpu',weights_only=False)
            for n in ('split','direct','direct2','split2')}
    report={'split_vs_direct':measure(states['split'],states['direct']),
            'direct_vs_direct2':measure(states['direct'],states['direct2']),
            'split_vs_split2':measure(states['split'],states['split2'])}
    sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
                  str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
    import tworld as T
    old,_,_=T.train('corrg','raw','suffix',7,6,torch.device('cuda'),lambda **kw:None,backbone='fmamba')
    old_state={'world':{k:v.cpu() for k,v in old.state_dict().items()},'order_rng':states['direct']['order_rng'],
               'history':[{'objective':float('nan')}]}
    report['direct_vs_legacy']=measure(states['direct'],old_state)
    (HERE/'verify_diagnose.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'max_abs':v['max_abs'],'different_tensor_count':v['different_tensor_count'],
                         'order_rng_equal':v['order_rng_equal']} for k,v in report.items()}),flush=True)

if __name__=='__main__':main()
