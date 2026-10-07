"""CPU numerical diagnosis of the frozen-health test's exact reproduction failure.

No fitting choices: recreate literal teval HUD ridge TRAIN/validation contract.
Compare seed7 batch4 GPU HUD outputs to check_damage's saved batch16 teacher
health on exactly those cases. Report all threshold crossings and absolute error;
do not relax the half-unit threshold or relabel the failed clock test valid.
"""
import json
from pathlib import Path
import torch
import teval as T
import stochdiag as SD
import check_damage_rule as DR
import h16_resume as R

HERE=Path(__file__).resolve().parent


def read(path,key):
    info=json.loads((path/f'{key}.json').read_text());f=path/info['file']
    assert R.file_hash(f)==info['sha256']
    return torch.load(f,map_location='cpu',weights_only=False)


def main():
    torch.set_num_threads(2)
    meta,tr,seeds=T.split();cache=torch.load(Path(str(T.CACHE).format('raw')),map_location='cpu',mmap=True,weights_only=False)
    val=torch.isin(meta['seed'],seeds[:len(seeds)//5]);fit=tr&~val
    toks=torch.cat([cache['ctx'][:,-1:],cache['fut']],1)
    vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
    rows=lambda m:m[:,None].expand(-1,17).flatten()
    hud=T.ridge(toks[:,:,63:81].flatten(0,1).flatten(1).float(),T.facts_of(vis)['hud'].flatten(0,1),rows(fit),rows(val))
    fut,_=SD.token_cache(torch.device('cpu'));seq=torch.cat([cache['ctx'],fut[:,0]],1)
    health=lambda x:hud(x.float().reshape(-1,18*192))[:,0]*9
    hp=vis[:,:,1512]*9;alive=~meta['future_dead'].cumsum(2).bool().any(1)
    drop=(hp[:,1:]<hp[:,:-1]-.5)&alive;same=(hp[:,1:]==hp[:,:-1])&alive
    m={k:v[:,0] for k,v in DR.masks(meta).items()}
    fresh=m['valid']&m['k3']&m['adjacent']&~m['win']&~m['adjwin']&m['drop2']
    selected=drop|fresh;si=torch.where(same.flatten())[0]
    selected.view(-1)[si[torch.randperm(len(si),generator=torch.Generator().manual_seed(20261007))[:512]]]=True
    pairs=torch.nonzero(selected);rr,kk=pairs.T;hcur=health(seq[rr,3+kk,63:81])
    small=HERE/'evals/resume/e17_health_clock'
    hud4=torch.empty(len(pairs),18,192);covered=torch.zeros(len(pairs),dtype=torch.bool)
    for p in small.glob('s7_true_history_*.json'):
        s=read(small,p.stem);hud4[s['ids']]=s['hud'];covered[s['ids']]=True
    assert bool(covered.all())
    old=next(Path('artifacts/eda/frozen_eval_resume_v1').glob('corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__damage_w15__*'))
    tf=torch.empty(len(meta['seed']),16)
    for lo in range(0,len(tf),16):
        s=read(old,f'batch_{lo}');tf[lo:lo+len(s['teacher'])]=s['teacher']
    p4=health(hud4);p16=tf[rr,kk];err=p4-p16
    crossed=(p4<hcur-.5)!=(p16<hcur-.5)
    result={'scope':__doc__,'sources':{f:R.file_hash(f) for f in (__file__,T.__file__,R.__file__)},
            'inputs':{str(p):R.file_hash(p) for p in (small/'contract.json',old/'contract.json')},
            'cases':len(pairs),'max_absolute_health_difference':float(err.abs().max()),
            'mean_absolute_health_difference':float(err.abs().mean()),
            'crossings':[{'root':int(rr[j]),'step':int(kk[j]),'damage':bool(drop[rr[j],kk[j]]),
                         'current':float(hcur[j]),'batch4':float(p4[j]),'batch16':float(p16[j]),
                         'difference':float(err[j])} for j in torch.where(crossed)[0]],
            'damage_catches_batch4':int((p4<hcur-.5)[drop[rr,kk]].sum()),
            'damage_catches_batch16':int((p16<hcur-.5)[drop[rr,kk]].sum()),
            'failed_exact_count_reproduction_preserved':True}
    R.atomic_json(HERE/'evals/e17_health_batch_parity.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
