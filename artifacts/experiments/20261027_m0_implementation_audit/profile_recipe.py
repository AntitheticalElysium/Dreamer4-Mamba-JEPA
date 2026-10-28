"""Short independent mechanical profile; not production P1 or a learning study."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR


def main():
    torch.set_num_threads(6);torch.manual_seed(1897)
    cfg=TR.load_cfg();split=D.load_split(cfg);P=cfg['data']['context_prefix']
    streams=D.Streams(split['train'],16,P,seed=1897,targets_offsets=D.frame_offsets(split['train']))
    targets=np.load('artifacts/saccade_m0/targets/change_targets.npy',mmap_mode='r')
    model=M.M0(cfg).cuda().train();model.denoiser.checkpoint=True
    TR.set_reward_values(model,split['train'])
    opt=torch.optim.AdamW(model.param_groups(),lr=1e-4,betas=(.9,.999),eps=1e-8)
    for g in opt.param_groups:g['lr']*=g['lr_scale']
    carry=model.belief.initial(16,'cuda');cache=torch.zeros(16,P+1,256,device='cuda')
    out=[]
    for steps,micro in ((64,1),(256,4)):
        times=[];norms=[];losses=[]
        torch.cuda.reset_peak_memory_stats()
        for update in range(8):
            torch.cuda.synchronize();start=time.perf_counter()
            chunk=streams.next(steps);opt.zero_grad(set_to_none=True)
            vals=[]
            for rows in np.array_split(np.arange(16),micro):
                terms,c,cc=TR.p1_losses(model,D.select(chunk,rows),M.carry_rows(carry,rows),cache[rows],targets,cfg)
                (model.harmoniser(TR.groups(terms))/micro).backward()
                M.carry_put(carry,rows,c);cache[rows]=cc
                vals.append({k:float(v.detach()) for k,v in terms.items()})
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1))
            opt.step();torch.cuda.synchronize()
            assert np.isfinite(gn)
            if update>=3:times.append(time.perf_counter()-start)
            norms.append(gn);losses.append(vals)
        out.append({'steps':steps,'microbatches':micro,'warmup_updates':3,'measured_updates':5,
                    'mean_seconds_per_update':float(np.mean(times)),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'grad_norms':norms,'last_terms':losses[-1]})
    result={'seed':1897,'production_world_seed':False,'world_parameters':sum(p.numel() for p in model.parameters()),
            'profiles':out,'scope':'Eight updates per shape, real streams and all losses; no quality/plateau claim.'}
    Path(__file__).with_name('recipe_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
