"""Resource and short-prefix-equivalence smoke for the proposed 64-frame fmamba world."""
import json
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from long_world import LongTWorld,rollout_loss
import tworld as T
import spatial as S
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr,autocast_context,optimizer_step,phase_optimizer
HERE=Path(__file__).resolve().parent

def main():
    torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    base=T.TWorld('corrg',backbone='fmamba').cuda().eval()
    long=LongTWorld().cuda().eval();long.load_state_dict(base.state_dict())
    s=F.layer_norm(torch.randn(1,6,81,192,device='cuda'),(192,))
    a=torch.randint(17,(1,6),device='cuda')
    with torch.no_grad():
        p=base(s,a)[0];q=long(s,a)[0]
    prefix_diff=float((p-q).abs().max())
    assert prefix_diff<=1e-6,prefix_diff
    del base,long,p,q,s,a
    config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    results=[]
    for b in (1,2,4,8):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        world=LongTWorld().cuda().train()
        opt=phase_optimizer([world],config)
        params=[p for g in opt.param_groups for p in g['params']]
        s=F.layer_norm(torch.randn(b,64,81,192,device='cuda'),(192,))
        a=torch.randint(17,(b,63),device='cuda')
        start=time.time()
        loss=norm=None
        try:
            with autocast_context(config):loss=rollout_loss(world,s,a)
            norm=optimizer_step(opt,loss,params,learning_rate=_phase_lr(config,0),
                                grad_clip=config.agent.grad_clip,strict=True,zero_grad=True)
            torch.cuda.synchronize()
            row={'batch':b,'ok':True,'seconds':round(time.time()-start,3),
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                 'loss':float(loss.detach()),'grad_norm':float(norm)}
        except torch.OutOfMemoryError as e:
            row={'batch':b,'ok':False,'seconds':round(time.time()-start,3),
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'error':str(e).splitlines()[0]}
        print(json.dumps(row),flush=True);results.append(row)
        del world,opt,params,s,a,loss,norm
        torch.cuda.empty_cache()
        if not row['ok']:break
    report={'prefix_max_abs':prefix_diff,'batches':results,
            'source':{'long_world':str(HERE/'long_world.py'),'short_world':str(Path(T.__file__))}}
    (HERE/'long_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'complete','prefix_max_abs':prefix_diff,'max_working_batch':max((x['batch'] for x in results if x['ok']),default=0)}),flush=True)

if __name__=='__main__':main()
