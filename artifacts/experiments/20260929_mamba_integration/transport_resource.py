"""Train-step time/memory smoke for exact carry transport, versus same initialized fmamba."""
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder'),str(HERE)]
import tworld as T  # noqa: E402
from transport_world import TransportTWorld  # noqa: E402
import spatial as S  # noqa: E402
from d4mj.config import config_from_dict  # noqa: E402
from d4mj.train import _phase_lr,autocast_context,optimizer_step,phase_optimizer  # noqa: E402

def main():
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    out={'status':'complete','steps':[]}
    for kind in ('fmamba','transport'):
      for b in (1,2,4):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        w=(T.TWorld('corrg',backbone='fmamba') if kind=='fmamba' else TransportTWorld()).cuda().train()
        opt=phase_optimizer([w],cfg)
        params=[p for g in opt.param_groups for p in g['params']]
        s=F.layer_norm(torch.randn(b,6,81,192,device='cuda'),(192,)).half()
        a=torch.randint(17,(b,5),device='cuda')
        began=time.time();loss=norm=None
        try:
          with autocast_context(cfg):loss=T.rollout_losses(w,s,a,'suffix')
          norm=optimizer_step(opt,loss,params,learning_rate=_phase_lr(cfg,0),
                              grad_clip=cfg.agent.grad_clip,strict=True,zero_grad=True)
          torch.cuda.synchronize()
          row={'kind':kind,'batch':b,'ok':True,'seconds':time.time()-began,
               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
               'objective':float(loss.detach()),'gradient_norm':float(norm)}
        except torch.OutOfMemoryError as ex:
          row={'kind':kind,'batch':b,'ok':False,'seconds':time.time()-began,
               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'error':str(ex).splitlines()[0]}
        out['steps'].append(row);print(json.dumps(row),flush=True)
        del w,opt,params,s,a,loss,norm
        torch.cuda.empty_cache()
        if not row['ok']:break
    HERE.joinpath('transport_resource.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'stage':'complete','rows':len(out['steps'])}),flush=True)
if __name__=='__main__':main()
