"""Paired, warmed six-frame training-step throughput/memory for exact carry transport."""
import hashlib,json,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder'),str(HERE)]
import tworld as T
from transport_world import TransportTWorld
import spatial as S
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr,autocast_context,optimizer_step,phase_optimizer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    out={'status':'complete','scope':'one cold plus three warmed full optimizer steps on synthetic six-frame inputs',
         'rows':[],'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'transport_world.py',HERE/'carry_transport.py',Path(T.__file__)]}}
    for kind in ('fmamba','transport'):
      for b in (1,4,16,40):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        w=(T.TWorld('corrg',backbone='fmamba') if kind=='fmamba' else TransportTWorld()).cuda().train()
        opt=phase_optimizer([w],cfg);params=[p for g in opt.param_groups for p in g['params']]
        s=F.layer_norm(torch.randn(b,6,81,192,device='cuda'),(192,)).half()
        a=torch.randint(17,(b,5),device='cuda')
        durations=[];peak=[];err=None
        for step in range(4):
          try:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with autocast_context(cfg):loss=T.rollout_losses(w,s,a,'suffix')
            norm=optimizer_step(opt,loss,params,learning_rate=_phase_lr(cfg,step),
                                grad_clip=cfg.agent.grad_clip,strict=True,zero_grad=True)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-started);peak.append(torch.cuda.max_memory_allocated())
          except torch.OutOfMemoryError as e:
            err=str(e).splitlines()[0];break
        row={'kind':kind,'batch':b,'ok':err is None,'cold_seconds':durations[0] if durations else None,
             'warm_seconds':durations[1:],'warm_median_seconds':statistics.median(durations[1:]) if len(durations)>1 else None,
             'max_peak_allocated_bytes':max(peak) if peak else None,'error':err}
        out['rows'].append(row);print(json.dumps(row),flush=True)
        del w,opt,params,s,a
        if 'loss' in locals():del loss
        if 'norm' in locals():del norm
        torch.cuda.empty_cache()
        if err:break
    HERE.joinpath('transport_resource_steady.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'stage':'complete','rows':len(out['rows'])}),flush=True)
if __name__=='__main__':main()
