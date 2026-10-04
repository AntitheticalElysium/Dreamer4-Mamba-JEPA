"""Read-only move-routing learning curve on the fixed diagnosis panel.

Reports held-out move/blocked labels, the corrg frame logit, and its actual
copy weights at the player and target cells. World and panel are not refit.
"""
import json, sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(ROOT/'artifacts/experiments/20260926_diagnosis'),str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
import teval as T
from onestep import classify
from compound import auc
from corrg_probe import probe_auc
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as S
TARGET=torch.tensor([31,30,32,22,40])
SCROLL=torch.tensor([0,3,4,1,2])
def run(pool,budget):
 device=torch.device('cuda')
 tag=f'int_corrg_{pool}_suffix_s7_fmamba_u18000'+(f'_at{budget}' if budget<18000 else '')
 path=ROOT/'artifacts/eda/levers_mamba_integration_v1'/f'{tag}.pt'
 world,st=T.load_world(path,device)
 meta,tr,_=T.split(); cl,_=classify(meta); ca=T.build_cache(pool,device)
 config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
 rows=[(i,a) for i in range(len(tr)) for a in range(1,5) if cl[i,a] in (0,1)]
 out={k:[] for k in ('frame','player_self','player_scroll','target_self','target_scroll','action','seed','moved','test','h_action','h_target','input_target')}
 with torch.no_grad(),autocast_context(config):
  for start in range(0,len(rows),16):
   sub=rows[start:start+16]; ii=torch.tensor([i for i,a in sub]);aa=torch.tensor([a for i,a in sub]); b=len(sub)
   ctx=ca['ctx'][ii].float().to(device);acts=torch.cat([ca['ctx_a'][ii],aa[:,None]],1).to(device)
   h,ha=world.backbone_full(ctx,acts); h=h[:,-1]; frame=world.frame(ha[:,-1]).float()[:,0]
   logits=world.choose(h).float()+frame[:,None,None]*torch.tensor([0,1,1,1,1,0],device=device)
   w=torch.softmax(logits,-1);ar=torch.arange(b,device=device);tgt=TARGET[aa].to(device);scroll=SCROLL[aa].to(device)
   out['frame'].append(frame.cpu());out['player_self'].append(w[:,31,0].cpu());out['player_scroll'].append(w[ar,31,scroll].cpu())
   out['target_self'].append(w[ar,tgt,0].cpu());out['target_scroll'].append(w[ar,tgt,scroll].cpu())
   out['h_action'].append(ha[:,-1].float().cpu());out['h_target'].append(h[ar,tgt].float().cpu());out['input_target'].append(ctx[ar,-1,tgt].float().cpu());out['action'].append(aa);out['seed'].append(meta['seed'][ii]);out['moved'].append((cl[ii,aa]==0));out['test'].append((~tr[ii]))
 out={k:torch.cat(v) for k,v in out.items()};te=out['test'];y=out['moved']
 result={'checkpoint_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'n_all':len(y),'n_test':int(te.sum()),'n_test_moved':int((te&y).sum()),'n_test_blocked':int((te&~y).sum()),'metrics':{}}
 for k in ('frame','player_self','player_scroll','target_self','target_scroll'):
  x=out[k];result['metrics'][k]={'auc_moved_test':float(auc(x[te],y[te])),'mean_test_moved':float(x[te&y].mean()),'mean_test_blocked':float(x[te&~y].mean())}
 for k in ('h_action','h_target','input_target'):
  result['metrics'][k]={'probe_auc_moved_test':float(probe_auc(out[k],y,~te,te))}
 # A copy-route decision is simply whether the direction-correct scroll candidate outweighs self at player cell.
 d=out['player_scroll']>out['player_self']; result['player_scroll_beats_self']={'moved':float(d[te&y].float().mean()),'blocked':float(d[te&~y].float().mean())}
 # Save only aggregate rows to avoid an unnecessarily large/reidentifiable artifact.
 return result
if __name__=='__main__':
 ans={str(budget):{pool:run(pool,budget) for pool in ('raw','ldad1','ldad10')} for budget in (12000,18000)}
 p=ROOT/'artifacts/experiments/20260929_mamba_integration/gate_curve.json';p.write_text(json.dumps(ans,indent=2)+'\n')
 print(json.dumps(ans,indent=2))
