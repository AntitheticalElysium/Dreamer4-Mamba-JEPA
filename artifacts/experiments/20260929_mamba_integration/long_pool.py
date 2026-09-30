"""Resumable, shared-ledger 64-frame patch-token cache for Raw/LDAD1/LDAD10.

Writes each arm to a staging directory, flushes data before advancing progress, verifies input hashes
and exact TRAIN/terminal labels, then hashes files and publishes by atomic directory rename.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
CKPTS={
 'raw':ROOT/'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt',
 'ldad1':ROOT/'artifacts/eda/levers_ldad_v1/raw_lam1/step-010000.pt',
 'ldad10':ROOT/'artifacts/eda/levers_ldad_v1/raw_lam10/step-010000.pt',
}
OUT=ROOT/'artifacts/eda/levers_mamba_long_pools_v1'
SHAPE=(19789,64,81,192)


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2)+'\n')
 os.replace(tmp,path)


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--pool',required=True,choices=tuple(CKPTS))
 p.add_argument('--batch',type=int,default=8)
 p.add_argument('--out',type=Path,default=OUT)
 p.add_argument('--limit',type=int,default=None,help='Resource smoke; staging remains resumable')
 a=p.parse_args()
 from d4mj.checkpoint import LEWM_BRIDGE_FORMAT, read_lewm_bridge, read_lewm_bundle
 from d4mj.config import config_from_dict
 from d4mj.data import load_joint_corpus
 from d4mj.world_api import ModelBundle
 ledger_path=HERE/'long_ledger.jsonl'
 ledger_meta=json.loads((HERE/'long_ledger.json').read_text())
 assert digest(ledger_path)==ledger_meta['ledger_sha256']
 rows=[json.loads(s) for s in ledger_path.read_text().splitlines()]
 assert len(rows)==SHAPE[0]
 ckpt=CKPTS[a.pool]
 ckpt_hash=digest(ckpt)
 src={'checkpoint_sha256':ckpt_hash,'ledger_sha256':ledger_meta['ledger_sha256'],
      'dataset_contract_sha256':json.loads((ROOT/'artifacts/lewm_m4_canonical/raw/dataset.json').read_text())['contract']['sha256'],
      'script_sha256':digest(Path(__file__)),'shape':list(SHAPE),'dtype':'float16','batch':a.batch}
 stage=a.out/f'{a.pool}.stage';final=a.out/a.pool
 if final.exists():raise SystemExit(f'final exists: {final}; refusing overwrite')
 stage.mkdir(parents=True,exist_ok=True)
 progress=stage/'progress.json'
 if progress.exists():
  old=json.loads(progress.read_text())
  assert old['contract']==src,'pool resume contract drift'
  start=old['next_row']
 else:
  start=0
  atomic_json(progress,{'contract':src,'next_row':0})
 raw=torch.load(ckpt,map_location='cpu',weights_only=False)
 payload=(read_lewm_bridge(ckpt) if raw.get('format')==LEWM_BRIDGE_FORMAT else
          read_lewm_bundle(ckpt) if 'format' in raw else raw)
 config=config_from_dict(payload['config'])
 bundle=ModelBundle.create(config)
 bundle.encoder.load_state_dict(payload['modules']['encoder'])
 enc=bundle.encoder.to('cuda').freeze()
 dataset=json.loads((ROOT/'artifacts/lewm_m4_canonical/raw/dataset.json').read_text())
 episodes,contract=load_joint_corpus(dataset['paths'],config)
 assert contract==dataset['contract']
 by_id={e.episode_id:e for e in episodes}
 for r in rows:
  e=by_id[r['episode_id']];s=r['start']
  assert e.split=='train' and e.uniform_eligible and s>=0 and s+64<=len(e)+1
  if r['terminal']:assert bool(e.terminated[-1]) and s+64==len(e)+1
 token_path=stage/'tokens.f16'
 expected=int(np.prod(SHAPE))*2
 if token_path.exists():
  assert token_path.stat().st_size==expected
  tokens=np.memmap(token_path,mode='r+',dtype=np.float16,shape=SHAPE)
 else:
  assert start==0
  tokens=np.memmap(token_path,mode='w+',dtype=np.float16,shape=SHAPE)
 stop=min(len(rows),a.limit or len(rows))
 assert start<=stop
 for b in range(start,stop,a.batch):
  chunk=rows[b:min(b+a.batch,stop)]
  frames=torch.as_tensor(np.stack([np.asarray(by_id[r['episode_id']].observations[r['start']:r['start']+64])
                                   for r in chunk])).to('cuda')
  with torch.no_grad():
   _,_,t,_,_=enc._hidden(frames)
   t=F.layer_norm(t.float(),(192,)).reshape(len(chunk),64,81,192).half().cpu().numpy()
  tokens[b:b+len(chunk)]=t
  tokens.flush()
  atomic_json(progress,{'contract':src,'next_row':b+len(chunk)})
  if b%512<a.batch:
   print(json.dumps({'stage':'encode','pool':a.pool,'done':b+len(chunk),'of':len(rows),
                     'peak_gpu_bytes':torch.cuda.max_memory_allocated()}),flush=True)
 if stop<len(rows):
  print(json.dumps({'stage':'paused','pool':a.pool,'done':stop,'of':len(rows)}),flush=True)
  return
 labels={k:[] for k in ('actions','reward_led','alive')}
 for r in rows:
  e=by_id[r['episode_id']];s=r['start']
  actions=torch.as_tensor(np.asarray(e.actions_taken[s:s+63])).long()
  rewards=torch.as_tensor(np.asarray(e.rewards[max(s-1,0):s+63]),dtype=torch.float64)
  led=rewards if s>0 else torch.cat([rewards.new_zeros(1),rewards])
  done=torch.as_tensor(np.asarray(e.terminated[s:s+63]),dtype=torch.bool)
  labels['actions'].append(actions)
  labels['reward_led'].append(led.float())
  labels['alive'].append(torch.cat([torch.ones(1,dtype=torch.bool),~done]))
 assert all(x.shape==(63,) for x in labels['actions'])
 assert all(x.shape==(64,) for x in labels['alive'])
 assert all(not bool(x[:-1].logical_not().any()) for x in labels['alive'])
 assert all(not bool(labels['alive'][i][-1]) for i,r in enumerate(rows) if r['terminal'])
 torch.save({'actions':torch.stack(labels['actions']),'reward_led':torch.stack(labels['reward_led']),
             'alive':torch.stack(labels['alive']),'terminal':torch.tensor([r['terminal'] for r in rows])},stage/'labels.pt')
 manifest={'status':'complete','contract':src,'tokens_sha256':digest(token_path),
           'labels_sha256':digest(stage/'labels.pt'),'row_count':len(rows),
           'terminal_count':sum(r['terminal'] for r in rows)}
 atomic_json(stage/'manifest.json',manifest)
 os.replace(stage,final)
 print(json.dumps({'stage':'complete','pool':a.pool,'tokens_sha256':manifest['tokens_sha256'],
                   'row_count':len(rows)}),flush=True)

if __name__=='__main__':main()
