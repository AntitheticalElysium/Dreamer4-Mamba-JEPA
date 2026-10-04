"""Source-bound DEV-only 128-frame encoding for matched full-history versus rolling-six evaluation."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from long_pool import CKPTS
from d4mj.checkpoint import LEWM_BRIDGE_FORMAT, read_lewm_bridge, read_lewm_bundle
from d4mj.config import config_from_dict
from d4mj.data import load_joint_corpus
from d4mj.world_api import ModelBundle

OUT=ROOT/'artifacts/eda/levers_mamba_long_eval_pools_v1'
SHAPE=(128,128,81,192)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pool',required=True,choices=tuple(CKPTS))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--batch',type=int,default=4)
    a=p.parse_args()
    ledger_path=HERE/'long_eval_ledger.jsonl'
    ledger_meta=json.loads((HERE/'long_eval_ledger.json').read_text())
    assert ledger_meta['split']=='dev' and ledger_meta['length']==128
    assert digest(ledger_path)==ledger_meta['ledger_sha256']
    rows=[json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(rows)==SHAPE[0]
    ckpt=CKPTS[a.pool]
    raw=torch.load(ckpt,map_location='cpu',weights_only=False)
    payload=(read_lewm_bridge(ckpt) if raw.get('format')==LEWM_BRIDGE_FORMAT else
             read_lewm_bundle(ckpt) if 'format' in raw else raw)
    config=config_from_dict(payload['config'])
    bundle=ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload['modules']['encoder'])
    encoder=bundle.encoder.to('cuda').freeze()
    dataset_path=ROOT/'artifacts/lewm_m4_canonical/raw/dataset.json'
    dataset=json.loads(dataset_path.read_text())
    episodes,contract=load_joint_corpus(dataset['paths'],config)
    assert contract==dataset['contract']
    by_id={e.episode_id:e for e in episodes}
    assert len({r['episode_id'] for r in rows})==len(rows)
    for r in rows:
        e=by_id[r['episode_id']]
        assert e.split=='dev' and r['start']>=0 and r['start']+128<=len(e)+1
    a.out.mkdir(parents=True,exist_ok=True)
    final=a.out/a.pool
    assert not final.exists(),f'eval pool exists: {final}'
    stage=a.out/f'{a.pool}.stage'
    assert not stage.exists(),f'incomplete stage exists: {stage}'
    stage.mkdir()
    token_path=stage/'tokens.f16'
    tokens=np.memmap(token_path,mode='w+',dtype=np.float16,shape=SHAPE)
    torch_actions=[]
    for start in range(0,len(rows),a.batch):
        chunk=rows[start:start+a.batch]
        frames=torch.as_tensor(np.stack([np.asarray(by_id[r['episode_id']].observations[r['start']:r['start']+128])
                                         for r in chunk])).to('cuda')
        with torch.no_grad():
            _,_,patch,_,_=encoder._hidden(frames)
            patch=F.layer_norm(patch.float(),(192,)).reshape(len(chunk),128,81,192).half().cpu().numpy()
        tokens[start:start+len(chunk)]=patch
        for r in chunk:
            e=by_id[r['episode_id']]
            acts=torch.as_tensor(np.asarray(e.actions_taken[r['start']:r['start']+127])).long()
            assert acts.shape==(127,)
            torch_actions.append(acts)
        tokens.flush()
        print(json.dumps({'stage':'encode','pool':a.pool,'done':start+len(chunk),'of':len(rows),
                          'peak_gpu_bytes':torch.cuda.max_memory_allocated()}),flush=True)
    atomic_save({'actions':torch.stack(torch_actions),'rows':rows},stage/'labels.pt')
    manifest={'status':'complete','split':'dev','final_opened':False,
              'ledger_sha256':ledger_meta['ledger_sha256'],
              'checkpoint_sha256':digest(ckpt),
              'dataset_contract_sha256':dataset['contract']['sha256'],
              'source_sha256':digest(Path(__file__)),'shape':list(SHAPE),
              'tokens_sha256':digest(token_path),'labels_sha256':digest(stage/'labels.pt')}
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    os.replace(stage,final)
    print(json.dumps({'stage':'complete','pool':a.pool,'tokens_sha256':manifest['tokens_sha256']}),flush=True)


if __name__=='__main__':
    main()
