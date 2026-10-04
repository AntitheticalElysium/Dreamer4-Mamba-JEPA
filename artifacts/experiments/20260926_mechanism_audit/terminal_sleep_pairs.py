"""Exact terminal-pair sleep/action contingency for the U/W bridge pool."""
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import torch

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260926_diagnosis')]
from sleep import asleep

def main():
    pool=torch.load(ROOT/'artifacts/eda/spatial_pool_v1/pool.pt',weights_only=False,mmap=True)
    want=defaultdict(list)
    for i,(eid,start) in enumerate(pool['ids']):
        if bool(pool['terminal'][i]):want[eid].append((i,start))
    pair=Counter();actions=Counter(); n=0
    for store in ('craftax_expert_store_v1','craftax_support_v2'):
        for shard in sorted((ROOT/'artifacts'/store).glob('shard-*.pt')):
            for e in torch.load(shard,weights_only=False,mmap=True)['episodes']:
                rows=want.get(e['episode_id'])
                if not rows:continue
                for i,start in rows:
                    flag=asleep(e['observations'][start+4:start+6]).tolist()
                    act=int(e['actions_taken'][start+4])
                    pair[f'{int(flag[0])}{int(flag[1])}']+=1
                    actions[f'{act}:{int(flag[0])}{int(flag[1])}']+=1
                    n+=1
    assert n==int(pool['terminal'].sum())
    out={'terminal_windows':n,'pair_prev_sleep_death_sleep':dict(pair),
         'actions_by_pair':dict(sorted(actions.items()))}
    (HERE/'terminal_sleep_pairs.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
