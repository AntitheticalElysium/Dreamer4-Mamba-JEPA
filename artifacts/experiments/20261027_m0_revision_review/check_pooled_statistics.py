"""Reproduce existing prototypes and report matched descriptive summaries."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'artifacts/experiments/20261023_system_design/eda_pooled_separation.py'


def summary(values,r,pool):
    x=np.stack([pool(v,r) for v in values])
    d=np.sqrt(((x[:,None]-x[None])**2).mean(-1))
    pair=d[np.triu_indices(len(x),1)]
    collision=(d==0)&~np.eye(len(x),dtype=bool)
    return {'prototypes':len(x),'pairs':len(pair),'minimum':float(pair.min()),
            'median_pairwise':float(np.median(pair)),
            'exact_collision_pairs':int((pair==0).sum()),
            'prototypes_with_any_collision':int(collision.any(1).sum())}


def main():
    start=time.monotonic(); scope=runpy.run_path(str(SOURCE))
    result={'scope':'raw prototype separation, not learned encoder or historical readability',
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'pooling':{}}
    for r in (7,3,2,1):
        hud={str(k):summary(list(v.values()),r,scope['pool'])
             for k,v in scope['proto_hud'].items() if len(v)>2}
        result['pooling'][str(r)]={
            'map':summary(list(scope['proto_map'].values()),r,scope['pool']),
            'hud_row_7_only':hud,
            'hud_median_of_position_minima':float(np.median([v['minimum'] for v in hud.values()])),
            'hud_median_of_position_pairwise_medians':float(np.median([v['median_pairwise'] for v in hud.values()]))}
    # Exact reachability through the three temporal layers in the 6-layer timing model.
    t=np.arange(64); mask=(t[None]<=t[:,None])&(t[None]>=t[:,None]-4)
    reach=np.eye(64,dtype=bool)
    for _ in range(3): reach=(reach.astype(int)@mask.astype(int))>0
    result['temporal_mask']={'per_layer_lookback':4,'temporal_layers':3,
        'earliest_frame_reaching_frame_63':int(np.flatnonzero(reach[63])[0]),
        'effective_lookback':int(63-np.flatnonzero(reach[63])[0])}
    result['seconds']=time.monotonic()-start
    (OUT/'pooled_statistics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
