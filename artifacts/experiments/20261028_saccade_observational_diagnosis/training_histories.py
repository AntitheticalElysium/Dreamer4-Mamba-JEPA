"""Replay stream metadata only, preserving exact recorded fixation histories at actual snapshots."""
from .common import *


if __name__=='__main__':
    cfg=TR.load_cfg();episodes=D.load_split(cfg)['train'];s=D.Streams(episodes,cfg['data']['streams'],cfg['data']['context_prefix'],seed=1)
    original=D.gather;D.gather=lambda episodes,records,offsets=None:records
    histories=[[] for _ in range(cfg['data']['streams'])];frames=0;next_checkpoint=1;out={};checks=[]
    try:
        while frames<cfg['p1']['frames']:
            steps=64 if frames<cfg['p1']['frames']/2 else 256;records=s.next(steps)
            for stream,row in enumerate(records):
                for r in row[-steps:]:
                    if histories[stream] and r[1]!=histories[stream][-1][1]:histories[stream]=[]
                    histories[stream].append(r)
            frames+=steps*cfg['data']['streams']
            if frames>=next_checkpoint*cfg['p1']['frames']*cfg['p1']['checkpoint_every_fraction']:
                if next_checkpoint in (1,5,10):
                    ck=torch.load(RUN/f'ckpt_{next_checkpoint:02d}.pt',map_location='cpu',weights_only=False,mmap=True)
                    assert json.dumps(s.state(),sort_keys=True)==json.dumps(ck['streams'],sort_keys=True)
                    rows=[]
                    for i,((e,uid),t) in enumerate(zip(s.cur,s.t)):
                        history=histories[i] if histories[i] and histories[i][-1][1]==uid else []
                        assert len(history)==t and [r[2] for r in history]==list(range(t))
                        rows.append({'episode':e,'uid':uid,'t':t,'history':[tuple(r) for r in history]})
                    out[next_checkpoint]=rows;checks.append({'checkpoint':next_checkpoint,'frames':frames,'stream_state_exact':True})
                    torch.save(out,OUT/'training_histories.pt');save('training_histories_checks.json',checks);print('snapshot',next_checkpoint,'depths',[r['t'] for r in rows],flush=True)
                next_checkpoint+=1
    finally:D.gather=original
