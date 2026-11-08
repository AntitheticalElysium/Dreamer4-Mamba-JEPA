"""Original cheap head, pre-crop C/S policy, isolated online learning state."""
import importlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saccade import model as M, train as TR

Q=importlib.import_module('artifacts.experiments.20261102_direct_chooser.direct')
HERE=Path(__file__).resolve().parent
TRACE=np.dtype([('target','<i4'),('episode','<i4'),('window','u1'),('uniform','u1'),
                ('C','u1'),('S','u1'),('probability','<f4')])


def config():return json.loads((HERE/'chooser.json').read_text())


def coarse(perception,frames):
    """Exact existing periphery/light operations; never compute a fine crop."""
    lead=frames.shape[:-3];f=frames.reshape(-1,63,63,3);x=f.float()/255
    v=M.out_map(perception.light(torch.cat([x.mean((1,2)),x.std((1,2))],-1)),perception.s)
    cells=f.reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float()
    cells=torch.round(F.adaptive_avg_pool2d(cells,perception.acuity)) if perception.acuity<7 else cells
    h=cells.reshape(len(f),81,-1)/127.5-1
    for layer,film in zip(perception.p_layers,perception.film):
        gain,shift=film(v).unsqueeze(1).chunk(2,-1);h=F.gelu(layer(h))*(1+gain)+shift
    p=M.out_map(perception.p_out(h),perception.s)
    return p.reshape(*lead,81,-1),v.reshape(*lead,-1)


def inputs(p,v,chunk,prefix):
    """All new target frames see only their own episode's preceding eight frames."""
    dev=p.device;n,L=p.shape[:2]
    target=torch.arange(prefix,L,device=dev);h=target[:,None]-torch.arange(8,0,-1,device=dev)[None]
    episode=torch.as_tensor(chunk['episode'],device=dev);valid=torch.as_tensor(chunk['valid'],device=dev)
    ok=valid[:,h]&(episode[:,h]==episode[:,target,None])
    ph=p[:,h].float()*ok[...,None,None];vh=v[:,h].float()*ok[...,None]
    pa=torch.as_tensor(chunk['prev_action'],device=dev);r=torch.as_tensor(chunk['reward'],device=dev)
    d=torch.as_tensor(chunk['terminated'],device=dev)
    ah=F.one_hot((pa[:,h]+1).clamp(0,17),18).float()*ok[...,None]
    ctx=torch.cat([vh,ah,(M.SYMLOG(r[:,h])*ok)[...,None],(d[:,h]*ok)[...,None],ok[...,None]],-1)
    return (ph.reshape(-1,8,81,16),ctx.reshape(-1,8*29),pa[:,prefix:].reshape(-1)),p[:,prefix:].reshape(-1,81,16).float()


class Controller:
    def __init__(self,model,cfg,policy,out_dir,device):
        self.model,self.cfg,self.policy,self.device=model,cfg,dict(policy),torch.device(device)
        assert self.policy['history']==8 and self.policy['uniform_floor']==.2
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(policy['head_seed']);self.head=Q.Head().to(self.device)
        assert sum(p.numel() for p in self.head.parameters())==101584
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'])
        self.gaze=torch.Generator(device=self.device).manual_seed(policy['gaze_seed'])
        self.batch=torch.Generator(device=self.device).manual_seed(policy['batch_seed'])
        self.frames=self.updates=self.rows=0;self.pending=None
        self.out=Path(out_dir);self.trace=self.out/'gaze.bin';self.log=self.out/'chooser.jsonl'
        if self.trace.exists() and self.trace.stat().st_size and not (self.out/'latest.pt').exists():
            raise RuntimeError('Gaze trace without a world checkpoint; inspect before restarting')
        self.centres=Q.CENTRES.to(self.device);self.win=Q.WIN.to(self.device)

    def state(self):
        assert self.pending is None
        return dict(config=self.policy,head=self.head.state_dict(),optimizer=self.opt.state_dict(),
                    gaze_rng=self.gaze.get_state(),batch_rng=self.batch.get_state(),
                    frames=self.frames,updates=self.updates,trace_rows=self.rows,
                    log_bytes=self.log.stat().st_size if self.log.exists() else 0)

    def load_state(self,state):
        assert state['config']==self.policy,'Fixation recipe differs on resume'
        self.head.load_state_dict(state['head']);self.opt.load_state_dict(state['optimizer'])
        self.gaze.set_state(state['gaze_rng'].cpu());self.batch.set_state(state['batch_rng'].cpu())
        self.frames,self.updates,self.rows=state['frames'],state['updates'],state['trace_rows']
        size=self.rows*TRACE.itemsize
        if size and (not self.trace.exists() or self.trace.stat().st_size<size):raise RuntimeError('Incomplete gaze trace')
        if self.trace.exists():
            with self.trace.open('r+b') as f:f.truncate(size)
        log_size=state['log_bytes']
        if log_size and (not self.log.exists() or self.log.stat().st_size<log_size):raise RuntimeError('Incomplete chooser log')
        if self.log.exists():
            with self.log.open('r+b') as f:f.truncate(log_size)

    @torch.no_grad()
    def prepare(self,chunk,streams):
        prefix=self.cfg['data']['context_prefix'];original=chunk['fixation'].copy()
        fr=torch.as_tensor(chunk['frames'],device=self.device);parts=[]
        flat=fr.flatten(0,1)
        for x in flat.split(self.policy['encoding_batch']):
            with TR.autocast(self.device,self.cfg):parts.append(coarse(self.model.perception,x))
        p=torch.cat([x[0] for x in parts]).reshape(*fr.shape[:2],81,16)
        v=torch.cat([x[1] for x in parts]).reshape(*fr.shape[:2],8)
        xx,y=inputs(p,v,chunk,prefix)
        eligible=torch.as_tensor(chunk['valid'][:,prefix:],device=self.device).flatten()&(xx[2]>=0)
        ids=torch.where(eligible)[0];self.pending=(tuple(x.detach() for x in xx),y.detach(),ids)
        rows=np.zeros(len(y),dtype=TRACE)
        rows['target']=chunk['target'][:,prefix:].reshape(-1);rows['episode']=chunk['episode'][:,prefix:].reshape(-1)
        uniform=(original[:,prefix:,0]-1)*7+original[:,prefix:,1]-1
        rows['uniform']=uniform.reshape(-1);rows['window']=rows['uniform'];rows['probability']=1/49
        cs=[];ss=[];entropy=[]
        for ix in ids.split(self.policy['scoring_batch']):
            x=tuple(z[ix] for z in xx)
            with torch.autocast(self.device.type,enabled=False):
                taken=self.head(x);noop=self.head((x[0],x[1],torch.zeros_like(x[2])))
                c=(taken-noop).square().sum(-1)[:,self.win].sum(-1)
                s=(y[ix]-taken).square().sum(-1)[:,self.win].sum(-1)
                assert torch.count_nonzero(c[x[2]==0])==0
                prob=Q.policy(c,s)
            if self.policy['policy']=='uniform_control':choice=torch.as_tensor(rows['uniform'][ix.cpu().numpy()].astype(np.int64),device=self.device)
            else:
                assert self.policy['policy']=='existing_combined_score_softmax'
                u=torch.rand(len(ix),1,device=self.device,generator=self.gaze,dtype=torch.float64)
                choice=torch.searchsorted(prob.double().cumsum(-1),u).squeeze(-1).clamp(max=48)
            ci=ix.cpu().numpy();rows['window'][ci]=choice.cpu().numpy()
            rows['C'][ci]=c.argmax(-1).cpu().numpy();rows['S'][ci]=s.argmax(-1).cpu().numpy()
            rows['probability'][ci]=prob.gather(1,choice[:,None]).squeeze(1).cpu().numpy()
            cs.append(float(c.mean()));ss.append(float(s.mean()));entropy.append(float(-(prob*prob.log()).sum(-1).mean()))
        chosen=torch.as_tensor(rows['window'].astype(np.int64),device=self.device)
        chunk['fixation'][:,prefix:]=self.centres[chosen].reshape(*chunk['fixation'][:,prefix:].shape).cpu().numpy()
        assert np.array_equal(chunk['fixation'][:,:prefix],original[:,:prefix])
        # Preserve the actual past observations when these frames become prefixes.
        for i,hist in enumerate(streams.hist):
            streams.hist[i]=[(*r[:3],*map(int,chunk['fixation'][i,-len(hist)+j])) for j,r in enumerate(hist)]
        with self.trace.open('ab') as f:rows.tofile(f)
        self.rows+=len(rows)
        self.description=dict(C=float(np.mean(cs)) if cs else 0.,S=float(np.mean(ss)) if ss else 0.,
            entropy=float(np.mean(entropy)) if entropy else math.log(49),
            window_histogram=np.bincount(rows['window'],minlength=49).tolist(),
            frames=len(rows),changed_fraction=float(np.mean(rows['window']!=rows['uniform'])))

    def learn(self,frames):
        xx,y,ids=self.pending;self.frames+=int(frames)
        goal=self.frames//self.policy['world_frames_per_head_update'];losses=[]
        total=self.cfg['p1']['frames']//self.policy['world_frames_per_head_update']
        while self.updates<goal:
            if not len(ids):self.updates+=1;continue
            step=self.updates;warm=min(1.,(step+1)/self.policy['warmup_updates'])
            frac=max(0.,(step-self.policy['warmup_updates'])/max(1,total-self.policy['warmup_updates']))
            lr=warm*(self.policy['minimum_learning_rate']+.5*(self.policy['learning_rate']-self.policy['minimum_learning_rate'])*(1+math.cos(math.pi*frac)))
            self.opt.param_groups[0]['lr']=lr
            ix=ids[torch.randint(len(ids),(self.policy['batch_size'],),device=self.device,generator=self.batch)]
            with torch.autocast(self.device.type,enabled=False):loss=(self.head(tuple(z[ix] for z in xx))-y[ix]).square().mean()
            self.opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.head.parameters(),self.policy['grad_clip']);self.opt.step()
            losses.append(float(loss.detach()));self.updates+=1
        self.pending=None
        with self.log.open('a') as f:f.write(json.dumps(dict(world_frames=self.frames,head_updates=self.updates,
            head_loss=float(np.mean(losses)) if losses else None,**self.description))+'\n')

    def close(self):
        torch.save(self.state(),self.out/'chooser_last.pt')
