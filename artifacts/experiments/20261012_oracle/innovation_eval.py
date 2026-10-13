"""Resume-safe inference for compact innovations; privileged posterior is labelled.

Prior modes receive observed/generated history and actions only. Posterior modes
receive the held-out recorded successor as a conditional-completion reference.
Past innovations are retained within a sampled trajectory, never re-sampled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TRITON_F32_DEFAULT','ieee')
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import loss_intervention as L
from d4mj.config import config_from_dict
from d4mj.train import autocast_context

METHODS=('posterior','zero_current','shuffle_current','prior_mode','prior_sample')


@torch.no_grad()
def historical(channel,belief,frames):
    logits=channel.posterior(belief[:,:-1],frames[:,:-1],frames[:,1:])
    return logits.argmax(-1)


@torch.no_grad()
def predict(world,channel,frames,actions,config,method,history_codes=None,*,successor=None,
            generator=None,shuffled_codes=None):
    belief=I.native_belief(world,frames,actions,config) if channel is not None else None
    code=None;rate=None
    if channel is None:
        delta=None
    else:
        past=historical(channel,belief,frames) if history_codes is None else history_codes
        assert past.shape==(*frames.shape[:1],frames.shape[1]-1,I.SLOTS)
        if method=='posterior':
            assert successor is not None
            code=channel.posterior(belief[:,-1],frames[:,-1],successor).argmax(-1)
            logits,_=channel.prior(belief[:,-1],code)
            rate=F.cross_entropy(logits.flatten(0,-2),code.flatten(),reduction='none').view(len(code),I.SLOTS).sum(-1)/torch.log(torch.tensor(2.,device=frames.device))
        elif method=='shuffle_current':
            assert shuffled_codes is not None
            code=shuffled_codes
        elif method=='zero_current':
            code=torch.zeros((len(frames),I.SLOTS),device=frames.device,dtype=torch.long)
        else:
            assert method in ('prior_mode','prior_sample') and successor is None, 'prior received a future'
            _,code=channel.prior(belief[:,-1],sample=method=='prior_sample',generator=generator)
        codes=torch.cat([past,code[:,None]],1)
        weights=F.one_hot(codes,I.CHOICES).float()
        if method=='zero_current':weights[:,-1]=0
        delta=channel.condition(belief,weights)
    with I.conditioned(world,delta),autocast_context(config):
        result=world(frames,actions)[0][:,-1].float()
    assert torch.isfinite(result).all()
    return result,code,rate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,nargs='+',default=[0,500,1500,3000])
    p.add_argument('--arms',nargs='+',default=list(T.ARMS))
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--horizon',type=int,choices=(1,16),default=1)
    p.add_argument('--limit-batches',type=int)
    p.add_argument('--run-directory',type=Path,default=T.OUT)
    a=p.parse_args();assert a.samples>=2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    tok,actions=panel['tokens'].float(),panel['actions']
    for update in a.updates:
        directory=a.run_directory/f'eval_u{update:04d}_h{a.horizon:02d}'
        directory.mkdir(parents=True,exist_ok=True)
        for arm in a.arms:
            checkpoint=a.run_directory/f'{arm}_u{update:04d}.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['arm']==arm and saved['update']==update
            world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16).cuda().eval()
            world.load_state_dict(saved['world'])
            channel=None if arm=='control'else I.Innovation().cuda().eval()
            if channel is not None:channel.load_state_dict(saved['channel'])
            methods=('control',) if channel is None else (METHODS if a.horizon==1 else ('posterior','prior_mode','prior_sample'))
            # The complete posterior/shuffle assignment is computed before
            # batching, so shuffling is across episodes and independent of batch.
            posterior_codes=None
            if channel is not None:
                codes=[]
                for start in range(0,len(tok),a.batch):
                    n=min(a.batch,len(tok)-start)
                    frames=tok[start:start+a.batch,1:16].cuda().repeat_interleave(a.samples,0)
                    ac=actions[start:start+a.batch,1:16].cuda().repeat_interleave(a.samples,0)
                    belief=I.native_belief(world,frames,ac,config)
                    with torch.no_grad():
                        codes.append(channel.posterior(belief[:,-1],frames[:,-1],tok[start:start+a.batch,16].cuda().repeat_interleave(a.samples,0)).argmax(-1).cpu().view(n,a.samples,I.SLOTS)[:,0])
                posterior_codes=torch.cat(codes)
            shuffled=torch.arange(len(tok)).roll(1)
            contract={'format':'learned_innovation_evaluation_v1','training_contract_id':saved['contract_id'],
                      'checkpoint_sha256':L.sha(checkpoint),'panel_sha256':L.sha(L.OUT/'panel.pt'),
                      'source_hashes':{str(Path(x).relative_to(L.ROOT)):L.sha(x)
                                       for x in (__file__,I.__file__,T.__file__,L.__file__,L.TW.__file__)},
                      'arm':arm,'update':update,'samples':a.samples,'batch':a.batch,'horizon':a.horizon,
                      'methods':methods,'seed':20261013,'precision':'BF16 model, FP32 outputs; persistent Triton kernels; TF32 off',
                      'truth':'reused200DEVepisodes; posterior sees recorded next tokens; prior sees no future',
                      'history':'15 frames; past codes retained inside each generated branch',
                      'numerical_control':'same repeated-root batch shape for ALL methods, including deterministic control; stored deterministic branches compressed only after inference',
                      'shuffle':'one-root cyclic derangement; current code only; historical codes unchanged'}
            contract=json.loads(json.dumps(contract));cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
            cp=directory/f'contract_{arm}.json'
            if cp.exists():assert json.loads(cp.read_text())==contract
            else:L.atomic(contract,cp,True)
            for method in methods:
                dst=directory/f'pred_{arm}_{method}.pt'
                if dst.exists():
                    old=torch.load(dst,map_location='cpu',weights_only=False)
                    assert old['contract_id']==cid
                    print(json.dumps({'stage':'inference_resume','update':update,'arm':arm,'method':method}),flush=True)
                    continue
                samples=a.samples if method=='prior_sample'else 1
                compute_samples=a.samples
                chunks=[]
                for start in range(0,len(tok),a.batch):
                    if a.limit_batches is not None and start//a.batch>=a.limit_batches:return
                    batch_path=directory/'batches'/f'{arm}_{method}_{start:03d}.pt'
                    if batch_path.exists():
                        chunk=torch.load(batch_path,map_location='cpu',weights_only=False)
                        assert chunk['contract_id']==cid
                        chunks.append(chunk);continue
                    stop=min(start+a.batch,len(tok));n=stop-start
                    frames=tok[start:stop,1:16].cuda().repeat_interleave(compute_samples,0)
                    ac=actions[start:stop,1:16].cuda().repeat_interleave(compute_samples,0)
                    belief=I.native_belief(world,frames,ac,config) if channel is not None else None
                    old_codes=historical(channel,belief,frames) if channel is not None else None
                    values,codes_list,rates=[],[],[]
                    generator=torch.Generator(device='cuda').manual_seed(20261013+start*104729)
                    for depth in range(a.horizon):
                        ac=actions[start:stop,depth+1:depth+16].cuda().repeat_interleave(compute_samples,0)
                        successor=tok[start:stop,depth+16].cuda().repeat_interleave(compute_samples,0) if method=='posterior'else None
                        sh=posterior_codes[shuffled[start:stop]].cuda().repeat_interleave(compute_samples,0) if method=='shuffle_current'else None
                        value,code,rate=predict(world,channel,frames,ac,config,method,old_codes,
                                               successor=successor,generator=generator,shuffled_codes=sh)
                        branches=value.cpu().view(n,compute_samples,81,192)
                        if samples==1:
                            assert torch.equal(branches,branches[:,:1].expand_as(branches)), 'deterministic repeats differ'
                        values.append(branches[:,:samples])
                        if code is not None:
                            codes_list.append(code.cpu().view(n,compute_samples,I.SLOTS)[:,:samples])
                            old_codes=torch.cat([old_codes[:,1:],code[:,None]],1)
                        if rate is not None:rates.append(rate.cpu().view(n,compute_samples)[:,:samples])
                        frames=torch.cat([frames[:,1:],value[:,None]],1)
                    chunk={'contract_id':cid,'start':start,'stop':stop,
                           'predictions':torch.stack(values,2),
                           'codes':torch.stack(codes_list,2) if codes_list else None,
                           'assignment_nll_bits':torch.stack(rates,2) if rates else None}
                    L.atomic(chunk,batch_path);chunks.append(chunk)
                    if start%(a.batch*10)==0:
                        print(json.dumps({'stage':'inference','update':update,'arm':arm,'method':method,
                                          'roots':stop,'samples':samples,'horizon':a.horizon}),flush=True)
                payload={'contract_id':cid,'roots':panel['roots'],'checkpoint_sha256':contract['checkpoint_sha256'],
                         'predictions':torch.cat([c['predictions']for c in chunks]),
                         'codes':torch.cat([c['codes']for c in chunks]) if channel is not None else None,
                         'assignment_nll_bits':torch.cat([c['assignment_nll_bits']for c in chunks]) if method=='posterior'else None}
                L.atomic(payload,dst)
                print(json.dumps({'stage':'inference_done','update':update,'arm':arm,'method':method,'shape':list(payload['predictions'].shape)}),flush=True)
            del world,channel;torch.cuda.empty_cache()


if __name__=='__main__':main()
