"""CPU-only multi-key successor fan. Simulator fields are evaluation-only.

Each existing DEV root receives 32 independent one-step futures for its recorded
action. This estimates full-state-conditional variation, not the additional
uncertainty of a partially observed model. Every root is an atomic resume unit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','4')
import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
import build as B
import loss_intervention as L

DIRECTORY=B.OUT/'learned_innovation/real_futures'


def cpu_encoder(path):
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMEncoder
    payload=read_lewm_bundle(path)  # Full canonical source/execution checks remain binding.
    enc=LeWMEncoder(config_from_dict(payload['config']))
    enc.load_state_dict(payload['modules']['encoder']);enc.freeze()
    @torch.no_grad()
    def encode(frames):
        return torch.cat([torch.nn.functional.layer_norm(enc._hidden(frames[j:j+64,None])[2].float(),(192,)).half()
                          for j in range(0,len(frames),64)])
    return encode


def layouts(states):
    player=np.asarray(states.player_position)
    out=np.zeros((len(player),63,4),np.uint8)
    for channel,group in enumerate((states.zombies,states.cows,states.skeletons,states.arrows)):
        local=np.asarray(group.position)-player[:,None]+np.array([3,4])
        present=np.asarray(group.mask)&(local>=0).all(-1)&(local<[7,9]).all(-1)
        f,k=np.nonzero(present)
        out[f,local[f,k,0]*9+local[f,k,1],channel]=1
    return torch.from_numpy(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys',type=int,default=32)
    p.add_argument('--limit',type=int,default=200)
    a=p.parse_args();assert a.keys>=2
    torch.set_num_threads(4)
    panel=torch.load(B.OUT/'panel.pt',map_location='cpu',weights_only=False)
    contract={'format':'innovation_real_successor_fan_v1','keys':a.keys,'seed':20261013,
              'panel_sha256':L.sha(B.OUT/'panel.pt'),'root_count':len(panel['roots']),
              'sources':{str(Path(x).relative_to(L.ROOT)):L.sha(x)
                         for x in (__file__,B.__file__,B.RP.__file__,L.__file__)},
              'condition':'exact current full simulator state; logged root action; independent RNG keys',
              'scope':'evaluation only; no simulator field reaches learned-innovation training',
              'encoder':'frozen Raw encoder, CPU float32 then per-token LayerNorm and fp16 as original panel'}
    contract['encoder_checkpoint_sha256']=L.sha(L.TW.S.CHECKPOINT)
    # B.encoder uses teval's frozen Raw checkpoint; pin it separately below.
    import teval as T
    contract['raw_encoder_checkpoint_sha256']=L.sha(T.ENCODERS['raw'])
    recorded=torch.load(T.ENCODERS['raw'],map_location='cpu',weights_only=False)['sources']['execution']
    torch.set_float32_matmul_precision(recorded['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=recorded['cuda_matmul_tf32']
    torch.backends.cudnn.allow_tf32=recorded['cudnn_tf32']
    torch.backends.cudnn.benchmark=recorded['cudnn_benchmark']
    torch.backends.cudnn.deterministic=recorded['cudnn_deterministic']
    torch.use_deterministic_algorithms(recorded['deterministic_algorithms'])
    os.environ['TRITON_F32_DEFAULT']=recorded['triton_f32_default']
    contract['encoder_execution']=recorded
    cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    cp=DIRECTORY/'contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:L.atomic(contract,cp,True)
    env,params,_,step,render=B.RP.env_and_render()
    @jax.jit
    def fan(s,action,keys):
        _,states,_,_,_=jax.vmap(lambda key:env.step(key,s,action,params))(keys)
        return states,jax.vmap(render)(states)
    encode=None
    for i,root in enumerate(panel['roots'][:a.limit]):
        dst=DIRECTORY/f'root_{i:03d}.pt'
        if dst.exists():
            old=torch.load(dst,map_location='cpu',weights_only=False)
            assert old['contract_id']==cid and old['root']==root
            continue
        fields=B.RP.episode_fields(root['shard'],root['slot'])
        assert fields['split']=='dev' and fields['episode_id']==root['episode_id']
        state=B.RP.advance_to(root['shard'],root['slot'],root['t'])
        assert np.array_equal(np.asarray(render(state)),fields['observations'][root['t']].numpy())
        action=fields['actions_taken'][root['t']]
        assert int(action)==int(panel['actions'][i,15])
        recorded_key=B.RP._slot_keys(root['shard'],root['slot'])[1][root['t']]
        _,recorded,_,_,_=step(recorded_key,state,jnp.asarray(int(action)))
        recorded_pixels=np.asarray(render(recorded))
        mismatch=int(np.abs(recorded_pixels.astype(int)-fields['observations'][root['t']+1].numpy().astype(int)).max())
        assert mismatch==0, (i,mismatch)
        keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(20261013),i),a.keys)
        states,pixels=fan(state,jnp.asarray(int(action)),keys)
        if encode is None:encode=cpu_encoder(T.ENCODERS['raw'])
        token=encode(torch.from_numpy(np.asarray(pixels).copy()))
        original=encode(torch.from_numpy(recorded_pixels.copy())[None])[0]
        encoder_diff=float((original.float()-panel['tokens'][i,16].float()).abs().max())
        assert encoder_diff<=.002, encoder_diff
        hud=torch.from_numpy(np.stack([np.asarray(getattr(states,'player_'+name))
                                     for name in ('health','food','drink','energy')],-1).copy()).float()
        L.atomic({'contract_id':cid,'root':root,'action':int(action),'keys':torch.from_numpy(np.asarray(keys).copy()),
                  'tokens':token,'layouts':layouts(states),'hud':hud,'player_position':torch.from_numpy(np.asarray(states.player_position).copy()),
                  'recorded_replay_pixel_maxdiff':mismatch,'original_encoder_maxabs':encoder_diff},dst)
        if i%10==0:print(json.dumps({'stage':'fan','roots':i+1,'total':len(panel['roots']),
                                     'original_encoder_maxabs':encoder_diff}),flush=True)
    paths=[DIRECTORY/f'root_{i:03d}.pt' for i in range(len(panel['roots']))]
    if all(x.exists()for x in paths):
        rows=[torch.load(x,map_location='cpu',weights_only=False)for x in paths]
        assert all(x['contract_id']==cid for x in rows)
        payload={'contract_id':cid,'roots':panel['roots'],'tokens':torch.stack([r['tokens']for r in rows]),
                 'layouts':torch.stack([r['layouts']for r in rows]),'hud':torch.stack([r['hud']for r in rows]),
                 'source_hashes':[L.sha(x)for x in paths],
                 'replay_maxdiff':max(r['recorded_replay_pixel_maxdiff']for r in rows),
                 'encoder_maxabs':max(r['original_encoder_maxabs']for r in rows)}
        L.atomic(payload,DIRECTORY/'fan.pt')
        print(json.dumps({'stage':'fan_done','roots':len(rows),'keys':a.keys,
                          'replay_maxdiff':payload['replay_maxdiff'],'encoder_maxabs':payload['encoder_maxabs']}),flush=True)


if __name__=='__main__':main()
