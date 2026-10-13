"""Does the code vary with actual RNG outcomes, at a fixed observed history?

Export native causal features briefly on GPU, then inspect posterior codes on
32 real successors per root on CPU. Labels are evaluation-only. No decoder fit.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import loss_intervention as L


def entropy(items):
    _,counts=np.unique(items,axis=0,return_counts=True)
    probability=counts/counts.sum()
    return float(-(probability*np.log2(probability)).sum())


def mi(codes,outcomes):
    return entropy(codes)+entropy(outcomes)-entropy(np.concatenate([codes,outcomes],1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('export','score'))
    p.add_argument('--updates',nargs='+',type=int,default=[0,500,1500])
    a=p.parse_args();torch.set_num_threads(4)
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    fan=torch.load(T.OUT/'real_futures/fan.pt',map_location='cpu',weights_only=False)
    assert fan['roots']==panel['roots']
    directory=T.OUT/'content';directory.mkdir(exist_ok=True)
    for update in a.updates:
        for arm in (('capacity',)if update==0 else ('capacity','rate')):
            source=T.OUT/f'{arm}_u{update:04d}.pt'
            checkpoint=torch.load(source,map_location='cpu',weights_only=False)
            dst=directory/f'{arm}_u{update:04d}.pt'
            source_hash=L.sha(source)
            if a.mode=='export':
                if dst.exists():
                    assert torch.load(dst,map_location='cpu',weights_only=False)['checkpoint_sha256']==source_hash
                    continue
                from d4mj.config import config_from_dict
                config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16).cuda().eval();world.load_state_dict(checkpoint['world'])
                channel=I.Innovation().cuda().eval();channel.load_state_dict(checkpoint['channel'])
                features,indices=[],[]
                for start in range(0,200,4):
                    frames=panel['tokens'][start:start+4,1:16].cuda().float()
                    actions=panel['actions'][start:start+4,1:16].cuda()
                    belief=I.native_belief(world,frames,actions,config)[:,-1]
                    features.append(belief.cpu())
                    with torch.no_grad():indices.append(channel.posterior(belief,frames[:,-1],panel['tokens'][start:start+4,16].cuda().float()).argmax(-1).cpu())
                L.atomic({'checkpoint_sha256':source_hash,'belief':torch.cat(features),'GPU_original_codes':torch.cat(indices),
                          'panel_sha256':L.sha(L.OUT/'panel.pt'),'sources':{str(Path(x).relative_to(L.ROOT)):L.sha(x)for x in (__file__,I.__file__,T.__file__)}},dst)
                del world,channel;torch.cuda.empty_cache()
                print(json.dumps({'stage':'content_export','arm':arm,'update':update}),flush=True)
            else:
                record=torch.load(dst,map_location='cpu',weights_only=False)
                assert record['checkpoint_sha256']==source_hash and record['panel_sha256']==L.sha(L.OUT/'panel.pt')
                channel=I.Innovation().eval();channel.load_state_dict(checkpoint['channel'])
                codes=[]
                with torch.no_grad():
                    original=channel.posterior(record['belief'],panel['tokens'][:,15].float(),panel['tokens'][:,16].float()).argmax(-1)
                    assert torch.equal(original,record['GPU_original_codes']),'CPU/GPU code assignment drift'
                    for root in range(200):
                        n=fan['tokens'].shape[1]
                        code=channel.posterior(record['belief'][root:root+1].expand(n,-1,-1),
                                               panel['tokens'][root:root+1,15].float().expand(n,-1,-1),fan['tokens'][root].float()).argmax(-1)
                        codes.append(code)
                codes=torch.stack(codes).numpy()
                hud=fan['hud'].numpy();occupancy=fan['layouts'].numpy().reshape(200,len(codes[0]),-1)
                counts=np.stack([[len(np.unique(c,axis=0)),entropy(c)]for c in codes])
                target=occupancy
                information=np.array([mi(c,o)for c,o in zip(codes,target)])
                rng=np.random.default_rng(20261013)
                null=np.mean([[mi(c,o[rng.permutation(len(o))])for c,o in zip(codes,target)]for _ in range(100)],0)
                flags=np.array([entropy(o)>0 for o in target])
                value={'checkpoint_sha256':source_hash,'fan_sha256':L.sha(T.OUT/'real_futures/fan.pt'),
                       'sources':record['sources'],'CPU_GPU_original_code_parity_exact':True,
                       'mean_distinct_codes_per_root':float(counts[:,0].mean()),
                       'roots_with_any_RNG_code_variation':int((counts[:,0]>1).sum()),
                       'mean_empirical_joint_code_entropy_bits_given_full_state':float(counts[:,1].mean()),
                       'roots_with_any_mob_occupancy_variation':int(flags.sum()),
                       'mob_variable_roots_with_constant_codes':int((flags&(counts[:,0]==1)).sum()),
                       'mob_code_MI_bits_mean':float(information.mean()),'permuted_MI_bits_mean':float(null.mean()),
                       'raw_per_root':{'distinct_codes':counts[:,0].tolist(),'code_entropy':counts[:,1].tolist(),
                                       'mob_code_MI':information.tolist(),'permuted_MI':null.tolist()},
                       'limits':'K32 plug-in entropy; upward-biased MI with permutation sensitivity; fixed full-state fans mix physical RNG and rendering RNG; no nuisance-separation claim'}
                L.atomic(value,directory/f'{arm}_u{update:04d}.json',True)
                L.atomic({'codes':torch.from_numpy(codes),'root_episode_ids':[r['episode_id']for r in panel['roots']]},directory/f'codes_{arm}_u{update:04d}.pt')
                print(json.dumps({k:v for k,v in value.items()if k not in ('raw_per_root','sources')}),flush=True)


if __name__=='__main__':main()
