"""Same-cell fovea reader tables and paired decoder summaries, frozen checkpoint."""
import time
import torch.nn.functional as F
from .common import *


def describe(y, pred, ep):
    return G.estimate(pred, np.ones(len(y), bool), ep)


def prepare(split):
    L=labels(split); enc=torch.load(OUT/f"real_{split}.pt",weights_only=False,mmap=True)
    grid=grids(L); off=np.r_[0,np.cumsum(L['ep_T']+1)]; hud=hud_values(L)
    tab={k:[] for k in ('p','g','v','px','low','cell','kind','terr','value','ep','night')}
    summary=[]
    for e,(r,ep) in enumerate(zip(enc,s1.episodes(L))):
        n=len(r['p']); cells=crop_cells(r['fix']); ar=np.arange(n)[:,None]
        frames=D.RP.episode_fields(ep.shard,ep.slot)['observations'].numpy()
        px=frames.reshape(n,9,7,9,7,3).transpose(0,1,3,2,4,5).reshape(n,81,147)[ar,cells]
        # Same RGB/channel order and rounded adaptive pooling as actual periphery input.
        q=torch.from_numpy(px.copy()).reshape(-1,7,7,3).permute(0,3,1,2).float()
        low=F.adaptive_avg_pool2d(q,3).round().byte().flatten(1).numpy()
        kgrid=np.pad(grid[e]['grid'].reshape(n,63),((0,0),(0,18)),constant_values=-1)
        terrain=np.pad(grid[e]['view'].reshape(n,63),((0,0),(0,18)),constant_values=-1)
        vals=np.full((n,81),-1); vals[:,HUD_CELLS]=hud[off[e]:off[e+1]]
        fields={'p':r['p'][ar,cells].half(),'g':r['g'].half(), 'v':r['v'][:,None].expand(-1,9,-1).half(),
                'px':px,'low':low.reshape(n,9,-1),'cell':cells,'kind':kgrid[ar,cells],
                'terr':terrain[ar,cells],'value':vals[ar,cells],
                'ep':np.full((n,9),e,np.int32),'night':np.repeat((L['s_light'][off[e]:off[e+1]]<.5)[:,None],9,1)}
        for key,z in fields.items():
            z=z if isinstance(z,torch.Tensor) else torch.from_numpy(z.copy())
            tab[key].append(z.reshape(n*9,*z.shape[2:]))
        if split=='dev':
            summary.append({**{k:r[k].numpy().ravel() for k in ('tsp_nll','rec_nll','tsp_mae','rec_mae','sigma_rgb')},
                            **{k:np.asarray(fields[k]).ravel() for k in ('cell','kind','value','ep','night')}})
    table={k:torch.cat(v) for k,v in tab.items()}; torch.save(table,OUT/f'crops_{split}.pt')
    if split=='dev':
        S={k:np.concatenate([r[k] for r in summary]) for k in summary[0]}
        masks={'all':np.ones(len(S['cell']),bool),'map':S['cell']<63,'HUD':S['cell']>=63,
               **{name:S['kind']==k for k,name in enumerate(KINDS)},
               **{name:S['cell']==c for name,c in zip(HUD,HUD_CELLS)}}
        out={}
        for name,base in masks.items():
            for ln,lm in (('all',True),('day',~S['night']),('night',S['night'])):
                m=base&lm
                if not m.any():continue
                out[f'{name}/{ln}']={'n':int(m.sum()),**{k:G.estimate(S[k],m,S['ep']) for k in ('tsp_nll','rec_nll','tsp_mae','rec_mae','sigma_rgb')},
                                    'paired_nll_gain':G.estimate(S['tsp_nll']-S['rec_nll'],m,S['ep'])}
        save('decoder_summary.json',out)
    print(split,len(table['cell']),flush=True)


if __name__=='__main__':
    prepare('train');prepare('dev')
