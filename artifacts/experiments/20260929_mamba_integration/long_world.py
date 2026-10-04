"""Length-64 continuation of the six-frame per-tile fmamba+corrg world.

The model parameters are unchanged from the short checkpoint. The six learned time embeddings repeat
periodically: positions 0..5 are bit-identical to the short model, and no new positional parameters
can encode the training depth. Spatial attention stays within a frame; Mamba-2 mixes time per token.
Optional fixed six-frame resets provide a same-frames/same-loss control for long recurrence.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
import tworld as T  # noqa: E402
import spatial as S  # noqa: E402

LENGTH=64
MAX_EVAL_LENGTH=128

class LongTWorld(T.TWorld):
    def __init__(self,reset_every=None):
        super().__init__('corrg',backbone='fmamba')
        assert reset_every in (None,S.W)
        self.reset_every=reset_every

    def backbone_full(self,s,a):
        b,t=s.shape[:2]
        assert 1<=t<=MAX_EVAL_LENGTH and a.shape==(b,t)
        pos=torch.arange(t,device=s.device)%S.W
        x=torch.cat([self.action(a)[:,:,None],self.embed(s)],2)+self.space+self.time[pos,None]
        for layer in self.layers:
            if self.reset_every is None:
                x=checkpoint(layer,x,None,use_reentrant=False) if torch.is_grad_enabled() else layer(x,None)
            else:
                chunks=[]
                for start in range(0,t,self.reset_every):
                    part=x[:,start:start+self.reset_every]
                    part=checkpoint(layer,part,None,use_reentrant=False) if torch.is_grad_enabled() else layer(part,None)
                    chunks.append(part)
                x=torch.cat(chunks,1)
        x=self.norm(x)
        return x[:,:,1:],x[:,:,0]


def rollout_loss(world,s,actions):
    """The same teacher L1 + final depth-two suffix L1 as T.rollout_losses, extended to L=64."""
    assert s.ndim==4 and s.shape[1:]==(LENGTH,81,192)
    assert actions.shape==(len(s),LENGTH-1)
    a=F.pad(actions,(0,1))
    pred,_,_=world(s,a)
    teacher=(pred[:,:LENGTH-1]-s[:,1:]).abs().mean()
    anchor=LENGTH-3
    first=pred[:,anchor]
    second,_,_=world(torch.cat([s[:,:anchor+1],first[:,None].to(s.dtype)],1),a[:,:anchor+2])
    generated=torch.stack([first,second[:,anchor+1]],1)
    return teacher+(generated-s[:,anchor+1:]).abs().mean()
