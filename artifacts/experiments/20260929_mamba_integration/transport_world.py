"""Prototype: per-tile Mamba-2 with exact screen-shift transport of conv and SSM carries.

No new parameters. Correspondence is estimated causally from consecutive available states, as in
fcanvas. This is a bounded 7x9 view: new border cells start from zero; off-screen memory is dropped.
The prototype exists to test whether this is materially different from fcanvas before training.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder'),str(HERE)]
import tworld as T  # noqa: E402
import spatial as S  # noqa: E402
from scroll import estimate  # noqa: E402
from carry_transport import transport  # noqa: E402


class TransportFactored(T.Factored):
    def forward(self,x,shifts):
        b,t,n,d=x.shape
        assert n==82 and shifts.shape==(b,t) and self.time=='fmamba'
        h=self.n1(x).flatten(0,1)
        x=x+self.space(h,h,h,need_weights=False)[0].view(b,t,n,d)
        h=self.n2(x)
        carry=self.mix.initial(b*n,device=x.device,dtype=h.dtype)
        out=[]
        for k in range(t):
            if k:
                carry=transport(carry,shifts[:,k].long(),b)
            y,carry=self.mix.step(h[:,k].reshape(b*n,1,d),carry)
            out.append(y.view(b,n,d))
        x=x+torch.stack(out,1)
        return x+self.mlp(self.n3(x))


class TransportTWorld(T.TWorld):
    def __init__(self,head='corrg'):
        super().__init__(head,backbone='fmamba')
        for layer in self.layers:
            layer.__class__=TransportFactored

    def backbone_full(self,s,a):
        b,t=s.shape[:2]
        assert 1<=t<=S.W and a.shape==(b,t)
        x=torch.cat([self.action(a)[:,:,None],self.embed(s)],2)+self.space+self.time[:t,None]
        shifts=F.pad(estimate(s[:,:-1],s[:,1:]),(1,0)) if t>1 else a.new_zeros((b,1))
        for layer in self.layers:
            x=checkpoint(layer,x,shifts,use_reentrant=False) if torch.is_grad_enabled() else layer(x,shifts)
        x=self.norm(x)
        return x[:,:,1:],x[:,:,0]
