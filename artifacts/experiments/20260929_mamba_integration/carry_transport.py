"""Transport both per-screen-slot Mamba-2 carries under a one-tile view shift.

Slot 0 is the action token; slots 1..63 are 7x9 map cells; slots 64..81 are HUD tokens.
For new map cells with no previous correspondence, initialize both convolution and SSM carry to zero.
This is a bounded-view mechanism: content that leaves the 7x9 view is deliberately discarded.
"""
import torch
from d4mj.mamba_recurrence import MambaCarry

SHIFTS=((0,0),(-1,0),(1,0),(0,-1),(0,1))
SLOTS=82


def _source_lut(device):
    lut=torch.empty(len(SHIFTS),SLOTS,dtype=torch.long,device=device)
    for code,(dr,dc) in enumerate(SHIFTS):
        lut[code,0]=0
        lut[code,64:]=torch.arange(64,SLOTS,device=device)
        for r in range(7):
            for c in range(9):
                lut[code,1+r*9+c]=1+(r+dr)*9+(c+dc) if 0<=r+dr<7 and 0<=c+dc<9 else -1
    return lut


def transport(carry:MambaCarry,shift:torch.Tensor,batch:int)->MambaCarry:
    """shift [B] in {none,up,down,left,right}; carry first dimension B*82.

    One indexed gather per carry rather than materializing all five candidate shifts. Invalid new
    border slots gather an arbitrary old slot but are then zeroed, with zero gradient to that slot.
    """
    if shift.shape!=(batch,) or shift.dtype!=torch.long or bool(((shift<0)|(shift>=len(SHIFTS))).any()):
        raise ValueError('invalid per-sample view shift')
    source=_source_lut(shift.device).index_select(0,shift)
    valid=source>=0
    flat=(torch.arange(batch,device=shift.device)[:,None]*SLOTS+source.clamp_min(0)).flatten()
    def move(t):
        assert t.shape[0]==batch*SLOTS and t.device==shift.device
        out=t.index_select(0,flat).reshape(batch,SLOTS,*t.shape[1:])
        mask=valid.reshape(batch,SLOTS,*([1]*(t.ndim-1)))
        return torch.where(mask,out,torch.zeros_like(out)).flatten(0,1)
    return MambaCarry(move(carry.conv),move(carry.ssm))
