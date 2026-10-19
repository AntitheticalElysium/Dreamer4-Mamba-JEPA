"""CPU-only exact full-budget sampler preview before any model updates."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import prefix_intervention as P


def main():
    pool = torch.load(P.TW.POOLS['rawlong']/'labels.pt',weights_only=False,map_location='cpu',mmap=True)
    valid = torch.from_numpy(np.load(P.OUT/'layouts_valid.npy'))
    main = torch.where(~pool['terminal']&valid)[0]
    terminal = torch.where(pool['terminal']&valid)[0]
    order = torch.Generator().manual_seed(11)
    exposure = torch.Generator().manual_seed(20261012)
    count = {str(p):{'terminal':0,'main':0} for p in P.EXPOSURES}
    minimum,maximum,ledger = 999,-1,'initial'
    for _ in range(3000):
        term = torch.rand(16,generator=order) < P.TW.TERMINAL_SHARE
        rows = torch.where(term,terminal[torch.randint(len(terminal),(16,),generator=order)],
                           main[torch.randint(len(main),(16,),generator=order)])
        starts = torch.where(term,48,torch.randint(0,49,(16,),generator=order))
        targets = torch.where(term,63,torch.randint(30,64,(16,),generator=exposure))
        p = P.EXPOSURES[int(torch.randint(len(P.EXPOSURES),(),generator=exposure))]
        count[str(p)]['terminal'] += int(term.sum()); count[str(p)]['main'] += int((~term).sum())
        minimum = min(minimum,int((targets-p-15).min())); maximum = max(maximum,int(targets.max()))
        raw = rows.numpy().tobytes()+starts.numpy().tobytes()+targets.numpy().tobytes()+p.to_bytes(4,'little')
        ledger = hashlib.sha256((b'' if ledger=='initial' else bytes.fromhex(ledger))+raw).hexdigest()
    assert minimum==0 and maximum==63
    result = {'status':'complete','source_inputs_sha256':{
        str(f.relative_to(P.ROOT)):P.LI.sha(f) for f in
        (Path(__file__),Path(P.__file__),P.TW.POOLS['rawlong']/'labels.pt',P.OUT/'layouts_valid.npy')},
        'updates':3000,'windows':16,'order_seed':11,'exposure_seed':20261012,
        'valid_main_rows':len(main),'valid_terminal_rows':len(terminal),
        'minimum_prefix_start':minimum,'maximum_target':maximum,
        'exposure_counts':count,'terminal_share_by_exposure':{
            p:v['terminal']/(v['terminal']+v['main']) for p,v in count.items()},
        'teacher_targets_per_arm':3000*16*15,'endpoint_targets_per_arm':3000*16,
        'ledger_chain':ledger,
        'scope':'exact CPU dry-run of declared sampler; numerical training or semantics not tested'}
    path = P.OUT/'generated_prefix_sampler_preview.json'
    P.LI.atomic(result,path,True)
    print(path)


if __name__ == '__main__': main()
