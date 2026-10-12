"""Report resumed/uninterrupted state equality and numerical weight drift."""
import argparse
import json
from pathlib import Path
import torch
import prefix_intervention as P


def tensor_differences(left,right,path='root'):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype,path
        assert torch.isfinite(left).all() and torch.isfinite(right).all(),path
        maximum=float((left.double()-right.double()).abs().max()) if left.numel() else 0.
        return {'tensor_count':1,'different_tensors':int(not torch.equal(left,right)),
                'max_abs':maximum,'max_abs_path':path}
    if isinstance(left,dict):
        assert left.keys()==right.keys(),path
        sub=[tensor_differences(left[k],right[k],path+'/'+str(k)) for k in left]
    elif isinstance(left,(tuple,list)):
        assert type(left)==type(right) and len(left)==len(right),path
        sub=[tensor_differences(l,r,path+'/'+str(i)) for i,(l,r) in enumerate(zip(left,right))]
    else:
        assert left==right,path
        return {'tensor_count':0,'different_tensors':0,'max_abs':0.,'max_abs_path':None}
    largest=max(sub,key=lambda x:x['max_abs']) if sub else {'max_abs':0.,'max_abs_path':None}
    return {'tensor_count':sum(x['tensor_count'] for x in sub),'different_tensors':sum(x['different_tensors'] for x in sub),
            'max_abs':largest['max_abs'],'max_abs_path':largest['max_abs_path']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('resumed',type=Path);parser.add_argument('reference',type=Path)
    parser.add_argument('--out',default='generated_prefix_resume_proof.json')
    parser.add_argument('--parameter-tolerance',type=float,default=1e-4)
    args=parser.parse_args()
    left=torch.load(args.resumed,weights_only=False,map_location='cpu')
    right=torch.load(args.reference,weights_only=False,map_location='cpu')
    assert left['contract_id']==right['contract_id'] and left['update']==right['update']
    exact={key:P.LI.restored_equal(left[key],right[key]) for key in
           ('order','exposure','torch_rng','cuda_rng','ledger_chain','sampling')}
    assert all(exact.values()),exact
    differences={key:tensor_differences(left[key],right[key],key) for key in ('worlds','oracles','optimizers')}
    assert max(differences[key]['max_abs'] for key in ('worlds','oracles'))<=args.parameter_tolerance,differences
    result={'status':'pass','contract_id':left['contract_id'],'update':left['update'],
            'exact_sampler_rng_ledger':exact,'numerical_state_differences':differences,
            'parameter_tolerance':args.parameter_tolerance,
            'bit_exact_weights':all(differences[key]['different_tensors']==0 for key in ('worlds','oracles')),
            'source_artifact_sha256':{str(f):P.LI.sha(f) for f in (Path(__file__),args.resumed,args.reference)},
            'scope':'short actual CUDA resume; exact deserialization/streams, numerical endpoints explicitly reported'}
    P.LI.atomic(result,P.OUT/args.out,True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
