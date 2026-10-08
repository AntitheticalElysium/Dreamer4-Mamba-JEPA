"""Run unchanged teval mathematics with JSON-stable immutable exports.

Historical integer horizon keys become strings in JSON. Sorting the Python
integer keys versus reloaded string keys produced different digests for identical
reports. Keep all trainer/Store source hashes intact; normalize only JSON exports
at this compatibility entrypoint. Reuse completed evidence only after verifying
its original sources, weights, runtime, actual input tensors, cached report and
exported raw rows. New evaluations bind this entrypoint in their own source ledger.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
L=HERE.parent/'20260927_levers'
sys.path.insert(0,str(L))
sys.path.insert(0,str(HERE.parent/'20260926_diagnosis'))
sys.path.insert(0,str(HERE.parent/'20260921_readout_ladder'))
import teval as T
import h16_resume as R
import spatial as S

original_atomic_json=R.atomic_json


def json_export(path,value,immutable=False):
    # Normalize exactly as the existing JSON serialization does. No rounding,
    # tolerance widening or changes to the hashed training/resume contracts.
    return original_atomic_json(path,json.loads(json.dumps(value)),immutable=immutable)


def completed(path,args,inputs_by_pool):
    st=torch.load(path,map_location='cpu',weights_only=False)
    pool=st['args']['pool']
    name=st['name']
    tag=name+(f'__snap_{args.snap.stem}'if args.snap else '')+('__hard'if args.hard else '')
    tag+=(''if args.window==5 else f'__w{args.window}')
    report=L/'evals'/f'{tag}__readout_v2.json'
    if not report.exists():return False
    inputs=inputs_by_pool.get(pool)
    if inputs is None:
        meta,train,seeds=T.split();cache=T.build_cache(pool,torch.device('cpu'))
        tensors={**{f'cache_{k}':v for k,v in cache.items()if isinstance(v,torch.Tensor)},
                 **{f'meta_{k}':v for k,v in meta.items()if isinstance(v,torch.Tensor)},
                 'train_roots':train,'train_seeds':seeds}
        if args.snap:tensors['snap_codes']=torch.load(args.snap,weights_only=False)['codes'].float()
        inputs={k:R.tensor_hash(v)for k,v in tensors.items()}
        inputs_by_pool[pool]=inputs
    bb=st['args'].get('backbone','full')
    options={'window':args.window,'hard':args.hard,'snap':str(args.snap),'batch':4 if bb in ('fmamba','fcanvas')else 16}
    runtime={'torch':str(torch.__version__),'numpy':R.np.__version__,'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(),'cpu_threads':torch.get_num_threads(),
             'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
             'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}
    pins={str(path):R.file_hash(path),str(S.CHECKPOINT):R.file_hash(S.CHECKPOINT)}
    for root in Path('artifacts/eda/frozen_eval_resume_v1').glob(name+f'__teval_w{args.window}__*'):
        c=json.loads((root/'contract.json').read_text())
        if c['options']!=options or c['inputs']!=inputs or c['runtime']!=runtime or c['checkpoints']!=pins:continue
        if any(R.file_hash(p)!=sha for p,sha in c['sources'].items()):continue
        # These are original numerical sources, not an unbound filename shortcut.
        store=R.Store(root,c)
        with store.lock():res=store.load('result')
        if res is None:continue
        rows=res.pop('_per_root')
        assert R.digest(json.loads(report.read_text()))==R.digest(json.loads(json.dumps(res))),report
        variants=[L/'evals'/f'{tag}_per_root.pt',L/'evals'/f'{tag}__readout_v2_per_root.pt']
        exported=[p for p in variants if p.exists()]
        assert exported,variants
        for p in exported:
            old=torch.load(p,map_location='cpu',weights_only=False)
            assert set(old)==set(rows),p
            assert all(torch.equal(old[k],v)if isinstance(v,torch.Tensor)else old[k]==v for k,v in rows.items()),p
        print(json.dumps({'stage':'verified_completed_reuse','name':name,'report_numeric_difference':0,
                          'raw_rows_exact':True,'original_contract':store.contract,'export_hook':R.file_hash(__file__)}),flush=True)
        return True
    # Existing incompatible evidence must not be overwritten by a new run.
    raise RuntimeError(f'Completed export has no matching current input/source/runtime record: {report}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('worlds',nargs='+',type=Path)
    p.add_argument('--snap',type=Path)
    p.add_argument('--hard',action='store_true')
    p.add_argument('--window',type=int,default=5)
    args=p.parse_args()
    pending=[];inputs={}
    for path in args.worlds:
        if not completed(path,args,inputs):pending.append(path)
    if not pending:return
    R.atomic_json=json_export
    argv=list(map(str,pending))+['--window',str(args.window)]
    if args.hard:argv+=['--hard']
    if args.snap:argv+=['--snap',str(args.snap)]
    T.main(argv)


if __name__=='__main__':main()
