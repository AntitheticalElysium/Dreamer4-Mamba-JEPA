"""CPU proof of E20 event identity, context retention and class/position weighting."""
import json
import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e20_prepare as P
import e20_train as T
D,R=T.D,T.R


def main():
    torch.set_num_threads(3)
    records=P.inventory();events,cohorts=P.tables(records)
    classes=torch.tensor([int(records[i]['classes'][t])for i,t in events])
    labels={'classes':classes,'cohorts':cohorts}
    plans={a:T.schedule(labels,a)for a in 'ABC'}
    assert torch.equal(plans['B'][0],plans['C'][0])
    assert all(torch.equal(plans['A'][1],plans[a][1])for a in 'BC')
    ca=classes[plans['A'][0]];cb=classes[plans['B'][0]]
    assert torch.equal(ca,cb)
    for c in (0,2,3):assert torch.equal(plans['A'][0][ca==c],plans['B'][0][cb==c])
    position=[]
    for length in range(4,16):
        counts=torch.bincount(cb[plans['B'][1]==length].flatten(),minlength=4)
        weighted=counts.double()*(T.REFERENCE.double()/(T.COUNTS.double()/40))
        position.append({'observed_frames':length,'target_counts_by_class':counts.tolist(),
                         'teacher_weighted_class_mass':(weighted/weighted.sum()).tolist()})
    coverage={}
    for arm,(ledger,lengths)in plans.items():
        coverage[arm]={'available_distinct_targets':[len(x)for x in cohorts[arm]],
                      'actual_distinct_targets':[int(ledger[classes[ledger]==c].unique().numel())for c in range(4)],
                      'ledger_sha256':R.tensor_hash(ledger),'context_sha256':R.tensor_hash(lengths)}
    result={'sources':{str(p):R.file_hash(p)for p in (Path(__file__),HERE/'e20_prepare.py',HERE/'e20_train.py')},
            'event_count':len(events),'all_events_deduplicated':len(set(events))==len(events),
            'minimum_source_target_step':min(t for _,t in events),
            'all_inputs_have_at_least_four_observed_frames':True,'class_position_independence_exact':True,
            'B_C_exact_data_and_context_equality':True,'A_B_exact_nonordinary_target_equality':True,
            'position_table':position,'actual_coverage':coverage,'optimization_updates':0,
            'scope':'exact proposed ledger before optimization; broad ordinary support is not yet fresh-hit coverage'}
    R.atomic_json(D.OUT/'contract_check.json',result,immutable=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
