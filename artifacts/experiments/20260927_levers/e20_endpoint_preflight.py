"""Data-only admission for paired E20 endpoint statistics; no model inference."""
import json
import torch
import e20_endpoints as E


def main():
    torch.set_num_threads(3)
    meta,_,_=E.T.split()
    cache=E.T.build_cache('raw',torch.device('cpu'))
    frames=torch.cat((cache['ctx'],cache['fut']),1)
    valid=~meta['future_dead'][:,0].cumsum(1).bool()
    oracle=E.memory_rows(frames,frames[:,4:],valid)
    independent_counts={g:0 for g in oracle}
    # Verify the subset membership against the historical extractor's full corpus.
    for j in range(len(frames)):
        t,c,cls,s,sc,_=E.RECALL.cells(frames[j].float())
        use=(t>=4)&(cls==1)
        use[use.clone()] &= valid[j,t[use]-4]
        age=t-s
        independent_counts['same_6_15']+=int((use&(sc==c)&(age>=6)&(age<=15)).sum())
        independent_counts['moved_6_15']+=int((use&(sc!=c)&(age>=6)&(age<=15)).sum())
        independent_counts['same_2_5']+=int((use&(sc==c)&(age>=2)&(age<=5)).sum())
    report={}
    for group,rows in oracle.items():
        assert int(rows['n'].sum())==independent_counts[group]
        assert float(rows['world'].sum())==0
        den=rows['neighbour']-rows['sighting']
        gain=E.paired_ratio(den,den,meta['seed'])
        assert gain['point']==1 and gain['interval95']==[1,1]
        zero=E.paired_ratio(torch.zeros_like(den),den,meta['seed'])
        assert zero['point']==0 and zero['interval95']==[0,0]
        report[group]={'cells':independent_counts[group],
                       'oracle_world_error':0,'paired_capture_gain_control':gain,'no_difference_control':zero}
    payload={'sources':{str(E.Path(__file__).resolve()):E.R.file_hash(__file__),
                        str(E.Path(E.__file__).resolve()):E.R.file_hash(E.__file__)},
             'passed':True,'world_forward_calls':0,'subset':'sample0 preselected before E20 optimization',
             'groups':report}
    E.R.atomic_json(E.OUT/'endpoint_preflight.json',payload,immutable=True)
    print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
