"""Independent real-state and copy controls for both E20 endpoint readers, CPU."""
import json
import torch
import e20_endpoints as E


def main():
    torch.set_num_threads(3)
    meta,fit,_,health,masks,frames,_,truth,_=E.inputs()
    real=health(frames[:,3:])
    ok=torch.ones(len(fit),16,dtype=torch.bool)
    oracle=E.health_report(meta,fit,masks,real,truth,real[:,1:],real[:,1:],ok)
    copy=E.health_report(meta,fit,masks,real,truth,real[:,:-1],real[:,:1].expand(-1,16,-1),ok)
    for v in copy['metrics'].values():
        assert v['teacher']['1.5']['all']['overall']['hits_drawn']==0
        assert v['teacher']['1.5']['all']['overall']['false_drops']==0
    result={'sources':E.TRAIN.source_pins(),'reader_sha256':E.R.file_hash(E.OUT/'health_reader.pt'),
            'frames_sha256':E.R.tensor_hash(frames),'metadata_sha256':E.R.file_hash(E.T.META),
            'world_forward_calls':0,'real_control':oracle['positive_controls'],
            'copy_control':{k:v['teacher']['1.5']for k,v in copy['metrics'].items()},
            'scope':'reused diagnostic roots; no fitting or world inference; C admission remains source holdout'}
    E.R.atomic_json(E.OUT/'readout_controls.json',result,immutable=True)
    print(json.dumps({k:v for k,v in result.items()if k!='sources'}),flush=True)


if __name__=='__main__':main()
