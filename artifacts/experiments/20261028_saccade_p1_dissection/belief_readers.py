"""Localise finite-reader losses: exact projected token vs recurrent output, matched 256-D inputs."""
import time
import copy
import torch.nn.functional as F
from .common import *

HEADS={'ach':22,'hud':160,'dep':20,'side':16}


def terms(o,y):
    h=o['hud'].reshape(-1,16,10); side=y['side'].reshape(-1);sp=o['side'].reshape(-1,4)
    return {'ach':F.binary_cross_entropy_with_logits(o['ach'],y['ach']),
            'hud_all':F.cross_entropy(h.reshape(-1,10),y['hud'].reshape(-1)),
            'body':F.cross_entropy(h[:,:4].reshape(-1,10),y['hud'][:,:4].reshape(-1)),
            'inventory':F.cross_entropy(h[:,4:].reshape(-1,10),y['hud'][:,4:].reshape(-1)),
            'dep':F.cross_entropy(o['dep'].reshape(-1,5),y['dep'].reshape(-1)),
            'side':F.cross_entropy(sp[side>=0],side[side>=0]) if (side>=0).any() else sp.sum()*0}


def objective(t,task):
    if task=='joint':return t['ach']+t['hud_all']+t['dep']+t['side']
    if task=='recent':return t['dep']+t['side']
    return t[task]


@torch.no_grad()
def predict(reader,X):
    rows={k:[] for k in HEADS}
    for q in X.split(16384):
        o=reader(q.cuda())
        for k,v in o.items():rows[k].append(v.cpu())
    return {k:torch.cat(v) for k,v in rows.items()}


def metric(o,Y,ep,task):
    hp=o['hud'].reshape(-1,16,10).argmax(-1).numpy();y=Y['hud'].numpy();res={}
    for j,name in enumerate(HUD):
        if task=='body' and j>=4 or task=='inventory' and j<4 or task=='recent':continue
        err=np.abs(hp[:,j]-y[:,j]);m=np.ones(len(y),bool);classes=np.unique(y[:,j])
        res[name]={'exact':G.estimate(err==0,m,ep),'mae':G.estimate(err,m,ep),'within_one':G.estimate(err<=1,m,ep),
                   'balanced_recall':float(np.mean([(hp[y[:,j]==k,j]==k).mean() for k in classes])),
                   'confusion':np.bincount(y[:,j]*10+hp[:,j],minlength=100).reshape(10,10).tolist()}
    if task in ('joint','recent'):
        dp=o['dep'].reshape(-1,4,5).softmax(-1).numpy();yd=Y['dep'].numpy()
        res['recent']={name:{f'{lo}-{hi}':G.average_precision((yd[:,j]==s+1).astype(int),dp[:,j,s+1])
                             for s,(lo,hi) in enumerate(s1.STRATA)} for j,name in enumerate(KINDS[1:])}
    if task=='joint':
        a=o['ach'].sigmoid().numpy();ya=Y['ach'].numpy()
        res['achievement_ap']=[G.average_precision(ya[:,j],a[:,j]) for j in range(22)]
    return res


if __name__=='__main__':
    before=fingerprints();start=time.time();out={}
    enc={k:torch.load(OUT/f'real_{k}.pt',weights_only=False,mmap=True) for k in ('train','dev')}
    L={k:labels(k) for k in enc};E={k:episode_rows(L[k]) for k in enc};Y={k:s1.belief_targets(L[k]) for k in enc}
    ti=torch.from_numpy(np.where(E['train']%10!=9)[0]);vi=torch.from_numpy(np.where(E['train']%10==9)[0])
    Ygpu={k:v.cuda() for k,v in Y['train'].items()}
    for representation in ('b','x','full_percept'):
        X={k:torch.cat([r[representation] for r in enc[k]]) if representation!='full_percept'
           else s1.frame_features(half_encoding(enc[k]))[:,:-17] for k in enc}
        XT=X['train'].cuda();XD=X['dev'];del X
        for task in ('joint','body','inventory','recent'):
            for seed in (0,1):
                torch.manual_seed(seed);reader=s1.MLP(XT.shape[1],HEADS).cuda();opt=torch.optim.AdamW(reader.parameters(),lr=1e-3,weight_decay=1e-4)
                rng=torch.Generator(device='cuda').manual_seed(seed);log=[];best={};states={};stopped=None
                for epoch in range(30):
                    reader.train()
                    for inds in torch.randperm(len(ti),device='cuda',generator=rng).split(4096):
                        rows=ti.cuda()[inds];t=terms(reader(XT[rows]),{k:v[rows] for k,v in Ygpu.items()})
                        loss=objective(t,task);opt.zero_grad();loss.backward();opt.step()
                    reader.eval()
                    with torch.no_grad():
                        vt=[]
                        for rows in vi.cuda().split(65536):
                            vt.append({k:float(v) for k,v in terms(reader(XT[rows]),{k:v[rows] for k,v in Ygpu.items()}).items()})
                    vals={k:float(np.mean([v[k] for v in vt])) for k in vt[0]};vals['joint']=float(objective(vals,'joint'));vals['recent']=vals['dep']+vals['side']
                    log.append(vals)
                    for target in (('joint','body','inventory','recent') if task=='joint' else (task,)):
                        if vals[target]<best.get(target,(float('inf'),-1))[0]:
                            best[target]=(vals[target],epoch);states[target]=copy.deepcopy(reader.state_dict())
                    if task=='joint' and stopped is None and epoch-best['joint'][1]>=3:
                        stopped=epoch;states['original_s1_stop']=copy.deepcopy(states['joint']);best['original_s1_stop']=best['joint']
                if task=='joint' and stopped is None:
                    states['original_s1_stop']=states['joint'];best['original_s1_stop']=best['joint'];stopped=29
                selection=('original_s1_stop','body','inventory','recent') if task=='joint' else (task,)
                for target in selection:
                    reader.load_state_dict(states[target]);o=predict(reader,XD)
                    scored='joint' if target=='original_s1_stop' else target
                    result=metric(o,Y['dev'],E['dev'],scored);key=f'{representation}/{task}/{target}/{seed}'
                    out[key]={'metrics':result,'best':best[target],'original_stopping_epoch':stopped,'validation':log,
                              'n_train':len(ti),'n_val':len(vi),'n_dev':len(XD),'input_dim':XT.shape[1]}
                    torch.save({'state':{k:v.cpu() for k,v in reader.state_dict().items()},'heads':HEADS,'input_dim':XT.shape[1],
                                'hud_prediction':o['hud'].reshape(-1,16,10).argmax(-1).byte()},OUT/f'belief_{representation}_{task}_{target}_{seed}.pt')
                    save('belief_readers.json',out)
                print(representation,task,seed,best,flush=True)
        del XT,XD;torch.cuda.empty_cache()
    assert fingerprints()==before
    save('belief_readers_run.json',{'seconds':time.time()-start,'world_sources_unchanged':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
