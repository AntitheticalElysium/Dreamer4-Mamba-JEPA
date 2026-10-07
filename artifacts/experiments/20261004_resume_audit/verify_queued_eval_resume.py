"""Exercise actual queued evaluator loops on CPU synthetic fixtures; no research result or GPU work."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
sys.path.insert(0, 'artifacts/experiments/20260927_levers')
sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
import check_damage as damage
import check_recall as recall
import teval as teval
import h16_resume as R
import d4mj.config as CFG
import spatial as Sp
import onestep

torch.set_num_threads(1)
real_device, real_load = torch.device, torch.load
config = SimpleNamespace(runtime=SimpleNamespace(device='cpu', precision='float32'))
HERE = Path(__file__).parent

def legacy(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

old_damage = legacy('old_damage', 'DAMAGE_BEFORE_RESUME.py')
old_recall = legacy('old_recall', 'RECALL_BEFORE_RESUME.py')
old_teval = legacy('old_teval', 'TEVAL_BEFORE_RESUME.py')

class World:
    head, backbone_kind = 'corrt', 'fmamba'
    time = torch.zeros(6, 1)
    def __init__(self): self.calls = 0
    def __call__(self, frames, actions):
        self.calls += 1
        out = frames.clone()
        grid = frames[..., :63, :].unflatten(-2, (7, 9))
        for a, dr, dc in ((1, 0, 1), (2, 0, -1), (3, 1, 0), (4, -1, 0)):
            shifted = torch.roll(grid, (dr, dc), dims=(-3, -2)).flatten(-3, -2)
            out[..., :63, :] = torch.where((actions == a)[..., None, None], shifted, out[..., :63, :])
        out += .0001
        return out, out, None

class Probes:
    def __init__(self, *args, **kw): pass
    def hud(self, x): return x[..., :4]
    def read(self, tokens, vis): return {'fixture_mean': float(tokens.float().mean())}

class InterruptStore(R.Store):
    interrupted = False
    def save(self, key, value, generation):
        super().save(key, value, generation)
        if not type(self).interrupted and (key == 'batch_0' or key == 'progress'):
            type(self).interrupted = True
            raise InterruptedError('injected after committed first evaluation batch')

def equal_results(a, b):
    a, b = dict(a), dict(b)
    ra, rb = a.pop('_per_root', None), b.pop('_per_root', None)
    assert R.digest(a) == R.digest(b)
    if ra is not None:
        assert set(ra) == set(rb)
        assert all(torch.equal(ra[k], rb[k]) if isinstance(ra[k], torch.Tensor) else ra[k] == rb[k] for k in ra)

with tempfile.TemporaryDirectory(prefix='queued-eval-resume-', dir='/tmp') as tmp:
    tmp = Path(tmp)
    torch.manual_seed(61004)
    roots, steps, width = 17, 20, 192
    canvas = torch.randn(15, 16, width)
    moves = [0, 2, 2, 1, 4, 2, 1, 3, 1, 2, 4, 1, 3, 2, 2, 1, 4, 1, 3]
    offsets, row, col = [(3, 3)], 3, 3
    for a in moves:
        dr, dc = {0:(0,0), 1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0)}[a]
        row += dr; col += dc; offsets.append((row, col))
    seq = torch.randn(steps, 81, width)
    for t, (r, c) in enumerate(offsets): seq[t, :63] = canvas[r:r+7, c:c+9].flatten(0, 1)
    hp = torch.tensor([.9,.9,.9,.9,.7,.7,.8,.8,.6,.6,.5,.5,.6,.6,.4,.4,.5,.5,.3,.3])
    seq[:, 63, 0] = hp
    seqs = seq[None].repeat(roots, 1, 1, 1).half()
    actions = torch.tensor(moves)[None].repeat(roots, 1)
    cache = {'ctx': seqs[:, :4].contiguous(), 'ctx_a': actions[:, :3].contiguous(),
             'fut_a': actions[:, 3:19].contiguous(), 'fut': seqs[:, 4:].contiguous(),
             'one': seqs[:, 4:5].repeat(1, 17, 1, 1)}
    fut5 = cache['fut'][:, None].repeat(1, 5, 1, 1, 1)
    visible = torch.zeros(roots, 5, 16, 1534); visible[..., 1512] = hp[4:]
    root_visible = torch.zeros(roots, 1534); root_visible[:, 1512] = hp[3]
    meta = {'seed': torch.arange(roots), 'root_visible': root_visible, 'future_visible': visible,
            'future_dead': torch.zeros(roots, 5, 16, dtype=torch.bool),
            'onestep_visible': torch.zeros(roots, 5, 17, 1534)}
    train = torch.arange(roots) % 2 == 0; seeds = meta['seed'][train]
    pool = {'tokens': seqs[:, :6], 'actions': actions[:, :5], 'terminal': torch.zeros(roots, dtype=torch.bool)}
    pool.update({k:torch.zeros(roots,6) for k in ('reward_led','reward_valid','alive','dh')})
    valid = torch.ones(roots, 5, 16, dtype=torch.bool)
    adjacent = valid.clone()
    adjwin = (torch.arange(roots) % 3 != 0)[:, None, None].expand_as(valid)
    win = (torch.arange(roots) % 3 == 2)[:, None, None].expand_as(valid)
    masks = {'valid': valid, 'k3': (torch.arange(16) >= 3)[None,None].expand_as(valid),
             'adjacent': adjacent, 'adjwin': adjwin, 'win': win,
             'drop2': (hp[4:] < hp[3:19] - .15)[None,None].expand_as(valid)}
    world_path = tmp / 'world.pt'; torch.save({'name': 'cpu_fixture', 'args': {'pool': 'raw'}}, world_path)
    fixture_config = tmp / 'config.pt'; torch.save({'config': {}}, fixture_config)
    current_worlds = []
    def load_world(*args):
        w = World(); current_worlds.append(w)
        return w, {'name':'cpu_fixture', 'args':{'pool':'raw'}}
    def torch_load(path, *args, **kw):
        if str(path).endswith('/pool.pt'): return pool
        return real_load(path, *args, **kw)
    def store_factory(root, interrupt=False):
        def factory(path, name, options, inputs, dependencies=()):
            contract = {'options': options, 'inputs': {k:R.tensor_hash(v) for k,v in inputs.items()}, 'checkpoint': R.file_hash(path)}
            cls = InterruptStore if interrupt else R.Store
            return cls(root / name, contract)
        return factory
    def run_main(module, argv):
        buffer = io.StringIO()
        with patch.object(sys, 'argv', [module.__file__, *argv]), contextlib.redirect_stdout(buffer): module.main()
        return json.loads(buffer.getvalue().splitlines()[-1])
    common = [patch.object(torch, 'device', lambda arg: real_device('cpu') if str(arg) == 'cuda' else real_device(arg)),
              patch.object(torch, 'load', torch_load), patch.object(torch.cuda, 'empty_cache', lambda: None),
              patch.object(CFG, 'config_from_dict', lambda _:config), patch.object(Sp, 'CHECKPOINT', fixture_config),
              patch.object(teval, 'split', lambda:(meta,train,seeds)), patch.object(teval, 'build_cache', lambda *a:cache),
              patch.object(teval, 'Probes', Probes), patch.object(teval, 'load_world', load_world),
              patch.object(old_teval, 'Probes', Probes),
              patch.object(damage.SD, 'token_cache', lambda *a:(fut5,None)),
              patch.object(damage.DR, 'masks', lambda *a:masks),
              patch.object(recall, 'futures', lambda *a:(seqs,actions,F.pad(torch.ones(roots,19,dtype=torch.bool),(1,0)))),
              patch.object(old_recall, 'futures', lambda *a:(seqs,actions,F.pad(torch.ones(roots,19,dtype=torch.bool),(1,0)))),
              patch.object(onestep, 'classify', lambda *a:((torch.arange(17) % len(onestep.CLASSES))[None].expand(roots,-1),None))]
    result = {}
    with contextlib.ExitStack() as stack:
        for p in common: stack.enter_context(p)
        for label, old, new, argv in [('damage',old_damage,damage,[str(world_path)]),
                                     ('recall_pool',old_recall,recall,[str(world_path)]),
                                     ('recall_futures',old_recall,recall,['--futures',str(world_path)]),
                                     ('recall_imagined',old_recall,recall,['--futures','--imagined',str(world_path)])]:
            reference = run_main(old, argv)
            with patch.object(R, 'frozen_eval_store', store_factory(tmp / (label+'_full'))):
                uninterrupted = run_main(new, argv)
            equal_results(reference, uninterrupted)
            InterruptStore.interrupted = False
            try:
                with patch.object(R, 'frozen_eval_store', store_factory(tmp / (label+'_split'), True)):
                    run_main(new, argv)
                raise AssertionError('interruption not reached')
            except InterruptedError: pass
            with patch.object(R, 'frozen_eval_store', store_factory(tmp / (label+'_split'))):
                resumed = run_main(new, argv)
                equal_results(reference, resumed)
                current_worlds.clear()
                completed = run_main(new, argv)
                equal_results(resumed, completed)
                assert sum(w.calls for w in current_worlds) == 0
            result[label] = {'legacy_parity':True, 'interruption_parity':True, 'completed_world_calls':0}
        kw = dict(batch=4, window=5)
        reference = old_teval.evaluate(World(),cache,meta,Probes(),train,seeds,~train,real_device('cpu'),**kw)
        full = R.Store(tmp/'teval_full', {'test':'teval'})
        got = teval.evaluate(World(),cache,meta,Probes(),train,seeds,~train,real_device('cpu'),resume_store=full,**kw)
        equal_results(reference,got)
        InterruptStore.interrupted = False
        split = InterruptStore(tmp/'teval_split', {'test':'teval'})
        try:
            teval.evaluate(World(),cache,meta,Probes(),train,seeds,~train,real_device('cpu'),resume_store=split,**kw)
            raise AssertionError('interruption not reached')
        except InterruptedError: pass
        resumed = R.Store(split.root, {'test':'teval'})
        w = World()
        got = teval.evaluate(w,cache,meta,Probes(),train,seeds,~train,real_device('cpu'),resume_store=resumed,**kw)
        equal_results(reference,got)
        assert w.calls == 4 * 33
        w = World()
        got = teval.evaluate(w,cache,meta,Probes(),train,seeds,~train,real_device('cpu'),resume_store=resumed,**kw)
        equal_results(reference,got); assert w.calls == 0
        result['teval'] = {'legacy_parity':True,'interruption_parity':True,'completed_world_calls':0}
    result['scope'] = 'Synthetic CPU fixtures exercise original and amended evaluator loops/storage; CUDA parity and research performance not tested'
    with patch.object(torch.cuda, 'get_device_name', lambda:'CPU contract fixture'):
        a = R.frozen_eval_store(world_path, 'contract_fixture_' + tmp.name, {'window':5},
                                {'scalar':torch.tensor(1), 'strided':torch.arange(18).reshape(3,6)[:,::2]})
        b = R.frozen_eval_store(world_path, 'contract_fixture_' + tmp.name, {'window':15},
                                {'scalar':torch.tensor(1), 'strided':torch.arange(18).reshape(3,6)[:,::2]})
    assert a.root != b.root
    manifest = json.loads((a.root/'contract.json').read_text())
    assert any(p.endswith('/tworld.py') for p in manifest['sources'])
    assert any(p.endswith('/onestep.py') for p in manifest['sources'])
    result['real_contract_helper'] = {'scalar_strided_hashes':True,'window_namespaces_differ':True,'lazy_forward_sources_bound':True}
    for st in (a,b):
        for p in st.root.iterdir(): p.unlink()
        st.root.rmdir()
    result['sources'] = {str(Path(m.__file__)):R.file_hash(m.__file__) for m in (damage,recall,teval,R)}
    R.atomic_json(Path(__file__).with_suffix('.json'), result)
    print(json.dumps(result, indent=2))
