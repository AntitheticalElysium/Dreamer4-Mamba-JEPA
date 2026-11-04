"""Output-isolated reuse of F2's evaluation instruments. No world/head fitting."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import copy
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from saccade import data as D, train as TR

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'20261103_p1f2'
U = importlib.import_module('artifacts.experiments.20261104_p1f21.union')


def setup():
    """Both checkpoint recipes can read the virtual store; original splits keep their original membership."""
    U.use(U.cfg_union())
    base = D.load_split
    def split(cfg):
        if cfg['data']['corpus'] == U.STORES[0]:
            c = copy.deepcopy(cfg); c['data']['corpus'] = U.CORPUS
            return {k: [e for e in v if e.shard < 420] for k, v in base(c).items()}
        return base(cfg)
    D.load_split = split


def invoke(module, args):
    sys.argv = [module, *map(str, args)]
    importlib.import_module(module).main()


def codex(what, path):
    old = importlib.import_module('artifacts.experiments.20261101_p1f15.readings_codex')
    out = HERE/'readings/codex'; out.mkdir(parents=True, exist_ok=True)
    old.HERE = HERE; old.BR.OUT = out
    if what == 'sampler':
        old.sampler(path)
        file = out/'sampler_results.json'; result = json.loads(file.read_text())
        result['models']['p1f21'] = result['models'].pop('p1f15')
        result['p1f21_wrapper_sha256'] = TR.file_sha256(__file__)
        file.write_text(json.dumps(result, indent=1))
    else:
        old.BR.WORLDS = {'p1f21': path}
        sys.argv = [sys.argv[0], what, 'p1f21']; old.BR.main()


def paired_summary():
    rs = importlib.import_module('artifacts.experiments.20261101_beta_tsp.readings_summary')
    new = HERE/'readings/codex'; combined = HERE/'readings/codex_vs_p1f2'
    combined.mkdir(parents=True, exist_ok=True)
    models = {}
    for source, name in [(BASE/'readings/codex', 'p1f2'), (new, 'p1f21')]:
        for f in source.glob(f'{name}_*.json'):
            dst = combined/f.name
            if not dst.exists():dst.symlink_to(f.resolve())
        models[name] = json.loads((source/'sampler_results.json').read_text())['models'][name]
    (combined/'sampler_results.json').write_text(json.dumps({'models': models}))
    rs.OUT, rs.A, rs.B, rs.HERE = combined, 'p1f2', 'p1f21', HERE
    rs.main()


def fovea(path):
    """Identical fixed58-scene instrument; F2 reference immutable, candidate readers fitted anew."""
    old = importlib.import_module('artifacts.experiments.20261103_p1f2.fovea_readings')
    w = importlib.import_module('artifacts.experiments.20261103_p1f16_review.local_uptake')
    out = HERE/'readings/fovea'; out.mkdir(parents=True, exist_ok=True)
    # The reused summary has historical slot names. Alias only inside this new output,
    # then rename all reported world keys explicitly; no old reference is modified.
    ref = (BASE/'readings/fovea/p1f2').resolve()
    identity = json.loads((BASE/'readings/fovea/identity.json').read_text())
    assert identity['candidate']['checkpoint_sha256'] == TR.file_sha256(BASE/'run/ckpt_10.pt')
    assert identity['baseline_npz_sha256'] == TR.file_sha256(BASE/'readings/fovea/p1f16/local.npz')
    link = out/'p1f16'
    if not link.exists():link.symlink_to(ref, target_is_directory=True)
    w.HERE = out; source, weights = w.fixed_cohort()
    ident = w.run_world('p1f2', path, source, weights)
    result = old.summary(out)
    result['worlds'] = {'p1f2': result['worlds']['p1f16'], 'p1f21': result['worlds']['p1f2']}
    result['labels'] = {'p1f16_directory': 'immutable F2 reference', 'p1f2_directory': 'new F2.1 candidate',
                        'paired_fraction_change': 'F2.1 minus F2'}
    result['candidate_identity'] = ident
    result['baseline_npz_sha256'] = TR.file_sha256(ref/'local.npz')
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


def own_interface(path, label_source='dev', smoke=False):
    """One-step closed-loop query gaze, with each world's OWN head and its own real-history acquisitions."""
    og = importlib.import_module('artifacts.experiments.20261104_f2_own_gaze.own_gaze')
    S, G, IF = og.S, og.G, og.IF
    source = S.LABELS/'s0_dev_labels.npz' if label_source == 'dev' else HERE/f'heldout_{label_source}.npz'
    L = dict(np.load(source))
    if smoke:L = S.first_episodes(L, 2)
    cls = IF.classes(L, np.random.default_rng(2031))
    if smoke:cls = {k: v[:2] for k, v in cls.items()}
    out = HERE/('eval_smoke' if smoke else 'readings')
    out.mkdir(exist_ok=True)
    dest = out/f'interface_own_{label_source}.json'
    result = {'label_source': str(source), 'label_sha256': TR.file_sha256(source),
              'query_gaze': 'chosen from each generated coarse sample, not recorded successor',
              'classes': {k: len(v) for k, v in cls.items()}, 'checkpoints': {}, 'differences': {}}
    rows = {}
    for name, ckpt in [('p1f2', BASE/'run/ckpt_10.pt'), ('p1f21', path)]:
        model, cfg = IF.load(ckpt); head = og.head_of(ckpt)
        W, index, forecasts = og.own_windows(model, head, L, cfg)
        enc = S.encode(model, L, cfg, windows=W)
        choose = og.closed_loop(L, forecasts, index)
        report = {'path': str(ckpt), 'sha256': TR.file_sha256(ckpt), 'classes': {}}
        for i, (key, tuples) in enumerate(cls.items()):
            if not tuples:continue
            choose.same.clear()
            o = IF.futures(model, enc, L, tuples, 4, cfg,
                           torch.Generator(device=og.DEV).manual_seed(3000+i), choose)
            ep = np.array([tuples[j][0] for j in o['tuple']])
            report['classes'][key] = IF.summary(o, ep)
            rows[name, key] = (o, ep)
        result['checkpoints'][name] = report
        dest.write_text(json.dumps(result, indent=1))
        del model, head, enc, forecasts, choose; torch.cuda.empty_cache()
    for key in cls:
        if ('p1f2', key) not in rows:continue
        (a, ep), (b, _) = rows['p1f2', key], rows['p1f21', key]
        result['differences'][key] = {}
        for q in ['unlock_p','exp_reward','neg_p','term_p']:
            z = G.estimate(b['gen_'+q]-a['gen_'+q], np.ones(len(ep),bool), ep)
            result['differences'][key][q] = {**z, 'ci95': [z['est']-1.96*z['se'], z['est']+1.96*z['se']]}
    dest.write_text(json.dumps(result, indent=1))


def main():
    setup()
    torch.set_num_threads(4)
    if sys.argv[1] == 'legacy':
        return invoke(sys.argv[2], sys.argv[3:])
    what, path = sys.argv[1], Path(sys.argv[2])
    if what.startswith('codex_'):return codex(what.removeprefix('codex_'), path)
    if what == 'summary':return paired_summary()
    if what == 'fovea':return fovea(path)
    if what == 'own':return own_interface(path, sys.argv[3], '--smoke' in sys.argv)
    if what in ['head', 'gaze']:
        name = 'head_readings' if what == 'head' else 'gaze_readings'
        module = importlib.import_module('artifacts.experiments.20261103_p1f2.'+name)
        module.HERE = HERE; sys.argv = [name, str(path)]; return module.main()
    raise ValueError(what)


if __name__ == '__main__':
    main()
