"""Fixed policy definitions and paired episode-level summaries; CPU, no CUDA import."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '20261102_direct_chooser'
CKDIR = HERE.parent / '20261101_p1f15/run'
CENTRES = np.array([(r, c) for r in range(1, 8) for c in range(1, 8)])
WIN = np.array([[(r+dr)*9+c+dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)] for r, c in CENTRES])
POLICIES = ('uniform', 'existing', 'C_softmax', 'S_softmax', 'probability_mixture',
            'C_direct', 'S_direct', 'direct_mixture')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def policies(c, s):
    c, s = np.asarray(c, dtype=np.float64), np.asarray(s, dtype=np.float64)
    assert c.shape == s.shape and c.shape[-1] == 49
    assert np.isfinite(c).all() and np.isfinite(s).all()
    assert (c >= 0).all() and (s >= 0).all()
    u = np.full_like(c, 1/49)
    def norm(x):
        den = x.mean(-1, keepdims=True)
        return np.divide(x, den, out=np.zeros_like(x), where=den > 1e-12)
    def direct(x):
        den = x.sum(-1, keepdims=True)
        return np.divide(x, den, out=u.copy(), where=den > 0)
    pc, ps = softmax(norm(c)), softmax(norm(s))
    dc, ds = direct(c), direct(s)
    raw = dict(uniform=u, existing=softmax((norm(c)+norm(s))/2),
               C_softmax=pc, S_softmax=ps, probability_mixture=(pc+ps)/2,
               C_direct=dc, S_direct=ds, direct_mixture=(dc+ds)/2)
    return {k: .8*v+.2*u for k, v in raw.items()}


def cohorts():
    # Stored metadata was fixed before the original head fits. No new cohort search.
    return {name: {k: z[k] for k in ('cls', 'ep', 't', 'counter', 'faced', 'light')}
            for name in ('saved', 'confirmation')
            for z in [np.load(OLD/f'predictions_611601_{name}.npz')]}


def summarize(meta, c, s, errors=None, draws=2000):
    """All policies share the SAME per-class episode bootstrap and observations."""
    pp = policies(c, s)
    remote = (WIN[None] == meta['counter'][:, None, None]).any(-1)
    contact = (WIN[None] == meta['faced'][:, None, None]).any(-1)
    masks = dict(remote=remote, contact=contact, union=remote | contact)
    vals, columns = [], []
    for name in POLICIES:
        p = pp[name]
        for metric, v in [(k+'_mass', (p*mask).sum(-1)) for k, mask in masks.items()] + [
            ('entropy', -(p*np.log(p)).sum(-1)),
            ('touch_hud_mass', p[:, (WIN >= 63).any(-1)].sum(-1)),
            ('max_probability', p.max(-1)),
            ('TV_existing', .5*np.abs(p-pp['existing']).sum(-1))]:
            vals.append(v); columns.append((name, metric))
    values = np.stack(vals, -1)
    result = {}
    selections = {str(cl): meta['cls'] == cl for cl in np.unique(meta['cls'])}
    random = meta['cls'] == 'random'
    selections.update(random_bright=random & (meta['light'] >= .7),
                      random_dark=random & (meta['light'] <= .3))
    for label, ix in selections.items():
        if not ix.any():
            continue
        _, inv = np.unique(meta['ep'][ix], return_inverse=True)
        counts = np.bincount(inv)
        sums = np.zeros((len(counts), values.shape[-1]))
        np.add.at(sums, inv, values[ix])
        rng = np.random.default_rng(611755)
        bs = rng.integers(len(counts), size=(draws, len(counts)))
        # Gather in modest blocks instead of a 2000 x episodes x metrics tensor.
        boot = np.concatenate([sums[b].sum(1)/counts[b].sum(1)[:, None] for b in np.array_split(bs, 40)])
        means = values[ix].mean(0)
        lo, hi = np.quantile(boot, [.025, .975], axis=0)
        entry = dict(n=int(ix.sum()), episodes=len(counts), policies={})
        for j, (name, metric) in enumerate(columns):
            q = dict(mean=float(means[j]), low=float(lo[j]), high=float(hi[j]))
            if metric.endswith('_mass'):
                for ref in ('existing', 'uniform'):
                    k = columns.index((ref, metric)); diff = boot[:, j]-boot[:, k]
                    dl, dh = np.quantile(diff, [.025, .975])
                    q['minus_'+ref] = dict(mean=float(means[j]-means[k]), low=float(dl), high=float(dh))
            entry['policies'].setdefault(name, {})[metric] = q
        entry['components'] = {}
        for sig, x in (('C', c), ('S', s)):
            chosen = x[ix].argmax(-1)
            entry['components'][sig] = {
                k+'_argmax': float(mask[ix][np.arange(ix.sum()), chosen].mean()) for k, mask in masks.items()}
            entry['components'][sig]['mean_score'] = float(x[ix].mean())
        if errors:
            entry['errors'] = {k: float(np.asarray(v)[ix].mean()) for k, v in errors.items()
                               if not (label.startswith('random') and k.startswith('counter_'))}
        result[label] = entry
    return result


def checks():
    rng = np.random.default_rng(611756)
    c, s = rng.random((12, 49)), rng.random((12, 49))
    pp = policies(c, s)
    for p in pp.values():
        assert np.allclose(p.sum(-1), 1) and (p >= .2/49-1e-15).all()
    assert np.allclose(pp['probability_mixture'], (pp['C_softmax']+pp['S_softmax'])/2)
    assert np.allclose(pp['direct_mixture'], (pp['C_direct']+pp['S_direct'])/2)
    for p in policies(np.zeros_like(c), np.zeros_like(s)).values():
        assert np.allclose(p, 1/49)
    # Reversing the score peak must reverse coverage, without relying on argmax ties.
    hot = np.zeros((1, 49)); hot[0, -1] = 1
    cold = np.flip(hot, -1).copy()
    assert policies(hot, hot)['existing'][0, -1] > .99 - .2
    assert policies(cold, cold)['existing'][0, -1] < .01
    # Constant positive scores also yield uniform, including direct normalization.
    for p in policies(np.ones_like(c), np.ones_like(s)).values():
        assert np.allclose(p, 1/49)
    # A disjoint pair of narrow peaks stays represented in the probability mixture.
    mix = policies(hot, cold)['probability_mixture']
    assert mix[0, 0] > .39 and mix[0, -1] > .39
    return dict(probability_normalization=True, exploration_floor=True, flat_uniform=True,
                mixture_identity=True, reversed_signal_detected=True)
