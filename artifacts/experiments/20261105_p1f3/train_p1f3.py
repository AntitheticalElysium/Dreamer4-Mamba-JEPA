"""P1-F3: F2's recipe (F1.6 world + the original online cheap-head fixation; F2's trainer, chooser, corpus, seed and budget
unchanged) with ONE change, the physical part's G1 term as an x-space loss (Dreamer 4, 2509.24527: "We formulate the
objective in data space"; Kingma & Gao 2023: our EDM weighting = x-loss + eps-loss), scaled to keep G1's physical
magnitude; the glimpse keeps EDM's weighting. From scratch, world seed 1; no automatic launch.

Weighting (cfg edm.g1_weighting, launcher-scoped: saccade/model.py, train.py and m0.yaml are pinned by F2.1's evaluation
seal; made canonical after the results, user 11-05): per element, periphery and lighting raw-output squared error x
SCALE x c_out(sigma_physical)^2 (= SCALE x the x-space error), glimpse unchanged. SCALE = 3.425: sum(raw) / sum(raw x
c_out^2) of the physical part on F2 ckpt_10's measured training-form term (consequence trace B4, 2,000 random frames x 4
draws; bootstrap 3.29-3.62; F2.1 3.46), so the glimpse keeps its 23.4% share. Applied only while TR.p1_losses or
TR.encoder_diagnostics runs (the training term and its logged diagnostics); the checkpoint monitor (TR.monitor) stays
in F2's EDM units. The reimplemented loss consumes the RNG exactly as model.denoise_loss does (checks.py)."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from saccade import model as M, train as TR
F2L = importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
T, C = F2L.T, F2L.C
HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '20261103_p1f2/run/ckpt_10.pt'
CONTROL_SHA = 'e0f6a24ec51d2bf021ca88ba8080f804595800dc9295aecf0e37c7da7d7e1234'
WORLD_SEED = 1
WEIGHTING = {'physical': 'x_space', 'scale': 3.425, 'glimpse': 'edm',
             'source': 'Dreamer 4 (2509.24527) x-space loss; scale = consequence trace B4 on F2 ckpt_10 (physical part)'}
OVERRIDE = {('edm', 'g1_weighting'): WEIGHTING}
PHYSICAL = (0, 2)                                                        # model._parts order (p, g, v): p and v physical
ORIG_LOSS, ORIG_P1, ORIG_DIAG = M.denoise_loss, TR.p1_losses, TR.encoder_diagnostics
ACTIVE = [False]


def denoise_loss(den, x0, sigma, prev_action, memory, fix, episode, valid, sd=1.0, noise=None, weights=None):
    """model.denoise_loss with a per-part weight on each element's squared error (same ops and RNG order). Inactive:
    the original function. Active: physical parts x SCALE c_out^2, glimpse x 1. `weights` (checks only) overrides."""
    if not ACTIVE[0] and weights is None:
        return ORIG_LOSS(den, x0, sigma, prev_action, memory, fix, episode, valid, sd, noise)
    noise = noise or tuple(torch.randn_like(t) for t in x0)
    pre = [M.precondition(sg, sd) for sg in M._parts(sigma)]
    xs = tuple(t + M._bcast(sg, t) * n for t, n, sg in zip(x0, noise, M._parts(sigma)))
    c_noise = (pre[0][3], pre[1][3]) if isinstance(sigma, tuple) else pre[0][3]
    Fo = den(tuple(M._bcast(c[2], t) * t for t, c in zip(xs, pre)), c_noise, prev_action, memory, fix, episode, valid)
    se = [((f.float() - (t - M._bcast(c[0], t) * s) / M._bcast(c[1], t)) ** 2).flatten(2).sum(-1)
          for f, t, s, c in zip(Fo, x0, xs, pre)]
    if weights is None:
        weights = [WEIGHTING['scale'] * pre[k][1].float() ** 2 if k in PHYSICAL else 1 for k in range(len(se))]
    return sum(s_ * w_ for s_, w_ in zip(se, weights)) / sum(t[0, 0].numel() for t in x0)


def _active(fn):
    def wrapped(*a, **k):                                                # re-entrant: encoder_diagnostics calls p1_losses
        before, ACTIVE[0] = ACTIVE[0], True
        try:
            return fn(*a, **k)
        finally:
            ACTIVE[0] = before
    return wrapped


def install():
    """Route model.denoise_loss through the weighted loss; active only inside the training term and its diagnostics."""
    M.denoise_loss, TR.p1_losses, TR.encoder_diagnostics = denoise_loss, _active(ORIG_P1), _active(ORIG_DIAG)


def recipe(cfg):
    cfg = F2L.recipe(copy.deepcopy(cfg))
    for (a, b), v in OVERRIDE.items():
        cfg[a][b] = copy.deepcopy(v)
    return cfg


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, 'F2 control changed'
    ck = torch.load(CONTROL, map_location='cpu', weights_only=False)
    old = copy.deepcopy(cfg)
    for a, b in OVERRIDE:
        del old[a][b]
    assert old == ck['cfg'] and ck['world_seed'] == WORLD_SEED, 'changes beyond the G1 weighting'
    saved, now = ck['provenance'], TR.run_provenance(cfg)
    assert now['source_sha256'] == saved['source_sha256'] and now['torch'] == saved['torch'], 'sources differ from F2'
    assert ck['fixation_controller']['config'] == C.config(), 'chooser differs from F2'
    return cfg


def identity(cfg, policy):
    files = [Path(__file__), HERE.parent / '20261103_p1f2/trainer.py', Path(C.__file__),
             HERE.parent / '20261103_p1f2/chooser.json', Path(C.Q.__file__)]
    return dict(recipe='P1-F2 + physical G1 as an x-space loss (glimpse EDM); from scratch', world_seed=WORLD_SEED,
                chooser=policy, g1_weighting=WEIGHTING, source_sha256={str(p.relative_to(TR.ROOT)): TR.file_sha256(p) for p in files},
                control=dict(path=str(CONTROL), sha256=CONTROL_SHA), mamba_deterministic=True,
                cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                provenance={k: v for k, v in TR.run_provenance(cfg).items() if k != 'git_commit'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--updates', type=int)
    ap.add_argument('--no-checkpoints', action='store_true'); args = ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    cfg = check_recipe(recipe(TR.load_cfg())); policy = C.config()
    install()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    expected = identity(cfg, policy); file = out / 'identity.json'
    if file.exists():
        assert json.loads(file.read_text()) == expected, 'Different launch identity on resume'
    else:
        assert not (out / 'latest.pt').exists(), 'Checkpoint without identity'
        file.write_text(json.dumps(expected, indent=2) + '\n')
    T.p1(cfg, WORLD_SEED, out, updates=args.updates, checkpoints=not args.no_checkpoints, controller_cfg=policy)


if __name__ == '__main__':
    main()
