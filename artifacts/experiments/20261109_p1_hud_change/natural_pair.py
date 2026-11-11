"""Natural thirst alias diagnostic on the selected checkpoint (Episode 376: t=654 vs t=658)."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
PARENT_REF = HERE.parent / '20261108_full_frame_change'


@torch.no_grad()
def main(ckpt_path=None, out_path=None):
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    torch.set_num_threads(2)
    dev = torch.device('cuda')
    ckpt = Path(ckpt_path or sys.argv[1]).resolve()
    out_file = Path(out_path or (sys.argv[2] if len(sys.argv) > 2 else HERE / 'readings/natural_pair.json')).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    IF = importlib.import_module('artifacts.experiments.20261030_actor_interface.interface')
    model, cfg = IF.load(str(ckpt))

    a = json.loads((PARENT_REF / 'SINGLE_PLANE_QUALIFICATION.json').read_text())['raw_history']['conflicts'][0]
    assert a['e'] == a['other_e']
    e = a['e']
    times = [a['t'], a['other_t']]

    L = dict(np.load(D.ROOT / 'artifacts/eda/s0/s0_dev_labels.npz'))
    off = np.r_[0, np.cumsum(L['ep_T'] + 1)]
    f = D.RP.episode_fields(int(L['ep_shard'][e]), int(L['ep_slot'][e]))
    frames, acts, rewards, term = [f[k].numpy() for k in ('observations', 'actions_taken', 'rewards', 'terminated')]
    t, u = times
    assert np.array_equal(frames[t - 7:t + 1], frames[u - 7:u + 1])
    assert np.array_equal(acts[t - 8:t + 1], acts[u - 8:u + 1])
    assert np.array_equal(rewards[t - 8:t], rewards[u - 8:u]) and np.array_equal(term[t - 8:t], term[u - 8:u])

    # Fit a qualified reader on the dev body digits using current checkpoint representations
    enc = S.encode(model, L, cfg)
    dev_split = D.load_split(cfg)['dev']
    ep_ids = np.repeat(np.arange(len(enc)), L['ep_T'] + 1)
    X = torch.cat([torch.cat([r['p'][:, 65], r['v']], 1) for r in enc]).float().to(dev)
    Y = torch.as_tensor(L['s_body'][:, 2], device=dev).long()
    held = torch.as_tensor((ep_ids % 5 == 4) | (ep_ids == e), device=dev)
    xt, xv, yt, yv = X[~held], X[held], Y[~held], Y[held]
    with torch.enable_grad():
        reader, _ = S.fit(lambda idx, val: (xv if val else xt)[idx],
                          lambda out, idx, val: torch.nn.functional.cross_entropy(out['y'], (yv if val else yt)[idx]),
                          len(yt), len(yv), {'y': 10}, 110819)
    reader.eval()
    pred = torch.cat([reader(x)['y'].argmax(-1) for x in xv.split(8192)])
    valid = float((pred == yv).float().mean())
    del X, Y, xt, xv, yt, yv

    common = D.fixation_prior(np.random.default_rng(24001), (8,))
    query = D.fixation_prior(np.random.default_rng(24002), ())
    K = 128
    out = []
    generated = []
    percepts = []
    for t in times:
        fx = D.fixation_prior(np.random.default_rng(24003), (t + 1,))
        fx[-8:] = common
        fix = torch.as_tensor(fx, device=dev)[None]
        act = torch.as_tensor(acts[:t + 1], device=dev, dtype=torch.long)[None]
        rw = torch.as_tensor(np.r_[0.0, rewards[:t]], device=dev, dtype=torch.float32)[None]
        td = torch.as_tensor(np.r_[False, term[:t]], device=dev)[None]
        prev = torch.cat([act.new_full((1, 1), -1), act[:, :-1]], 1)
        with TR.autocast(dev, cfg):
            P = model.perception(torch.as_tensor(frames[:t + 1], device=dev)[None], fix)
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(P), rw, td, act), model.belief.initial(1, dev))
            bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            cache = model.denoiser.new_cache()
            gen = torch.Generator(device=dev).manual_seed(24004)
            for s in range(t - 7, t + 1):
                cache = M.context_pass(model.denoiser, tuple(z[:, s:s + 1] for z in P.parts()), prev[:, s:s + 1],
                                       model.belief.memory(bp[:, s:s + 1]), fix[:, s:s + 1], cache, cfg, gen)
            cache = {'t': cache['t'], 'kv': [tuple(z.expand(K, *z.shape[1:]) for z in kv) for kv in cache['kv']]}
            xs = M.sample(model.denoiser, [tuple(z.shape[2:]) for z in P.parts()], act[:, -1:].expand(K, -1),
                          model.belief.memory(y[:, -1:]).expand(K, -1, -1, -1),
                          torch.as_tensor(query, device=dev)[None, None].expand(K, 1, -1),
                          cache, cfg, torch.Generator(device=dev).manual_seed(24005))
            real = model.perception(torch.as_tensor(frames[t:t + 2], device=dev),
                                    torch.as_tensor(query, device=dev)[None].expand(2, -1))
        true = L['s_body'][off[e] + t:off[e] + t + 2, 2]
        read = reader(torch.cat([real.p[:, 65], real.v], 1).float())['y'].argmax(-1).cpu().numpy()
        assert np.array_equal(read, true)
        prob = reader(torch.cat([xs[0][:, 0, 65], xs[2][:, 0]], 1).float())['y'].softmax(-1)
        digits = prob.argmax(-1).cpu().numpy()
        out.append({'e': e, 't': t, 'current': int(true[0]), 'true_next': int(true[1]),
                    'generated_drop': float((digits < true[0]).mean()),
                    'generated_digits': np.bincount(digits, minlength=10).tolist(),
                    'mean_reader_confidence': float(prob.max(-1).values.mean())})
        generated.append((xs[0].cpu(), xs[2].cpu(), y[:, -1].float().cpu()))
        percepts.append(tuple(z[:, -8:].cpu() for z in P.parts()))
    assert all(torch.equal(x, y) for x, y in zip(*percepts))
    result = {'checkpoint': str(ckpt), 'checkpoint_sha256': TR.file_sha256(ckpt),
              'reader_holdout_accuracy': valid, 'recent_full_frames_actions_outcomes_exact': True,
              'recent_percepts_exact': True, 'K': K, 'cases': out,
              'contrast': out[0]['generated_drop'] - out[1]['generated_drop']}
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
