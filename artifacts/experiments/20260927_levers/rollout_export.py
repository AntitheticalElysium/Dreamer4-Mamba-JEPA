"""Part I of the 10-10 program (NOTEBOOK "PROGRAM", predeclared 09:10): save every variant's predictions on the diagnosis futures
once, so that every readout (scoreboard.py, the Part II diagnostics) reads the same tensors.
Data: teval's Raw diagnosis cache (1,002 held roots: 4 context frames + actions, the factual 16-step future, the 17 one-step successors
under key 0) and stochdiag's five futures (sample 0 = the factual one). Protocol = teval.evaluate's (window min(4 + k, cap), cap = the
world's frames - 1; 4 frames for the one-step successors), so a deterministic world's `gen` / `tf` / `one` here are teval's.
Per variant, in artifacts/eda/scoreboard_v1/<variant>/ (fp16 / int16 memmaps):
  gen     [R,16,81,192]  self-fed rollout of the factual 16 actions (each imagined frame fed back)
  tf      [R,16,81,192]  teacher-forced step at every depth: frame k predicted from the TRUE frames before it
  one     [R,17,81,192]  the 17 one-step successors of the root (4 context frames)
  w_gen, w_tf [R,16,81,6]  corrt routing weights of the predicted frame (self, up, down, left, right, generate)
  codes_gen, codes_tf [R,16,4], codes_one [R,17,4]  Delta codes used (stage-A variants)
  meta.json  world / stage-A / prior files and their sha256, script sha256, protocol
Variants:
  --world NAME=world.pt              a tworld checkpoint (deterministic)
  --prior NAME=prior.pt [--samples S] [--postcodes]   a dworld stage-B prior and its stage-A file, exported as
        NAME_zeroed      Delta None (the stage-A decoder alone)
        NAME_sample<j>   every predicted transition's 4 codes sampled by the prior; gen: from the imagined history (e22_imagine's
                         rollout and RNG: one generator, seed 20261009, consumed only by gen, in e22_imagine's order: variants
                         zeroed, sample0.., batches of 32 in root order, depths in order -- so its summaries must reproduce
                         e22_imagine exactly); tf / one: from the true history (past transitions carry the posterior's codes of the
                         true transitions), drawn from a SECOND generator (seed 20261010) so gen's draws are untouched
        NAME_postcodes   every predicted transition carries the posterior's code of the TRUE transition (gen: imagined frames fed
                         back, as e22_imagine --postcodes; tf: P2's conditioning; one: the posterior on (root, action, key-0 successor))
Resume: a variant with DONE is skipped; the gen generator's state is saved after each prior variant and restored.
Usage: rollout_export.py [--world NAME=path] ... [--prior NAME=path --samples S --postcodes] [--roots N (tests)] [--device cpu]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
H, N = 16, 17


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


class Store:
    def __init__(self, d, R, stage_a):
        d.mkdir(parents=True, exist_ok=True)
        mm = lambda n, shape, dt: np.lib.format.open_memmap(d / f'{n}.npy', mode='w+', dtype=dt, shape=shape)
        self.gen, self.tf = mm('gen', (R, H, 81, 192), np.float16), mm('tf', (R, H, 81, 192), np.float16)
        self.one = mm('one', (R, N, 81, 192), np.float16)
        self.w_gen, self.w_tf = mm('w_gen', (R, H, 81, 6), np.float16), mm('w_tf', (R, H, 81, 6), np.float16)
        if stage_a:
            self.codes_gen, self.codes_tf = mm('codes_gen', (R, H, 4), np.int16), mm('codes_tf', (R, H, 4), np.int16)
            self.codes_one = mm('codes_one', (R, N, 4), np.int16)

    def flush(self):
        for v in self.__dict__.values():
            v.flush()


def weights(world, b):
    w = getattr(world, 'last_weights', None)
    return w[:, -1].float().cpu().view(b, 81, 6).numpy() if w is not None else np.zeros((b, 81, 6), np.float32)


def main():
    import teval as T
    import stochdiag as SD
    import check_e16 as CE
    import dworld as DW
    from d4mj.config import config_from_dict
    import spatial as Sp
    args = sys.argv[1:]
    dev = torch.device(args[args.index('--device') + 1] if '--device' in args else 'cuda')
    S = int(args[args.index('--samples') + 1]) if '--samples' in args else 4
    worlds = [args[i + 1].split('=', 1) for i, a in enumerate(args) if a == '--world']
    priors = [args[i + 1].split('=', 1) for i, a in enumerate(args) if a == '--prior']
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5, _ = SD.token_cache(torch.device('cpu'))
    R = int(args[args.index('--roots') + 1]) if '--roots' in args else len(meta['seed'])
    ctx, ca, fa, fut, one_true = cache['ctx'][:R], cache['ctx_a'][:R], cache['fut_a'][:R], cache['fut'][:R], cache['one'][:R]
    # true frames: teval's cache for the deterministic variants (teval's own tensors), stochdiag's sample 0 for the prior variants
    # (e22_imagine's). They are the same frames encoded in different batches: 4.9e-6 of elements differ by one fp16 step (max 0.002).
    true_frames = torch.cat([ctx, fut5[:R, 0]], 1)                                                # [R, 20, 81, 192] (e22_imagine's)
    acts_all = torch.cat([ca, fa], 1)                                                             # [R, 19]
    own = sha(Path(__file__))
    out = OUT if '--roots' not in args else OUT.parent / 'scoreboard_test'

    def write_meta(d, info):
        (d / 'meta.json').write_text(json.dumps({**info, 'script_sha256': own, 'roots': R, 'protocol':
                                                 'teval.evaluate: 4 context frames, window min(4 + k, cap); one: 4 frames'}, indent=1))
        (d / 'DONE').write_text('1')

    for name, path in worlds:                                                                     # deterministic tworld checkpoints
        d = out / name
        if (d / 'DONE').exists():
            print(json.dumps({'variant': name, 'status': 'exists'}), flush=True); continue
        world, st = T.load_world(Path(path), dev)
        cap = world.time.shape[0] - 1
        st_ = Store(d, R, False)
        bs = 4 if st['args'].get('backbone') in ('fmamba', 'fcanvas') else 16                    # teval_export's batch (memory)
        with torch.no_grad():
            for i in range(0, R, bs):
                b = min(bs, R - i)
                c4 = ctx[i:i + b].float()
                fan = c4.repeat_interleave(N, 0)
                acts = torch.cat([ca[i:i + b].repeat_interleave(N, 0), torch.arange(N).repeat(b)[:, None]], 1)
                st_.one[i:i + b] = T.step(world, fan, acts, dev, config).view(b, N, 81, 192).half().numpy()
                g_frames, t_frames, hist = [c4[:, j] for j in range(4)], [c4[:, j] for j in range(4)], [ca[i:i + b, j] for j in range(3)]
                for k in range(H):
                    w = 4 if k == 0 else cap
                    a = torch.stack(hist[-(w - 1):] + [fa[i:i + b, k]], 1)
                    g = T.step(world, torch.stack(g_frames[-w:], 1), a, dev, config)
                    st_.w_gen[i:i + b, k] = weights(world, b)
                    t = T.step(world, torch.stack(t_frames[-w:], 1), a, dev, config)
                    st_.w_tf[i:i + b, k] = weights(world, b)
                    st_.gen[i:i + b, k], st_.tf[i:i + b, k] = g.half().numpy(), t.half().numpy()
                    g_frames.append(g); t_frames.append(fut[i:i + b, k].float()); hist.append(fa[i:i + b, k])
        st_.flush()
        write_meta(d, {'variant': name, 'kind': 'world', 'world': str(path), 'world_sha256': sha(path), 'world_name': st['name'],
                       'window_cap': cap})
        print(json.dumps({'variant': name, 'status': 'saved'}), flush=True)
        del world; torch.cuda.empty_cache()

    for name, path in priors:                                                                     # dworld stage-B prior + stage A
        st_b = torch.load(path, map_location='cpu', weights_only=False)
        world, post, st_a = DW.load_a(Path(st_b['stage_a']), dev)
        prior = DW.Prior().to(dev); prior.load_state_dict(st_b['prior']); prior.eval()
        cap = world.time.shape[0] - 1
        g = torch.Generator(device=dev).manual_seed(20261009)
        g2 = torch.Generator(device=dev).manual_seed(20261010)
        variants = ['zeroed'] + [f'sample{j}' for j in range(S)] + (['postcodes'] if '--postcodes' in args else [])
        info = {'kind': 'prior', 'prior': str(path), 'prior_sha256': sha(path), 'stage_a': st_b['stage_a'],
                'stage_a_sha256': sha(st_b['stage_a']), 'stage_a_name': st_a['name'], 'window_cap': cap}
        for vname in variants:
            d = out / f'{name}_{vname}'
            if (d / 'DONE').exists():
                gs = json.loads((d / 'meta.json').read_text()).get('generators_after')
                if gs:
                    g.set_state(torch.tensor(gs[0], dtype=torch.uint8)); g2.set_state(torch.tensor(gs[1], dtype=torch.uint8))
                print(json.dumps({'variant': d.name, 'status': 'exists'}), flush=True); continue
            st_ = Store(d, R, True)
            with torch.no_grad():
                for i in range(0, R, 32):                                                        # e22_imagine's batch
                    b = min(32, R - i)
                    tf_ = true_frames[i:i + b].float().to(dev); ac = acts_all[i:i + b].to(dev)
                    pc_all = post.quantizer(post.encode(tf_[:, :-1].flatten(0, 1), ac.flatten(), tf_[:, 1:].flatten(0, 1)))[1].view(b, -1, DW.K)
                    pc = pc_all[:, :3]
                    # gen: e22_imagine's loop exactly (generator g)
                    imagined, icodes = [tf_[:, j] for j in range(4)], [pc[:, j] for j in range(3)]
                    for k in range(H):
                        c = 3 + k
                        iw = min(len(imagined), cap)
                        win, wa = torch.stack(imagined[-iw:], 1), ac[:, c + 1 - iw:c + 1]
                        if vname == 'zeroed':
                            world.delta = None
                            nxt = T.step(world, win, wa, dev, config).to(dev)
                            scodes = torch.zeros(b, DW.K, dtype=torch.long, device=dev)
                        elif vname == 'postcodes':
                            scodes = pc_all[:, c]
                            nxt = CE.step(world, post, win, wa, torch.stack(icodes[len(icodes) - (iw - 1):] + [scodes], 1), dev, config).to(dev)
                        else:
                            isn = torch.stack(imagined[-DW.BLOCKS:], 1)
                            scodes, _ = prior.sample(isn, ac[:, c + 1 - isn.shape[1]:c + 1], torch.stack(icodes[-(isn.shape[1] - 1):], 1), generator=g)
                            nxt = CE.step(world, post, win, wa, torch.stack(icodes[len(icodes) - (iw - 1):] + [scodes], 1), dev, config).to(dev)
                        st_.w_gen[i:i + b, k] = weights(world, b)
                        st_.gen[i:i + b, k] = nxt.half().cpu().numpy(); st_.codes_gen[i:i + b, k] = scodes.cpu().numpy()
                        imagined.append(nxt); icodes.append(scodes)
                    # tf: the true window, past transitions = the posterior's true codes (generator g2 for the prior's draw)
                    for k in range(H):
                        c = 3 + k
                        iw = min(c + 1, cap)
                        win, wa = tf_[:, c + 1 - iw:c + 1], ac[:, c + 1 - iw:c + 1]
                        if vname == 'zeroed':
                            world.delta = None
                            nxt = T.step(world, win, wa, dev, config)
                            scodes = torch.zeros(b, DW.K, dtype=torch.long, device=dev)
                        else:
                            if vname == 'postcodes':
                                scodes = pc_all[:, c]
                            else:
                                lo = max(0, c + 1 - DW.BLOCKS)
                                scodes, _ = prior.sample(tf_[:, lo:c + 1], ac[:, lo:c + 1], pc_all[:, lo:c], generator=g2)
                            nxt = CE.step(world, post, win, wa, torch.cat([pc_all[:, c + 1 - iw:c], scodes[:, None]], 1), dev, config)
                        st_.w_tf[i:i + b, k] = weights(world, b)
                        st_.tf[i:i + b, k] = nxt.half().numpy(); st_.codes_tf[i:i + b, k] = scodes.cpu().numpy()
                    # one: the 17 actions from the root (4 context frames), 4 roots x 17 at a time (memory)
                    for j in range(0, b, 4):
                        bj = min(4, b - j)
                        fan = tf_[j:j + bj, :4].repeat_interleave(N, 0)
                        acts = torch.cat([ac[j:j + bj, :3].repeat_interleave(N, 0), torch.arange(N, device=dev).repeat(bj)[:, None]], 1)
                        pcf = pc[j:j + bj].repeat_interleave(N, 0)
                        if vname == 'zeroed':
                            world.delta = None
                            o = T.step(world, fan, acts, dev, config)
                            oc = torch.zeros(bj * N, DW.K, dtype=torch.long)
                        else:
                            if vname == 'postcodes':
                                succ = one_true[i + j:i + j + bj].float().to(dev).flatten(0, 1)          # key-0 successors
                                oc = post.quantizer(post.encode(fan[:, -1], acts[:, -1], succ))[1].view(bj * N, DW.K)
                            else:
                                oc, _ = prior.sample(fan, acts, pcf, generator=g2)
                            o = CE.step(world, post, fan, acts, torch.cat([pcf, oc[:, None]], 1), dev, config)
                        st_.one[i + j:i + j + bj] = o.view(bj, N, 81, 192).half().numpy()
                        st_.codes_one[i + j:i + j + bj] = oc.view(bj, N, DW.K).cpu().numpy()
            st_.flush()
            write_meta(d, {**info, 'variant': d.name, 'mode': vname,
                           'generators_after': [g.get_state().tolist(), g2.get_state().tolist()]})
            print(json.dumps({'variant': d.name, 'status': 'saved'}), flush=True)
        del world, post, prior; torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
