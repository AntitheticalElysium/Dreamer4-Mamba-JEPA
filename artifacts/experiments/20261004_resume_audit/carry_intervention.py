"""CPU reference interventions on the actual two Mamba carry components; declared in PLAN.md."""
import dataclasses
import json
import sys
import types
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sighting_intervention as S
from d4mj.mamba_recurrence import MambaCarry


def main():
    torch.set_num_threads(2)
    pool = torch.load(S.TW.POOLS['raw'] / 'pool.pt', weights_only=False, mmap=True)
    saved = torch.load(S.OUT / 'sighting_rows.pt', weights_only=False)
    ledger0 = saved['ledger']
    same0 = ledger0[:, 2] == ledger0[:, 4]
    picks = []
    rg = torch.Generator().manual_seed(S.SEED)
    for same in (True, False):
        ids = torch.where(same0 == same)[0]
        picks.append(ids[torch.randperm(len(ids), generator=rg)[:64]])
    ledger = ledger0[torch.cat(picks)]
    arrays, actions, truth, _, sighting, donor = S.inputs(pool, ledger)
    ids = [pool['ids'][int(i)][0] for i in ledger[:, 0]]
    names = {v: i for i, v in enumerate(sorted(set(ids)))}
    clusters = torch.tensor([names[v] for v in ids])
    results, raw = {}, {}
    for seed in (7, 8):
        path = Path(f'artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s{seed}_fmamba_u36000.pt')
        world, st = S.T.load_world(path, torch.device('cpu'))
        for l in world.layers:
            l.mix.settings = dataclasses.replace(l.mix.settings, backend='reference')
        ordinary = [l.mix.forward for l in world.layers]
        originals = [l.mix.scan for l in world.layers]
        active = {'kind': 'split', 'slots': None}

        def wrapper(mix, x, carry=None, *, backend=None, original):
            assert carry is None
            assert x.shape[1] > 1
            prefix, state = original(x[:, :-1], backend='reference')
            kind = active['kind']
            mask = torch.zeros(x.shape[0], dtype=torch.bool)
            slots = active['slots']
            if kind == 'unrelated_both':
                slots = (slots - 1 + 17) % 63 + 1
            mask[torch.arange(len(slots)) * 82 + slots] = True
            conv, ssm = state.conv, state.ssm
            if kind in ('conv', 'both', 'unrelated_both'):
                conv = conv.clone(); conv[mask] = 0
            if kind in ('ssm', 'both', 'unrelated_both'):
                ssm = ssm.clone(); ssm[mask] = 0
            final, state = original(x[:, -1:], MambaCarry(conv, ssm), backend='reference')
            return torch.cat([prefix, final], 1), state

        pred = torch.empty(len(ledger), 5, 2, 192)
        max_parity = 0.
        with torch.no_grad():
            for t in ledger[:, 1].unique().tolist():
                inds = torch.where(ledger[:, 1] == t)[0]
                for lo in range(0, len(inds), 4):
                    batch = inds[lo:lo + 4].tolist()
                    acts = torch.stack([actions[i] for i in batch])
                    active['slots'] = ledger[batch, 2] + 1
                    for j, kind in enumerate(('split', 'ssm', 'conv', 'both', 'unrelated_both')):
                        active['kind'] = kind
                        for l, fn in zip(world.layers, originals):
                            l.mix.forward = types.MethodType(lambda mix, x, carry=None, backend=None, fn=fn:
                                wrapper(mix, x, carry, backend=backend, original=fn), l.mix)
                        for cond in (0, 1):
                            frames = torch.stack([arrays[i][cond] for i in batch])
                            out = world(frames, acts)[0][:, -1]
                            pred[batch, j, cond] = out[torch.arange(len(batch)), ledger[batch, 2]]
                            if kind == 'split':
                                for l, fn in zip(world.layers, ordinary):
                                    l.mix.forward = fn
                                direct = world(frames, acts)[0][:, -1]
                                max_parity = max(max_parity, float((out - direct).abs().max()))
                                for l, fn in zip(world.layers, originals):
                                    l.mix.forward = types.MethodType(lambda mix, x, carry=None, backend=None, fn=fn:
                                        wrapper(mix, x, carry, backend=backend, original=fn), l.mix)
                print(json.dumps({'stage': 'carry', 'seed': seed, 'input_frames': t, 'cases': len(inds)}), flush=True)
        assert max_parity < 1e-4, f'split reference mismatch {max_parity}'
        err = (pred - truth[:, None, None]).square().sum(-1)
        by = {'split_max_abs_error': max_parity, 'checkpoint_sha256': S.sha(path)}
        for same in (True, False):
            m = (ledger[:, 2] == ledger[:, 4]) == same
            values = {}
            for j, kind in enumerate(('split', 'ssm', 'conv', 'both', 'unrelated_both')):
                delta = pred[:, j, 1] - pred[:, j, 0]
                pull = (delta * (donor - sighting)).sum(-1) / (donor - sighting).square().sum(-1).clamp_min(1e-12)
                values[kind] = {'original_error': float(err[m, j, 0].mean()),
                                'reset_error_change': S.interval((err[:, j, 0] - err[:, 0, 0])[m], clusters[m]),
                                'sighting_swap_error_change': S.interval((err[:, j, 1] - err[:, j, 0])[m], clusters[m]),
                                'donor_pull': S.interval(pull[m], clusters[m])}
            by['same' if same else 'moved'] = values
        results[st['name']] = by
        raw[st['name']] = pred
        print(json.dumps({'stage': 'carry_result', 'seed': seed, 'result': by}), flush=True)
    results['_provenance'] = {'script_sha256': S.sha(__file__), 'plan_sha256': S.sha(S.OUT / 'PLAN.md'),
                              'parent_rows_sha256': S.sha(S.OUT / 'sighting_rows.pt')}
    torch.save({'ledger': ledger, 'predictions': raw, 'truth': truth, 'sighting': sighting, 'donor': donor}, S.OUT / 'carry_rows.pt')
    (S.OUT / 'carry_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
