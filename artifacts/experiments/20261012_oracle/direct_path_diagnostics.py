"""Frozen inference interventions: routing, generation and source-copy bookkeeping.

No optimizer, fitting, threshold changes or training labels. Simulator source
identities are evaluation-only. Interventions diagnose paths, not repairs.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import direct_model as M
import direct_train as T
import direct_eval as E
import direct_coherence as C
import direct_analysis as A
import loss_intervention as L
from d4mj.train import autocast_context

OUT = T.OUT / 'diagnostics/paths'
VARIANTS = ('base', 'routing_fixed', 'generation_fixed', 'hard_destination_choice',
            'copy_only', 'generation_only', 'remove_offsource_source_copy')


def candidates(frames, generated):
    grid = frames.view(len(frames), 9, 9, 192)
    padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
    parts = [grid] + [padded[:, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in L.TW.NEIGHBOURS]
    return torch.stack([x.reshape(frames.shape) for x in parts] + [generated], -2)


def finish(parts, weights):
    return F.layer_norm((parts * weights[..., None]).sum(-2), (192,))


def source_slots(unit):
    offsets = [(0, 0), *L.TW.NEIGHBOURS]
    return [(p, offsets.index(((unit['source'] // 9) - p // 9,
                              (unit['source'] % 9) - p % 9))) for p in unit['points']]


@torch.no_grad()
def export():
    config = T.settings()
    data, inputs = E.cohort('exploratory')
    report = json.loads((T.OUT / 'eval_exploratory_u3000/analysis.json').read_text())
    units = [u for u in report['arms']['energy']['source_units'] if u['primary'] and u['dual_reader_clean']]
    assert len(units) == 14 and len({u['root'] for u in units}) == 14
    checkpoint = T.OUT / 'training/energy_u3000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name, digest in saved['contract']['sources_inputs'].items():
        assert L.sha(L.ROOT / name) == digest, name
    reference = torch.load(T.OUT / 'eval_exploratory_u3000/pred_energy.pt', map_location='cpu', weights_only=False, mmap=True)
    contract = {'format': 'direct_frozen_path_diagnostics_v1', 'question': 'Which deployed completion path produces incoherent source/destination samples?',
                'checkpoint_sha256': L.sha(checkpoint), 'base_prediction_sha256': L.sha(T.OUT / 'eval_exploratory_u3000/pred_energy.pt'),
                'inputs': inputs, 'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p) for p in (__file__, M.__file__, E.__file__, C.__file__)},
                'variants': list(VARIANTS), 'units': units, 'samples': 64, 'batch': 4,
                'routing_or_generation_fixed': 'hold that path at the same root first-noise draw; vary the other path with the original64 noises',
                'hard_destination_choice': 'argmax candidate independently per destination; still no source constraint',
                'source_intervention': 'remove only routes that copy this source into other destinations; renormalize remaining candidates',
                'scope': 'post-hoc fixed-world mechanism diagnostics; no optimization or proposed-repair verdict'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (OUT / 'contract.json').exists():
        assert json.loads((OUT / 'contract.json').read_text()) == contract
    else:
        L.atomic(contract, OUT / 'contract.json', True)
    world = M.DirectWorld().cuda().eval()
    world.load_state_dict(saved['world'])
    bank = torch.load(A.BANK, map_location='cpu', weights_only=False)
    real_head = torch.load(A.REAL, map_location='cpu', weights_only=False)['heads']['real']
    for start in sorted({u['root'] // 4 * 4 for u in units}):
        path = OUT / 'batches' / f'{start:04d}.pt'
        if path.exists():
            assert torch.load(path, map_location='cpu', weights_only=False)['contract_id'] == cid
            continue
        context = data['context_tokens'][start:start + 4].cuda().float()
        frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
        ac = F.pad(data['actions'][start:start + 4].cuda(), (0, 1))
        generator = torch.Generator().manual_seed(2026101402 + start)
        noises = torch.cat([M.draw_noise((len(context), 16), generator) for _ in range(32)]).cuda()
        with autocast_context(config):
            h, ha = world.backbone_full(frames, ac)
        selected = [u for u in units if start <= u['root'] < start + 4]
        local = [u['root'] - start for u in selected]
        values = {name: [] for name in VARIANTS}
        bookkeeping = []
        for k, noise in enumerate(noises):
            with autocast_context(config):
                output, weights, generated = world.complete(frames, ac, h, ha, noise)
            out, w, gen = output[:, 14], weights[:, 14], generated[:, 14]
            if k == 0:
                fixed_w, fixed_gen = w.clone(), gen.clone()
            parts = candidates(frames[:, 14], gen)
            fixed_parts = parts.clone(); fixed_parts[..., -1, :] = fixed_gen
            hard = F.one_hot(w.argmax(-1), 6).to(w.dtype)
            copy = w.clone(); copy[..., -1] = 0; copy = copy / copy.sum(-1, keepdim=True).clamp_min(1e-12)
            removed = w.clone()
            for u in selected:
                for p, slot in source_slots(u):
                    if p != u['source']:
                        removed[u['root'] - start, p, slot] = 0
            removed = removed / removed.sum(-1, keepdim=True).clamp_min(1e-12)
            variants = {'base': out, 'routing_fixed': finish(parts, fixed_w),
                        'generation_fixed': finish(fixed_parts, w), 'hard_destination_choice': finish(parts, hard),
                        'copy_only': finish(parts, copy), 'generation_only': F.layer_norm(gen, (192,)),
                        'remove_offsource_source_copy': finish(parts, removed)}
            for name, value in variants.items():
                values[name].append(value[local].cpu())
            trace = []
            for u in selected:
                b = u['root'] - start
                slots = source_slots(u)
                sw = torch.stack([w[b, p, slot] for p, slot in slots])
                points = u['points']; z = (parts[b, points] * w[b, points, :, None]).sum(-2)
                inv_std = torch.rsqrt(z.var(-1, unbiased=False) + 1e-5)
                kind = C.READER_TO_LAYOUT.index(u['kind'])
                alpha = (real_head['w'][:-1, kind] / real_head['sd']).cuda().double()
                bias = real_head['w'][-1, kind].double() - (real_head['mu'].double() * alpha.cpu()).sum()
                centered = parts[b, points].double() - parts[b, points].double().mean(-1, keepdim=True)
                terms = (centered @ alpha) * w[b, points].double() * inv_std[:, None].double()
                predicted = terms.sum(-1) + bias.cuda()
                decoded = C._linear(real_head, out[b, points].cpu())[..., kind]
                error = float((predicted.cpu() - decoded).abs().max())
                assert error < 2e-5, error
                source_terms = torch.stack([terms[j, slot] for j, (_, slot) in enumerate(slots)])
                trace.append({'source_copy_weights': sw.cpu(), 'source_copy_mass': float(sw.sum()),
                              'generation_weights': w[b, points, -1].cpu(), 'source_copy_score_terms': source_terms.cpu(),
                              'generation_score_terms': terms[:, -1].cpu(), 'total_scores': predicted.cpu(),
                              'score_decomposition_maxabs': error})
            bookkeeping.append(trace)
        predictions = {name: torch.stack(rows, 1) for name, rows in values.items()}
        replay = float((predictions['base'] - reference['predictions'][[u['root'] for u in selected]]).abs().max())
        assert replay == 0, replay
        L.atomic({'contract_id': cid, 'units': selected, 'predictions': predictions, 'bookkeeping': bookkeeping,
                  'base_saved_prediction_maxabs': replay}, path)
        print(json.dumps({'stage': 'paths', 'batch_start': start, 'sources': len(selected), 'base_maxabs': replay}), flush=True)
    return contract


def analyze():
    torch.set_num_threads(4)
    contract = json.loads((OUT / 'contract.json').read_text())
    bank = torch.load(A.BANK, map_location='cpu', weights_only=False)
    head = torch.load(A.REAL, map_location='cpu', weights_only=False)['heads']['real']
    units, predictions, traces = [], {name: [] for name in VARIANTS}, []
    for path in sorted((OUT / 'batches').glob('*.pt')):
        row = torch.load(path, map_location='cpu', weights_only=False)
        units.extend(row['units'])
        for name in VARIANTS:
            predictions[name].append(row['predictions'][name])
        for j in range(len(row['units'])):
            traces.append({name: torch.stack([r[j][name] if isinstance(r[j][name], torch.Tensor) else torch.tensor(r[j][name])
                                              for r in row['bookkeeping']]) for name in row['bookkeeping'][0][j]})
    assert len(units) == 14
    groups = list(range(14))
    result = {'contract_sha256': L.sha(OUT / 'contract.json'), 'source_units': units, 'variants': {}, 'source_bookkeeping': []}
    base_flags = C.decode(torch.cat(predictions['base']), bank, head)['mob']['real']
    for name, rows in predictions.items():
        decoded = C.decode(torch.cat(rows), bank, head)
        result['variants'][name] = {}
        for reader, flags in decoded['mob'].items():
            metrics = {key: [] for key in ('valid', 'absent', 'multiple', 'source_duplicate', 'valid_move')}
            events = []
            for j, unit in enumerate(units):
                ev = C._event(flags[j], unit['points'], unit['kind']); events.append(ev.tolist())
                metrics['valid'].append(float((ev < 5).double().mean()))
                metrics['absent'].append(float((ev == 5).double().mean()))
                metrics['multiple'].append(float((ev == 6).double().mean()))
                metrics['valid_move'].append(float(((ev > 0) & (ev < 5)).double().mean()))
                dup = flags[j, :, unit['source'], unit['kind']] & flags[j, :, unit['points'][1:], unit['kind']].any(-1)
                metrics['source_duplicate'].append(float(dup.double().mean()))
            result['variants'][name][reader] = {'per_root': metrics, 'intervals': {k: A.interval(v, groups) for k, v in metrics.items()}, 'events': events}
    for j, unit in enumerate(units):
        ev = C._event(base_flags[j], unit['points'], unit['kind'])
        tr = traces[j]
        result['source_bookkeeping'].append({'unit': unit, 'trace': {k: v.tolist() for k, v in tr.items()},
            'mass_summary': {label: float(tr['source_copy_mass'][mask].mean()) if mask.any() else None
                             for label, mask in (('valid', ev < 5), ('invalid', ev >= 5), ('multiple', ev == 6), ('absent', ev == 5))}})
    L.atomic(result, OUT / 'analysis.json', True)
    print(json.dumps({'stage': 'path_analysis', 'variants': {n:{r:{k:v['mean'] for k,v in q['intervals'].items()} for r,q in readers.items()} for n,readers in result['variants'].items()}}), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('export', 'analyze')); a = p.parse_args()
    export() if a.mode == 'export' else analyze()
