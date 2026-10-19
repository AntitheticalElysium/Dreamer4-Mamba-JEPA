"""Frozen post-hoc route/source-domain/LN localization; no optimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
from torch.nn import functional as F

import direct_analysis as A
import direct_coherence as C
import direct_model as D
import direct_train as T
import loss_intervention as L
import paired_analysis as PA
import paired_eval as E
import paired_model as P
from d4mj.train import autocast_context

OUT = PA.OUT/'diagnostics/failure_localization'
PREREG = Path(__file__).with_name('PAIRED_FAILURE_PREREG.md')
VARIANTS = ('base', 'hard_source_routes', 'scored_sources_only', 'unscored_sources_only')


def centered_rms(value):
    value = value.float()
    return (value-value.mean(-1, keepdim=True)).square().mean(-1).sqrt()


def copy_candidates(frames, generated):
    grid = frames.float().reshape(*frames.shape[:2], 9, 9, 192)
    pad = F.pad(grid, (0, 0, 1, 1, 1, 1))
    cands = [grid]+[pad[:, :, 1+dr:10+dr, 1+dc:10+dc] for dr, dc in P.NEIGHBOURS]
    return torch.stack([value.reshape(frames.shape) for value in cands]+[generated], -2)


def altered_packet(packet, variant, scored):
    result = dict(packet)
    if variant == 'hard_source_routes':
        result['routes'] = F.one_hot(packet['routes'].argmax(-1), 5).to(packet['routes'].dtype)
    elif variant in ('scored_sources_only', 'unscored_sources_only'):
        mask = scored if variant == 'scored_sources_only' else ~scored
        result['gate'] = packet['gate']*mask[None, None, :, None]
    else:
        assert variant == 'base'
    return result


@torch.no_grad()
def export(limit_batches=None):
    assert PREREG.is_file(), 'post-hoc diagnostic must precede computation'
    data, inputs = E.cohort('exploratory')
    final = PA.OUT/'eval_exploratory_u3000'
    checkpoints = {arm: PA.OUT/'training'/f'{arm}_u3000.pt' for arm in P.ARMS}
    contract = {'format': 'paired_failure_localization_v1', 'roots': data['roots'],
                'samples': 64, 'batch': 4, 'target_index': 14, 'variants': VARIANTS,
                'seed_rule': '2026101402 + batch_start', 'inputs': inputs,
                'checkpoints': {arm: L.sha(path) for arm, path in checkpoints.items()},
                'predictions': {arm: L.sha(final/f'pred_{arm}.pt') for arm in P.ARMS},
                'sources': {str(Path(path).relative_to(L.ROOT)): L.sha(path) for path in
                            (__file__, PREREG, P.__file__, D.__file__, E.__file__, PA.__file__, C.__file__)},
                'scope': 'Frozen inference interventions; no trained repairs or criterion changes'}
    contract = json.loads(json.dumps(contract))
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract_path = OUT/'contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract
    else:
        L.atomic(contract, contract_path, True)
    config = T.settings()
    scored = torch.zeros(81, dtype=torch.bool, device='cuda'); scored[list(P.INTERIOR)] = True
    for arm in P.ARMS:
        saved = A.load(checkpoints[arm])
        assert saved['arm'] == arm and saved['update'] == 3000
        for path, digest in saved['contract']['sources_inputs'].items():
            assert L.sha(L.ROOT/path) == digest, path
        cached = A.load(final/f'pred_{arm}.pt')
        world = E.model(arm).cuda().eval(); world.load_state_dict(saved['world'])
        variants = VARIANTS if arm == 'paired' else ('base',)
        for batch_index, start in enumerate(range(0, len(data['roots']), 4)):
            path = OUT/'batches'/f'{arm}_{start:04d}.pt'
            if path.exists():
                row = A.load(path)
                assert row['contract_id'] == cid and row['start'] == start
            else:
                context = data['context_tokens'][start:start+4].cuda().float()
                frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
                actions = F.pad(data['actions'][start:start+4].cuda(), (0, 1))
                rng = torch.Generator().manual_seed(2026101402+start)
                noises = torch.cat([D.draw_noise((len(context), 16), rng) for _ in range(32)]).cuda()
                outputs = {variant: [] for variant in variants}
                pres = {variant: [] for variant in variants}
                edits = {variant: [] for variant in variants}
                with autocast_context(config):
                    belief, action_belief = world.backbone_full(frames, actions)
                    reference = P.PairedWorld.reference(world, frames, actions, belief, action_belief)
                assert torch.equal(F.layer_norm(reference[:, 14].float(), (192,)).cpu(),
                                   cached['reference'][start:start+len(context)])
                for noise in noises:
                    with autocast_context(config):
                        if arm == 'direct':
                            output, weights, generated = world.complete(frames, actions, belief, action_belief, noise)
                            pre = (weights[..., None]*copy_candidates(frames, generated)).sum(-2)
                            assert torch.equal(F.layer_norm(pre.float(), (192,)), output)
                            values = {'base': (output, pre, pre-reference)}
                        else:
                            packet = world.event_packets(belief, noise)
                            values = {}
                            for variant in variants:
                                changed = altered_packet(packet, variant, scored)
                                delta, _ = world.completion.edits(belief, changed)
                                pre = reference+delta
                                values[variant] = (F.layer_norm(pre.float(), (192,)), pre, delta)
                    for variant, (output, pre, delta) in values.items():
                        outputs[variant].append(output[:, 14].cpu())
                        pres[variant].append(pre[:, 14].float().cpu())
                        edits[variant].append(centered_rms(delta[:, 14]).cpu())
                predictions = {key: torch.stack(values, 1) for key, values in outputs.items()}
                replay = float((predictions['base']-cached['predictions'][start:start+len(context)]).abs().max())
                assert replay == 0, replay
                summaries = {}
                for variant in variants:
                    pre = torch.stack(pres[variant], 1)
                    centered = pre-pre.mean(-1, keepdim=True)
                    summaries[variant] = {'pre_ln_variance': pre.var(1, unbiased=False).mean(-1),
                                           'pre_ln_centered_variance': centered.var(1, unbiased=False).mean(-1),
                                           'post_ln_variance': predictions[variant].float().var(1, unbiased=False).mean(-1),
                                           'edit_centered_rms': torch.stack(edits[variant], 1).mean(1),
                                           'pre_ln_centered_rms': centered_rms(pre).mean(1),
                                           'mean_ln_inverse_variance': (centered.square().mean(-1)+1e-5).reciprocal().mean(1)}
                row = {'contract_id': cid, 'start': start, 'roots': data['roots'][start:start+len(context)],
                       'reference_centered_rms': centered_rms(reference[:, 14]).cpu(),
                       'reference': F.layer_norm(reference[:, 14].float(), (192,)).cpu(),
                       'predictions': predictions, 'summaries': summaries, 'base_replay_maxabs': replay}
                L.atomic(row, path)
            print(json.dumps({'stage': 'failure_localization_export', 'arm': arm,
                              'completed_roots': start+len(row['roots']), 'total': len(data['roots'])}), flush=True)
            if limit_batches and batch_index+1 >= limit_batches:
                return
        del world; torch.cuda.empty_cache()


def analyze():
    torch.set_num_threads(4)
    contract = json.loads((OUT/'contract.json').read_text())
    for name, digest in contract['sources'].items():
        assert L.sha(L.ROOT/name) == digest, name
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    bundle = A.exploratory_bundle(); groups = A.clusters(bundle['roots'])
    assert bundle['roots'] == contract['roots']
    bank, head = A.load(A.BANK), A.load(A.REAL)['heads']['real']
    regions = {'scored35': list(P.INTERIOR), 'other_map': [i for i in range(63) if i not in P.INTERIOR],
               'HUD': list(range(63, 81)), 'all81': list(range(81))}
    result = {'format': 'paired_failure_localization_analysis_v1', 'contract_id': cid,
              'scope': contract['scope'], 'arms': {},
              'limitations': ['Inference hardening is not retrained hard routing',
                              'Normalization ratios describe the frozen map, not a sole causal explanation',
                              'Active-source masks are inference controls, not learned selective computation']}
    for arm in P.ARMS:
        rows = [A.load(p) for p in sorted((OUT/'batches').glob(f'{arm}_*.pt'))]
        assert rows and all(row['contract_id'] == cid for row in rows)
        assert [root for row in rows for root in row['roots']] == bundle['roots']
        reference = torch.cat([row['reference'] for row in rows])
        reference_rms = torch.cat([row['reference_centered_rms'] for row in rows])
        result['arms'][arm] = {'base_replay_maxabs': max(row['base_replay_maxabs'] for row in rows),
                               'reference_rms_per_cell': reference_rms.tolist(), 'variants': {}}
        for variant in (VARIANTS if arm == 'paired' else ('base',)):
            pred = torch.cat([row['predictions'][variant] for row in rows])
            report = C.analyze(pred, bundle['root_tokens'], bundle['fan_tokens'], bundle['root_layout'],
                               bundle['fan_layout'], bank, head, shifts=bundle['shifts'], actions=bundle['actions'],
                               root_tiles=bundle['root_tiles'], real_tiles=bundle['fan_tiles'])
            source = PA.source_metrics(report, pred, bundle, bank, head, reference)
            summary = {key: torch.cat([row['summaries'][variant][key] for row in rows])
                       for key in rows[0]['summaries'][variant]}
            magnitude = {}
            for region, cells in regions.items():
                per_root = {key: value[:, cells].double().mean(-1) for key, value in summary.items()}
                per_root['reference_centered_rms'] = reference_rms[:, cells].double().mean(-1)
                per_root['pre_ln_std'] = per_root['pre_ln_variance'].sqrt()
                per_root['post_ln_std'] = per_root['post_ln_variance'].sqrt()
                q = {key: A.interval(value.tolist(), groups) for key, value in per_root.items()}
                q['per_root'] = {key: value.tolist() for key, value in per_root.items()}
                q['root_quantiles'] = {key: {'median': float(value.median()),
                                             'p95': float(torch.quantile(value, .95)), 'max': float(value.max())}
                                       for key, value in per_root.items()}
                q['cell_quantiles'] = {key: {'median': float(value[:, cells].double().median()),
                                             'p95': float(torch.quantile(value[:, cells].double().flatten(), .95)),
                                             'max': float(value[:, cells].max())}
                                       for key, value in {**summary, 'reference_centered_rms': reference_rms}.items()}
                contributions = summary['post_ln_variance'][:, cells].double().sum(-1)
                total_contribution = float(contributions.sum())
                q['largest_post_variance_roots'] = [
                    {'index': int(i), 'root': bundle['roots'][int(i)],
                     'fraction_of_region_variance': float(contributions[i])/total_contribution if total_contribution else None,
                     **{key: float(value[i]) for key, value in per_root.items()}}
                    for i in contributions.argsort(descending=True)[:10]]
                q['reference_centered_rms'] = A.interval(reference_rms[:, cells].double().mean(-1).tolist(), groups)
                pre = float(summary['pre_ln_variance'][:, cells].sum())
                post = float(summary['post_ln_variance'][:, cells].sum())
                q['aggregate_post_to_pre_variance_ratio'] = post/pre if pre > 0 else None
                low = reference_rms[:, cells] < .1
                q['reference_rms_below_point1_fraction'] = float(low.double().mean())
                q['low_reference_post_variance_mean'] = float(summary['post_ln_variance'][:, cells][low].mean()) if low.any() else None
                q['other_reference_post_variance_mean'] = float(summary['post_ln_variance'][:, cells][~low].mean()) if (~low).any() else None
                magnitude[region] = q
            result['arms'][arm]['variants'][variant] = {
                'report': report, 'source_metrics': source,
                'source_summary': {reader: {name: A.interval(values, groups) for name, values in data['raw_per_root'].items()}
                                   for reader, data in source.items()},
                'magnitude_regions': magnitude, 'raw_magnitudes': {key: value.tolist() for key, value in summary.items()}}
            del pred
    L.atomic(result, OUT/'analysis.json', True)
    print(json.dumps({'stage': 'failure_localization_analyzed', 'path': str(OUT/'analysis.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'analyze'))
    parser.add_argument('--limit-batches', type=int)
    args = parser.parse_args()
    export(args.limit_batches) if args.mode == 'export' else analyze()
