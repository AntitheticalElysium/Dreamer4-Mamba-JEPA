"""Resumable frozen paired-event interventions and result-gated fresh collection.

These inference-only interventions use fixed evaluation sources. They neither
fit readers nor change the model, loss, data or checkpoint selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

# CPU analysis commands pass CUDA_VISIBLE_DEVICES=''; an ordinary export must
# not inherit direct_analysis's CPU-only default before CUDA initializes.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import torch
from torch.nn import functional as F

import direct_analysis as A
import direct_coherence as C
import direct_joint_shuffle as J
import direct_model as M
import direct_train as T
import loss_intervention as L
import paired_analysis as PA
import paired_eval as E
import paired_model as P
from d4mj.train import autocast_context

OUT = PA.OUT
VARIANTS = ('base_first', 'fully_sampled', 'single_event', 'untied_removal', 'untied_arrival', 'event_off')


def clone_packet(packet):
    return {key: value.clone() for key, value in packet.items()}


def replace_sources(base, other, selected, start):
    result = clone_packet(base)
    for unit in selected:
        b, source = unit['root']-start, unit['source']
        for key in result:
            result[key][b, 14, source] = other[key][b, 14, source]
    return result


@torch.no_grad()
def export(update=3000, limit_batches=None):
    assert update == 3000, 'causal diagnostics use declared final checkpoint only'
    directory = OUT/'diagnostics'/f'u{update:04d}'
    data, inputs = E.cohort('exploratory')
    analysis_path = OUT/f'eval_exploratory_u{update:04d}'/'analysis.json'
    report = json.loads(analysis_path.read_text())
    units = [u for u in report['arms']['paired']['source_units'] if u['primary'] and u['reader_clean']]
    units = [u for u in units if len(C._event(data['fan_layout'][u['root']], u['points'], u['kind']).unique()) >= 2]
    assert units and len({u['root'] for u in units}) == len(units), 'one isolated source per current root'
    checkpoint = OUT/'training'/f'paired_u{update:04d}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name, digest in saved['contract']['sources_inputs'].items():
        assert L.sha(L.ROOT/name) == digest, name
    reference = torch.load(OUT/f'eval_exploratory_u{update:04d}'/'pred_paired.pt', map_location='cpu', weights_only=False, mmap=True)
    contract = {'format': 'paired_frozen_event_diagnostics_v1', 'update': update, 'units': units,
                'checkpoint_sha256': L.sha(checkpoint), 'analysis_sha256': L.sha(analysis_path),
                'base_prediction_sha256': L.sha(OUT/f'eval_exploratory_u{update:04d}'/'pred_paired.pt'),
                'inputs': inputs, 'variants': list(VARIANTS), 'samples': 64, 'batch': 4,
                'sources': {str(Path(s).relative_to(L.ROOT)): L.sha(s) for s in
                            (__file__, P.__file__, E.__file__, PA.__file__, C.__file__, J.__file__)},
                'intervention': 'replace exactly one source packet at t14 after spatial attention; all other packets and deterministic belief fixed to draw0',
                'untied': 'single-event removal with draw0 arrivals, and draw0 removal with single-event arrivals',
                'reference_limit': 'Reference-invalid corrections do not establish movement causality; reference-valid stratum reported separately',
                'scope': 'Exploratory final-checkpoint diagnostics; no model selection or optimization'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cp = directory/'contract.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract
    else:
        L.atomic(contract, cp, True)
    world = P.PairedWorld().cuda().eval(); world.load_state_dict(saved['world'])
    config = T.settings()
    for index, start in enumerate(sorted({u['root']//4*4 for u in units})):
        path = directory/'batches'/f'{start:04d}.pt'
        if path.exists():
            assert torch.load(path, map_location='cpu', weights_only=False)['contract_id'] == cid
        else:
            context = data['context_tokens'][start:start+4].cuda().float()
            frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
            ac = F.pad(data['actions'][start:start+4].cuda(), (0, 1))
            rng = torch.Generator().manual_seed(2026101402+start)
            noises = torch.cat([M.draw_noise((len(context), 16), rng) for _ in range(32)]).cuda()
            selected = [u for u in units if start <= u['root'] < start+4]
            local = [u['root']-start for u in selected]
            outputs = {name: [] for name in VARIANTS}; traces = []
            with autocast_context(config):
                h, ha = world.backbone_full(frames, ac)
                fixed = world.event_packets(h, noises[0])
                base, _, _ = world.complete_from_packets(frames, ac, h, ha, fixed)
            for noise in noises:
                with autocast_context(config):
                    packet = world.event_packets(h, noise)
                    single = replace_sources(fixed, packet, selected, start)
                    off = clone_packet(fixed)
                    for unit in selected:
                        off['gate'][unit['root']-start, 14, unit['source']] = 0
                    full, routes, aux = world.complete_from_packets(frames, ac, h, ha, packet, trace=True)
                    variants = {'base_first': base,
                                'fully_sampled': full,
                                'single_event': world.complete_from_packets(frames, ac, h, ha, single)[0],
                                'untied_removal': world.complete_from_packets(frames, ac, h, ha, fixed, removal_packets=single)[0],
                                'untied_arrival': world.complete_from_packets(frames, ac, h, ha, fixed, arrival_packets=single)[0],
                                'event_off': world.complete_from_packets(frames, ac, h, ha, off)[0]}
                for name, value in variants.items():
                    outputs[name].append(value[local, 14].cpu())
                arriving_mass = torch.zeros_like(packet['gate'])
                contributor_count = torch.zeros_like(packet['gate'])
                for edge, offset in enumerate(P.NEIGHBOURS, 1):
                    mass = packet['gate']*routes[..., edge:edge+1]
                    arriving_mass += P.shift_grid(mass, *offset)
                    contributor_count += P.shift_grid((mass > .1).float(), *offset)
                tr = []
                for unit in selected:
                    b, source, points = unit['root']-start, unit['source'], unit['points']
                    probs = routes[b, 14, source]
                    tr.append({'gate': packet['gate'][b, 14, source].cpu(), 'routes': probs.cpu(),
                               'max_route': probs.max().cpu(),
                               'entropy': -(probs[probs > 0]*probs[probs > 0].log()).sum().cpu(),
                               'split_mass': (1-probs.square().sum()).cpu(),
                               'source_edit_norm': aux['source_edits'][b, 14, source].norm().cpu(),
                               'arrival_edit_norms': aux['arrivals'][b, 14, source].norm(dim=-1).cpu(),
                               'arriving_mass_at_points': arriving_mass[b, 14, points, 0].cpu(),
                               'contributors_above_point1_at_points': contributor_count[b, 14, points, 0].cpu()})
                traces.append(tr)
            predictions = {name: torch.stack(rows, 1) for name, rows in outputs.items()}
            replay = float((predictions['fully_sampled']-reference['predictions'][[u['root'] for u in selected]]).abs().max())
            assert replay == 0, replay
            L.atomic({'contract_id': cid, 'units': selected, 'predictions': predictions, 'traces': traces,
                      'reference': reference['reference'][[u['root'] for u in selected]],
                      'fully_sampled_replay_maxabs': replay}, path)
        print(json.dumps({'stage': 'paired_event_diagnostics', 'batch_start': start, 'sources': len(units)}), flush=True)
        if limit_batches and index+1 >= limit_batches:
            return


def unit_readings(predictions, units, bank, head, root_tokens):
    decoded = C.decode(predictions, bank, head)['mob']
    shifts = C._camera_shifts(root_tokens, predictions)
    result = {}
    for reader, flags in decoded.items():
        rows = {name: [] for name in ('valid', 'absent', 'multiple', 'duplicate', 'valid_move')}
        events = []
        for j, unit in enumerate(units):
            ev = C._event(flags[j], unit['points'], unit['kind'])
            registered = (shifts[j] == 0).all(-1)
            event = torch.where(registered, ev, 7); events.append(event.tolist())
            rows['valid'].append(float((event < 5).double().mean()))
            rows['absent'].append(float((event == 5).double().mean()))
            rows['multiple'].append(float((event == 6).double().mean()))
            rows['valid_move'].append(float(((event > 0)&(event < 5)).double().mean()))
            duplicate = flags[j, :, unit['source'], unit['kind']] & flags[j, :, unit['points'][1:], unit['kind']].any(-1)
            rows['duplicate'].append(float(duplicate.double().mean()))
        result[reader] = {'per_root': rows, 'events': events, 'intervals': {name: A.interval(v) for name, v in rows.items()}}
    return result, decoded


def analyze(update=3000):
    torch.set_num_threads(4)
    directory = OUT/'diagnostics'/f'u{update:04d}'
    contract = json.loads((directory/'contract.json').read_text())
    for name, digest in contract['sources'].items():
        assert L.sha(L.ROOT/name) == digest, name
    rows = [torch.load(p, map_location='cpu', weights_only=False) for p in sorted((directory/'batches').glob('*.pt'))]
    assert rows and sum(len(row['units']) for row in rows) == len(contract['units']), 'diagnostics incomplete'
    units = [u for row in rows for u in row['units']]
    data, _ = E.cohort('exploratory')
    group_keys = A.clusters([data['roots'][u['root']] for u in units])
    root_tokens = data['root_tokens'][[u['root'] for u in units]]
    bank, head = A.load(A.BANK), A.load(A.REAL)['heads']['real']
    predictions = {name: torch.cat([row['predictions'][name] for row in rows]) for name in VARIANTS}
    readings, decoded = {}, {}
    for name, pred in predictions.items():
        readings[name], decoded[name] = unit_readings(pred, units, bank, head, root_tokens)
    reference = torch.cat([row['reference'] for row in rows])
    ref_flags = C.decode(reference[:, None], bank, head)['mob']
    ref_shift = C._camera_shifts(root_tokens, reference[:, None])
    causal, routing = {}, {}
    for reader in ('real', 'bank'):
        causal_rows = []
        for j, unit in enumerate(units):
            kind, source = unit['kind'], unit['source']
            others = unit['points'][1:]
            base = decoded['base_first'][reader][j]
            altered = decoded['single_event'][reader][j]
            source_clear = base[:, source, kind] & ~altered[:, source, kind]
            destination_create = (~base[:, others, kind] & altered[:, others, kind]).any(-1)
            base_event = torch.tensor(readings['base_first'][reader]['events'][j])
            event = torch.tensor(readings['single_event'][reader]['events'][j])
            changed = base_event != event
            source_clear_and_destination = source_clear & destination_create
            valid_move_switch = (base_event == 0) & (event > 0) & (event < 5)
            ref_event = int(C._event(ref_flags[reader][j], unit['points'], kind)[0])
            ref_valid = ref_event < 5 and bool((ref_shift[j, 0] == 0).all())
            inside = torch.zeros(81, dtype=torch.bool); inside[unit['points']] = True
            untouched = (predictions['single_event'][j, :, ~inside]-predictions['base_first'][j, :, ~inside]).abs().max().item()
            causal_rows.append({'unit': unit, 'reference_valid': ref_valid, 'reference_event': ref_event,
                                'event_changed_fraction': changed.double().mean().item(),
                                'source_cleared_fraction': source_clear.double().mean().item(),
                                'destination_created_fraction': destination_create.double().mean().item(),
                                'joint_clear_create_fraction': source_clear_and_destination.double().mean().item(),
                                'valid_stay_to_move_fraction': valid_move_switch.double().mean().item(),
                                'untouched_tokens_maxabs': untouched,
                                'event_transition_counts': torch.bincount(base_event*8+event, minlength=64).reshape(8, 8).tolist()})
        causal[reader] = {'rows': causal_rows,
                          'reference_valid_sources': sum(row['reference_valid'] for row in causal_rows),
                          'mechanism_scope': 'Full source packet intervention at fixed deterministic context; no causal claim if reference-valid sources show no valid movement switch'}
        routing[reader] = []
        for row in rows:
            for j, unit in enumerate(row['units']):
                index = next(i for i, u in enumerate(units) if u == unit)
                ev = torch.tensor(readings['fully_sampled'][reader]['events'][index])
                values = {key: torch.stack([trace[j][key] for trace in row['traces']]) for key in row['traces'][0][j]}
                summary = {}
                for label, mask in (('valid_move', (ev > 0)&(ev < 5)), ('stay', ev == 0), ('invalid', ev >= 5)):
                    summary[label] = {'samples': int(mask.sum()),
                                      **{key: value[mask].float().mean(0).tolist() if mask.any() else None for key, value in values.items()}}
                routing[reader].append({'unit': unit, 'by_decoded_event': summary,
                                        'route_offsets': [list(x) for x in P.OFFSETS]})
    # Reuse the historical decoded source-cell shuffle exactly: 128 permutations
    # preserve each of the five cell marginals and isolate occupancy dependence.
    # Camera registration remains binding in the main evaluator; it is not
    # changed into a second shuffle-dependent quantity for this mechanism test.
    dependence = {}
    for reader in ('real', 'bank'):
        original, altered, detail = [], [], []
        for j, unit in enumerate(units):
            local = decoded['fully_sampled'][reader][j, :, unit['points'], unit['kind']]
            actual = J.stats(local)
            rng = torch.Generator().manual_seed(2026101403+j)
            draws = []
            for repetition in range(128):
                shuffled = J.permuted(local, rng)
                assert torch.equal(local.sum(0), shuffled.sum(0)), 'shuffle changed marginal'
                draws.append(J.stats(shuffled))
            mean = J.means(draws)
            original.append(actual['valid_outcome_probability'])
            altered.append(mean['valid_outcome_probability'])
            detail.append({'unit': unit, 'actual': actual, 'shuffled_mean': mean})
        diff = A.subtract(original, altered)
        dependence[reader] = {'unshuffled': original, 'shuffled': altered, 'difference': diff,
                              'interval': A.interval(diff, group_keys), 'source_details': detail,
                              'repetitions': 128, 'seed_rule': '2026101403 + source_order',
                              'definition': 'Occupancy exactly-one validity, matching direct_joint_shuffle.py; camera enforced separately'}
    # Selectivity includes both spread and mean fidelity. A low-variance biased
    # output is not a selective stochastic model. These masks are evaluation-
    # only and were never available to the training objective.
    final = json.loads((OUT/f'eval_exploratory_u{update:04d}'/'analysis.json').read_text())
    selectivity = {}
    real_mean = data['fan_tokens'].float().mean(1)
    all_groups = A.clusters(data['roots'])
    for arm in P.ARMS:
        payload = A.load(OUT/f'eval_exploratory_u{update:04d}'/f'pred_{arm}.pt')
        mean_error = (payload['predictions'].float().mean(1)-real_mean).double().square().mean(-1)
        variation = final['arms'][arm]['sampled_variation']
        selectivity[arm] = {}
        for name in ('scored_domain', 'scored_semantic_stable', 'scored_mob_uncertain', 'HUD'):
            mask = torch.tensor(variation['masks'][name], dtype=torch.bool)
            counts = mask.sum(-1)
            error = [float(mean_error[i, mask[i]].mean()) if counts[i] else None for i in range(len(mask))]
            predicted_variance = variation[name]['variance_per_root']
            real_variance = variation[name]['real_variance_per_root']
            denominator = sum(real_variance)
            selectivity[arm][name] = {'mean_to_real_fan_mse_per_root': error,
                                      'mean_to_real_fan_mse_interval': A.interval(error, all_groups),
                                      'predicted_variance_per_root': predicted_variance,
                                      'real_variance_per_root': real_variance,
                                      'aggregate_variance_ratio': sum(predicted_variance)/denominator if denominator else None,
                                      'variation_share_interval': A.interval(variation[name]['share_per_root'], all_groups),
                                      'eligible_cells_per_root': counts.tolist()}
    selectivity['paired_minus_direct'] = {}
    for name in ('scored_domain', 'scored_semantic_stable', 'scored_mob_uncertain', 'HUD'):
        mean_diff = A.subtract(selectivity['paired'][name]['mean_to_real_fan_mse_per_root'],
                               selectivity['direct'][name]['mean_to_real_fan_mse_per_root'])
        variance_diff = A.subtract(selectivity['paired'][name]['predicted_variance_per_root'],
                                   selectivity['direct'][name]['predicted_variance_per_root'])
        selectivity['paired_minus_direct'][name] = {'mean_mse_difference': A.interval(mean_diff, all_groups),
                                                    'variance_amount_difference': A.interval(variance_diff, all_groups)}
    result = {'format': 'paired_event_diagnostics_analysis_v1', 'contract_sha256': L.sha(directory/'contract.json'),
              'variants': readings, 'routing': routing, 'causal_single_source': causal, 'cell_shuffle': dependence,
              'selectivity_and_mean_fidelity': selectivity,
              'scope': contract['scope'], 'limitations': ['Single source intervention is an ownership test, not identification of environmental causal variables',
                                                          'Soft concentration is descriptive; semantic movement and no collision remain binding',
                                                          'Reference-invalid repairs alone do not demonstrate stochastic movement causality']}
    L.atomic(result, directory/'analysis.json', True)
    print(json.dumps({'stage': 'paired_diagnostics_analyzed', 'path': str(directory/'analysis.json')}), flush=True)


def fresh_gate(path, training):
    report = json.loads(Path(path).read_text())
    if report.get('allows_confirmation') is not True:
        raise ValueError('fresh collection requires explicit allows_confirmation=True')
    record = json.loads((Path(training)/'training.json').read_text())
    if record.get('completed') is not True or record['update'] != 3000:
        raise ValueError('both matched 3000-update arms must be complete')
    digests = {}
    for arm in P.ARMS:
        cp = Path(training)/f'{arm}_u3000.pt'
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        assert saved['arm'] == arm and saved['update'] == 3000 and saved['contract_id'] == record['contract_id']
        evidence = report['prediction_contracts'][arm]['contract']
        assert evidence['training_contract_id'] == record['contract_id'] and evidence['checkpoint_sha256'] == L.sha(cp)
        digests[arm] = L.sha(cp)
    return {'report_sha256': L.sha(path), 'training_contract_id': record['contract_id'], 'checkpoint_hashes': digests,
            'paired_gate_source_sha256': L.sha(__file__), 'paired_prereg_sha256': L.sha(PA.PREREG)}


def collect_fresh(result):
    import direct_fresh as FRESH
    # The original collector is unchanged and source-pinned. Only its hardcoded
    # arm-identity gate is replaced by the equally strict matched-study gate.
    previous = FRESH.gate_report
    FRESH.gate_report = fresh_gate
    try:
        return FRESH.collect(result, directory=OUT/'fresh', training=OUT/'training')
    finally:
        FRESH.gate_report = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    q = p.add_subparsers(dest='mode', required=True)
    for name in ('export', 'analyze'):
        v = q.add_parser(name); v.add_argument('--updates', type=int, default=3000)
        if name == 'export':
            v.add_argument('--limit-batches', type=int)
    v = q.add_parser('collect-fresh'); v.add_argument('--exploratory-result', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'export':
        export(a.updates, a.limit_batches)
    elif a.mode == 'analyze':
        analyze(a.updates)
    else:
        collect_fresh(a.exploratory_result)


if __name__ == '__main__':
    main()
