"""Frozen CPU readings for direct versus paired continuous event completion.

All physical margins and fresh coverage requirements are inherited unchanged
from direct_generator/criteria.json. New comparisons were declared before code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import torch

import direct_analysis as A
import direct_coherence as C

ROOT, BASE = A.ROOT, A.BASE
OUT = BASE / 'paired_event'
PREREG = Path(__file__).with_name('PAIRED_PREREG.md')
HISTORICAL_CRITERIA = A.OUT / 'criteria.json'
ARMS = ('direct', 'paired')


def prepare_criteria(path=OUT/'criteria.json'):
    old = A.verify_criteria(HISTORICAL_CRITERIA)
    criteria = {key: value for key, value in old.items() if key != 'criteria_id'}
    criteria.update({'format': 'paired_frozen_criteria_v1',
                     'historical_criteria_id': old['criteria_id'],
                     'historical_criteria_sha256': A.sha(HISTORICAL_CRITERIA),
                     'primary': 'upper95(paired-direct)<0 on scored35 joint occupancy ES plus valid source/movement and joint source-event comparisons',
                     'matched_rules': {'valid_sources': 'lower95(paired-direct)>0',
                                       'valid_cow_movement': 'lower95(paired-direct)>0',
                                       'cow_joint_registered_event_brier': 'upper95(paired-direct)<0; independent reader negative mean'},
                     'mechanism_rules': 'Reference/packet/routing/shuffle/selectivity are diagnostic; no post-hoc numerical pass thresholds',
                     'scope': 'one-step controlled stationary-camera endpoint coordination; no actor, H16 or uncertainty decomposition'})
    criteria['sources'] = {**old['sources'], str(PREREG.relative_to(ROOT)): A.sha(PREREG),
                           str(Path(__file__).relative_to(ROOT)): A.sha(__file__),
                           str(HISTORICAL_CRITERIA.relative_to(ROOT)): A.sha(HISTORICAL_CRITERIA)}
    criteria['criteria_id'] = hashlib.sha256(json.dumps(criteria, sort_keys=True).encode()).hexdigest()
    if Path(path).exists():
        assert json.loads(Path(path).read_text()) == criteria, 'criteria drift'
    else:
        A.atomic(criteria, path)
    print(json.dumps({'stage': 'paired_criteria_frozen', 'criteria_id': criteria['criteria_id'], 'path': str(path)}), flush=True)
    return criteria


def verify_criteria(path):
    criteria = json.loads(Path(path).read_text())
    for name, digest in criteria['sources'].items():
        assert A.sha(ROOT/name) == digest, f'frozen criteria drift: {name}'
    payload = {key: value for key, value in criteria.items() if key != 'criteria_id'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == criteria['criteria_id']
    return criteria


def source_metrics(report, predictions, bundle, bank, head, reference=None):
    """Keep invalid/camera-mismatched samples in law and movement denominators."""
    decoded = C.decode(predictions, bank, head)['mob']
    shifts = torch.tensor(report['sampled_camera']['shift_per_root'])
    ref_flags = C.decode(reference[:, None], bank, head)['mob'] if reference is not None else None
    ref_shifts = C._camera_shifts(bundle['root_tokens'], reference[:, None]) if reference is not None else None
    n = len(bundle['roots'])
    output = {}
    for reader, flags in decoded.items():
        rows = {name: [[] for _ in range(n)] for name in
                ('valid', 'valid_move', 'cow_valid_move', 'cow_registered_event_brier',
                 'source_duplicate', 'absent_or_multiple', 'reference_valid',
                 'reference_invalid_sample_valid', 'reference_valid_sample_valid')}
        details = []
        for u in report['source_units']:
            if not u['primary'] or not u['reader_clean']:
                continue
            i, p, kind = u['root'], u['points'], u['kind']
            truth = C._event(bundle['fan_layout'][i], p, kind)
            if len(truth.unique()) < 2:
                continue
            event = C._event(flags[i], p, kind)
            registered = (shifts[i] == 0).all(-1)
            joint = torch.where(registered, event, 7)
            valid = (event < 5) & registered
            moving = (event > 0) & (event < 5) & registered
            duplicate = flags[i, :, p[0], kind] & flags[i, :, p[1:], kind].any(-1)
            values = {'valid': float(valid.double().mean()), 'valid_move': float(moving.double().mean()),
                      'source_duplicate': float(duplicate.double().mean()),
                      'absent_or_multiple': float((event >= 5).double().mean())}
            brier = float(C._categorical_brier(joint[None], truth[None], 8)[0])
            if kind == 1:
                values['cow_valid_move'] = values['valid_move']
                values['cow_registered_event_brier'] = brier
            ref_event = None
            if ref_flags is not None:
                ref_event = int(C._event(ref_flags[reader][i], p, kind)[0])
                ref_valid = ref_event < 5 and bool((ref_shifts[i, 0] == 0).all())
                values['reference_valid'] = float(ref_valid)
                values['reference_valid_sample_valid' if ref_valid else 'reference_invalid_sample_valid'] = values['valid']
            for name, value in values.items():
                rows[name][i].append(value)
            details.append({'root': i, 'source': u['source'], 'kind': kind, 'events': event.tolist(),
                            'registered_events': joint.tolist(), 'real_events': truth.tolist(),
                            'sample_frequencies': [(joint == j).double().mean().item() for j in range(8)],
                            'real_frequencies': [(truth == j).double().mean().item() for j in range(8)],
                            'reference_event': ref_event, 'registered_event_brier': brier})
        output[reader] = {'raw_per_root': {key: [float(np.mean(v)) if v else None for v in rs] for key, rs in rows.items()},
                          'source_details': details}
    return output


def execution(payload):
    path = OUT/'validation.json'
    proof = json.loads(path.read_text()) if path.exists() else {}
    source_bound = bool(proof.get('sources')) and all(A.sha(ROOT/p) == digest for p, digest in proof.get('sources', {}).items())
    passed = proof.get('passed') is True and source_bound and payload.get('noise_replay_maxabs') == 0
    return {'passed': passed, 'validation_sha256': A.sha(path) if path.exists() else None,
            'source_bound': source_bound, 'repeat_exact': payload.get('noise_replay_maxabs') == 0,
            'final_noise_output_maxabs': float((payload['predictions']-payload['predictions'][:, :1]).abs().max())}


def analyze(directory, fresh=None, criteria_path=OUT/'criteria.json'):
    criteria = verify_criteria(criteria_path)
    bundle = A.load(fresh) if fresh else A.exploratory_bundle()
    groups = A.clusters(bundle['roots'])
    bank, head = A.load(A.BANK), A.load(A.REAL)['heads']['real']
    reports, payloads, hashes, contracts, paths, coverage, sources = {}, {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        path = Path(directory)/f'pred_{arm}.pt'
        payload = A.load(path)
        assert payload['arm'] == arm and payload['roots'] == bundle['roots']
        pred = payload['predictions']
        assert pred.shape == (len(bundle['roots']), criteria['samples'], 81, 192)
        assert not pred.is_cuda and torch.isfinite(pred).all()
        for name, digest in payload['contract']['sources'].items():
            assert A.sha(ROOT/name) == digest, f'evaluation source drift: {name}'
        actions = bundle['actions']
        if actions.ndim == 2:
            actions = actions[:, -1]
        report = C.analyze(pred, bundle['root_tokens'], bundle['fan_tokens'], bundle['root_layout'], bundle['fan_layout'],
                           bank, head, shifts=bundle['shifts'], actions=actions,
                           root_tiles=bundle['root_tiles'], real_tiles=bundle['fan_tiles'])
        alternatives, coverage[arm] = A.source_alternatives(report, pred, bundle, bank, head)
        report['actual_alternative_coverage_per_root'] = alternatives
        report['scored_reader_disagreement_per_root'] = A.scored_reader_disagreement(pred, bank, head)
        sources[arm] = source_metrics(report, pred, bundle, bank, head, payload.get('reference'))
        reports[arm], payloads[arm], hashes[arm] = report, payload, A.sha(path)
        contracts[arm] = {key: payload[key] for key in ('contract_id', 'contract', 'update', 'costs') if key in payload}
        paths[arm] = execution(payload)
    paired, checks, physical = {}, {}, {}
    for reader in ('real', 'bank'):
        def contrast(name, x, y):
            d = A.subtract(x, y)
            paired[reader+'_'+name] = {'raw_per_root': d, 'interval': A.interval(d, groups)}
            return paired[reader+'_'+name]['interval']
        rd, rp = reports['direct']['raw_per_root'][reader], reports['paired']['raw_per_root'][reader]
        contrast('joint_energy_difference', rp['scored_joint_occupancy_energy_reader_reference'], rd['scored_joint_occupancy_energy_reader_reference'])
        for name in ('valid', 'valid_move', 'cow_valid_move', 'cow_registered_event_brier'):
            contrast(name+'_difference', sources['paired'][reader]['raw_per_root'][name], sources['direct'][reader]['raw_per_root'][name])
    upper = lambda x: x['ci95'][1] is not None and x['ci95'][1] < 0
    lower = lambda x: x['ci95'][0] is not None and x['ci95'][0] > 0
    checks['primary_distribution'] = upper(paired['real_joint_energy_difference']['interval'])
    checks['valid_sources_improve'] = lower(paired['real_valid_difference']['interval'])
    checks['valid_cow_movement_improves'] = lower(paired['real_cow_valid_move_difference']['interval'])
    checks['cow_joint_law_improves'] = upper(paired['real_cow_registered_event_brier_difference']['interval'])
    rp = reports['paired']['raw_per_root']['real']
    for name, metric in (('source_absent_or_multiple', None), ('source_duplicate', 'source_and_other_destination_fraction'),
                         ('source_camera_mismatch', 'camera_mismatch_fraction')):
        values = A.add(rp['primary_clean_absent_fraction'], rp['primary_clean_multiple_destination_fraction']) if metric is None else rp['primary_clean_'+metric]
        reading = A.interval(values, groups)
        physical[name] = {'raw_per_root': values, 'interval': reading, 'margin': criteria['margins'][name]}
        checks[name] = reading['ci95'][1] is not None and reading['ci95'][1] <= criteria['margins'][name]
    static = A.subtract(reports['paired']['static_content']['scored_sampled_corruption_per_root'], reports['direct']['static_content']['scored_sampled_corruption_per_root'])
    paired['static_corruption_difference'] = {'raw_per_root': static, 'interval': A.interval(static, groups)}
    checks['static_fidelity'] = paired['static_corruption_difference']['interval']['ci95'][1] <= criteria['margins']['static_paired_increase']
    alt = reports['paired']['actual_alternative_coverage_per_root']['real']
    alternative_interval = A.interval(alt, groups)
    checks['actual_alternatives'] = alternative_interval['ci95'][0] is not None and alternative_interval['ci95'][0] >= criteria['minimum_actual_alternative_coverage']
    disagreement = A.subtract(reports['paired']['scored_reader_disagreement_per_root'], reports['direct']['scored_reader_disagreement_per_root'])
    paired['reader_disagreement_difference'] = {'raw_per_root': disagreement, 'interval': A.interval(disagreement, groups)}
    clean_roots = {u['root'] for u in reports['paired']['source_units'] if u['primary'] and u['reader_clean']}
    clean_diff = [x if i in clean_roots else None for i, x in enumerate(paired['real_joint_energy_difference']['raw_per_root'])]
    paired['reader_valid_source_joint_energy_difference'] = {'raw_per_root': clean_diff, 'interval': A.interval(clean_diff, groups)}
    bank_brier = paired['bank_cow_registered_event_brier_difference']['interval']['mean']
    checks['reader_sensitivity'] = (paired['bank_joint_energy_difference']['interval']['mean'] < 0
                                    and bank_brier is not None and bank_brier < 0
                                    and paired['reader_valid_source_joint_energy_difference']['interval']['mean'] is not None
                                    and paired['reader_valid_source_joint_energy_difference']['interval']['mean'] < 0
                                    and paired['reader_disagreement_difference']['interval']['ci95'][1] <= criteria['margins']['reader_disagreement_paired_increase'])
    reasons = [key+' failed' for key, passed in checks.items() if not passed]
    valid_execution = all(row['passed'] for row in paths.values())
    final = {payloads[arm]['update'] for arm in ARMS} == {criteria['primary_checkpoint']}
    missing = alternative_interval['roots'] == 0 or paired['real_cow_valid_move_difference']['interval']['roots'] == 0
    fresh_coverage = (fresh is not None and bundle.get('coverage_met') is True and len(bundle['roots']) >= 128
                      and coverage['paired']['variable_clean_roots'] >= 20 and coverage['paired']['variable_clean_cow_roots'] >= 20)
    if not final:
        verdict = 'milestone_diagnostic'
    elif not valid_execution:
        verdict = 'inconclusive_execution'; reasons.append('missing or failed source/numerical/live-noise proof')
    elif missing:
        verdict = 'inconclusive_coverage'
    elif all(checks.values()) and fresh is None:
        verdict = 'exploratory_promising'
    elif all(checks.values()) and fresh_coverage:
        verdict = 'confirmed_success'
    elif fresh is not None and not fresh_coverage:
        verdict = 'inconclusive_coverage'
    else:
        verdict = 'coherent_completion_not_established'
    # Decompose deterministic correction from stochastic spread; both are in LN output coordinates.
    references = {}
    for arm, payload in payloads.items():
        ref = payload['reference'].double(); pred = payload['predictions'].double()
        correction = pred.mean(1)-ref
        spread = (pred-pred.mean(1, keepdim=True)).square().mean(1)
        actions = bundle['actions']
        if actions.ndim == 2:
            actions = actions[:, -1]
        reference_report = C.analyze(payload['reference'][:, None].expand(-1, 2, -1, -1), bundle['root_tokens'], bundle['fan_tokens'],
                                     bundle['root_layout'], bundle['fan_layout'], bank, head,
                                     shifts=bundle['shifts'], actions=actions,
                                     root_tiles=bundle['root_tiles'], real_tiles=bundle['fan_tiles'])
        references[arm] = {'source_metrics': {r: {k: A.interval(v, groups) for k, v in q['raw_per_root'].items()
                                                   if k.startswith('reference_')} for r, q in sources[arm].items()},
                           'report': reference_report,
                           'systematic_correction_mse_per_root': correction.square().mean((1, 2)).tolist(),
                           'stochastic_variance_per_root': spread.mean((1, 2)).tolist(),
                           'scope': 'Correction to a potentially invalid deterministic baseline; not identified physical causes'}
    result = {'format': 'paired_analysis_v1', 'criteria_id': criteria['criteria_id'], 'criteria_sha256': A.sha(criteria_path),
              'verdict': verdict, 'allows_confirmation': verdict == 'exploratory_promising', 'checks': checks,
              'failed_conditions': reasons, 'coverage': coverage, 'execution': paths, 'physical': physical,
              'paired': paired, 'source_metrics': sources, 'reference_diagnostics': references,
              'actual_alternatives': {'raw_per_root': alt, 'interval': alternative_interval},
              'prediction_hashes': hashes, 'prediction_contracts': contracts,
              'fresh_bundle_sha256': A.sha(fresh) if fresh else None, 'arms': reports,
              'scope': criteria['scope'], 'negative_validity_requires': ['Frozen real-reader controls', 'Live post-LN independent noise/gradients and exact resume',
                                                                         'Optimization curves before budget-sufficiency claim']}
    destination = Path(directory)/'analysis.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == result, 'analysis evidence drift'
    else:
        A.atomic(result, destination)
    print(json.dumps({'stage': 'paired_analysis_done', 'verdict': verdict, 'checks': checks, 'path': str(destination)}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare-criteria'); q.add_argument('--criteria', type=Path, default=OUT/'criteria.json')
    q = sub.add_parser('analyze'); q.add_argument('--directory', type=Path, required=True)
    q.add_argument('--fresh', type=Path); q.add_argument('--criteria', type=Path, default=OUT/'criteria.json')
    a = p.parse_args(); torch.set_num_threads(4)
    prepare_criteria(a.criteria) if a.command == 'prepare-criteria' else analyze(a.directory, a.fresh, a.criteria)


if __name__ == '__main__':
    main()
