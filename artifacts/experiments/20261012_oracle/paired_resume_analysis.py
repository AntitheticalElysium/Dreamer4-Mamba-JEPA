"""Metadata-only JSON compatibility for the frozen paired-event evaluator.

Frozen scientific sources remain unchanged. Its cache check compared tuple
metadata with JSON lists. This runner converts ONLY four named metadata fields
before that existing check, proves full initial-report equality, and retains all
frozen source, input, reader, criterion and numerical checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch

import direct_analysis as A
import direct_coherence as C
import paired_analysis as P

FIELDS = ('event_names', 'registered_event_names', 'layout_types', 'scored_map_cells')
PROOF = P.OUT/'evaluation_resume_compatibility.json'
INITIAL = P.OUT/'eval_exploratory_u0000'


def source_contract(criteria_path):
    criteria = P.verify_criteria(criteria_path)
    saved = json.loads((INITIAL/'analysis.json').read_text())
    assert saved['criteria_id'] == criteria['criteria_id'], 'criterion identity changed'
    assert saved['criteria_sha256'] == A.sha(criteria_path), 'criterion bytes changed'
    files = [Path(__file__), Path(P.__file__), Path(C.__file__), Path(A.__file__), Path(criteria_path),
             P.HISTORICAL_CRITERIA, INITIAL/'analysis.json', INITIAL/'pred_direct.pt',
             INITIAL/'pred_paired.pt', P.OUT/'validation.json']
    contract = {'format': 'paired_metadata_resume_compatibility_v1',
                'criterion_id': criteria['criteria_id'],
                'normalization': 'tuple-to-list only at the four top-level C.analyze report metadata fields',
                'fields': list(FIELDS),
                'sources_inputs': {str(path.relative_to(P.ROOT)): A.sha(path) for path in files},
                'frozen_scientific_source_changes': False, 'criterion_changes': False,
                'initial_report_changes': False, 'numerical_tolerance': 0,
                'scope': 'JSON metadata representation only; no statistics, model, metric, reader, cache input or evaluation cohort change'}
    contract['contract_id'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract, saved


def numerical_counts(a, b):
    """Count exact numeric comparisons; no tolerance or float coercion."""
    compared, different = 0, 0
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in a:
            n, d = numerical_counts(a[key], b[key]); compared += n; different += d
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            n, d = numerical_counts(x, y); compared += n; different += d
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and not isinstance(b, bool)
        compared, different = 1, int(a != b)
    return compared, different


def prove(contract, saved, changes, criteria_path):
    if PROOF.exists():
        evidence = json.loads(PROOF.read_text())
        assert evidence['contract'] == contract, 'compatibility proof source/input drift'
        assert evidence['full_initial_report_exact'] is True and evidence['numerical_mismatch_count'] == 0
        assert evidence['metadata_fields_converted'] == 16
        return evidence, None
    before = A.sha(INITIAL/'analysis.json')
    start = len(changes)
    # Existing frozen analyze now checks its entire live result against the
    # existing cache. It cannot write that cache because it already exists.
    result = P.analyze(INITIAL, criteria_path=criteria_path)
    assert result == saved, 'metadata normalization did not establish full exact equality'
    assert A.sha(INITIAL/'analysis.json') == before, 'initial evidence was modified'
    converted = changes[start:]
    assert len(converted) == 16 and {entry['field'] for entry in converted} == set(FIELDS)
    compared, different = numerical_counts(saved, result)
    assert different == 0
    evidence = {'contract': contract, 'full_initial_report_exact': True,
                'numeric_scalar_pairs_compared': compared, 'numerical_mismatch_count': different,
                'metadata_fields_converted': len(converted), 'conversions': converted,
                'initial_analysis_sha256_before': before, 'initial_analysis_sha256_after': A.sha(INITIAL/'analysis.json'),
                'cache_reuse_basis': 'Full result equality plus exact source, criterion, validation and prediction bytes; not shape compatibility'}
    A.atomic(evidence, PROOF)
    return evidence, result


def run(directory, fresh=None, criteria_path=P.OUT/'criteria.json'):
    torch.set_num_threads(4)
    contract, saved = source_contract(criteria_path)
    original = C.analyze
    changes = []

    def normalize_metadata(*args, **kwargs):
        report = original(*args, **kwargs)
        for field in FIELDS:
            assert field in report and isinstance(report[field], (tuple, list)), field
            if isinstance(report[field], tuple):
                value = report[field]
                report[field] = list(value)
                # This changes only the container type; metadata values remain
                # exactly identical. All numerical arrays are untouched.
                assert tuple(report[field]) == value
                changes.append({'report_call': len(changes)//len(FIELDS), 'field': field,
                                'from': 'tuple', 'to': 'list', 'values_exact': True})
        return report

    C.analyze = normalize_metadata
    try:
        evidence, initial_result = prove(contract, saved, changes, criteria_path)
        if Path(directory).resolve() == INITIAL.resolve() and fresh is None and initial_result is not None:
            result = initial_result
        else:
            result = P.analyze(Path(directory), fresh, criteria_path)
    finally:
        C.analyze = original
    print(json.dumps({'stage': 'paired_metadata_compatibility', 'proof_sha256': A.sha(PROOF),
                      'criterion_id': contract['criterion_id'], 'numerical_mismatch_count': evidence['numerical_mismatch_count'],
                      'verdict': result['verdict'], 'directory': str(directory)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--criteria', type=Path, default=P.OUT/'criteria.json')
    args = parser.parse_args()
    run(args.directory, args.fresh, args.criteria)


if __name__ == '__main__':
    main()
