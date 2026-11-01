"""Verify observational artifacts and preservation, without loading a world on GPU."""
import ast
import datetime
import hashlib
import json
import re
import subprocess

from .common import OUT, ROOT, OLD, fingerprints, save


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    opening = read('opening_identity.json')
    closing = fingerprints()
    assert opening == closing
    assert json.loads((OLD / 'closing_identity.json').read_text()) == closing
    assert subprocess.run(['git', 'diff', '--quiet', '--', 'saccade/'], cwd=ROOT).returncode == 0
    for p in OUT.glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    invalid_links = []
    for text in [read_path.read_text() for read_path in (OUT / 'REPORT.md', OUT / 'PROTOCOL.md')]:
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target and not (OUT / target.split('#')[0]).exists():
                invalid_links.append(target)
    assert not invalid_links, invalid_links
    assert all(x['stream_state_exact'] for x in read('training_histories_checks.json'))
    assert set(read('sampler_observation.json')['instrumentation_max_differences'].values()) == {0.0}
    n = read('production_outcomes.json')['all']
    assert n['n_histories'] == 30000
    assert round(n['termination']['recorded']['est'] * 30000) == 84
    assert round(n['termination']['generated']['est'] * 30000) == 1
    assert round(n['negative_reward']['recorded']['est'] * 30000) == 573
    assert round(n['negative_reward']['generated']['est'] * 30000) == 183
    assert len(list(OUT.glob('danger_*.pt'))) == 24
    assert len(read('activation_readout.json')) == 16
    assert read('likelihood_normalisation.json')['fraction_bins_abs_nll_error_gt_01'] == 0
    save('closing_identity.json', closing)
    selected = sorted([*OUT.glob('*.py'), *OUT.glob('*.json'), *OUT.glob('*.md'), *OUT.glob('*.png')])
    manifest = {
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checkpoint_reader_canonical_hashes_unchanged': True,
        'canonical_saccade_git_diff_empty': True,
        'all_scripts_parse': True,
        'all_report_local_links_exist': True,
        'native_hook_parity_exact': True,
        'saved_stream_states_exact_at_all_three_snapshots': True,
        'world_training_runs': 0,
        'world_optimizer_steps': 0,
        'model_interventions': 0,
        'new_measurement_reader_fits': 40,
        'world_seeds': 1,
        'untouched_final_split_used': False,
        'failed_controls_retained': [
            'Direct binary health reader full-percept positive control',
            'Validation-recall threshold transfer with saturated probabilities',
            'HUD reference support, including real terminal examples',
            'Unsupported generated entity labels',
        ],
        'artifact_sha256': {p.name: sha(p) for p in selected if p.name != 'analysis_manifest.json'},
    }
    save('analysis_manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'artifact_sha256'}, indent=2))
