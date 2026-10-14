"""Observation-only, resumable bright-view TRAIN ledger for direct completion.

The first RGB frame selects a window. A fixed interior scoring domain, not a
future-dependent movement mask, excludes one-step newly revealed map cells.
Brightness is an operational selection rule; it does not certify simulator day.
Only this module and its data manifests are written by the preparation process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / 'artifacts/eda/levers_mamba_long_pools_v1/raw'
LEDGER = ROOT / 'artifacts/experiments/20260929_mamba_integration/long_ledger.jsonl'
DATASET = ROOT / 'artifacts/lewm_m4_canonical/raw/dataset.json'
OUT = ROOT / 'artifacts/eda/oracle_v1/direct_generator/data'
FRAMES = 16
BRIGHT_Q75 = 150
MAP_ROWS = 7
GRID = 9
INTERIOR = [r * GRID + c for r in range(1, 6) for c in range(1, 8)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(obj, path, as_json=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w' if as_json else 'wb') as stream:
        if as_json:
            json.dump(obj, stream, indent=2)
            stream.write('\n')
        else:
            torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bright_statistics(rgb):
    """Batch uint8 RGB -> exact map max-channel 75th percentile (linear q75)."""
    rgb = np.asarray(rgb)
    assert rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3
    assert rgb.shape[1] == rgb.shape[2] and rgb.shape[1] % GRID == 0
    patch = rgb.shape[1] // GRID
    maximum = rgb[:, :MAP_ROWS * patch].max(-1).reshape(len(rgb), -1)
    return np.quantile(maximum, .75, axis=1).astype(np.float32)


def read_manifest(directory=OUT):
    """Training entry point: require completed, byte-bound eligible ledger."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert sha(directory / 'windows.pt') == manifest['windows_sha256']
    assert sha(POOL / 'manifest.json') == manifest['contract']['inputs']['pool_manifest']
    assert sha(POOL / 'labels.pt') == manifest['contract']['inputs']['pool_labels']
    assert sha(LEDGER) == manifest['contract']['inputs']['long_ledger']
    windows = torch.load(directory / 'windows.pt', weights_only=False, map_location='cpu')
    assert windows.dtype == torch.int64 and windows.shape == (manifest['eligible_windows'], 2)
    assert torch.equal(windows, windows.unique(dim=0, sorted=True)), 'ledger duplicates/order drift'
    return manifest, windows


def prepare(directory=OUT, stop_after_shards=None):
    torch.set_num_threads(4)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pool_manifest = json.loads((POOL / 'manifest.json').read_text())
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()]
    assert sha(LEDGER) == pool_manifest['contract']['ledger_sha256']
    assert len(rows) == pool_manifest['row_count'] == pool_manifest['contract']['shape'][0]
    assert pool_manifest['contract']['shape'][1:] == [64, 81, 192]
    assert sha(POOL / 'labels.pt') == pool_manifest['labels_sha256']
    pool = torch.load(POOL / 'labels.pt', map_location='cpu', mmap=True, weights_only=False)
    assert len(pool['terminal']) == len(rows)
    source_paths = json.loads(DATASET.read_text())['paths']
    source_manifests = {path: json.loads((ROOT / path / 'manifest.json').read_text()) for path in source_paths}
    contract = {
        'format': 'direct_bright_view_data_v1',
        'inputs': {
            'script': sha(Path(__file__)), 'dataset': sha(DATASET),
            'long_ledger': sha(LEDGER), 'pool_manifest': sha(POOL / 'manifest.json'),
            'pool_labels': pool_manifest['labels_sha256'],
            'tokens_sha256': pool_manifest['tokens_sha256'],
            'source_manifests': {path: sha(ROOT / path / 'manifest.json') for path in source_paths},
        },
        'selection': {
            'only_first_window_RGB_frame': True,
            'statistic': '75th percentile of per-pixel max(R,G,B), map rows only',
            'threshold_in_uint8_units': BRIGHT_Q75,
            'map_geometry': 'top7 of9 image patch rows; no HUD pixels',
            'future_or_simulator_fields': False,
            'label': 'operational bright-view; not simulator-certified daylight',
        },
        'windows': {'frames': FRAMES, 'starts': [0, 64 - FRAMES], 'all_actions_retained': True,
                    'sampler': 'uniform over unique eligible (pool_row,start) entries',
                    'terminal_oversampling': False},
        'scored_tokens': INTERIOR,
        'scoring_scope': 'fixed35 interior map cells; HUD and edge tokens not scored',
        'limitations': ['RGB brightness also selects terrain appearance and sleep appearance',
                        'no claim that nuisance is absent or four-way uncertainty is separated',
                        'first input may be bright while later frames darken; this is reported, not filtered',
                        'uniform windows overlap and duplicate factual transitions; counts are not independent',
                        'fixed interior is visible before any one-tile scroll, but occluded facts can remain unknown'],
    }
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (directory / 'contract.json').exists():
        assert json.loads((directory / 'contract.json').read_text()) == contract, 'selection/input contract drift'
    else:
        atomic(contract, directory / 'contract.json', True)
    if (directory / 'manifest.json').exists():
        manifest, _ = read_manifest(directory)
        assert manifest['contract_id'] == contract_id
        return manifest
    by_episode = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_episode[row['episode_id']].append((row_index, row))
        assert bool(pool['terminal'][row_index]) == row['terminal']
    found = set()
    cache_records = []
    all_windows = []
    action_counts = np.zeros(17, dtype=np.int64)
    row_counts = Counter()
    source_counts = Counter()
    selected_episodes = set()
    terminal_windows = 0
    selected_first_hist = []
    target_bright = 0
    target_total = 0
    unique_transition_count = 0
    started = time.time()
    shard_number = 0
    for source_index, (source, source_manifest) in enumerate(source_manifests.items()):
        for shard_index, shard in enumerate(source_manifest['shards']):
            path = ROOT / source / shard['file']
            stat = path.stat()
            fingerprint = {'sha256': shard['sha256'], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            cache_path = directory / 'cache' / f'source{source_index}_shard{shard_index:04d}.pt'
            if cache_path.exists():
                cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                assert cached['contract_id'] == contract_id and cached['source'] == fingerprint
            else:
                assert sha(path) == shard['sha256'], f'RGB/action source digest mismatch: {path}'
                payload = torch.load(path, mmap=True, weights_only=False, map_location='cpu')
                cached_episodes = {}
                for fields in payload['episodes']:
                    eid = fields['episode_id']
                    if eid not in by_episode:
                        continue
                    assert fields['split'] == 'train' and fields['uniform_eligible'], 'non-TRAIN input'
                    observations = fields['observations']
                    actions = fields['actions_taken'].cpu().numpy()
                    assert observations.dtype == torch.uint8 and observations.ndim == 4
                    wanted = set()
                    for row_index, row in by_episode[eid]:
                        start = row['start']
                        assert start >= 0 and start + 64 <= len(observations)
                        assert np.array_equal(actions[start:start + 63], pool['actions'][row_index].numpy())
                        if row['terminal']:
                            assert start + 64 == len(observations) and bool(fields['terminated'][-1])
                        wanted.update(range(start, start + 64))
                    indices = np.array(sorted(wanted), dtype=np.int64)
                    q75 = np.full(len(observations), -1, dtype=np.float32)
                    digest = hashlib.sha256()
                    digest.update(indices.tobytes())
                    for offset in range(0, len(indices), 128):
                        selected = indices[offset:offset + 128]
                        frames = observations[selected].numpy()
                        q75[selected] = bright_statistics(frames)
                        digest.update(frames.tobytes())
                    cached_episodes[eid] = {'q75': torch.from_numpy(q75),
                                            'actions': torch.from_numpy(actions.copy()),
                                            'RGB_frames_sha256': digest.hexdigest(),
                                            'scanned_first_and_target_frames': len(indices)}
                cached = {'contract_id': contract_id, 'source': fingerprint, 'episodes': cached_episodes}
                atomic(cached, cache_path)
            cache_records.append({'path': str(cache_path.relative_to(ROOT)), 'sha256': sha(cache_path),
                                  'source_path': str(path.relative_to(ROOT)), 'source': fingerprint})
            for eid, values in cached['episodes'].items():
                assert eid not in found
                found.add(eid)
                q75 = values['q75'].numpy()
                actions = values['actions'].numpy()
                touched = np.zeros(len(actions), dtype=bool)
                for row_index, row in by_episode[eid]:
                    starts = np.arange(65 - FRAMES, dtype=np.int64)
                    absolute = row['start'] + starts
                    assert (q75[absolute] >= 0).all()
                    keep = starts[q75[absolute] >= BRIGHT_Q75]
                    if not len(keep):
                        continue
                    selected_episodes.add(eid)
                    row_counts[row_index] += len(keep)
                    source_counts[source] += len(keep)
                    terminal_windows += len(keep) * int(row['terminal'])
                    all_windows.extend((row_index, int(start)) for start in keep)
                    selected_first_hist.extend(q75[row['start'] + keep].tolist())
                    target_indices = row['start'] + keep[:, None] + np.arange(1, FRAMES)[None]
                    target_bright += int((q75[target_indices] >= BRIGHT_Q75).sum())
                    target_total += target_indices.size
                    transition_indices = target_indices - 1
                    action_counts += np.bincount(actions[transition_indices].flatten(), minlength=17)
                    touched[transition_indices] = True
                unique_transition_count += int(touched.sum())
            shard_number += 1
            if shard_number % 20 == 0:
                print(json.dumps({'stage': 'RGB_manifest', 'shards_done': shard_number,
                                  'TRAIN_episodes_seen': len(found), 'eligible_windows': len(all_windows),
                                  'seconds': time.time() - started}), flush=True)
            if stop_after_shards and shard_number >= stop_after_shards:
                print(json.dumps({'stage': 'paused', 'shards_done': shard_number}), flush=True)
                return None
    assert found == set(by_episode), 'TRAIN episode ledger incompletely covered'
    windows = torch.tensor(sorted(all_windows), dtype=torch.int64)
    assert windows.ndim == 2 and windows.shape[1] == 2 and len(windows) > 0
    assert torch.equal(windows, windows.unique(dim=0, sorted=True))
    atomic(windows, directory / 'windows.pt')
    manifest = {
        'status': 'complete', 'contract_id': contract_id, 'contract': contract,
        'windows_sha256': sha(directory / 'windows.pt'),
        'eligible_windows': len(windows), 'candidate_windows': len(rows) * (65 - FRAMES),
        'TRAIN_pool_rows': len(rows), 'eligible_pool_rows': len(row_counts),
        'TRAIN_episodes': len(found), 'eligible_episodes': len(selected_episodes),
        'terminal_metadata_windows': terminal_windows,
        'source_windows': dict(source_counts),
        'all_action_transition_exposures': {str(i): int(n) for i, n in enumerate(action_counts)},
        'unique_factual_transitions': unique_transition_count,
        'scored_transition_exposures': target_total,
        'target_first_frame_bright_fraction': target_bright / target_total,
        'selected_first_RGB_q75_quantiles': dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                                    np.quantile(selected_first_hist, [0, .25, .5, .75, 1]).tolist())),
        'shard_caches': cache_records,
        'actual_source_hash_verification': 'every source shard verified against store manifest before cache creation',
        'resumption': 'per-shard caches retain per-episode first/target RGB profiles and source fingerprint',
    }
    assert action_counts.sum() == target_total
    atomic(manifest, directory / 'manifest.json', True)
    print(json.dumps({'stage': 'complete', **{k: v for k, v in manifest.items()
                                             if k not in ('contract', 'shard_caches')}}), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=OUT)
    parser.add_argument('--stop-after-shards', type=int)
    args = parser.parse_args()
    prepare(args.directory, args.stop_after_shards)
