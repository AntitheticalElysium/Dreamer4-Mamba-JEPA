"""CPU-only, result-gated fresh confirmation fan for the two direct generators.

No collection occurs without an explicit allows_confirmation=True report and
completed, frozen two-arm checkpoints. New episodes/futures are evaluation-only;
the existing real-state readers and thresholds remain frozen. Roots are selected
by observed RGB; simulator labels only establish evaluation coverage afterwards.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import direct_data as D
import direct_coherence as C

ROOT = D.ROOT
sys.path.insert(0, str(ROOT))  # standalone CLI must resolve artifacts and d4mj
OUT = ROOT / 'artifacts/eda/oracle_v1/direct_generator'
RESERVATION = D.OUT / 'new_seed_confirmation_reservation.json'
BANK = ROOT / 'artifacts/eda/oracle_v1/snap_bank_readers.pt'
REAL_READER = ROOT / 'artifacts/eda/oracle_v1/learned_innovation/eval_u3000_h01/mob_heads.pt'
ENCODER = ROOT / 'artifacts/lewm_m4_canonical/raw/joint/step-010000.pt'
BASE_SEED = 2026101400
ENVS, LIMIT, KEYS = 24, 2500, 32
EPSILONS = (.1, .25, .5, 1.)
MIN_BRIGHT, MIN_PRIMARY, MIN_COW = 128, 20, 20
MAX_BRIGHT, MAX_EPISODES = 512, 1536


def client_time():
    return datetime.now(ZoneInfo('Australia/Sydney')).isoformat()


def root_time(episode_id, steps):
    """One preselected root per episode; no RGB/semantic outcome search."""
    if steps < 15:
        return None
    value = hashlib.sha256(f'direct_fresh_root:{episode_id}'.encode()).digest()
    return 14 + int.from_bytes(value[:8], 'little') % (steps - 14)


def stopping(counts):
    success = (counts['bright_roots'] >= MIN_BRIGHT and counts['primary_roots'] >= MIN_PRIMARY
               and counts['cow_roots'] >= MIN_COW)
    cap = counts['bright_roots'] >= MAX_BRIGHT or counts['episodes'] >= MAX_EPISODES
    return success, success or cap


def gate_report(path, training):
    """Fail before importing JAX/collector or creating fresh-data files."""
    path, training = Path(path), Path(training)
    report = json.loads(path.read_text())
    if report.get('allows_confirmation') is not True:
        raise ValueError('fresh collection requires explicit allows_confirmation=True')
    record = json.loads((training / 'training.json').read_text())
    if record.get('completed') is not True or record['update'] != 3000:
        raise ValueError('both matched 3000-update arms must be complete')
    checkpoints = {arm: training / f'{arm}_u3000.pt' for arm in ('point', 'energy')}
    digests = {}
    for arm, checkpoint in checkpoints.items():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['arm'] == arm and payload['update'] == 3000
        assert payload['contract_id'] == record['contract_id']
        digests[arm] = D.sha(checkpoint)
    if 'checkpoint_hashes' in report:
        reported_hashes = report['checkpoint_hashes']
    else:
        evidence = report.get('prediction_contracts', {})
        reported_hashes = {arm: evidence[arm]['contract']['checkpoint_sha256']
                           for arm in ('point', 'energy')}
        assert all(evidence[arm]['contract']['training_contract_id'] == record['contract_id']
                   for arm in ('point', 'energy')), 'approval report training identity mismatch'
    assert reported_hashes == digests, 'approval report checkpoint identity mismatch'
    return {'report_sha256': D.sha(path), 'training_contract_id': record['contract_id'],
            'checkpoint_hashes': digests}


def source_contract(approval):
    reservation = json.loads(RESERVATION.read_text())
    assert reservation['base_seed'] == BASE_SEED
    for relative, expected in reservation['collector_sources'].items():
        assert D.sha(ROOT / relative) == expected, f'preregistered source drift: {relative}'
    return {'format': 'direct_fresh_confirmation_v1', 'approval': approval,
            'reservation_sha256': D.sha(RESERVATION), 'sources': reservation['collector_sources'],
            'base_seed': BASE_SEED, 'envs': ENVS, 'limit': LIMIT, 'keys': KEYS,
            'epsilons': list(EPSILONS), 'root_rule': 'one SHA256-selected root t in[14,steps-1] per episode',
            'RGB_rule': 'first15-frame-history observed RGB map q75>=150; no future condition',
            'stopping': {'minimum_bright': MIN_BRIGHT, 'minimum_valid_variable_primary': MIN_PRIMARY,
                         'minimum_valid_variable_cow': MIN_COW,
                         'maximum_bright': MAX_BRIGHT, 'maximum_episodes': MAX_EPISODES,
                         'unit': 'first processed episode boundary; all bright roots retained'},
            'scope': 'new-seed full-state conditional evaluation; no fresh fitting or model selection',
            'reader_selection': 'bank frozen plus mob_heads.pt heads.real frozen; root source detected and real event agreement>=.95',
            'coverage_selection': 'no-camera-shift, isolated near-player, no DO, neighborhood within35 trained cells; all32 branches stay/one-neighbor; >=2 events',
            'CPU_only': True, 'preregistered_client_date': '2026-10-14'}


def layouts(states):
    player = np.asarray(states.player_position)
    single = player.ndim == 1
    if single:
        player = player[None]
    result = np.zeros((len(player), 63, 4), dtype=np.uint8)
    for channel, name in enumerate(('zombies', 'cows', 'skeletons', 'arrows')):
        group = getattr(states, name)
        position, mask = np.asarray(group.position), np.asarray(group.mask)
        if single:
            position, mask = position[None], mask[None]
        local = position - player[:, None] + np.array([3, 4])
        present = mask & (local >= 0).all(-1) & (local < [7, 9]).all(-1)
        frame, entity = np.nonzero(present)
        result[frame, local[frame, entity, 0] * 9 + local[frame, entity, 1], channel] = 1
    value = torch.from_numpy(result)
    return value[0] if single else value


def tiles(states):
    maps, player = np.asarray(states.map), np.asarray(states.player_position)
    single = maps.ndim == 2
    if single:
        maps, player = maps[None], player[None]
    rows, cols = np.indices((7, 9))
    rr = player[:, 0, None] + rows.flatten()[None] - 3
    cc = player[:, 1, None] + cols.flatten()[None] - 4
    valid = (rr >= 0) & (cc >= 0) & (rr < maps.shape[1]) & (cc < maps.shape[2])
    values = maps[np.arange(len(maps))[:, None], rr.clip(0, maps.shape[1] - 1), cc.clip(0, maps.shape[2] - 1)]
    values = np.where(valid, values, 1)  # verified Craftax Classic OUT_OF_BOUNDS enum
    output = torch.from_numpy(values.astype(np.int64).copy())
    return output[0] if single else output


def eligibility(record, bank, head):
    root, fan = record['root_layout'][None].bool(), record['fan_layout'][None].bool()
    units, coverage = C._source_units(root, fan, record['shifts'][None], record['actions'][-1:])
    real = C.decode(record['fan_tokens'][None], bank, head)['mob']['real'][0]
    current = C.decode(record['root_tokens'][None], bank, head)['mob']['real'][0]
    valid = []
    for unit in units:
        event = C._event(real, unit['points'], unit['kind'])
        clean = bool(current[unit['source'], unit['kind']]) and float((event == unit['real_events']).double().mean()) >= .95
        variable = unit['real_events'].unique().numel() >= 2
        unit = {**unit, 'reader_clean': clean, 'variable': variable}
        valid.append(unit)
    primary = [unit for unit in valid if unit['primary'] and unit['reader_clean'] and unit['variable']]
    return {'units': valid, 'coverage': coverage, 'valid_variable_primary': bool(primary),
            'valid_variable_cow': any(unit['kind'] == 1 for unit in primary)}


def cpu_encoder():
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMEncoder
    raw = torch.load(ENCODER, map_location='cpu', weights_only=False)
    execution = raw['sources']['execution']
    torch.set_float32_matmul_precision(execution['float32_matmul_precision'])
    torch.use_deterministic_algorithms(execution['deterministic_algorithms'])
    os.environ['TRITON_F32_DEFAULT'] = execution['triton_f32_default']
    payload = read_lewm_bundle(ENCODER)
    encoder = LeWMEncoder(config_from_dict(payload['config']))
    encoder.load_state_dict(payload['modules']['encoder'])
    encoder.to('cpu').freeze()
    @torch.no_grad()
    def encode(rgb):
        assert rgb.device.type == 'cpu' and rgb.dtype == torch.uint8
        return torch.cat([torch.nn.functional.layer_norm(encoder._hidden(rgb[i:i+64, None])[2].float(), (192,)).half()
                          for i in range(0, len(rgb), 64)])
    return encode


def collect(exploratory_result, directory=OUT / 'fresh', training=OUT / 'training', stop_after_episodes=None):
    torch.set_num_threads(4)
    approval = gate_report(exploratory_result, training)
    contract = source_contract(approval)
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cp = directory / 'contract.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, 'fresh confirmation contract drift'
    else:
        D.atomic(contract, cp, True)
    # Import simulator/policy only after the explicit scientific result gate.
    import jax
    import jax.numpy as jnp
    from artifacts.collect_support import LAYER, PARAMS, rollout_batch
    sys.path.insert(0, str(ROOT / 'd4_mamba_jepa'))
    from expert.ppo_expert import load_expert
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_env import make_craftax_env_from_name
    assert all(device.platform == 'cpu' for device in jax.devices()), 'JAX must use CPU only'
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    params = env.default_params
    network, policy_params, _ = load_expert(PARAMS, obs_dim=int(env.observation_space(params).shape[0]),
                                          action_dim=int(env.action_space(params).n), layer_size=LAYER)
    renderer = make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))
    frame = jax.jit(lambda state: jnp.clip(jnp.round(renderer(state)), 0, 255).astype(jnp.uint8))
    render_v = jax.vmap(frame)
    runner = jax.jit(lambda key, epsilon: rollout_batch(network, policy_params, env, params,
                                                       render_v, key, ENVS, epsilon, LIMIT))
    @jax.jit
    def replay(reset_key, env_keys, actions, stop):
        _, initial = env.reset(reset_key, params)
        def body(carry, xs):
            state, index = carry
            key, action = xs
            _, successor, _, _, _ = env.step(key, state, action, params)
            state = jax.tree.map(lambda new, old: jnp.where(index < stop, new, old), successor, state)
            return (state, index + 1), None
        return jax.lax.scan(body, (initial, 0), (env_keys, actions))[0][0]
    @jax.jit
    def fan(state, action, keys):
        _, futures, _, _, _ = jax.vmap(lambda key: env.step(key, state, action, params))(keys)
        return futures, jax.vmap(frame)(futures)
    bank = torch.load(BANK, map_location='cpu', weights_only=False)
    head = torch.load(REAL_READER, map_location='cpu', weights_only=False)['heads']['real']
    encode = cpu_encoder()
    counts = {'episodes': 0, 'bright_roots': 0, 'primary_roots': 0, 'cow_roots': 0}
    root_paths, done = [], False
    for round_index in range((MAX_EPISODES + ENVS - 1) // ENVS):
        round_path = directory / 'episodes' / f'round_{round_index:03d}.pt'
        master_key = jax.random.PRNGKey(BASE_SEED + round_index)
        epsilon = EPSILONS[round_index % len(EPSILONS)]
        if round_path.exists():
            batch = torch.load(round_path, map_location='cpu', weights_only=False)
            assert batch['contract_id'] == cid and batch['round'] == round_index
        else:
            first, _, output, reset_keys = runner(master_key, jnp.asarray(epsilon))
            rgb, actions, rewards, dones = [np.asarray(x) for x in output[:4]]
            episodes = []
            for slot in range(ENVS):
                finished = np.flatnonzero(dones[:, slot])
                steps = int(finished[0]) + 1 if len(finished) else LIMIT
                eid = f'direct-fresh:{BASE_SEED}:{round_index}:{slot}'
                pixels = np.concatenate([np.asarray(first)[slot][None], rgb[:steps, slot]])
                episodes.append({'episode_id': eid, 'steps': steps,
                                 'observations': torch.from_numpy(pixels.copy()),
                                 'actions': torch.from_numpy(actions[:steps, slot].astype(np.int64).copy()),
                                 'rewards': torch.from_numpy(rewards[:steps, slot].copy()),
                                 'terminated_or_truncated': bool(len(finished)), 'root_t': root_time(eid, steps)})
            batch = {'contract_id': cid, 'round': round_index, 'epsilon': epsilon,
                     'master_key': torch.from_numpy(np.asarray(master_key).copy()),
                     'reset_keys': torch.from_numpy(np.asarray(reset_keys).copy()), 'episodes': episodes}
            D.atomic(batch, round_path)
        key, _ = jax.random.split(master_key)
        step_keys = jax.random.split(key, LIMIT)
        slot_keys = jax.vmap(lambda key: jax.random.split(jax.random.split(key, 4)[3], ENVS))(step_keys)
        for slot, episode in enumerate(batch['episodes']):
            counts['episodes'] += 1
            t = episode['root_t']
            bright = t is not None and float(D.bright_statistics(episode['observations'][t-14:t-13].numpy())[0]) >= D.BRIGHT_Q75
            if bright:
                destination = directory / 'roots' / f'round_{round_index:03d}_slot_{slot:02d}.pt'
                if destination.exists():
                    record = torch.load(destination, map_location='cpu', weights_only=False)
                    assert record['contract_id'] == cid and record['root']['episode_id'] == episode['episode_id']
                else:
                    bucket = next(size for size in (64, 128, 256, 512, 1024, 2500) if size >= max(t, 1))
                    acts = np.zeros(bucket, np.int32)
                    acts[:min(episode['steps'], bucket)] = episode['actions'][:bucket].numpy()
                    state = replay(jnp.asarray(batch['reset_keys'][slot].numpy()), slot_keys[:bucket, slot], jnp.asarray(acts), t)
                    action = int(episode['actions'][t])
                    current_pixels = np.asarray(frame(state))
                    assert np.array_equal(current_pixels, episode['observations'][t].numpy()), 'fresh root replay mismatch'
                    _, recorded, _, _, _ = env.step(slot_keys[t, slot], state, jnp.asarray(action), params)
                    recorded_pixels = np.asarray(frame(recorded))
                    assert np.array_equal(recorded_pixels, episode['observations'][t+1].numpy()), 'fresh successor replay mismatch'
                    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(BASE_SEED),
                                                             2**16 + round_index * ENVS + slot), KEYS)
                    futures, pixels = fan(state, jnp.asarray(action), keys)
                    context = episode['observations'][t-14:t+1]
                    tokens = encode(torch.cat([context, torch.from_numpy(recorded_pixels.copy())[None],
                                               torch.from_numpy(np.asarray(pixels).copy())]))
                    root_position = torch.from_numpy(np.asarray(state.player_position).copy()).long()
                    future_position = torch.from_numpy(np.asarray(futures.player_position).copy()).long()
                    record = {'contract_id': cid, 'root': {'episode_id': episode['episode_id'], 'round': round_index,
                                                          'slot': slot, 't': t, 'evaluation_only': True},
                              'context_tokens': tokens[:15], 'actions': episode['actions'][t-14:t+1],
                              'root_tokens': tokens[14], 'fan_tokens': tokens[16:],
                              'root_layout': layouts(state), 'fan_layout': layouts(futures),
                              'root_tiles': tiles(state), 'fan_tiles': tiles(futures),
                              'root_position': root_position, 'fan_position': future_position,
                              'shifts': future_position - root_position,
                              'recorded_next_tokens': tokens[15], 'recorded_next_layout': layouts(recorded),
                              'recorded_next_tiles': tiles(recorded),
                              'recorded_next_position': torch.from_numpy(np.asarray(recorded.player_position).copy()).long(),
                              'future_keys': torch.from_numpy(np.asarray(keys).copy()),
                              'recorded_key': torch.from_numpy(np.asarray(slot_keys[t, slot]).copy()),
                              'reset_key': batch['reset_keys'][slot], 'episode_actions': episode['actions'],
                              'episode_env_keys': torch.from_numpy(np.asarray(slot_keys[:episode['steps'], slot]).copy()),
                              'full_root_state': jax.device_get(state),
                              'root_replay_pixel_maxdiff': 0, 'recorded_replay_pixel_maxdiff': 0}
                    record['source_eligibility'] = eligibility(record, bank, head)
                    D.atomic(record, destination)
                counts['bright_roots'] += 1
                counts['primary_roots'] += int(record['source_eligibility']['valid_variable_primary'])
                counts['cow_roots'] += int(record['source_eligibility']['valid_variable_cow'])
                root_paths.append(destination)
            enough, done = stopping(counts)
            D.atomic({'contract_id': cid, 'counts': counts, 'round': round_index, 'slot': slot,
                      'source_config_and_RNG': 'master seed plus round/slot; all episode actions/keys atomic',
                      'client_time': client_time(), 'coverage_met': enough}, directory / 'progress.json', True)
            if done:
                break
            if stop_after_episodes and counts['episodes'] >= stop_after_episodes:
                print(json.dumps({'stage': 'interrupted', 'counts': counts, 'client_time': client_time()}), flush=True)
                return None
        print(json.dumps({'stage': 'fresh_collection', 'counts': counts, 'client_time': client_time()}), flush=True)
        if done:
            break
    assert root_paths, 'confirmation infeasible: no RGB-bright roots; no historical fallback'
    records = [torch.load(path, map_location='cpu', weights_only=False) for path in root_paths]
    names = ('context_tokens', 'actions', 'root_tokens', 'fan_tokens', 'root_layout', 'fan_layout',
             'root_tiles', 'fan_tiles', 'root_position', 'fan_position', 'shifts', 'recorded_next_tokens',
             'recorded_next_layout', 'recorded_next_tiles', 'recorded_next_position', 'future_keys')
    panel = {name: torch.stack([row[name] for row in records]) for name in names}
    enough, _ = stopping(counts)
    panel.update({'contract_id': cid, 'contract': contract, 'roots': [row['root'] for row in records],
                  'source_eligibility': [row['source_eligibility'] for row in records],
                  'root_file_sha256': [D.sha(path) for path in root_paths], 'counts': counts,
                  'coverage_met': enough, 'root_replay_pixel_maxdiff': 0, 'recorded_replay_pixel_maxdiff': 0})
    D.atomic(panel, directory / 'fresh.pt')
    D.atomic({'contract_id': cid, 'status': 'coverage_met' if enough else 'inconclusive_coverage',
              'counts': counts, 'fresh_sha256': D.sha(directory / 'fresh.pt'),
              'client_time': client_time(), 'no_historical_fallback': True}, directory / 'collection.json', True)
    return panel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploratory-result', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=OUT / 'fresh')
    parser.add_argument('--training', type=Path, default=OUT / 'training')
    parser.add_argument('--stop-after-episodes', type=int)
    args = parser.parse_args()
    collect(args.exploratory_result, args.directory, args.training, args.stop_after_episodes)
