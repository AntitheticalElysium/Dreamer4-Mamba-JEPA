"""Resumable matched-loss oracle evaluation, fixed real-token readers.

GPU CLI: --updates 0|500|1500|3000 --prefix loss_intervention.
Evaluate identical 200 DEV roots, teacher-forced and self-fed H16. Independent
root batches resume after interruption; final predictions retain FP32 outputs.
No projection, threshold refit, or model-training operations occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
if os.environ['TRITON_F32_DEFAULT'] != 'ieee':
    raise RuntimeError('this evaluation requires TRITON_F32_DEFAULT=ieee')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import snap_eval as S
import oracle_eval as OE
import d_existing_compare as C
import scoreboard as SB

OUT, H, W = S.OUT, 16, 15
ARMS = ('D', 'L1', 'no_max')
EVENT_METRICS = {'damage_caught': ('caught', 'damage'), 'damage_false_drop': ('false', 'unchanged'),
                 'damage2_caught': ('caught2', 'damage2'), 'fresh_hit_caught': ('fresh_caught', 'fresh_hit'),
                 'fresh_no_hit_drawn': ('fresh_drawn', 'fresh_no_hit')}


def log(**row):
    print(json.dumps(row), flush=True)


def hashed_contract(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def infer_batch(world, cond, tok, act, layouts, config):
    tensors = {key: [] for key in ('tf', 'gen', 'tf_weights', 'gen_weights', 'tf_generated', 'gen_generated')}

    def record(mode, output):
        tensors[mode].append(output)
        weights = world.last_weights[:, -1].detach().float().cpu()
        generated = world.last_generated[:, -1].detach().float().cpu()
        assert weights.shape == (len(tok), 81, 6), 'correspondence weight shape changed'
        assert generated.shape == (len(tok), 81, 192), 'generated-candidate shape changed'
        assert torch.isfinite(weights).all() and torch.isfinite(generated).all()
        tensors[mode + '_weights'].append(weights)
        tensors[mode + '_generated'].append(generated)

    for k in range(1, H + 1):
        record('tf', OE.step(world, cond, tok[:, k:k + W].float(), act[:, k:k + W],
                            layouts[:, k + 1:k + 1 + W], torch.device('cuda'), config))
    frames = [tok[:, j].float() for j in range(1, 16)]
    for k in range(1, H + 1):
        output = OE.step(world, cond, torch.stack(frames[-W:], 1), act[:, k:k + W],
                         layouts[:, k + 1:k + 1 + W], torch.device('cuda'), config)
        record('gen', output)
        frames.append(output)
    tensors = {key: torch.stack(values, 1) for key, values in tensors.items()}
    for key in ('', '_weights', '_generated'):
        assert torch.equal(tensors['tf' + key][:, 0], tensors['gen' + key][:, 0]), 'same observed depth1 differs between TF and gen'
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in tensors.values())
    return tensors


def health_masks(visible):
    """Existing check_damage/check_damage_rule definitions on this panel.

    Health uses native 0..9 units, damage cut -0.5, ordinary hit <=-2.
    Fresh means adjacent after player displacement, no hit >=2 and no
    player-centered zombie adjacency in the previous three observed frames.
    All 200 panel roots have 15 observed frames before the root.
    """
    hp = (visible[..., 1512].float() * 9).round()
    tiles = visible[..., :1071].reshape(len(visible), 32, 7, 9, 17).argmax(-1)
    zombie = visible[..., 1071:1512].reshape(len(visible), 32, 7, 9, 7)[..., 0] > 0
    agree = []
    shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    a, b = tiles[:, 15:31], tiles[:, 16:32]
    for dr, dc in shifts:
        r0, r1, c0, c1 = max(0, -dr), 7 - max(0, dr), max(0, -dc), 9 - max(0, dc)
        agree.append((b[..., r0:r1, c0:c1] == a[..., r0 + dr:r1 + dr, c0 + dc:c1 + dc]).float().mean((-1, -2)))
    agree = torch.stack(agree, -1)
    disp = torch.tensor(shifts)[agree.argmax(-1)]
    rr = torch.arange(7)[:, None].expand(7, 9)
    cc = torch.arange(9)[None].expand(7, 9)
    distance = (rr - (3 + disp[..., 0, None, None])).abs() + (cc - (4 + disp[..., 1, None, None])).abs()
    adjacent = ((distance == 1) & zombie[:, 15:31]).any((-1, -2))
    adjacent_now = (zombie & (((rr - 3).abs() + (cc - 4).abs()) == 1)).any((-1, -2))
    historical_hit = torch.zeros_like(adjacent)
    historical_adjacent = torch.zeros_like(adjacent)
    dh = hp[:, 1:] - hp[:, :-1]
    for j in (1, 2, 3):
        historical_hit |= dh[:, 15 - j:31 - j] <= -2
        historical_adjacent |= adjacent_now[:, 15 - j:31 - j]
    historical_adjacent |= adjacent_now[:, 15:31]
    alive = (hp[:, 16:32] > 0) & (hp[:, 15:31] > 0)
    delta = hp[:, 16:32] - hp[:, 15:31]
    fresh = alive & (agree.max(-1).values >= .9) & adjacent & ~historical_hit & ~historical_adjacent
    return {'damage': ((delta < -.5) & alive).numpy(), 'damage2': ((delta <= -2) & alive).numpy(),
            'unchanged': ((delta == 0) & alive).numpy(), 'fresh': fresh.numpy(), 'alive': alive.numpy(),
            'true_delta': delta.numpy(), 'ambiguously_registered': int((alive & (agree.max(-1).values < .9)).sum())}


def health_events(next_health, current_health, masks):
    drop = next_health < current_health - .5
    out = {'damage': masks['damage'].astype(np.int64), 'caught': (drop & masks['damage']).astype(np.int64),
           'damage2': masks['damage2'].astype(np.int64), 'caught2': (drop & masks['damage2']).astype(np.int64),
           'unchanged': masks['unchanged'].astype(np.int64), 'false': (drop & masks['unchanged']).astype(np.int64),
           'fresh_hit': (masks['fresh'] & masks['damage2']).astype(np.int64),
           'fresh_caught': (masks['fresh'] & masks['damage2'] & drop).astype(np.int64),
           'fresh_no_hit': (masks['fresh'] & ~masks['damage2']).astype(np.int64),
           'fresh_drawn': (masks['fresh'] & ~masks['damage2'] & drop).astype(np.int64)}
    true_class = np.digitize(masks['true_delta'], [-1.5, -.5, .5], right=True)
    pred_class = np.digitize(next_health - current_health, [-1.5, -.5, .5], right=True)
    for c, name in enumerate(('le-2', '-1', '0', 'ge+1')):
        selected = masks['alive'] & (true_class == c)
        out['class_' + name] = selected.astype(np.int64)
        out['class_' + name + '_correct'] = (selected & (pred_class == c)).astype(np.int64)
    return out


def paired_analysis(modes, align, roots, facts, draws=2000):
    ids = np.asarray([r['episode_id'] for r in roots])
    episodes, inv = np.unique(ids, return_inverse=True)
    n = len(episodes)
    weights = np.random.default_rng(20261012).multinomial(n, np.full(n, 1 / n), size=draws)
    common = np.logical_and.reduce(list(align.values()))
    metrics = {'recall': ('true_positive', 'true'), 'precision': ('true_positive', 'predicted'),
               'fp_per_true': ('false_positive', 'true'), 'census': ('predicted', 'true'),
               'arrive_recall': ('arrive_true_positive', 'arrived'), 'residue': ('residue', 'left')}
    arms = [arm for arm in ARMS if arm + '/tf' in modes]
    contrasts = [(arm + '/gen', arm + '/tf') for arm in arms]
    if 'D' in arms:
        contrasts += [(arm + '/' + mode, 'D/' + mode) for arm in arms if arm != 'D' for mode in ('tf', 'gen')]
    if 'INIT_D/tf' in modes:
        contrasts += [(arm + '/' + mode, 'INIT_D/' + mode) for arm in arms for mode in ('tf', 'gen')]
    result = {'roots': len(ids), 'episodes': n, 'bootstrap': {'draws': draws, 'seed': 20261012, 'unit': 'episode'},
              'common_aligned_definition': 'intersection of every evaluated arm self-fed alignment; selected sensitivity population',
              'alignment': {arm: x.mean(0).tolist() for arm, x in align.items()}, 'populations': {}}
    regions = {'all': OE.MAPC, 'near': SB.NEAR4, 'interior': SB.INTER, 'edge': SB.EDGE}

    def cluster_sum(root_values):
        arr = np.zeros((n, *root_values.shape[1:]), dtype=root_values.dtype)
        np.add.at(arr, inv, root_values)
        return arr

    def estimate(numerator, denominator):
        bn, bd = weights @ numerator, weights @ denominator
        values = np.full(draws, np.nan)
        np.divide(bn, bd, out=values, where=bd > 0)
        total_den = float(denominator.sum())
        point = float(numerator.sum() / total_den) if total_den else None
        ci = np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None
        return {'estimate': point, 'ci95': ci, 'undefined_draws': int((~np.isfinite(values)).sum())}, values

    for population, mask in [('all', np.ones_like(common)), ('common_aligned', common)]:
        result['populations'][population] = {}
        for group, ks in S.GROUPS.items():
            select = mask[:, ks]
            out = {'selected_root_steps': int(select.sum()), 'selected_roots': int(select.any(1).sum()),
                   'selected_episodes': int(len(np.unique(ids[select.any(1)]))), 'models': {}, 'contrasts': {}}
            points, samples = {}, {}
            for key, mode in modes.items():
                cnt = {k: (v[:, ks] * select[:, :, None]).sum(1) for k, v in mode['counts'].items()}
                grouped = {k: cluster_sum(v) for k, v in cnt.items()}
                mob_out = {}
                for j, mob in enumerate(OE.TYPES):
                    vals = {'counts': {k: int(v[:, j].sum()) for k, v in cnt.items()}}
                    for metric, (num, den) in metrics.items():
                        vals[metric], z = estimate(grouped[num][:, j], grouped[den][:, j])
                        points[key, mob, metric], samples[key, mob, metric] = vals[metric]['estimate'], z
                    mob_out[mob] = vals
                semantic = {}
                denom = cluster_sum(select.sum(1))
                for metric, arr in mode['extras'].items():
                    value = cluster_sum((arr[:, ks] * select).sum(1))
                    divisor = len(regions[metric[5:]]) if metric.startswith('tile_') else 1
                    semantic[metric], z = estimate(value, denom * divisor)
                    points[key, 'semantic', metric], samples[key, 'semantic', metric] = semantic[metric]['estimate'], z
                for j, name in enumerate(('health', 'food', 'drink', 'energy')):
                    y = facts['hud'][:, ks, j].numpy()[select]
                    semantic[name + '_r2'] = (float(1 - semantic[name + '_sse']['estimate'] / max(y.var(), 1e-9)) if len(y) else None)
                events = {}
                event_grouped = {k: cluster_sum((v[:, ks] * select).sum(1)) for k, v in mode['damage'].items()}
                event_metrics = {**EVENT_METRICS, **{'class_' + name + '_accuracy': ('class_' + name + '_correct', 'class_' + name)
                                 for name in ('le-2', '-1', '0', 'ge+1')}}
                for metric, (num, den) in event_metrics.items():
                    events[metric], z = estimate(event_grouped[num], event_grouped[den])
                    events[metric].update(numerator=int(event_grouped[num].sum()), denominator=int(event_grouped[den].sum()))
                    points[key, 'health_events', metric], samples[key, 'health_events', metric] = events[metric]['estimate'], z
                out['models'][key] = {'mobs': mob_out, 'semantic': semantic, 'health_events': events}
            for a, b in contrasts:
                co = {}
                for typ in list(OE.TYPES) + ['semantic', 'health_events']:
                    co[typ] = {}
                    metric_names = modes[a]['extras'] if typ == 'semantic' else event_metrics if typ == 'health_events' else metrics
                    for metric in metric_names:
                        pa, pb = points[a, typ, metric], points[b, typ, metric]
                        z = samples[a, typ, metric] - samples[b, typ, metric]
                        co[typ][metric] = {'difference': pa - pb if pa is not None and pb is not None else None,
                                          'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None,
                                          'undefined_draws': int((~np.isfinite(z)).sum())}
                out['contrasts'][a + '_minus_' + b] = co
            result['populations'][population][group] = out
    return result


def analyse(directory, panel, visible, readers, prediction_paths, contract, draws):
    facts = SB.AllProbes.facts(visible[:, 16:32])
    truth = torch.stack([panel['layouts'][..., OE.LAYOUT_CH[t]] > 0 for t in OE.TYPES], -1)
    truth[:, :, 31] = False
    damage_masks = health_masks(visible)
    root_health = readers.hud(panel['tokens'][:, 15, 63:].flatten(1).float())[:, 0].numpy()
    modes, align = {}, {}
    updates = next(iter(contract.values()))['body']['updates']
    analysis_paths = dict(prediction_paths)
    if updates:
        initial_dir = directory.parent / 'eval_u0000'
        initial_path = initial_dir / 'pred_D.pt'
        initial_manifest = initial_dir / 'batches/D/contract.json'
        if not initial_path.exists() or not initial_manifest.exists():
            raise FileNotFoundError('later milestones require the current-run FP32 u0 D baseline evaluation')
        initial_contract = json.loads(initial_manifest.read_text())
        assert initial_contract['updates'] == 0 and initial_contract['arm'] == 'D'
        assert initial_contract['sources_inputs'] == next(iter(contract.values()))['body']['sources_inputs'], 'baseline evaluation source/input drift'
        contract['INIT_D'] = {'id': hashed_contract(initial_contract), 'body': initial_contract,
                              'prediction_sha256': S.sha(initial_path)}
        analysis_paths['INIT_D'] = initial_path
    for arm, path in [('TRUE', None)] + list(analysis_paths.items()):
        if arm == 'TRUE':
            predictions = {'tf': panel['tokens'][:, 16:32].float(), 'gen': panel['tokens'][:, 16:32].float()}
        else:
            predictions = torch.load(path, weights_only=False, map_location='cpu')
            assert predictions['contract_id'] == contract[arm]['id'], 'final prediction contract drift'
            assert predictions['tf'].dtype == predictions['gen'].dtype == torch.float32
            align[arm] = predictions['aligned'].numpy()
        for mode in ('tf', 'gen'):
            key = arm + '/' + mode
            if arm == 'TRUE' and mode == 'gen':
                modes[key] = modes['TRUE/tf']; continue
            counts, extras, _ = C.decode(predictions[mode], truth[:, 16:32], truth[:, 15:31], readers, facts)
            next_health = extras['health_bias'] + facts['hud'][..., 0].numpy()
            if mode == 'tf':
                assert 'TRUE/tf' in modes or arm == 'TRUE'
                true_next = next_health if arm == 'TRUE' else modes['TRUE/tf']['extras']['health_bias'] + facts['hud'][..., 0].numpy()
                current_health = np.concatenate([root_health[:, None], true_next[:, :-1]], 1)
            else:
                current_health = np.concatenate([root_health[:, None], next_health[:, :-1]], 1)
            modes[key] = {'counts': counts, 'extras': extras, 'damage': health_events(next_health, current_health, damage_masks)}
            log(stage='decoded', arm=arm, mode=mode)
        del predictions
    if set(prediction_paths) == set(ARMS) and updates == 0:
        for key in modes['D/tf']['counts']:
            assert all(np.array_equal(modes['D/tf']['counts'][key], modes[a + '/tf']['counts'][key]) and
                       np.array_equal(modes['D/gen']['counts'][key], modes[a + '/gen']['counts'][key]) for a in ARMS)
    result = paired_analysis(modes, align, panel['roots'], facts, draws)
    result.update({'format': 'oracle_loss_intervention_eval_v1', 'contracts': contract,
                   'scope': 'matched loss continuation at one shared trained oracle world; one training seed; reused DEV roots; no actor claim',
                   'fixed_thresholds': readers.thr.tolist(), 'prediction_precision': 'FP32',
                   'health_event_definitions': {'source': 'check_damage.py and check_damage_rule.py', 'units': 'native hearts 0..9',
                      'drop': 'predicted next health < reader current health -0.5', 'true_damage': 'native delta <-0.5, alive after',
                      'damage2': 'native delta <=-2, alive after', 'tf_current': 'reader of real current token',
                      'gen_current': 'reader of previous generated token; real root at depth1',
                      'fresh': 'adjacent after displacement; no >=2 hit and no current-center adjacency in preceding3frames',
                      'ambiguous_displacement_steps': damage_masks['ambiguously_registered']}})
    rows = {'format': 'oracle_loss_intervention_rows_v1', 'roots': panel['roots'], 'contracts': contract,
            'counts': {k: {ck: v.tolist() for ck, v in x['counts'].items()} for k, x in modes.items()},
            'semantic': {k: {ek: v.tolist() for ek, v in x['extras'].items()} for k, x in modes.items()},
            'health_events': {k: {ek: v.tolist() for ek, v in x['damage'].items()} for k, x in modes.items()},
            'alignment': {k: v.tolist() for k, v in align.items()}}
    S.atomic(rows, directory / 'rows.json', True)
    S.atomic(result, directory / 'analysis.json', True)
    log(stage='analysis_done', output=str(directory / 'analysis.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, required=True, choices=(0, 500, 1500, 3000))
    parser.add_argument('--prefix', default='loss_intervention')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--stop-after-batches', type=int)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--cpu-analysis-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.draws < 1 or (args.stop_after_batches is not None and args.stop_after_batches < 1):
        parser.error('batch size, draws and stop-after-batches must be positive')
    if Path(args.prefix).name != args.prefix:
        parser.error('prefix must be a directory name')
    if len(set(args.arms)) != len(args.arms):
        parser.error('arms cannot be repeated')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261012)
    started = time.time()
    run = OUT / args.prefix
    directory = run / f'eval_u{args.updates:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    panel_path, reader_path, visible_path = [OUT / name for name in ('panel.pt', 'snap_bank_readers.pt', 'snap_visible.pt')]
    panel = torch.load(panel_path, weights_only=False, map_location='cpu')
    visible = torch.load(visible_path, weights_only=False, map_location='cpu')
    assert panel['roots'] == visible['roots'], 'replay root mismatch'
    readers = S.Readers(torch.load(reader_path, weights_only=False, map_location='cpu'))
    source_paths = [Path(__file__), Path(OE.__file__), Path(S.__file__), Path(C.__file__),
                    ROOT/'artifacts/experiments/20261012_oracle/oracle_world.py']
    source_paths += [ROOT / 'artifacts/experiments/20260927_levers' / f'{name}.py'
                     for name in ('tworld', 'scoreboard', 'scroll', 'driftanat')]
    source_paths += [ROOT/'d4mj/train.py', ROOT/'d4mj/mamba_recurrence.py']
    source_paths += [ROOT/'artifacts/experiments/20260927_levers/check_damage.py',
                     ROOT/'artifacts/experiments/20260927_levers/check_damage_rule.py']
    from d4mj.config import config_from_dict
    import spatial as Sp
    source_paths.append(Path(Sp.__file__))
    files = {str(path.relative_to(ROOT)): S.sha(path) for path in source_paths + [panel_path, reader_path, visible_path, Path(Sp.CHECKPOINT)]}
    config = config_from_dict(torch.load(Sp.CHECKPOINT, weights_only=False, map_location='cpu')['config'])
    from scroll import estimate
    from driftanat import offsets
    tok, act, lay = panel['tokens'], panel['actions'], panel['layouts']
    true_shift = torch.stack([estimate(tok[:, i].float(), tok[:, i + 1].float()) for i in range(15, 31)], 1)
    true_offsets = offsets(true_shift)
    contracts, paths, completed_batches = {}, {}, 0
    for arm in args.arms:
        checkpoint = run / f'{arm}_u{args.updates:04d}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'milestone snapshot not available: {checkpoint}')
        contract = {'format': 'oracle_loss_intervention_eval_contract_v1', 'sources_inputs': files,
                    'model_sha256': S.sha(checkpoint), 'model_path': str(checkpoint.relative_to(ROOT)),
                    'roots': panel['roots'], 'arm': arm, 'updates': args.updates, 'batch_size': args.batch_size,
                    'precision': 'BF16 model autocast / FP32 saved outputs, IEEE Triton, TF32 disabled',
                    'feedback': 'raw generated outputs; no projection', 'reader': 'fixed TRAIN coefficients and thresholds'}
        contract_id = hashed_contract(contract)
        contracts[arm] = {'id': contract_id, 'body': contract}
        batch_dir = directory / 'batches' / arm
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest = batch_dir / 'contract.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == contract, 'evaluation resume source/input/model drift'
        else:
            S.atomic(contract, manifest, True)
        prediction_path = directory / f'pred_{arm}.pt'
        paths[arm] = prediction_path
        if args.cpu_analysis_only:
            if not prediction_path.exists():
                raise FileNotFoundError(f'CPU analysis requires completed predictions: {prediction_path}')
            continue
        world = cond = None
        pieces = []
        for start in range(0, len(tok), args.batch_size):
            end = min(start + args.batch_size, len(tok))
            path = batch_dir / f'batch_{start:04d}_{end:04d}.pt'
            if path.exists():
                piece = torch.load(path, weights_only=False, map_location='cpu')
                assert piece['contract_id'] == contract_id and piece['start'] == start and piece['end'] == end, 'batch contract drift'
                assert all(piece[key].dtype == torch.float32 for key in ('tf', 'gen', 'tf_weights', 'gen_weights', 'tf_generated', 'gen_generated'))
                log(stage='batch_resume', arm=arm, updates=args.updates, roots_done=end, roots_total=len(tok))
            else:
                if world is None:
                    world, cond = OE.load(checkpoint, torch.device('cuda'))
                    assert cond is not None, 'matched intervention snapshot lost oracle conditioning'
                sl = slice(start, end)
                tensors = infer_batch(world, cond, tok[sl], act[sl], lay[sl], config)
                aligned = S.alignment(tensors['gen'], tok[sl, 15].float(), true_offsets[sl])
                piece = {'contract_id': contract_id, 'start': start, 'end': end, **tensors, 'aligned': aligned}
                S.atomic(piece, path)
                restored = torch.load(path, weights_only=False, map_location='cpu')
                assert all(torch.equal(piece[k], restored[k]) for k in ('tf', 'gen', 'tf_weights', 'gen_weights', 'tf_generated', 'gen_generated', 'aligned')), 'save/load changed outputs'
                piece = restored
                completed_batches += 1
                log(stage='batch_done', arm=arm, updates=args.updates, roots_done=end, roots_total=len(tok),
                    seconds=round(time.time() - started, 1), gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3))
            pieces.append(piece)
            if args.stop_after_batches and completed_batches >= args.stop_after_batches:
                log(stage='interruption_check', resumable=True, arm=arm, updates=args.updates, roots_done=end)
                return
        if world is not None:
            del world, cond
            torch.cuda.empty_cache()
        final = {'contract_id': contract_id, **{key: torch.cat([x[key] for x in pieces])
                 for key in ('tf', 'gen', 'tf_weights', 'gen_weights', 'tf_generated', 'gen_generated', 'aligned')}}
        S.atomic(final, prediction_path)
        log(stage='world_done', arm=arm, updates=args.updates, roots=len(tok), seconds=round(time.time() - started, 1))
        del pieces, final
    analyse(directory, panel, visible['visible'], readers, paths, contracts, args.draws)
    log(stage='done', updates=args.updates, arms=args.arms, seconds=round(time.time() - started, 1))


if __name__ == '__main__':
    main()
