"""Frozen local/shared-noise interventions averaged over four reference draws.

No training or semantic supervision. Fixed references are noise indices0..3;
all64 original noises vary the other channel. Exact simulator source labels are
used only in the diagnostic readout, never by the completion model.
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

OUT = T.OUT / 'diagnostics/noise_channels'
MODES = ('local_fixed', 'shared_fixed')


@torch.no_grad()
def export():
    config = T.settings()
    data, inputs = E.cohort('exploratory')
    analysis = json.loads((T.OUT / 'eval_exploratory_u3000/analysis.json').read_text())
    units = [u for u in analysis['arms']['energy']['source_units'] if u['primary'] and u['dual_reader_clean']]
    checkpoint = T.OUT / 'training/energy_u3000.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name, digest in saved['contract']['sources_inputs'].items():
        assert L.sha(L.ROOT / name) == digest, name
    contract = {'format': 'direct_noise_channels_v1', 'checkpoint': L.sha(checkpoint), 'inputs': inputs,
                'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p) for p in (__file__, M.__file__, E.__file__, C.__file__)},
                'units': units, 'modes': list(MODES), 'references': [0, 1, 2, 3], 'samples': 64,
                'intervention': 'Hold local4 or shared4 dimensions at one original draw; vary the complementary dimensions with all original draws',
                'scope': 'Frozen conditional path test, not a trained architecture comparison; four reference draws are not an exhaustive variance decomposition'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (OUT / 'contract.json').exists():
        assert json.loads((OUT / 'contract.json').read_text()) == contract
    else: L.atomic(contract, OUT / 'contract.json', True)
    world = M.DirectWorld().cuda().eval(); world.load_state_dict(saved['world'])
    for start in sorted({u['root'] // 4 * 4 for u in units}):
        path = OUT / 'batches' / f'{start:04d}.pt'
        if path.exists():
            assert torch.load(path, map_location='cpu', weights_only=False)['contract_id'] == cid
            continue
        selected = [u for u in units if start <= u['root'] < start + 4]
        local = [u['root'] - start for u in selected]
        context = data['context_tokens'][start:start + 4].cuda().float()
        frames = torch.cat([context, torch.zeros_like(context[:, :1])], 1)
        ac = F.pad(data['actions'][start:start + 4].cuda(), (0, 1))
        rng = torch.Generator().manual_seed(2026101402 + start)
        noises = torch.cat([M.draw_noise((len(context), 16), rng) for _ in range(32)]).cuda()
        with autocast_context(config): h, ha = world.backbone_full(frames, ac)
        variants = {}
        for mode in MODES:
            reference_rows = []
            for reference in range(4):
                rows = []
                for k in range(64):
                    noise = noises[k].clone()
                    channels = slice(0, 4) if mode == 'local_fixed' else slice(4, 8)
                    noise[..., channels] = noises[reference, ..., channels]
                    with autocast_context(config): out = world.complete(frames, ac, h, ha, noise)[0]
                    rows.append(out[local, 14].cpu())
                reference_rows.append(torch.stack(rows, 1))
            variants[mode] = torch.stack(reference_rows, 1)
        L.atomic({'contract_id': cid, 'units': selected, 'predictions': variants}, path)
        print(json.dumps({'stage': 'noise_channels', 'start': start, 'sources': len(selected)}), flush=True)


def analyze():
    torch.set_num_threads(4)
    bank = torch.load(A.BANK, map_location='cpu', weights_only=False)
    head = torch.load(A.REAL, map_location='cpu', weights_only=False)['heads']['real']
    rows = [torch.load(path, map_location='cpu', weights_only=False) for path in sorted((OUT / 'batches').glob('*.pt'))]
    units = [u for row in rows for u in row['units']]; assert len(units) == 14
    result = {'contract_sha256': L.sha(OUT / 'contract.json'), 'units': units, 'modes': {}}
    for mode in MODES:
        pred = torch.cat([row['predictions'][mode] for row in rows])
        decoded = C.decode(pred, bank, head)
        result['modes'][mode] = {}
        for reader, flags in decoded['mob'].items():
            metrics = {key: [] for key in ('valid', 'absent', 'multiple', 'source_duplicate', 'valid_move', 'alternative_root_coverage')}
            reference_means = {key: [[] for _ in range(4)] for key in metrics}
            for i, unit in enumerate(units):
                event = C._event(flags[i], unit['points'], unit['kind'])
                duplicate = flags[i, :, :, unit['source'], unit['kind']] & flags[i, :, :, unit['points'][1:], unit['kind']].any(-1)
                values = {'valid': (event < 5).double().mean(-1), 'absent': (event == 5).double().mean(-1),
                          'multiple': (event == 6).double().mean(-1), 'source_duplicate': duplicate.double().mean(-1),
                          'valid_move': ((event > 0) & (event < 5)).double().mean(-1),
                          'alternative_root_coverage': torch.tensor([float(row[row < 5].unique().numel() >= 2) for row in event])}
                for key, value in values.items():
                    metrics[key].append(float(value.mean()))
                    for j in range(4): reference_means[key][j].append(float(value[j]))
            result['modes'][mode][reader] = {'per_root': metrics, 'intervals': {k: A.interval(v, list(range(14))) for k,v in metrics.items()},
                                           'reference_means': {k:[sum(v)/14 for v in ref] for k,ref in reference_means.items()}}
    L.atomic(result, OUT / 'analysis.json', True)
    print(json.dumps({'stage': 'noise_analysis', 'modes': {m:{r:{k:v['mean'] for k,v in data['intervals'].items()} for r,data in q.items()} for m,q in result['modes'].items()}}), flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','analyze'));a=p.parse_args()
    export() if a.mode=='export' else analyze()
