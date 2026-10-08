"""E21 tables straight from the result files (2026-10-08): no hand transcription.
Reads health_chain_v1/evidence_<tags>.json (subset readouts), health_chain_v1/gradient_<world>.json (allocation), the damage
panel's JSON line in levers_logs/<damage log> (check_damage.py --window 15) and 20260927_levers/compare.json (teval vs C0).
Prints markdown tables: per world (and snapshot) B2 (generator), B1 (phase in h63, drawable), emitted hits / fresh / false;
allocation shares; damage panel; teval paired differences.
Usage: health_e21_table.py <evidence tag> ... [--damage <log name>] [--compare <A tag>]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
LOGS = ROOT / 'artifacts/eda/levers_logs'
TAIL = '_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000'
SHORT = {'corrt_rawlong_teacher_s7_fmamba_L16b40_from36000': 'M16 s7', 'corrt_rawlong_teacher_s8_fmamba_L16b40_from36000': 'M16 s8',
         'corrt_rawlong_teacher_s7_fmamba' + TAIL: 'C0', 'corrt_rawlong_teacher_s7_fmamba_gl' + TAIL: 'G',
         'corrt_rawlong_teacher_s7_fmamba_ev' + TAIL: 'E', 'corrt_rawlong_teacher_s7_fmamba_gl_ev' + TAIL: 'GE',
         'corrt_rawlong_teacher_s7_fmamba_di' + TAIL: 'D',
         'corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000': "E' (LP-FT)",
         'corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000': 'oracle (M16 + oracle phase/adjacency)',
         'corrt_raw_teacher_s7_fmamba_u36000': 'M6 s7 (from scratch, 6-frame)',
         'corrt_raw_teacher_s7_fmamba_gl_ev_u36000': 'GE from scratch, stage 1',
         'corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000': 'GE from scratch (stage 2)'}


def short(name):
    """world (or world__w15, world_atNNNN__w15) -> M16 s7 / C0 / G / E / GE / D [@NNNN]; unknown names unchanged."""
    base = name.replace('__w15', '').replace('__w10', ' row9').replace('__w5', '')
    row = ' row9' if base.endswith(' row9') else ''
    base = base.replace(' row9', '')
    at = ''
    if '_at' in base and base.rsplit('_at', 1)[1].isdigit():
        base, at = base.rsplit('_at', 1)
        at = f' @{at}'
    return SHORT.get(base, base) + at + row


def main():
    args = sys.argv[1:]
    tags = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or not args[i - 1].startswith('--'))]
    print('| world | gen beats copy (hits) | gen L1 hits / unch | since = 6 in h63 | drawable from h63 | + oracle phase, adj | '
          'emitted hits /1,211 | fresh /207 | false /1,728 | event-head AUC |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for tag in tags:
        r = json.loads((OUT / f'evidence_{tag}.json').read_text())
        for k, v in r.items():
            if k in ('reference', 'positive_control_true_token63_history'):
                continue
            g, d, em, i = v['generator63'], v['drawable'], v['emitted'], v['ingredients_h63']
            ev = v.get('event63', {}).get('auc_hit_vs_unchanged', '-')
            print(f"| {short(k)} | {g['beats_copy_on_hits']:.3f} | {g['l1_hits']:.3f} / {g['l1_unchanged']:.3f} | {i.get('since_eq6', float('nan')):.3f} | "
                  f"{d['h63']['all'][0]} | {d['h63+oracle_since_adj']['all'][0]} | {em['all'][0]} | {em['fresh'][0]} | {em['unchanged_false'][0]} | {ev} |")
    grads = sorted(OUT.glob('gradient_*.json'))
    if grads:
        print('\n| world | objective | hits: share of objective | hits: backbone gradient share | gen term share | event term share | event at token-63 hits |')
        print('|---|---|---|---|---|---|---|')
        for p in grads:
            r = json.loads(p.read_text())
            so, gr = r['share_of_objective'], r['backbone_grad_norm_ratio']
            print(f"| {short(r['world'])}{' (Delta-IRIS loss)' if p.stem.endswith('deltairis') else ''} | {r['objective']:.4f} | {so['tok63_hit']:.5f} | "
                  f"{gr['tok63_hit']:.5f} | {gr.get('gen_term', '-')} | {gr.get('event_term', '-')} | {gr.get('event_tok63_hit', '-')} |")
    if '--damage' in args:
        line = [l for l in (LOGS / args[args.index('--damage') + 1]).read_text().splitlines() if l.startswith('{') and '"readings"' in l][-1]
        r = json.loads(line)
        print('\n| world | teacher caught /337 | false drop rate | fresh caught /31 | beside, no hit in window: caught / drawn without hit | -2 class accuracy |')
        print('|---|---|---|---|---|---|')
        for k, v in r.items():
            if not (isinstance(v, dict) and 'teacher_by_history' in v):     # skip the panel's own summary keys
                continue
            t, h = v['teacher'], v['teacher_by_history']
            print(f"| {short(k)} | {t['caught']:.3f} | {t['false_drop']:.4f} | {h['fresh']['caught']:.3f} | "
                  f"{h['beside_no_hit']['caught']:.3f} / {h['beside_no_hit']['drawn_without_hit']:.3f} | {v['health_change_accuracy']['le-2']['acc']:.3f} |")
    if '--compare' in args:
        a = args[args.index('--compare') + 1]
        r = json.loads((HERE / 'compare.json').read_text())
        print('\n| comparison | onestep_all B - A [95%] | gen_16 B - A [95%] |')
        print('|---|---|---|')
        for k, v in r.items():
            if k.startswith(a) or a in k.split(' -> ')[0]:
                o, g = v['onestep_all'], v['gen_16']
                print(f"| {short(k.split(' -> ')[0])} -> {short(k.split(' -> ')[1])} | {o['B_minus_A']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] | "
                      f"{g['B_minus_A']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] |")


if __name__ == '__main__':
    main()
