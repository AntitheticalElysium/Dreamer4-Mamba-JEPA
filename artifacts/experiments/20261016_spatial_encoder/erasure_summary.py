"""Tables from erasure_diagnosis.py's per-checkpoint files (R63): one row per run and step.
retention = refit AP on the predicted expected state / refit AP on the root's own state (copy): 1 = the predictor keeps the mob
information its input carried; < 1 = it destroys some.
Usage: erasure_summary.py -> prints; artifacts/eda/spatial_encoder_v1/erasure/summary.json
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402

D = EV.OUT / 'erasure'


def main():
    rows = []
    for f in sorted(D.glob('*@*.json')):
        run, step = re.match(r'(.+)@(\d+)\.json', f.name).groups()
        d = json.loads(f.read_text())
        r, p, dv = d['representation'], d['predictor_d1'], d['dev_in_distribution']
        row = {'run': run, 'step': int(step)}
        for st in ('backbone_ln', 'deployed'):
            x = r[st]; s = 'bb' if st == 'backbone_ln' else 'dep'
            row.update({f'{s}_zAP': x['zombie_ap'], f'{s}_cAP': x['cow_ap'], f'{s}_tile': x['tile_acc'],
                        f'{s}_mobvar': x['var_share_mob_within_tile'], f'{s}_tilevar': x['var_share_tile'],
                        f'{s}_mobE': x['change_energy']['mob_change']['energy_share'],
                        f'{s}_mobE_rel': x['change_energy']['mob_change']['mean_per_cell_rel_neither'],
                        f'{s}_driftE': x['change_energy']['neither']['energy_share']})
        for m, k in (('zombie', 'z'), ('cow', 'c')):
            q = p[m]
            row.update({f'{k}_exp_ap': q['exp']['ap'], f'{k}_copy_ap': q['copy']['ap'], f'{k}_real_ap': q['real']['ap'],
                        f'{k}_bestK_ap': q['best_of_K']['ap'], f'{k}_exp_per_frame': q['exp']['per_frame'],
                        f'{k}_refit_exp': q['refit_ap']['exp'], f'{k}_refit_copy': q['refit_ap']['copy'], f'{k}_refit_real': q['refit_ap']['real'],
                        f'{k}_retention': round(q['refit_ap']['exp'] / q['refit_ap']['copy'], 3) if q['refit_ap']['copy'] else None,
                        f'{k}_dev_refit_exp': dv[m]['exp']['refit_ap'], f'{k}_dev_refit_copy': dv[m]['copy']['refit_ap'],
                        f'{k}_dev_retention': round(dv[m]['exp']['refit_ap'] / dv[m]['copy']['refit_ap'], 3) if dv[m]['copy']['refit_ap'] else None})
        bg = p['background']
        row.update({'bg_tile_exp': bg['tile_acc_exp'], 'bg_tile_copy': bg['tile_acc_copy'], 'bg_changed_exp': bg['tile_acc_changed_exp'],
                    'bg_changed_copy': bg['tile_acc_changed_copy'], 'err_free_rel_copy': bg['sq_err_exp_mob_free_rel_copy'],
                    'err_mob_rel_copy': bg['sq_err_exp_mob_rel_copy']})
        rows.append(row)
    rows.sort(key=lambda x: (x['run'], x['step']))
    (D / 'summary.json').write_text(json.dumps(rows, indent=1))
    groups = {
        'representation (Tier A test frames)': ['bb_zAP', 'dep_zAP', 'bb_cAP', 'dep_cAP', 'bb_mobvar', 'dep_mobvar', 'bb_tilevar', 'dep_tilevar',
                                                'bb_mobE_rel', 'dep_mobE_rel', 'dep_driftE'],
        'predictor d1, zombie': ['z_exp_ap', 'z_copy_ap', 'z_bestK_ap', 'z_exp_per_frame', 'z_refit_exp', 'z_refit_copy', 'z_retention', 'z_dev_retention'],
        'predictor d1, cow': ['c_exp_ap', 'c_copy_ap', 'c_bestK_ap', 'c_exp_per_frame', 'c_refit_exp', 'c_refit_copy', 'c_retention', 'c_dev_retention'],
        'background': ['bg_tile_exp', 'bg_tile_copy', 'bg_changed_exp', 'bg_changed_copy', 'err_free_rel_copy', 'err_mob_rel_copy']}
    for title, cols in groups.items():
        print(f'\n## {title}\n| run@step | ' + ' | '.join(cols) + ' |')
        for x in rows:
            print(f"| {x['run']}@{x['step']} | " + ' | '.join('' if x[c] is None else f'{x[c]:.3g}' for c in cols) + ' |')


if __name__ == '__main__':
    main()
