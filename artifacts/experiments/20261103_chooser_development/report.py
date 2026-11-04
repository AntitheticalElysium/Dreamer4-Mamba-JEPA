"""Descriptive tables for the fixed campaign; no selection or automatic follow-up."""
import importlib
import json

C=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')


def main():
    lines=['# Chooser acquisition, adaptation and input readings', '',
           'All models are frozen-world diagnostic heads. No policy was used to train a world.',
           'No result here establishes improvement from looking or causal identification.', '',
           'Policies and budgets were fixed in PLAN.md before these fits.', '',
           '## E1: fixed checkpoint10, existing-policy remote probability', '',
           '| Head seed / update | Drink | Stone | Coal | Iron |', '|---|---:|---:|---:|---:|']
    for seed in (611601,611602):
        for step in (0,100,500,1000,2000,3000,4000):
            p=C.HERE/f'e1_{seed}_u{step:04d}_confirmation.json'
            if not p.exists():continue
            r=json.loads(p.read_text())
            cells=[f"{100*r[c]['policies']['existing']['remote_mass']['mean']:.1f}%" for c in ('drink','stone','coal','iron')]
            lines.append(f'| {seed} / {step} | '+' | '.join(cells)+' |')
    lines+=['', '## E2: cumulative-budget-matched endpoint remote probability', '',
            '| Seed / world / head | Drink | Stone | Coal | Iron |','|---|---:|---:|---:|---:|']
    for seed in (611601,611602):
        for ck in (1,5,10):
            for arm in (('carry',) if ck==1 else ('carry','fresh')):
                suffix='s4000' if arm=='carry' else 'endpoint'
                p=C.HERE/f'e2_{arm}_{seed}_ck{ck:02d}_{suffix}_confirmation.json'
                if not p.exists():continue
                r=json.loads(p.read_text())
                cells=[f"{100*r[c]['policies']['existing']['remote_mass']['mean']:.1f}%" for c in ('drink','stone','coal','iron')]
                lines.append(f'| {seed} / {ck} / {arm} | '+' | '.join(cells)+' |')
    lines+=['','## E3: capacity-matched information contract, endpoint remote probability','',
            '| Seed / input | Drink | Stone | Coal | Iron |','|---|---:|---:|---:|---:|']
    for seed in (611601,611602):
        for arm in ('coarse','rich'):
            p=C.HERE/f'e3_{arm}_{seed}_endpoint_confirmation.json'
            if not p.exists():continue
            r=json.loads(p.read_text())
            cells=[f"{100*r[c]['policies']['existing']['remote_mass']['mean']:.1f}%" for c in ('drink','stone','coal','iron')]
            lines.append(f'| {seed} / {arm} | '+' | '.join(cells)+' |')
    lines+=['', 'These tables only index the full evidence. Per-class C/S rankings, counter-only C,',
            'counter error/copy/target-step energy, all policies, contact/union coverage and',
            'episode-bootstrap intervals are in the matching JSON and NPZ files.',
            'Raw token error is not comparable across changed representations without its',
            'target scale; neither raw error nor a fitted negative proves absent information.',
            'E3 rich history is a diagnostic, not a production collector recommendation.',
            'Discuss the full readings before any learned-fixation world run.']
    (C.HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
