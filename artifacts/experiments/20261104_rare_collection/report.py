"""Readable coverage accounting from committed collection batches; CPU only."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    m=json.loads((ROOT/'artifacts/craftax_coverage_v1/manifest.json').read_text())
    old=json.loads((HERE.parent/'20261104_coverage_and_gaze/COVERAGE.json').read_text())['summary']
    new=m.get('coverage',{}).get('train',{})
    if not new:raise RuntimeError('No committed collection batch yet')
    n0=old['transitions'];n1=new['transitions']
    lines=['# Broad collection coverage', '',
        '**Collection status:** '+('complete' if m['complete'] else 'running; partial counts')+'.', '',
        f"Committed {m['rounds']}/{m['contract']['rounds']} batches, {m['episodes']:,} episodes, "
        f"{m['transitions']:,} transitions; {new['episodes']:,} episodes/{n1:,} transitions after fixed monitor exclusion.",
        '', 'All original dev/final data remain unchanged. These are acquisition counts, not learning or model-quality results.', '',
        '| Achievement | Old train unlocks | Added train unlocks | Combined | Old / combined per million transitions |',
        '|---|---:|---:|---:|---:|']
    for k,v in old['events'].items():
        if not k.startswith('unlock_'):continue
        name=k.removeprefix('unlock_');a=v['unique'];b=new['unlocks'].get(name,0)
        lines.append(f'|{name}|{a:,}|{b:,}|{a+b:,}|{1e6*a/n0:.1f} / {1e6*(a+b)/(n0+n1):.1f}|')
    lines+=['','| Event | Old train | Added train |','|---|---:|---:|']
    for oldkey,newkey in [('damage','damage'),('death','deaths'),('risk','risk'),('risk_death','risk_deaths')]:
        lines.append(f"|{oldkey}|{old['events'][oldkey]['unique']:,}|{new[newkey]:,}|")
    rp=ROOT/'artifacts/craftax_risk_v1/manifest.json'
    if rp.exists():
        risk=json.loads(rp.read_text());z=risk.get('coverage',{}).get('train',{})
        if z:
            lines+=['','## Additional danger collection','',
                f"Status: {'complete' if risk['complete'] else 'running, partial'}; {risk['rounds']}/{risk['contract']['rounds']} batches, "
                f"{z['episodes']:,} train episodes/{z['transitions']:,} train transitions.",'',
                '| Event | Old train | Advanced addition | Danger addition | Old / all combined per million transitions |',
                '|---|---:|---:|---:|---:|']
            for ok,nk in [('damage','damage'),('death','deaths'),('risk','risk'),('risk_death','risk_deaths')]:
                a=old['events'][ok]['unique'];b=new[nk];c=z[nk]
                lines.append(f'|{ok}|{a:,}|{b:,}|{c:,}|{1e6*a/n0:.1f} / {1e6*(a+b+c)/(n0+n1+z["transitions"]):.1f}|')
            lines+=['', 'The all-combined rate is an accounting diagnostic, not an adopted sampling mixture. '
                'No model run has been launched on the concatenation. Preserve or deliberately change danger '
                'frequencies before choosing the training recipe; do not let long competent episodes dilute them silently.']
    lines+=['', 'Absolute counts and rates both matter. A rare unlock count can increase while its mixture frequency '
        'barely changes if the collection adds many ordinary transitions. Different source episodes and later-state '
        'action coverage matter beyond repeated events. Whole episodes are kept; there is no post-outcome acceptance filter.', '',
        'The manifest also reports all inventory increases and prior-value histograms, body increases/decreases, '
        'day/night unlocks, visible creature frames, component-specific counts and the monitor partition. '
        'Per-transition exogenous flags and behavior probabilities are in audit sidecars.', '',
        'Remaining limits: this frozen expert may still miss behaviors; an absent state/action combination is not '
        'fixed merely by collecting more expert play. Compare the final report and propensities before selecting '
        'the training mixture. Fixed-frame-budget training will give fewer repetitions of a larger corpus. '
        'No sampling weights, model losses, acuity, architecture or overnight training are changed by collection.', '']
    (HERE/'COVERAGE.md').write_text('\n'.join(lines))
    print(lines[4])


if __name__=='__main__':main()
