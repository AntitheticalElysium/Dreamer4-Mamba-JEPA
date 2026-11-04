"""Export the predeclared source-cell confirmation reading, both seeds separately."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/saccade_projection_matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
z = json.loads((HERE / 'results.json').read_text())
modes = ['fine', 'native', 'retained', 'mixed', 'recovered_bf16']
labels = ['Raw fine\ncrop', 'Projected fine\n(native code)',
          'Same retained info\nin crop coordinates', 'Actual BF16\nmixed input', 'BF16 input\nrecovered*']
fig, ax = plt.subplots(figsize=(10.6, 4.9))
x = np.arange(len(modes))
for j, seed in enumerate(['611701', '611702']):
    rows = [z[seed]['confirmation']['primary']['source'][m] for m in modes]
    values = np.array([q['reduction_percent'] for q in rows])
    ci = np.array([q['reduction_percent_CI'] for q in rows])
    ax.errorbar(x + (j - .5) * .13, values,
        yerr=[values - ci[:, 0], ci[:, 1] - values], fmt='o', capsize=4,
        markersize=6, label=f'Reader seed {j + 1}', color=['#236caa', '#d87729'][j])
ax.axhline(0, color='#888', lw=1)
ax.set_xticks(x, labels)
ax.set_ylabel('Next-fine error reduction at entity cells (%)')
ax.set_ylim(-4, 32)
ax.set_title('Much of the apparent projection loss is reversible', loc='left', fontweight='bold', pad=15)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper right', frameon=False)
fig.text(.09, .03,
    'Confirmation: 1,824 night/entity/no-scroll frames, 102 episodes. Paired episode-bootstrap 95% intervals.\n'
    '*Known coarse contribution subtracted; native Mamba does not receive coarse tokens separately. Frozen checkpoint.',
    fontsize=9, color='#444')
fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.25)
fig.savefig(HERE / 'projection_recovery.png', dpi=170)
fig.savefig(HERE / 'projection_recovery.pdf')
plt.close(fig)
