"""Plot the fixed held-out trajectory; use identical axes within each panel."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/"TRAJECTORY.json").read_text())
u=json.loads((P/"UNTRAINED.json").read_text())
steps=(1,3,5,6,8,10)
frames=np.array((0,*steps))*2.56
fig,ax=plt.subplots(1,2,figsize=(9,3.45),layout="constrained")
for arm,color,label in (("hud","#00648a","HUD"),("multi","#ac3d2a","Multi-step")):
    qs=[u[arm]]+[r["checkpoints"][f"{arm}_{i}"] for i in steps]
    for key,ls,kind in (("ridge_bp","-","post-action"),("ridge_b","--","post-percept")):
        ax[0].plot(frames,[v[key]["sleep_duration"] for v in qs],marker="o",ms=3,ls=ls,color=color,label=f"{label} · {kind}")
    # A best-case bound within the tested readouts, not the average or an estimate of all information.
    ax[1].plot(frames,[max(v[m][k] for m in ("ridge_bp","mlp_recent_bp") for k in ("hunger","thirst","fatigue")) for v in qs],
               marker="o",ms=3,color=color,label=label)
ax[0].axhline(r["recent"]["ridge"]["sleep_duration"],color="#626262",ls=":",label="Recent cues only")
ax[1].axhline(0,color="#626262",ls=":")
ax[1].set_ylim(-.035,.02)
ax[1].text(.03,.94,"Exact visible-history reference: R² = 1 (off scale)",transform=ax[1].transAxes,fontsize=8,va="top")
ax[0].set_title("Time asleep, after >8 steps")
ax[1].set_title("Best awake-clock readout")
for a in ax:
    a.axvline(12.8,color="#aaaaaa",alpha=.8,ls=":")
    a.set_xlabel("Training frames (millions)")
    a.set_ylabel("Held-out R²")
    a.grid(axis="y",alpha=.2)
    a.legend(frameon=False,fontsize=8,loc="lower right")
fig.savefig(P/"belief_trajectory.png",dpi=175)
