#!/usr/bin/env bash
# MATCHED_ENCODER_PREREG.md evaluation: MoP backbone readers; R0c at 6k; at each snapshot (6k, 12k; 36k if continued) the
# registered evaluations with each arm's own encoder spec / readers, then matched_encoder_analysis.py; at 12k the continuation
# rule (stop the training unit if it says so).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
E="$(cd ../../eda && pwd)"; OUT=$E/spatial_encoder_v1; LV=$E/levers_tworlds_v1; M=$E/matched_encoder_v1
MOPCK=$OUT/stage2_mop_sig_s1/step-010000.pt; ENC="spatial-bb:$MOPCK"; RUN=corrt_mop_s1_teacher_s7_fmamba_di_u36000
[ -f "$OUT/probes2_mopbb_s1.pt" ] || EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $PY spatial_eval_v2.py "mopbb_s1=$ENC"
P=$OUT/probes2_mopbb_s1.pt
for st in 6000 12000 36000; do
  ck=$LV/${RUN}_at$st.pt; [ $st = 36000 ] && ck=$LV/$RUN.pt
  until [ -f "$ck" ]; do
    if [ $st = 36000 ] && [ -f "$M/stopped_at_12k" ]; then echo matched_eval_done_stopped; exit 0; fi
    sleep 120
  done
  sleep 30
  lab=M$((st / 1000))k
  export EVAL_DEVICE=cuda OMP_NUM_THREADS=4
  if [ $st = 6000 ]; then                                                       # R0c: finite objective, map error below copy
    $PY - "$ck" "$ENC" <<'PYEOF'
import json, sys, torch
sys.path.insert(0, '.')
import spatial_eval as EV, objective_audit as OA
from d_adapter import DWorld
ck, enc_spec = sys.argv[1], sys.argv[2]
hist = torch.load(ck, map_location='cpu', weights_only=False)['history']
enc = EV.PatchState(enc_spec); dw = DWorld(ck); ep = ec = 0.; n = 0
for f in sorted(OA.DIAG.glob('seed-*.pt')):
    for r in torch.load(f, weights_only=False):
        if bool(r['future_dead'][0][0]): continue
        ctx = enc(r['context']); nxt = dw.predict(ctx[None], torch.cat([r['context_actions'], r['future_actions'][0:1]])[None])[0].cpu()
        real = enc(r['future_frames'][0][0:1])[0].cpu()
        ep += float((nxt[:63] - real[:63]).pow(2).sum()); ec += float((ctx[-1, :63].cpu() - real[:63]).pow(2).sum()); n += 1
    if n >= 300: break
res = {'objective_finite': all(r['objective'] == r['objective'] and abs(r['objective']) < float('inf') for r in hist),
       'map_sq_err_pred_over_copy': ep / ec, 'units': n}
res['R0c_pass'] = res['objective_finite'] and res['map_sq_err_pred_over_copy'] < 1
json.dump(res, open(EV.OUT.parent / 'matched_encoder_v1' / 'r0c_6k.json', 'w'), indent=1); print(json.dumps(res))
PYEOF
  fi
  $PY location_dynamics.py "$lab=$ENC,d:$ck,$P"
  $PY consequence_check.py "$lab=$ENC,d:$ck,$P"
  EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY matched_encoder_analysis.py --budget $((st / 1000))k --raw D$((st / 1000))k --mop $lab
  if [ $st = 12000 ]; then
    if $PY -c "import json,sys; sys.exit(0 if json.load(open('$M/matched_12k.json'))['continuation']['decision']=='stop at 12k' else 1)"; then
      touch "$M/stopped_at_12k"; systemctl --user stop matched-train || true; echo STOPPED_AT_12K_BY_RULE
    fi
  fi
done
echo matched_eval_done
