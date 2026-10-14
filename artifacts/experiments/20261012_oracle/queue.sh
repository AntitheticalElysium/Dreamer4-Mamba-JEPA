# 10-12: after the D-loss pair (oracle_train_od / _cd), train the L1 pair concurrently, then evaluate all four arms on the DEV panel.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=artifacts/eda/levers_tworlds_v1; L=artifacts/experiments/20261012_oracle; G=artifacts/eda/levers_logs; W=artifacts/eda/oracle_v1/worlds
M16=$V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
while systemctl --user is-active --quiet oracle_train_od || systemctl --user is-active --quiet oracle_train_cd; do sleep 30; done
.venv/bin/python $L/oracle_world.py --arm oracle --recon l1 --init $M16 > $G/oracle_train_ol.log 2>&1 &
.venv/bin/python $L/oracle_world.py --arm control --recon l1 --init $M16 > $G/oracle_train_cl.log 2>&1 &
wait
echo END >> $G/oracle_train_ol.log; echo END >> $G/oracle_train_cl.log
A=""
for f in $W/*_u3000_b16.pt; do n=$(basename $f .pt | cut -d_ -f1,2); A="$A $n=$f"; done
.venv/bin/python $L/oracle_eval.py $A > $G/oracle_eval_arms.log 2>&1
echo END >> $G/oracle_eval_arms.log
