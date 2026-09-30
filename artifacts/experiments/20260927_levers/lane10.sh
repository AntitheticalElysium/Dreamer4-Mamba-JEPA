source artifacts/experiments/20260927_levers/lib.sh
# E5f: ITC's training and decoding on corrt (where.py: generator starved, entering cells filled from the zero pad,
# HUD never updated). One switch each at the standard 6k budget, then matched 18k (corrg u18000 showed 6k under-trains).
W=artifacts/eda/levers_tworlds_v1
arm() { local name=$1; shift; [ -f $W/$name.pt ] || job tworld_$name 3600 $PY $L/tworld.py --head corrt --pool raw --loss suffix "$@"
        [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt; }
arm corrt_raw_suffix_s7_gl --gen-loss
arm corrt_raw_suffix_s7_itc --regions itc
arm corrt_raw_suffix_s7_gl_itc --gen-loss --regions itc
job compare_itc 0 $PY $L/compare.py corrt_raw_suffix_s7:corrt_raw_suffix_s7_gl corrt_raw_suffix_s7:corrt_raw_suffix_s7_itc \
  corrt_raw_suffix_s7:corrt_raw_suffix_s7_gl_itc corrt_raw_suffix_s7_gl:corrt_raw_suffix_s7_gl_itc
JAX_PLATFORMS=cpu job where_itc 1500 $PY $L/where.py $W/corrt_raw_suffix_s7_gl.pt $W/corrt_raw_suffix_s7_itc.pt $W/corrt_raw_suffix_s7_gl_itc.pt
arm corrt_raw_suffix_s7_u18000 --updates 18000
arm corrt_raw_suffix_s7_gl_itc_u18000 --gen-loss --regions itc --updates 18000
job compare_itc18 0 $PY $L/compare.py corrg_raw_suffix_s7_u18000:corrt_raw_suffix_s7_u18000 \
  corrt_raw_suffix_s7_u18000:corrt_raw_suffix_s7_gl_itc_u18000 corrt_raw_suffix_s7_gl_itc:corrt_raw_suffix_s7_gl_itc_u18000
echo "$(date '+%F %T') LANE10_DONE" >> $LOGDIR/lanes.log
