# Runtime dissection / evidence preservation

The approved eight fits are complete. Initial compact-head smoke found a TF32
shape-rounding difference4.26e-5; IEEE arithmetic verified the masking/folding
identity to2.38e-7. Its failure evidence is retained as failed_attempt_01_*.

The first trained-head timing check required raw forecast difference<3e-6 and
stopped at8.94e-6. This was not promoted to a runtime conclusion. bench_v2.py
separately proves float64 folding equality (observed0) and checks actual broad
dev-policy distortion, maximum1.47e-5 total variation. Thus a gate was not merely
widened after failing: model algebra and relevant policy differences are measured.
Original COST.json/detail logs remain. Revised eager/compiled-Mamba evidence is
in COST_V2.json and cost_v2_*.json. Both complete with unchanged world weights;
native replay corrected-belief errors stay below0.9% relative in these chunks.

The optimized recurrent step reduces extrapolated added collection time from8.45h
to3.42h at25.6M frames, half16x64 and half4x256. This is measured extrapolation,
not an irreducible lower bound or a full-run ETA. After that optimization, repeated
head+score+sampling costs~1.3-1.4ms per small stream batch, versus~35ms for1024
frames at once. Kernel/host launch overhead remains substantial.

**Final bounded mechanical check, declared after profiling:** compile the exact
head+score function as well as the recurrent reference step. Keep multinomial
sampling, every-frame belief refresh, coarse/fine inputs, world/heads and policy
unchanged. Check same-input policy TV<1e-4 before using compiled timing. Same fixed
three-chunk cases / two steady-state measurements per shape. Hard wall cap180s;
partial/failure preserved, no further optimization follows automatically. Separate
file cost_full_compile.json, original measurements retained. This changes no
scientific arm and triggers no head/world training. Interpret quality independently
of runtime; neither inconsistency nor slow first implementation proves impossibility.

Completed:51.4s including startup/compilation, unchanged world weights. Same-input
scorer-policy maximum TV3.07e-6. Collection1B16x64=0.205s and4x256=0.702s per1024
new frames, versus matched1C0.041/0.040s. Added collection extrapolation2.87h for the
declared full budget. No further optimization launched. This is an implementation
measurement, not a demonstrated lower bound; no claim of being unoptimizable.
