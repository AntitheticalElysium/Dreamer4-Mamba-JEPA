Frozen diagnostic, not new training. Measure what the original MSE sees on the already-qualified hunger pairs:
two genuine histories, same recent inputs and action sequence, different real endpoint. For H=2,4,8 use the
stored blind branch and an uninterrupted observed teacher, on original saved weights. Compare with the
counterfactual unrebased teacher split, holding students fixed. Decompose MSE into shared prediction and
between-history contrast, with a pair-mean student as a mathematical control (not a deployable model).
Positive control: substitute the actual teacher endpoint; it must have zero error and faithfully express
the endpoint contrast. Test whether teacher contrast chiefly appears when the differing digit is seen.
Broad control: on fixed ordinary DEV windows, check whether real starting carry helps match the original
teacher compared with a state rebuilt from only the recent eight inputs. This reads the auxiliary in its
own units and does not certify useful world prediction. No labels are used by the learner. No further fits.
