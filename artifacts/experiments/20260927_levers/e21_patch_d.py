"""Applies E21 arm D (Delta-IRIS tokenizer loss in token space) to tworld.py: --recon {l1, deltairis}, default l1 (unchanged)."""
import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
rep("""targets (it scored S.W - 1: unchanged for every 6-frame run). `--snapshot-every N` saves the world every N updates (6000 before).
""", """targets (it scored S.W - 1: unchanged for every 6-frame run). `--snapshot-every N` saves the world every N updates (6000 before).
E21 arm D (predeclared 17:50): `--recon deltairis` replaces the teacher L1 by Delta-IRIS's tokenizer loss in token space
(vmicheli/delta-iris tokenizer.py): 1.0 x mean L2 + 0.1 x mean L1 + 0.01 x the mean over frames of the frame's worst element L2
(over its 81 x 192 token elements). Default `l1` is unchanged.
""")
rep("""def rollout_losses(world, s, a, loss, gen_loss=False, weight=None, faced=None, event=False):""",
    """def rollout_losses(world, s, a, loss, gen_loss=False, weight=None, faced=None, event=False, recon="l1"):""")
rep("""    teacher = err.mean()
""", """    teacher = err.mean() if recon == "l1" else \\
        err.pow(2).mean() + 0.1 * err.mean() + 0.01 * err.pow(2).flatten(2).max(-1).values.mean()      # E21 D: Delta-IRIS
""")
rep("""          init=None, state_every=STATE_EVERY, event=False, snapshot_every=6000):""",
    """          init=None, state_every=STATE_EVERY, event=False, snapshot_every=6000, recon="l1"):""")
rep("""                objective = rollout_losses(world, s, a, loss, gen_loss, weight, faced, event)""",
    """                objective = rollout_losses(world, s, a, loss, gen_loss, weight, faced, event, recon)""")
rep("""    parser.add_argument("--snapshot-every", type=int, default=6000, help="E21: --snapshots interval (default 6000)")
""", """    parser.add_argument("--snapshot-every", type=int, default=6000, help="E21: --snapshots interval (default 6000)")
    parser.add_argument("--recon", default="l1", choices=("l1", "deltairis"), help="E21 D: teacher-forced reconstruction loss")
""")
rep("""+ ("_gl" if args.gen_loss else "") + ("_ev" if args.event else "") \\""",
    """+ ("_gl" if args.gen_loss else "") + ("_ev" if args.event else "") + ("_di" if args.recon == "deltairis" else "") \\""")
rep("""                                 args.event, args.snapshot_every)""", """                                 args.event, args.snapshot_every, args.recon)""")
open(p, 'w').write(s)
print('patched', p)
