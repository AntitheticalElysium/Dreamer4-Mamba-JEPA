"""One-shot OnSuccess handler: select the declared P1 checkpoint and exec full S1.

Only the bound, successfully completed training invocation may trigger this handler. It does not start training,
retry evaluation or proceed to P2. CPU-only validation happens before S1 acquires the GPU.
Usage: python -m saccade.eval.after_p1 RUN_DIR WORLD_SEED
"""
import json
import math
import os
from pathlib import Path
import sys


def require_success(env):
    expected = {"MONITOR_UNIT": env["SACCADE_P1_UNIT"],
                "MONITOR_INVOCATION_ID": env["SACCADE_P1_INVOCATION"],
                "MONITOR_SERVICE_RESULT": "success", "MONITOR_EXIT_CODE": "exited", "MONITOR_EXIT_STATUS": "0"}
    if any(env.get(key) != value for key, value in expected.items()):
        raise RuntimeError("S1 requires successful exit of the bound P1 invocation")
    return expected


def select_checkpoint(records, cfg):
    """Final unless ANY three successive comparisons strictly increase; then global minimum, earliest tie.

    Four monitor observations are needed for three increases. Equality breaks the rising streak. This uses only
    the fixed held-out G1 monitor, never semantic probe results or dev/final outcomes.
    """
    budget = cfg["p1"]["frames"]
    fraction = cfg["p1"]["checkpoint_every_fraction"]
    count = round(1 / fraction)
    if not math.isclose(count * fraction, 1) or [r["checkpoint"] for r in records] != list(range(1, count + 1)):
        raise ValueError("Expected all numbered checkpoint monitor readings exactly once and in order")
    spacing = budget * fraction  # same multiplication order as train.py; avoid 300 < 300.00000000000006
    if any(r["frames"] < (i + 1) * spacing for i, r in enumerate(records)):
        raise ValueError("P1 checkpoint frame budget is incomplete")
    losses = [float(r["monitor"]["g1"]) for r in records]
    if not all(math.isfinite(x) for x in losses):
        raise ValueError("Checkpoint monitor losses must be finite")
    if any(r["monitor"]["windows"] != cfg["p1"]["monitor"]["windows"] for r in records):
        raise ValueError("Checkpoint monitor window counts differ from the fixed recipe")
    rising, triggered = 0, False
    for before, after in zip(losses, losses[1:]):
        rising = rising + 1 if after > before else 0
        triggered |= rising >= 3
    index = min(range(count), key=losses.__getitem__) if triggered else count - 1
    return {"checkpoint": index + 1, "monitor_loss": losses[index], "three_increases_triggered": triggered,
            "rule": "final unless any three successive strict increases; then global minimum, earliest tie",
            "monitor_history": [{"checkpoint": r["checkpoint"], "frames": r["frames"], "loss": loss}
                                for r, loss in zip(records, losses)]}


def exec_s1(command, log_path):
    # Exec drops validation/optimizer tensors and makes the unit's exit status S1's exit status.
    log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    os.execv(sys.executable, command)


def main():
    parent = require_success(os.environ)
    run, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
    out = run / "s1"
    out.mkdir(exist_ok=True)
    selection, result = out / "selection.json", out / "results.json"
    if selection.exists() or result.exists():
        raise RuntimeError("S1 follow-up already attempted; inspect its artifacts before a manual retry")

    from saccade import train as TR
    import torch

    cfg = TR.load_cfg()
    if TR.file_sha256(TR.ROOT / "saccade/eval/s1.py") != os.environ["SACCADE_S1_SHA256"]:
        raise RuntimeError("S1 source changed after the follow-up was queued")
    records = [r for line in (run / "log.jsonl").read_text().splitlines()
               if "monitor" in (r := json.loads(line))]
    decision = select_checkpoint(records, cfg)
    for r in records:
        if not (run / f"ckpt_{r['checkpoint']:02d}.pt").is_file():
            raise FileNotFoundError(f"Missing numbered checkpoint {r['checkpoint']}")
    final = torch.load(run / f"ckpt_{len(records):02d}.pt", map_location="cpu", weights_only=False)
    TR.validate_resume(final, cfg, seed, TR.run_provenance(cfg))
    if final["state"]["frames"] < cfg["p1"]["frames"]:
        raise ValueError("Final checkpoint has not completed P1")
    chosen = run / f"ckpt_{decision['checkpoint']:02d}.pt"
    command = [sys.executable, "-u", "-m", "saccade.eval.s1", str(chosen), str(result)]
    decision.update(parent=parent, checkpoint_path=str(chosen), checkpoint_sha256=TR.file_sha256(chosen),
                    evaluation_source_sha256=os.environ["SACCADE_S1_SHA256"], command=command)
    selection.write_text(json.dumps(decision, indent=2) + "\n")
    print(f"Selected {chosen.name} by the declared monitor rule; starting full S1 (no episode limits)", flush=True)
    exec_s1(command, out / "eval.log")


if __name__ == "__main__":
    main()
