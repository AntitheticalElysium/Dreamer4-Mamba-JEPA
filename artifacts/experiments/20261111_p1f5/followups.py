"""Queue the F4 movement, mining and sleep readouts after the sealed F5 service."""
import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "readings/followups"
UNIT = "saccade-p1f5.service"
ROUTES = "artifacts.experiments.20261107_f4_followups.routes_f4"
SLEEP = "artifacts.experiments.20261107_danger_adversarial"
STAGES = ("movement", "mining", "sleep")


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def routes(checkpoint, out):
    rf = importlib.import_module(ROUTES)
    rf.CKPT, rf.OUT = str(checkpoint), out

    def setup():
        # routes_f4.setup asserted model.denoiser.shortcut/.dense; since df702338 shortcut forcing with dense attention
        # is the only denoiser and those attributes no longer exist. Same load, same encode, the config-level check kept.
        model, cfg = rf.CV.IF.load(rf.CKPT)
        assert cfg["edm"]["objective"] == "shortcut_forcing" and cfg["edm"]["sampler"]["steps"] == 4
        labels = dict(rf.np.load(rf.CV.LABELS / "s0_dev_labels.npz"))
        encoded = rf.S.encode(model, labels, cfg)
        # The historical pair builder uses this EDM value as a cache dictionary key.
        # Shortcut forcing still uses its configured context_signal to construct every cache.
        cfg = {**cfg, "edm": {"inference_context_sigma": .1, **cfg["edm"]}}
        return model, cfg, labels, encoded

    rf.setup = setup
    return rf


def mining(checkpoint, out):
    import numpy as np
    import torch

    rf = routes(checkpoint, out)
    rt, cv = rf.RT, rf.CV
    model, cfg, labels, enc = rf.setup()
    cv.HUD = {slot: cv.HUD[slot] for slot in rt.SLOT.values()}
    readers, valid = cv.fit_readers(model, cfg, labels, enc, 0)
    bank_path = rt.HERE / "readings/bank.pt"
    bank = torch.load(bank_path, weights_only=False, map_location="cpu")
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    lo = cfg["edm"]["inference_context_sigma"]
    result = {"bank_sha256": sha256(bank_path), "draws": rt.K,
              "conditions": ["DO/DO", "NOOP/DO", "DO/NOOP", "NOOP/NOOP"], "readers": valid}
    with torch.no_grad():
        for kind, slot in rt.SLOT.items():
            rows, episode, rec = [], [], []
            for r in bank:
                if r["kind"] != kind:
                    continue
                p, successors, _, bp, caches, fx = rt.pair_state(model, cfg, enc, r)
                real = enc[r["e"]]["p"][r["t"] - 1].to(cv.DEV).float()
                assert float((p.p[0].float() - real).abs().max()) < .04
                read = lambda p, v: readers[slot](torch.cat([p.float(), v.float()], -1))["y"].softmax(-1).argmax(-1)
                count = r["truth"]["count"][1]
                rec.append(read(successors.p[:, slot], successors.v).tolist() == [count + 1, count])
                rates = []
                for action, memory in rt.COND:
                    xs = rf.sample(model, cfg, shapes, r["actions"][action], bp[memory:memory + 1],
                                   fx, caches[lo][action], rt.K, r["j"])
                    rates.append(float((read(xs[0][:, 0, slot], xs[2][:, 0]) == count + 1).float().mean()))
                rows.append(rates)
                episode.append(r["e"])
            assert len(rows) == rt.KINDS[kind], (kind, len(rows))
            result[kind] = {"pairs": len(rows), "episode_ids": episode, "per_pair_rates": rows,
                            "recorded_reader_check": float(np.mean(rec)),
                            "factorial": rt.factorial(np.asarray(rows), np.full(len(rows), -1.), np.asarray(episode))}
    return result


def sleep(checkpoint, out):
    import shutil
    import numpy as np
    import torch

    original = importlib.import_module(f"{SLEEP}.sleep")
    analysis = importlib.import_module(f"{SLEEP}.analyse")
    bank = original.HERE / "bank.pt"
    shutil.copyfile(bank, out / "bank.pt")
    original.HERE, original.CKPT = out, str(checkpoint)
    original.run()
    z = torch.load(out / "run.pt", map_location="cpu", weights_only=False)
    a, b, h = z["generated"], z["truth"], z["H"]
    assert z["checkpoint_sha256"] == sha256(checkpoint)
    assert (len(z["rows"]), z["K"], h) == (32, 8, 48)
    active = np.concatenate([np.ones((*b.shape[:2], 1), bool),
                             np.cumprod(~b[:, :, :-1, 3].astype(bool), 2).astype(bool)], 2)
    result = {k: z[k] for k in ("H", "K", "readers_valid", "prefix_sham_max")}
    result.update(sources=len(z["rows"]), bank_sha256=sha256(bank))
    times = {}
    for name, q, live, d in (("f5", a, a[..., 3].astype(bool), a[..., 2].astype(bool)),
                              ("simulator", b, active, b[..., 3].astype(bool))):
        times[name] = {}
        result[name] = {}
        for event, mask in (("energy9", live & (q[..., 0] >= 9)),
                            ("awake", live & (q[..., 1] < .5) & ~d), ("terminal", live & d)):
            t = analysis.first(mask, h)
            times[name][event] = t
            result[name][event] = {"probability_by_step": {str(k): analysis.interval(t <= k) for k in (12, 16, 32, 48)},
                                   "median_step_or_Hplus1_if_unobserved": float(np.median(t)),
                                   "restricted_mean_waiting_steps": analysis.interval(np.minimum(t, h)),
                                   "censored_fraction": float((t > h).mean())}
    ix = np.random.default_rng(110711).integers(0, len(z["rows"]), (2000, len(z["rows"])))
    for event in ("energy9", "awake"):
        model_wait, true_wait = (np.minimum(times[k][event], h).mean(1) for k in ("f5", "simulator"))
        ratio = model_wait[ix].mean(1) / true_wait[ix].mean(1)
        result[event + "_restricted_wait_ratio"] = {"mean": float(model_wait.mean() / true_wait.mean()),
                                                   "ci95": np.percentile(ratio, [2.5, 97.5]).tolist()}
    m = (a[:, :, :-1, 0] >= 9) & (a[:, :, :-1, 1] > .5) & (a[:, :, :-1, 3] > .5) & (a[:, :, 1:, 3] > .5)
    result["generated_energy9_rule"] = {"eligible_transitions": int(m.sum()),
                                       "next_awake_fraction": float((a[:, :, 1:, 1][m] < .5).mean()) if m.any() else None}
    return result


def run_stage(name, checkpoint):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    if name == "movement":
        rf = routes(checkpoint, out)
        rf.movement()
        raw = out / "movement_f4.json"
        result = json.loads(raw.read_text())
        assert result["pairs"] == 64 and result["K"] == 16
        raw.unlink()
        result["bank_sha256"] = sha256(rf.FZ.OUT / "movement_bank.pt")
    else:
        result = (mining if name == "mining" else sleep)(checkpoint, out)
    result["checkpoint_sha256"] = sha256(checkpoint)
    write(out / "result.json", result)
    print(name, "complete", flush=True)


def parent_selection(invocation, started, seal):
    print(f"Waiting for {UNIT} ({invocation}) to release the GPU", flush=True)
    while True:
        raw = subprocess.check_output(["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "InvocationID"], text=True)
        service = dict(line.split("=", 1) for line in raw.splitlines())
        if service["ActiveState"] in ("inactive", "failed"):
            break
        if service.get("InvocationID") != invocation:
            raise RuntimeError("The active F5 invocation changed")
        time.sleep(15)
    state = json.loads((HERE / "STATUS.json").read_text())
    if (state["started_at"], state["source_pins_sha256"]) != (started, seal):
        raise RuntimeError("F5 launch identity changed")
    selected = state["selection"]
    checkpoint = Path(selected["checkpoint_path"]).resolve()
    if checkpoint.parent != HERE / "run" or sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("F5 selection differs from its checkpoint")
    return checkpoint, selected["checkpoint_sha256"]


def queue(invocation, started, seal):
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "STATUS.json"
    state = json.loads(status.read_text()) if status.exists() else {}
    identity = {"parent_invocation": invocation, "parent_started_at": started, "source_pins_sha256": seal}
    if state.setdefault("identity", identity) != identity:
        raise RuntimeError("Follow-ups belong to a different F5 run")
    state["phase"] = "waiting"
    write(status, state)
    try:
        checkpoint, sha = parent_selection(invocation, started, seal)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, RuntimeError) as e:
        state.update(phase="blocked", error=str(e))
        write(status, state)
        raise
    state.pop("error", None)
    if state.setdefault("checkpoint_sha256", sha) != sha:
        raise RuntimeError("Follow-ups belong to a different F5 checkpoint")
    state["phase"] = "evaluation"
    write(status, state)
    failed = []
    for name in STAGES:
        result = OUT / name / "result.json"
        if state.get(name) == "complete" and result.exists() and json.loads(result.read_text())["checkpoint_sha256"] == sha:
            continue
        state[name] = "running"
        write(status, state)
        try:
            with (OUT / f"{name}.log").open("a") as log:
                subprocess.run([sys.executable, "-u", "-m", __spec__.name, "stage", name, str(checkpoint)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads(result.read_text())["checkpoint_sha256"] == sha
        except (OSError, subprocess.CalledProcessError, AssertionError) as e:
            state[name] = "failed"
            failed.append(name)
            print(name, str(e), file=sys.stderr, flush=True)
        else:
            state[name] = "complete"
        write(status, state)
    state["phase"] = "failed" if failed else "complete"
    write(status, state)
    if failed:
        raise RuntimeError("Incomplete follow-ups: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    p = subs.add_parser("queue")
    p.add_argument("invocation")
    p.add_argument("started")
    p.add_argument("seal")
    p = subs.add_parser("stage")
    p.add_argument("name", choices=STAGES)
    p.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "queue":
        queue(args.invocation, args.started, args.seal)
    else:
        run_stage(args.name, args.checkpoint)


if __name__ == "__main__":
    main()
