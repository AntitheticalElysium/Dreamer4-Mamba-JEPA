"""CPU-only tests for overnight completion and predeclared checkpoint selection."""
import copy
import json

import pytest

from saccade.eval.after_p1 import require_success, select_checkpoint


CFG = {"p1": {"frames": 1000, "checkpoint_every_fraction": .1, "monitor": {"windows": 2000}}}


def readings(losses):
    return [{"checkpoint": i + 1, "frames": 100 * (i + 1), "monitor": {"g1": x, "windows": 2000}}
            for i, x in enumerate(losses)]


@pytest.mark.parametrize("losses,selected,triggered", [
    ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 10, False),
    ([1, 2, 3, 1, 2, 3, 1, 2, 3, 2], 10, False),  # two increases do not trigger
    ([1, 2, 3, 3, 4, 5, 4, 3, 2, 2], 10, False),  # equality breaks a streak
    ([3, 1, 2, 3, 4, 3, 3, 2, 2, 2], 2, True),
    ([3, 1, 2, 3, 4, 3, 2, .5, 1, 1], 8, True),   # later minimum is eligible
    ([3, 1, 2, 3, 4, 3, 2, 1, 2, 2], 2, True),    # earliest tied minimum
])
def test_selection_is_final_or_global_minimum_by_declared_rule(losses, selected, triggered):
    result = select_checkpoint(readings(losses), CFG)
    assert result["checkpoint"] == selected
    assert result["three_increases_triggered"] == triggered


@pytest.mark.parametrize("fault", ["missing", "duplicate", "order", "frames", "nan", "windows"])
def test_incomplete_or_incomparable_readings_never_start_s1(fault):
    records = readings(list(range(10, 0, -1)))
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif fault == "order":
        records[0], records[1] = records[1], records[0]
    elif fault == "frames":
        records[-1]["frames"] = 999
    elif fault == "nan":
        records[4]["monitor"]["g1"] = float("nan")
    elif fault == "windows":
        records[4]["monitor"]["windows"] = 4
    with pytest.raises(ValueError):
        select_checkpoint(records, CFG)


def successful_exit():
    return {"SACCADE_P1_UNIT": "train.service", "SACCADE_P1_INVOCATION": "run1",
            "MONITOR_UNIT": "train.service", "MONITOR_INVOCATION_ID": "run1",
            "MONITOR_SERVICE_RESULT": "success", "MONITOR_EXIT_CODE": "exited", "MONITOR_EXIT_STATUS": "0"}


def test_successful_bound_invocation_is_accepted():
    assert require_success(successful_exit())["MONITOR_EXIT_STATUS"] == "0"


@pytest.mark.parametrize("key,value", [("MONITOR_UNIT", "other.service"), ("MONITOR_INVOCATION_ID", "another-run"),
                                      ("MONITOR_SERVICE_RESULT", "oom-kill"), ("MONITOR_EXIT_CODE", "killed"),
                                      ("MONITOR_EXIT_STATUS", "7"), ("MONITOR_EXIT_STATUS", None)])
def test_failure_manual_stop_or_other_invocation_is_rejected(key, value):
    env = successful_exit()
    if value is None:
        del env[key]
    else:
        env[key] = value
    with pytest.raises(RuntimeError, match="bound P1 invocation"):
        require_success(env)


def test_handler_selects_once_and_requests_full_s1_without_episode_limits(tmp_path, monkeypatch):
    from saccade.eval import after_p1 as A
    from saccade import train as TR
    import torch

    env = {**successful_exit(), "SACCADE_S1_SHA256": "identity"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    rows = readings([3, 1, 2, 3, 4, 3, 3, 2, 2, 2])
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for i in range(1, 11):
        (tmp_path / f"ckpt_{i:02d}.pt").write_bytes(b"checkpoint fixture")
    monkeypatch.setattr(TR, "load_cfg", lambda: CFG)
    monkeypatch.setattr(TR, "file_sha256", lambda p: "identity")
    monkeypatch.setattr(TR, "run_provenance", lambda cfg: {})
    validations = []
    monkeypatch.setattr(TR, "validate_resume", lambda *args: validations.append(args))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"state": {"frames": 1000}})
    monkeypatch.setattr(A.sys, "argv", ["after_p1", str(tmp_path), "1"])
    launches = []
    monkeypatch.setattr(A, "exec_s1", lambda command, log: launches.append((command, log)))
    A.main()
    assert len(validations) == len(launches) == 1 and validations[0][2] == 1
    command, log = launches[0]
    assert len(command) == 6 and command[1:4] == ["-u", "-m", "saccade.eval.s1"]
    assert command[4] == str(tmp_path / "ckpt_02.pt") and command[5] == str(tmp_path / "s1/results.json")
    assert log == tmp_path / "s1/eval.log"
    assert len(list(tmp_path.glob("ckpt_*.pt"))) == 10
    assert json.loads((tmp_path / "s1/selection.json").read_text())["three_increases_triggered"]
    with pytest.raises(RuntimeError, match="already attempted"):
        A.main()
    assert len(launches) == 1
