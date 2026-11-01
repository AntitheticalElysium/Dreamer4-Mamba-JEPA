"""CPU-only live proof: attach OnSuccess to an already-running throwaway service, then clean up both cases."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

OUT = Path(__file__).resolve().parent


def ctl(*args, check=True):
    return subprocess.run(["systemctl", "--user", *args], check=check, capture_output=True, text=True)


def properties(unit, *names):
    text = ctl("show", unit, *[f"-p{x}" for x in names]).stdout
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


if __name__ == "__main__":
    before = properties("saccade-p1-world1.service", "MainPID", "InvocationID")
    runtime = Path(os.environ["XDG_RUNTIME_DIR"]) / "systemd/user"
    results = []
    with tempfile.TemporaryDirectory(prefix="saccade-s1-hook-") as tmp:
        tmp = Path(tmp)
        producer = tmp / "producer.py"
        producer.write_text("import sys,time\nfrom pathlib import Path\nwhile not Path(sys.argv[1]).exists(): time.sleep(.05)\nsys.exit(int(sys.argv[2]))\n")
        consumer = tmp / "consumer.py"
        consumer.write_text("import os,sys,json\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('MONITOR_')}))\n")
        for exit_code in (0, 7):
            stem = "saccade-hook-test-" + uuid.uuid4().hex[:8]
            parent, child = stem + "-p.service", stem + "-e.service"
            release, marker = tmp / (stem + ".release"), tmp / (stem + ".json")
            unit = tmp / child
            unit.write_text(f"[Unit]\nAfter={parent}\n[Service]\nType=exec\nExecStart={sys.executable} {consumer} {marker}\n")
            dropdir = runtime / (parent + ".d")
            try:
                ctl("link", "--runtime", str(unit))
                subprocess.run(["systemd-run", "--user", f"--unit={parent}", sys.executable, str(producer), str(release), str(exit_code)],
                               check=True, capture_output=True, text=True)
                initial = properties(parent, "ActiveState", "MainPID", "InvocationID")
                assert initial["ActiveState"] == "active"
                dropdir.mkdir(parents=True)
                (dropdir / "50-s1.conf").write_text(f"[Unit]\nOnSuccess={child}\n")
                ctl("daemon-reload")
                attached = properties(parent, "MainPID", "InvocationID", "OnSuccess")
                assert attached["MainPID"] == initial["MainPID"] and attached["InvocationID"] == initial["InvocationID"]
                assert attached["OnSuccess"] == child and not marker.exists()
                release.touch()
                for _ in range(100):
                    if marker.exists() or (exit_code and properties(parent, "ActiveState")["ActiveState"] == "failed"):
                        break
                    time.sleep(.1)
                if exit_code == 0:
                    payload = json.loads(marker.read_text())
                    assert payload["MONITOR_SERVICE_RESULT"] == "success"
                    assert payload["MONITOR_EXIT_CODE"] == "exited" and payload["MONITOR_EXIT_STATUS"] == "0"
                    assert payload["MONITOR_INVOCATION_ID"] == initial["InvocationID"]
                    assert payload["MONITOR_UNIT"] == parent
                else:
                    time.sleep(.3)
                    assert not marker.exists()
                    payload = properties(parent, "ActiveState", "Result", "ExecMainStatus")
                    assert payload["ExecMainStatus"] == "7"
                results.append({"producer_exit": exit_code, "handler_ran": marker.exists(), "metadata": payload,
                                "attached_without_restart": True})
            finally:
                ctl("stop", parent, child, check=False)
                ctl("reset-failed", parent, child, check=False)
                (dropdir / "50-s1.conf").unlink(missing_ok=True)
                dropdir.rmdir()
                (runtime / child).unlink(missing_ok=True)
                ctl("daemon-reload")
    after = properties("saccade-p1-world1.service", "MainPID", "InvocationID")
    assert before == after
    report = {"cases": results, "production_p1_identity_unchanged": after}
    (OUT / "systemd_hook.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
