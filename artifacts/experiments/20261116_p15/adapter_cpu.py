"""CPU test of the P1.5 record adapter (no GPU: training runs). (1) the adapter check on a fresh world with the record
(Mamba-2 reference backend in place of Mamba-3: the record plumbing does not depend on the mixer); (2) a record-free
world: the committed (HEAD) s1.encode / one_step / interface.futures against the edited ones, bit for bit."""
import copy, importlib, importlib.util, json, subprocess, sys, types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
CPU = torch.device("cpu")
from saccade import model as M, train as TR
from saccade.eval import s1 as S
IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
EV = importlib.import_module("artifacts.experiments.20261116_p15.evaluation")
P15 = importlib.import_module("artifacts.experiments.20261116_p15.train")
F61 = importlib.import_module("artifacts.experiments.20261114_p1f61.train")
SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
S.DEV = IF.DEV = CPU


def cpu_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["belief"]["mixer"] = "mamba2"
    cfg["belief"].pop("mamba3", None)
    cfg["belief"]["backend"] = "reference"
    return cfg


def world(cfg, seed):
    torch.manual_seed(seed)
    m = M.M0(cfg).eval()
    for p in m.parameters():                                             # zero-initialised outputs would hide routes
        if p.dim() >= 2 and p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    return m


L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
eps = [i for i, t in enumerate(L["ep_T"]) if 150 <= t <= 400][:2]
Ls = SL.subset(L, eps)
out = {"episodes": eps, "ep_T": Ls["ep_T"].tolist()}

# (1) the adapter check, record world
cfg = cpu_cfg(P15.recipe())
model = world(cfg, 0)
enc = S.encode(model, Ls, cfg)
res = EV.check(model, cfg, enc, Ls, CPU)
out["check"] = res
print(json.dumps(res, indent=1), flush=True)

# one_step on the record world runs (shapes), with the guard on
eps_, ts, gen = S.one_step(model, enc, cfg, 8, np.random.default_rng(0))
out["one_step_record_world_ok"] = [list(v.shape) for v in gen.values()]

# (2) record-free world: HEAD s1 / interface vs edited, bit for bit
def head_module(rel, name):
    src = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, text=True)
    mod = types.ModuleType(name); mod.__file__ = str(ROOT / rel); mod.__package__ = name.rpartition(".")[0]
    exec(compile(src, str(ROOT / rel), "exec"), mod.__dict__)
    return mod

S0 = head_module("saccade/eval/s1.py", "saccade.eval.s1_head"); S0.DEV = CPU
IF0 = head_module("artifacts/experiments/20261030_actor_interface/interface.py", "artifacts.experiments.20261030_actor_interface.interface_head")
IF0.DEV = CPU; IF0.S = S0
cfg0 = cpu_cfg(F61.recipe())
old = world(cfg0, 1)
assert old.record is None
e_new, e_old = S.encode(old, Ls, cfg0), S0.encode(old, Ls, cfg0)
out["old_world_encode_bitwise"] = all(torch.equal(a[k], b[k]) for a, b in zip(e_new, e_old) for k in b) and \
    all(set(a) == set(b) for a, b in zip(e_new, e_old))
o_new = S.one_step(old, e_new, cfg0, 8, np.random.default_rng(0))[2]
o_old = S0.one_step(old, e_old, cfg0, 8, np.random.default_rng(0))[2]
out["old_world_one_step_bitwise"] = all(torch.equal(o_new[k], o_old[k]) for k in o_old)
tuples = [(0, 8), (1, 40)]
f_new = IF.futures(old, e_new, Ls, tuples, 2, cfg0, torch.Generator().manual_seed(0))
f_old = IF0.futures(old, e_old, Ls, tuples, 2, cfg0, torch.Generator().manual_seed(0))
out["old_world_futures_bitwise"] = all(np.array_equal(f_new[k], f_old[k]) for k in f_old)
print(json.dumps({k: v for k, v in out.items() if k != "check"}, indent=1), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
