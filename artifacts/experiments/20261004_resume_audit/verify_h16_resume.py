"""CPU-only interruption equivalence and corruption/contract controls for the real H16 evaluator."""
import json
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_h16_traj as C
import h16_resume as R

spec = importlib.util.spec_from_file_location("legacy_h16_traj", Path(__file__).with_name("H16_TRAJ_BEFORE_RESUME.py"))
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

torch.set_num_threads(1)
device = torch.device("cpu")
config = SimpleNamespace(runtime=SimpleNamespace(precision="float32", device="cpu"))
out = {}


class World:
    delta = None

    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    def __call__(self, frames, acts):
        self.calls += 1
        if self.calls == self.fail:
            raise InterruptedError("injected mid-batch interruption")
        y = frames + acts[..., None, None].float() * 0.001
        if self.delta is not None:
            y = y + self.delta * 0.01
        return y, y + 0.1, None


class Quantizer:
    def __call__(self, y):
        return None, y.long() % 8

    def embed(self, y):
        return y.float()


class Post:
    quantizer = Quantizer()

    def encode(self, x, a, y):
        return torch.stack([a, a + 1, a + 2, a + 3], -1)

    def condition(self, x):
        return x.mean(-1)[:, None, None].expand(-1, 81, 2)


class Prior:
    def sample(self, frames, acts, codes, generator):
        # Exercise both the explicit stream and ambient RNG, and preserve P(end).
        k = torch.randint(8, (len(frames), 4), generator=generator)
        p = torch.rand(len(frames), generator=generator) * 0.9 + torch.rand(len(frames)) * 0.1
        return k, p


class InterruptStore(R.Store):
    def save(self, key, value, generation):
        super().save(key, value, generation)
        if generation == 200:
            raise InterruptedError("injected after atomic optimizer checkpoint")


with tempfile.TemporaryDirectory(prefix="h16-resume-", dir="/tmp") as tmp:
    root = Path(tmp)
    torch.manual_seed(61234)
    data = {"ctx": torch.randn(5, 4, 81, 2), "acts": torch.randint(0, 17, (5, 3)),
            "cont": torch.randint(0, 17, (5, 16)), "seed": torch.arange(5)}
    ref = C.features(World(), config, data, root / "full.f16", 2, device, batch=2)
    try:
        C.features(World(fail=20), config, data, root / "split.f16", 2, device, batch=2, resumable=True)
    except InterruptedError:
        pass
    w = World()
    got = C.features(w, config, data, root / "split.f16", 2, device, batch=2, resumable=True)
    out["deterministic_feature_max_abs"] = float((got - ref).abs().max())
    out["resumed_feature_world_calls"] = w.calls  # Two remaining batches, not three.
    w = World()
    C.features(w, config, data, root / "split.f16", 2, device, batch=2, resumable=True)
    out["completed_features_world_calls"] = w.calls
    try:
        R.FeatureCache(root / "split.f16", got.shape, batch=3)
        raise AssertionError("layout mismatch accepted")
    except RuntimeError:
        out["layout_mismatch_rejected"] = True
    mm = np.memmap(root / "split.f16", dtype=np.float16, mode="r+", shape=got.shape)
    mm[0, 0, 0, 0] += 1; mm.flush()
    try:
        R.FeatureCache(root / "split.f16", got.shape, batch=2)
        raise AssertionError("corrupt committed feature accepted")
    except RuntimeError:
        out["corrupt_feature_rejected"] = True

    sys.modules["dworld"] = SimpleNamespace(K=4, BLOCKS=21, S=SimpleNamespace(D=2))
    torch.manual_seed(778)
    gen = torch.Generator().manual_seed(987)
    ref, pref = C.features_e16(World(), Post(), Prior(), config, data, root / "stoch_full.f16", 2, device, gen, batch=2)
    gref, tref = gen.get_state().clone(), torch.get_rng_state().clone()
    torch.manual_seed(778)
    gen = torch.Generator().manual_seed(987)
    try:
        C.features_e16(World(fail=20), Post(), Prior(), config, data, root / "stoch_split.f16", 2, device, gen,
                       batch=2, resumable=True)
    except InterruptedError:
        pass
    torch.manual_seed(99)  # Simulate unrelated work before the restart.
    gen = torch.Generator().manual_seed(99)
    got, pgot = C.features_e16(World(), Post(), Prior(), config, data, root / "stoch_split.f16", 2, device, gen,
                              batch=2, resumable=True)
    out["stochastic_feature_max_abs"] = float((got - ref).abs().max())
    out["stochastic_end_max_abs"] = float((pgot - pref).abs().max())
    out["stochastic_generator_equal"] = bool(torch.equal(gen.get_state(), gref))
    out["stochastic_global_rng_equal"] = bool(torch.equal(torch.get_rng_state(), tref))

    torch.manual_seed(444)
    x, xa = torch.randn(4, 17, 16, 6), torch.randn(3, 17, 16, 6)
    P, Pa = torch.rand(4, 17, 16).sort(-1).values, torch.rand(3, 17, 16).sort(-1).values
    for traj in (False, True):
        key = "trajectory" if traj else "snapshot"
        full = R.Store(root / (key + "_full"), {"arm": key})
        ref, bref = C.fit(x, P, xa, Pa, traj, 1, device, steps=400, store=full, key="head")
        old, bold = legacy.fit(x, P, xa, Pa, traj, 1, device, steps=400)
        out[key + "_legacy_head_max_abs"] = max(float((ref.state_dict()[k] - old.state_dict()[k]).abs().max())
                                                for k in ref.state_dict())
        out[key + "_legacy_best_equal"] = bref == bold
        split = InterruptStore(root / (key + "_split"), {"arm": key})
        try:
            C.fit(x, P, xa, Pa, traj, 1, device, steps=400, store=split, key="head")
        except InterruptedError:
            pass
        resumed = R.Store(split.root, {"arm": key})
        got, bgot = C.fit(x, P, xa, Pa, traj, 1, device, steps=400, store=resumed, key="head")
        out[key + "_head_max_abs"] = max(float((ref.state_dict()[k] - got.state_dict()[k]).abs().max())
                                         for k in ref.state_dict())
        out[key + "_best_equal"] = bref == bgot
        s1, s2 = full.load("head"), resumed.load("head")
        out[key + "_batch_rng_equal"] = bool(torch.equal(s1["generator"], s2["generator"]))
        out[key + "_optimizer_max_abs"] = max(float((s1["optimizer"]["state"][k][v] - s2["optimizer"]["state"][k][v]).abs().max())
                                               for k in s1["optimizer"]["state"] for v in ("exp_avg", "exp_avg_sq"))
        torch.manual_seed(123)
        C.fit(x, P, xa, Pa, traj, 1, device, steps=400, store=resumed, key="head")
        try:
            R.Store(split.root, {"arm": "different"})
            raise AssertionError("contract mismatch accepted")
        except RuntimeError:
            out["contract_mismatch_rejected"] = True
        state_path = resumed.root / "head-400.pt"
        with state_path.open("ab") as f:
            f.write(b"corrupt")
        try:
            resumed.load("head")
            raise AssertionError("corrupt head accepted")
        except RuntimeError:
            out["corrupt_head_rejected"] = True

    out["constant_auc"] = C.auc(torch.ones(4), torch.tensor([0, 1, 0, 1]))
    out["tied_auc"] = C.auc(torch.tensor([0., 1., 1., 2.]), torch.tensor([0, 0, 1, 1]))
    expected_auc = (1 + .5 + 1 + 1) / 4
    assert out["tied_auc"] == expected_auc and out["constant_auc"] == .5
    y = torch.randn(130, 17, 16, 6).half(); mu = y.float().mean((0, 1, 2)); sd = y.float().std((0, 1, 2))
    whole = ((y.float() - mu) / sd).half()
    chunked = torch.cat([((y[i:i + 64].float() - mu) / sd).half() for i in range(0, len(y), 64)])
    out["standardization_max_abs"] = float((whole - chunked).abs().max())
    out["tensor_hash_changes_on_content"] = R.tensor_hash(torch.tensor([1])) != R.tensor_hash(torch.tensor([2]))

assert all(v == 0 for k, v in out.items() if k.endswith("max_abs"))
assert all(v for k, v in out.items() if k.endswith("equal") or k.endswith("rejected"))
assert out["resumed_feature_world_calls"] == 32 and out["completed_features_world_calls"] == 0
out["scope"] = "CPU exact interruption equivalence and legacy numerical parity; CUDA equivalence not tested (GPU left to running jobs)"
out["sources"] = {str(Path(C.__file__)): R.file_hash(C.__file__), str(Path(R.__file__)): R.file_hash(R.__file__)}
R.atomic_json(Path(__file__).with_suffix(".json"), out)
print(json.dumps(out, indent=2))
