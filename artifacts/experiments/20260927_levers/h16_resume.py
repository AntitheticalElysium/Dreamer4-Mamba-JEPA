"""Hash-bound, atomic resume storage for check_h16_traj (no model mathematics)."""
import contextlib
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode())
    buf = memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast("B")
    for i in range(0, len(buf), 1024 * 1024):
        h.update(buf[i:i + 1024 * 1024])
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise RuntimeError(f"Refusing to overwrite different evidence: {path}")
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _sync_dir(path.parent)


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _sync_dir(path.parent)


class Store:
    def __init__(self, root, contract):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = digest(contract)
        with self.lock():
            atomic_json(self.root / "contract.json", contract, immutable=True)

    @contextlib.contextmanager
    def lock(self):
        with (self.root / "writer.lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise RuntimeError(f"Another evaluator owns {self.root}") from e
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def load(self, key):
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        info = json.loads(p.read_text())
        if info["contract"] != self.contract:
            raise RuntimeError(f"Resume contract differs: {p}")
        state = self.root / info["file"]
        if file_hash(state) != info["sha256"]:
            raise RuntimeError(f"Resume state is corrupt: {state}")
        return torch.load(state, map_location="cpu", weights_only=False)

    def save(self, key, value, generation):
        name = f"{key}-{generation}.pt"
        path = self.root / name
        atomic_torch(path, value)
        atomic_json(self.root / f"{key}.json", {"contract": self.contract, "file": name, "sha256": file_hash(path)})


def rng_state(device):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def frozen_eval_store(path, name, options, inputs, dependencies=()):
    """Resume contract for the other queued, frozen-world batch evaluations."""
    repo = Path.cwd().resolve()
    sources = {p.resolve() for p in Path("d4mj").rglob("*.py")}
    # load_world imports TWorld lazily. Bind its forward dependencies even before the first world is loaded.
    here = Path(__file__).resolve().parent
    sources.update(here / f"{name}.py" for name in ("tworld", "teval", "scroll", "h16_resume"))
    sources.add(here.parent / "20260921_readout_ladder" / "spatial.py")
    sources.update(here.parent / "20260926_diagnosis" / f"{name}.py" for name in ("onestep", "compound"))
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            p = Path(filename).resolve()
            if p.is_file() and p.is_relative_to(repo) and ".venv" not in p.relative_to(repo).parts:
                sources.add(p)
    contract = {"version": "frozen-eval-resume-v1", "options": options,
                "inputs": {k: tensor_hash(v) for k, v in inputs.items()},
                "sources": {str(p): file_hash(p) for p in sorted(sources)},
                "checkpoints": {str(p): file_hash(p) for p in (Path(path), *map(Path, dependencies))},
                "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                            "cudnn_deterministic": torch.backends.cudnn.deterministic,
                            "cudnn_benchmark": torch.backends.cudnn.benchmark}}
    return Store(Path("artifacts/eda/frozen_eval_resume_v1") / f"{name}__{digest(contract)[:16]}", contract)


class FeatureCache:
    """Committed root batches only; flush data before atomically publishing hashes/progress/RNG."""
    def __init__(self, path, shape, *, batch, auxiliary=False):
        self.path, self.shape = Path(path), tuple(shape)
        self.batch, self.auxiliary = batch, auxiliary
        self.meta = self.path.with_suffix(".progress.json")
        self.aux_path = self.path.with_suffix(".end.f32")
        expected = {"shape": list(shape), "batch": batch, "auxiliary": auxiliary}
        if self.meta.exists():
            self.info = json.loads(self.meta.read_text())
            if self.info["layout"] != expected:
                raise RuntimeError(f"Feature layout differs: {self.path}")
            for p, size in ((self.path, np.prod(shape) * 2), (self.aux_path, np.prod(shape[:3]) * 4)):
                if p == self.aux_path and not auxiliary:
                    continue
                if not p.exists() or p.stat().st_size != size:
                    raise RuntimeError(f"Missing/truncated feature storage: {p}")
            self.mm = np.memmap(self.path, dtype=np.float16, mode="r+", shape=shape)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # An uncommitted allocation from an interrupted first batch is safe to regenerate.
            self.mm = np.memmap(self.path, dtype=np.float16, mode="w+", shape=shape)
            self.info = {"layout": expected, "next": 0, "chunks": [], "extra": None}
        self.end = np.memmap(self.aux_path, dtype=np.float32,
                             mode="r+" if self.meta.exists() else "w+", shape=shape[:3]) if auxiliary else None
        previous = 0
        for chunk in self.info["chunks"]:
            a, b = chunk["start"], chunk["stop"]
            if a != previous or not a < b <= shape[0] or self._hash(a, b) != chunk["sha256"]:
                raise RuntimeError(f"Corrupt/noncontiguous committed feature batch: {self.path}, {a}:{b}")
            previous = b
        if previous != self.info["next"]:
            raise RuntimeError(f"Invalid feature progress: {self.meta}")

    def _hash(self, a, b):
        h = hashlib.sha256(memoryview(self.mm[a:b]).cast("B"))
        if self.end is not None:
            h.update(memoryview(self.end[a:b]).cast("B"))
        return h.hexdigest()

    @property
    def start(self):
        return self.info["next"]

    def commit(self, a, b, extra=None):
        if a != self.start or b <= a:
            raise RuntimeError("Feature commits must extend the contiguous prefix")
        for mm, path in ((self.mm, self.path), (self.end, self.aux_path)):
            if mm is not None:
                mm.flush()
                with path.open("rb") as f:
                    os.fsync(f.fileno())
        self.info["chunks"].append({"start": a, "stop": b, "sha256": self._hash(a, b)})
        self.info.update(next=b, extra=extra)
        atomic_json(self.meta, self.info)

    def tensors(self):
        if self.start != self.shape[0]:
            raise RuntimeError("Features are incomplete")
        return torch.from_numpy(self.mm), torch.from_numpy(self.end) if self.end is not None else None
