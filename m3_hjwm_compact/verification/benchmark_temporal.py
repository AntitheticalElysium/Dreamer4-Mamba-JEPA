"""Reproducible short-sequence temporal-backend benchmark.

Run from the repository root, for example:

    .venv/bin/python m3_hjwm_compact/verification/benchmark_temporal.py \
        --backend gru mamba2 --length 16 32 64 128

The script prints JSON only. It uses the exact adapter exercised by the model and
does not replace a failed backend with another implementation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import ModelConfig, TemporalModel  # noqa: E402


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def clear_state(value) -> None:
    if isinstance(value, torch.Tensor):
        value.zero_()
    elif isinstance(value, (list, tuple)):
        for item in value:
            clear_state(item)


def timed(fn, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def make_backend(name: str, dim: int, state: int, headdim: int, depth: int):
    cfg = ModelConfig(
        token_dim=dim,
        spatial_heads=4,
        temporal_backend=name,
        temporal_depth=depth,
        mamba_d_state=state,
        mamba_headdim=headdim,
        predictor="deterministic",
    )
    return TemporalModel(cfg)


def benchmark_case(
    backend_name: str,
    length: int,
    batch: int,
    streams: int,
    dim: int,
    state_dim: int,
    headdim: int,
    depth: int,
    repeats: int,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    backend = make_backend(backend_name, dim, state_dim, headdim, depth)
    backend = backend.to(device=device, dtype=dtype)
    x = torch.randn(batch, length, streams, dim, device=device, dtype=dtype)

    result: dict[str, object] = {
        "backend": backend_name,
        "length": length,
        "batch": batch,
        "streams": streams,
        "dim": dim,
        "state_dim": state_dim,
        "headdim": headdim,
        "parameters": sum(parameter.numel() for parameter in backend.parameters()),
    }

    try:
        backend.eval()
        with torch.no_grad():
            sequence, _ = backend.sequence(x)
            synchronize()
        result["sequence_finite"] = bool(torch.isfinite(sequence).all())

        def inference_once():
            with torch.no_grad():
                backend.sequence(x)

        inference_times = timed(inference_once, repeats=repeats)
        result["sequence_inference"] = quantiles(inference_times)
        result["sequence_tokens_per_second"] = (
            batch * streams * length / (statistics.median(inference_times) / 1000.0)
        )

        recurrent = backend.init_state(batch, streams, device, dtype)
        with torch.no_grad():
            backend.step(x[:, 0], recurrent)
            synchronize()
        recurrent = backend.init_state(batch, streams, device, dtype)
        with torch.no_grad():
            recurrent_outputs = []
            synchronize()
            start = time.perf_counter()
            for index in range(length):
                output, recurrent = backend.step(x[:, index], recurrent)
                recurrent_outputs.append(output)
            synchronize()
            recurrent_ms = (time.perf_counter() - start) * 1000.0
            stepped = torch.stack(recurrent_outputs, 1)
        difference = (stepped.float() - sequence.float()).abs()
        result["recurrent_finite"] = bool(torch.isfinite(stepped).all())
        result["equivalence_max_abs"] = float(torch.nan_to_num(difference).max())
        result["equivalence_mean_abs"] = float(torch.nan_to_num(difference).mean())
        result["equivalence_atol_rtol_0_05"] = bool(
            torch.allclose(stepped.float(), sequence.float(), atol=0.05, rtol=0.05)
        )
        result["recurrent_total_ms"] = recurrent_ms
        result["recurrent_step_ms"] = recurrent_ms / length
        result["cache_mib"] = tensor_bytes(recurrent.cache) / 2**20

        backend.train()
        # Exclude one-time Triton/CuTe compilation from steady-state training
        # latency and memory. Compilation support is probed separately.
        warm_input = x.detach().clone().requires_grad_(True)
        warm_output, _ = backend.sequence(warm_input)
        warm_output.float().square().mean().backward()
        synchronize()
        for parameter in backend.parameters():
            parameter.grad = None
        del warm_input, warm_output
        training_input = x.detach().clone().requires_grad_(True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        start = time.perf_counter()
        training_output, _ = backend.sequence(training_input)
        training_loss = training_output.float().square().mean()
        if torch.isfinite(training_loss):
            training_loss.backward()
            synchronize()
            result["training_finite"] = bool(
                torch.isfinite(training_input.grad).all()
                and all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in backend.parameters()
                )
            )
        else:
            result["training_finite"] = False
        result["training_step_ms"] = (time.perf_counter() - start) * 1000.0
        result["training_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["training_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    except Exception as exc:  # Preserve backend failures as benchmark evidence.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        del backend, x
        torch.cuda.empty_cache()
    return result


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", nargs="+", default=["gru", "mamba2"])
    parser.add_argument("--length", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--streams", type=int, default=66)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the target-hardware benchmark")

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "mamba_ssm": package_version("mamba-ssm"),
        "tilelang": package_version("tilelang"),
        "apache_tvm_ffi": package_version("apache-tvm-ffi"),
        "quack_kernels": package_version("quack-kernels"),
        "dtype": "bfloat16",
    }
    cases = [
        benchmark_case(
            backend,
            length,
            args.batch,
            args.streams,
            args.dim,
            args.state,
            args.headdim,
            args.depth,
            args.repeats,
        )
        for backend in args.backend
        for length in args.length
    ]
    print(json.dumps({"environment": environment, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
