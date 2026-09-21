"""Component-scoped M0-M3 gates. Failures never imply a verdict on the architecture."""

import copy
import hashlib

import numpy as np
import importlib.util
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .data import JointSampler, audit_episodes
from .lewm import SIGReg, joint_loss
from .lewm_config import LeWMConfig
from .gates import ComponentGateError
from .mamba_recurrence import MambaCarry
from .sources import ROOT, lewm_source_manifest, tensor_state_digest
from .world_api import ModelBundle


def _assert_close(a, b):
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


# Sealed 2026-09-05 on RTX 3060, pinned Mamba + Triton 3.7.1. See the
# calibration records in docs/evidence/m0_m3. No tolerance is fit at gate time.
# SSM comparison is absolute-only: large relative errors near zero are unhelpful.
NUMERICAL_PROFILE = "rtx3060_mamba_f577286d_v2"
NUMERICAL_TOLERANCES = {
    "reference_fp32": {"output": (1e-5, 1e-4), "conv": (1e-5, 1e-4), "ssm": (1e-5, 0.0),
                       "gradient": (1e-5, 1e-3)},
    "triton_fp32": {"output": (5e-4, 1e-4), "conv": (1e-5, 1e-4), "ssm": (1e-4, 0.0),
                    "gradient": (2e-5, 1e-3)},
    # In a stack, later convolution inputs include preceding layers' SSD rounding.
    # Keep the single-mixer convolution check tighter; only this propagation gets 1e-4.
    "triton_world_fp32": {"output": (5e-4, 1e-4), "conv": (1e-4, 1e-4), "ssm": (1e-4, 0.0)},
    "triton_bf16": {"output": (1e-2, 5e-3), "conv": (1e-5, 0.0), "ssm": (1e-3, 0.0),
                    "gradient": (2e-3, 5e-3)},
    "triton_reference_fp32": {"output": (1e-3, 1e-4), "conv": (1e-5, 1e-4),
                              "ssm": (1e-3, 0.0), "gradient": (5e-5, 1e-3)},
    "triton_reference_bf16": {"output": (1e-2, 5e-3), "conv": (1e-5, 0.0),
                              "ssm": (1e-3, 0.0), "gradient": (2e-3, 5e-3)},
}


def numerical_check(left, right, profile, quantity):
    atol, rtol = NUMERICAL_TOLERANCES[profile][quantity]
    try:
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{profile}/{quantity}: {error}") from error
    return float((left.detach().float()-right.detach().float()).abs().max())


def mixer_numerical_audit(mixer, precision: str) -> dict:
    """Independent equations, actual source operator and incoming-state gradients."""
    device = mixer.core.in_proj.weight.device
    backend = mixer.settings.backend
    if precision == "bf16" and backend != "triton":
        raise ComponentGateError("recurrence_precision", "BF16 reference backend has no sealed profile")
    profile = f"{backend}_{precision}"
    rows = []
    # The CUDA check includes a real chunk boundary, not only T < chunk_size.
    lengths = (2, 17, 65, mixer.settings.chunk_size+1) if backend == "triton" else (9, 65)
    for length in lengths:
        rng = torch.Generator(device=device).manual_seed(904+length)
        x = torch.randn(2, length, mixer.settings.width, device=device, generator=rng, requires_grad=True)
        empty = mixer.initial(2, device=device)
        incoming = MambaCarry(torch.randn(empty.conv.shape, device=device, generator=rng).requires_grad_(),
                              (.01*torch.randn(empty.ssm.shape,device=device,generator=rng)).requires_grad_())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            y, final = mixer.scan(x, incoming)
            carry, parts = incoming, []
            for t in range(length):
                part, carry = mixer.step(x[:, t:t+1], carry)
                parts.append(part)
            stepped = torch.cat(parts, 1)
            reference, ref_state = mixer.scan(x, incoming, backend="reference")
        row = {"length": length}
        for name, left, right in (("output",y,stepped),("conv",final.conv,carry.conv),("ssm",final.ssm,carry.ssm)):
            row[f"scan_step_{name}_max_abs"] = numerical_check(left,right,profile,name)
        variables = [x, incoming.conv, incoming.ssm, *mixer.parameters()]
        def gradients(output, state):
            objective = output.float().square().mean()+state.conv.float().square().mean()+state.ssm.square().mean()
            return torch.autograd.grad(objective,variables)
        full_grad, step_grad, ref_grad = gradients(y,final), gradients(stepped,carry), gradients(reference,ref_state)
        row["scan_step_gradient_max_abs"] = max(numerical_check(a,b,profile,"gradient")
                                                  for a,b in zip(full_grad,step_grad))
        if full_grad[1].abs().sum() == 0 or full_grad[2].abs().sum() == 0:
            raise AssertionError("gradient to incoming recurrence state was lost")
        comparison = f"triton_reference_{precision}" if backend == "triton" else profile
        row["reference_output_max_abs"] = numerical_check(y,reference,comparison,"output")
        row["reference_ssm_max_abs"] = numerical_check(final.ssm,ref_state.ssm,comparison,"ssm")
        row["reference_gradient_max_abs"] = max(numerical_check(a,b,comparison,"gradient")
                                                 for a,b in zip(full_grad,ref_grad))
        rows.append(row)
    return {"precision": precision, "profile": profile, "rows": rows, "incoming_state_gradients": True}


def objective_audit(config: LeWMConfig) -> dict:
    path = ROOT / "sources/lucas-maes__le-wm/module.py"
    spec = importlib.util.spec_from_file_location("_lewm_source_objective", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    device = config.runtime.device
    generator = torch.Generator(device=device).manual_seed(27)
    values = torch.randn(4, 8, 12, device=device, generator=generator, requires_grad=True)
    ours = SIGReg(17, 1024).to(device)
    reference = source.SIGReg(knots=17, num_proj=1024).to(device)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(321)
        expected = reference(values)
    actual = ours(values, torch.Generator(device=device).manual_seed(321))
    _assert_close(actual, expected)
    _assert_close(torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0])
    offset = torch.randn(1, 8, 12, device=device, generator=generator) * 3
    r = values-values.mean(0, keepdim=True)
    shifted = values+offset
    shifted = shifted-shifted.mean(0, keepdim=True)
    a = ours(r, torch.Generator(device=device).manual_seed(7))
    b = ours(shifted, torch.Generator(device=device).manual_seed(7))
    _assert_close(a, b)
    raw = ours(values+offset, torch.Generator(device=device).manual_seed(7))
    if torch.isclose(raw, a):
        raise AssertionError("the regularization treatment is not separated")
    return {"source_value": float(expected.detach()), "absolute_error": float((actual-expected).detach().abs()),
            "centering_offset_error": float((a-b).detach().abs())}


def window_audit(bundle: ModelBundle) -> dict:
    """The source predictor's mechanics: bounded window, causality, streaming parity.

    The Mamba audit asserts on conv/SSM carries that this backend does not have. What it
    must establish instead is the finite-window contract: one step equals the rolling scan,
    an evicted pair cannot reach the prediction, and a future action cannot reach the past.
    AdaLN-Zero leaves the predictor an identity at initialization, so the gate temporarily
    wakes a *copy* of the gates -- the audited weights are never touched.
    """
    import copy

    bundle.eval()
    c, w = bundle.config, bundle.world
    device = next(w.parameters()).device
    context = w.context
    probe = copy.deepcopy(w).to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(509)
    for block in probe.predictor.transformer.layers:
        gate = block.adaLN_modulation[-1]
        with torch.no_grad():
            gate.weight.copy_(torch.randn(gate.weight.shape, generator=generator).to(device) * 0.05)
            gate.bias.copy_(torch.randn(gate.bias.shape, generator=generator).to(device) * 0.05)

    length = context + 4
    z = torch.randn(2, length + 1, 1, c.encoder.latent_dim, device=device,
                    generator=torch.Generator(device=device).manual_seed(510))
    actions = torch.randint(c.dynamics.n_actions, (2, length), device=device,
                            generator=torch.Generator(device=device).manual_seed(511))
    with torch.no_grad():
        scanned = probe.teacher(z, actions)
        state = probe.start(z[:, :1])
        for index in range(actions.shape[1]):
            state, _ = probe.observe_latent(state, actions[:, index:index + 1], z[:, index + 1:index + 2])
        _assert_close(state.history, scanned.features[:, -1:])
        _assert_close(state.latent, scanned.state.latent)

        step = actions[:, :1]
        base, _ = probe.advance(probe.teacher(z[:, :context + 1], actions[:, :context]).state, step)
        evicted = z.clone(); evicted[:, 0] += 7.0
        moved, _ = probe.advance(probe.teacher(evicted[:, :context + 1], actions[:, :context]).state, step)
        if not torch.equal(base.latent, moved.latent):
            raise ValueError("window_audit: an evicted pair still reached the prediction")
        active = z.clone(); active[:, context] += 7.0
        changed, _ = probe.advance(probe.teacher(active[:, :context + 1], actions[:, :context]).state, step)
        if torch.equal(base.latent, changed.latent):
            raise ValueError("window_audit: an active pair did not reach the prediction")
        later = actions.clone(); later[:, -1] = (later[:, -1] + 1) % c.dynamics.n_actions
        _assert_close(probe.teacher(z, later).predicted[:, :-1], scanned.predicted[:, :-1])
    return {"backend": "sdpa", "context_pairs": context, "window_forgets_evicted_pairs": True,
            "streaming_matches_scan": True, "causal": True,
            "predictor_parameters": sum(p.numel() for p in w.predictor.parameters()),
            "package_parameters": sum(p.numel() for p in w.parameters() if p.requires_grad),
            "numerical_profile": "sdpa_fp32"}


def recurrence_audit(bundle: ModelBundle) -> dict:
    """Longer-than-training context, incoming carry gradients, source and chunk checks."""
    if getattr(bundle.config, "family", "lewm_mamba") == "lewm_transformer":
        return window_audit(bundle)
    bundle.eval()
    c, w = bundle.config, bundle.world
    device = next(w.parameters()).device
    rng = torch.Generator(device=device).manual_seed(500)
    profile = "triton_world_fp32" if c.dynamics.backend == "triton" else "reference_fp32"
    errors = []
    def compare_states(left, right):
        for i, (a, b) in enumerate(zip(bundle.state_tensors(left), bundle.state_tensors(right))):
            quantity = "ssm" if i >= 3 and i % 2 else "conv" if i >= 2 else "output"
            errors.append(numerical_check(a,b,profile,quantity))
    z = torch.randn(2, 18, 1, c.encoder.latent_dim, device=device, generator=rng)
    # Stacked frame-gap actions when the recipe strides; (2,17) at stride 1.
    stack = c.joint.stride
    a = torch.randint(c.dynamics.n_actions, (2, 17) if stack == 1 else (2, 17, stack),
                      device=device, generator=rng)
    with torch.no_grad():
        full = w.teacher(z, a)
        state = bundle.start(z[:, :1])
        for i in range(a.shape[1]):
            state, _ = w.observe_latent(state, a[:, i:i+1], z[:, i+1:i+2])
        compare_states(full.state, state)
        prefix = w.teacher(z[:, :8], a[:, :7]).state
        chunked = w.teacher(z[:, 7:], a[:, 7:], state=prefix)
        compare_states(full.state, chunked.state)
        changed = a.clone()
        changed[:, 7] = (changed[:, 7]+1) % c.dynamics.n_actions
        causal = w.teacher(z, changed)
        _assert_close(full.features[:, :8], causal.features[:, :8])
        root = bundle.prefill(z[:, :8], a[:, :7])
        original = [v.clone() for v in bundle.state_tensors(root)]
        buffers = tensor_state_digest(w.state_dict())
        first, _ = bundle.advance(root, a[:, 7:8])
        bundle.advance(bundle.fork(root), changed[:, 7:8])
        again, _ = bundle.advance(root, a[:, 7:8])
        for before, after in zip(original, bundle.state_tensors(root)):
            _assert_close(before, after)
        for before, after in zip(bundle.state_tensors(first), bundle.state_tensors(again)):
            _assert_close(before, after)
        if tensor_state_digest(w.state_dict()) != buffers:
            raise AssertionError("branch changed world buffers")
        if first.step != root.step+1:
            raise AssertionError("accepted transition did not advance exactly once")
        perturbed = z.clone()
        perturbed[:, 0] += 1
        other = w.teacher(perturbed, a).state
        memory_effect = float((other.history-full.state.history).abs().max())
        if memory_effect <= 1e-8:
            raise AssertionError("old context has no measurable effect with current observation fixed")

    mixer = w.layers[0].mixer
    numerical = [mixer_numerical_audit(mixer, "fp32")]
    if c.runtime.precision != "fp32":
        numerical.append(mixer_numerical_audit(mixer, c.runtime.precision))
    x = torch.randn(2, 9, c.dynamics.width, device=device, generator=rng)
    # Independent oracle: execute upstream's actual inference step, using its own
    # provided PyTorch fallbacks on CPU. This does NOT validate the CUDA kernels.
    import mamba_ssm.modules.mamba2 as source
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
    core = copy.deepcopy(mixer.core)
    src_state = mixer.initial(2, device=device)
    with torch.no_grad():
        if device.type == "cpu":
            with patch.object(source, "selective_state_update", None), patch.object(source, "causal_conv1d_update", None), \
                 patch.object(core.norm, "forward", side_effect=lambda v, gate: rms_norm_ref(
                     v, core.norm.weight, None, z=gate, eps=core.norm.eps, norm_before_gate=False)):
                expected = []
                for t in range(x.shape[1]):
                    out, conv, ssm = core.step(x[:, t:t+1], src_state.conv, src_state.ssm)
                    src_state = MambaCarry(conv, ssm)
                    expected.append(out)
                expected = torch.cat(expected, 1)
        else:
            expected = core(x)
        actual, _ = mixer.scan(x)
        _assert_close(actual, expected)
    return {"context_frames": 18, "training_frames": c.joint.frames,
            "numerical_profile": NUMERICAL_PROFILE, "numerical_checks": numerical,
            "world_state_max_abs": max(errors),
            "state_gradient_checked": True, "source_step_error": float((actual-expected).abs().max()),
            "memory_effect": memory_effect, "cuda_kernels_checked": device.type == "cuda"}


def normalization_audit(bundle: ModelBundle) -> dict:
    bundle.eval()
    e = bundle.config.encoder
    device = next(bundle.encoder.parameters()).device
    frames = torch.randint(256, (2, 4, e.resolution, e.resolution, 3), device=device, dtype=torch.uint8,
                           generator=torch.Generator(device=device).manual_seed(77))
    before = tensor_state_digest(bundle.encoder.state_dict())
    with torch.no_grad():
        whole = bundle.encode(frames)
        for b in range(2):
            for t in range(4):
                _assert_close(whole[b:b+1, t:t+1], bundle.encode(frames[b:b+1, t:t+1]))
    if before != tensor_state_digest(bundle.encoder.state_dict()):
        raise AssertionError("eval encoding changed normalization buffers")
    return {"singleton_batch_parity": True, "buffers_immutable": True}


def resource_preflight(bundle, episodes) -> dict:
    """Actual declared batch and optimizer step, without changing a training model."""
    from .train import set_phase_mode, joint_optimizer, autocast_context

    c = bundle.config
    set_phase_mode(bundle, "joint")
    sampler = JointSampler(episodes, c, torch.Generator().manual_seed(c.seed + 19))
    optimizer = joint_optimizer(bundle)
    regularizer = SIGReg(c.joint.knots, c.joint.projections).to(c.runtime.device)
    rng = torch.Generator(device=c.runtime.device).manual_seed(c.seed + 20)
    cuda = c.runtime.device == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elapsed, encoder_gradient, predictor_gradient = [], [], []
    for _ in range(3):
        start = time.perf_counter()
        batch = sampler.sample().to(c.runtime.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(c):
            loss = joint_loss(bundle.encoder, bundle.world, batch.frames, batch.actions, regularizer, rng, c)
        loss.total.backward()
        encoder_gradient.append(float(bundle.encoder.projector[0].weight.grad.norm()))
        # The first trainable weight the prediction path reaches. For the source predictor
        # that cannot be an attention or action weight: AdaLN-Zero gates both to exactly
        # zero at initialization by design, so a zero there is correct, not a dead graph.
        probe = (bundle.world.predictor_projector.net[0]
                 if getattr(c, "family", "lewm_mamba") == "lewm_transformer"
                 else bundle.world.pair_projection)
        predictor_gradient.append(float(probe.weight.grad.norm()))
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, c.joint.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    if min(encoder_gradient) <= 0 or min(predictor_gradient) <= 0:
        raise AssertionError("joint encoder or dynamics received no gradient")
    allocated = torch.cuda.max_memory_allocated() if cuda else None
    reserved = torch.cuda.max_memory_reserved() if cuda else None
    if cuda and max(allocated, reserved) > c.runtime.memory_budget_bytes:
        raise ComponentGateError("joint_memory", f"peak {max(allocated,reserved)} exceeds {c.runtime.memory_budget_bytes}")
    return {"actual_batch": c.joint.batch, "frames": c.joint.frames,
            "precision": c.runtime.precision, "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved, "warm_step_seconds": sum(elapsed[1:])/2,
            "encoder_gradient": encoder_gradient[-1], "predictor_gradient": predictor_gradient[-1],
            "gpu_validated": cuda,
            "device_name": torch.cuda.get_device_name() if cuda else "cpu",
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
            "optimizer_steps": 3, "purpose": "resource_only_not_a_learning_result"}


def joint_checks(config: LeWMConfig, episodes, report: dict):
    """Family-specific probes; gates.py owns dependency handling and reporting."""
    from .gates import Gate

    bundle = None
    def sources():
        report["sources"] = lewm_source_manifest(config)
        return {"versions": report["sources"]["versions"]}
    def device():
        if config.runtime.device == "cuda" and not torch.cuda.is_available():
            raise ComponentGateError("device", "CUDA unavailable in this process")
        return config.runtime.device
    def construct():
        nonlocal bundle
        bundle = ModelBundle.create(config)
        return {"encoder_parameters": sum(p.numel() for p in bundle.encoder.parameters()),
                "world_parameters": sum(p.numel() for p in bundle.world.parameters())}
    return (
        Gate("source_identity", sources),
        Gate("dataset", lambda: {"split_counts": audit_episodes(episodes,config)["split_counts"]}),
        Gate("device", device),
        Gate("objective_source",lambda: objective_audit(config),("source_identity","device")),
        Gate("model_construction",construct,("source_identity","dataset","device")),
        Gate("recurrence",lambda: recurrence_audit(bundle),("model_construction",)),
        Gate("normalization",lambda: normalization_audit(bundle),("model_construction",)),
        Gate("joint_resource",lambda: resource_preflight(bundle,episodes),
             ("objective_source","recurrence","normalization")),
    )


def covariance_summary(values):
    """Finite empirical spectrum; report scale separately from effective rank."""
    x = values.detach().cpu().double().reshape(-1, values.shape[-1])
    if not bool(torch.isfinite(x).all()):
        raise ComponentGateError("screen_finiteness", "nonfinite exported features")
    centered = x-x.mean(0)
    spectrum = torch.linalg.eigvalsh(centered.T@centered/max(1,len(x)-1)).clamp_min(0).flip(0)
    total = spectrum.sum()
    probability = spectrum/total.clamp_min(1e-30)
    rank = torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum()) if total > 0 else torch.tensor(0.)
    return {"coordinate_variance": float(total/x.shape[-1]), "mean_norm": float(x.mean(0).norm()),
            "effective_rank": float(rank), "eigenvalues": spectrum.tolist()}


@torch.no_grad()
def screen_features(bundle, windows, settings):
    """Frozen exported/CLS features and teacher predictions on one fixed sample."""
    from .lewm_config import window_layout

    bundle.eval()
    before = tensor_state_digest(bundle.encoder.state_dict())
    z_rows, cls_rows, prediction, marginal = [], [], [], []
    actions = windows["actions"]
    permuted = actions.roll(1, 0)
    predicted_at = window_layout(bundle.config.joint)[1]
    for start in range(0, len(actions), settings.encode_batch):
        end = start+settings.encode_batch
        frames = windows["frames"][start:end].to(bundle.device)
        with torch.autocast(device_type=bundle.device.type, enabled=False):
            z, cls = bundle.encoder.projected_and_cls(frames)
            # A widened centering window encodes frames the rollout never predicts.
            # Keep the prediction frames alone, exactly as `joint_loss` does: the
            # outgoing actions, the labels and every downstream reading are indexed
            # on the prediction transitions, not on the encoded window.
            if predicted_at != tuple(range(z.shape[1])):
                z, cls = z[:, list(predicted_at)], cls[:, list(predicted_at)]
            predicted = bundle.world.teacher(z, actions[start:end].to(bundle.device)).predicted
            shuffled = bundle.world.teacher(z, permuted[start:end].to(bundle.device)).predicted
        z_rows.append(z[:, :, 0].cpu()); cls_rows.append(cls.cpu())
        prediction.append(predicted[:, :, 0].cpu()); marginal.append(shuffled[:, :, 0].cpu())
    if tensor_state_digest(bundle.encoder.state_dict()) != before:
        raise ComponentGateError("normalization", "screen changed encoder buffers")
    z = torch.cat(z_rows)
    return {"projected": z, "cls": torch.cat(cls_rows), "prediction": torch.cat(prediction),
            "permuted_prediction": torch.cat(marginal), "persistence": z[:, :-1]}


def screen_prediction_report(features, clusters, settings, *, variance):
    target = features["projected"][:, 1:]
    errors = {name: (features[name]-target).square().mean((1,2))
              for name in ("prediction", "permuted_prediction", "persistence")}
    def difference(other):
        delta = errors["prediction"]-errors[other]
        values = torch.stack([delta[clusters == k].mean() for k in clusters.unique(sorted=True)])
        indices = torch.randint(len(values), (settings.bootstrap_draws, len(values)),
                                generator=torch.Generator().manual_seed(settings.seed+300))
        return {"difference": float(values.mean()),
                "interval": values[indices].mean(1).quantile(torch.tensor([.025,.975])).tolist(), "unit": "episode"}
    return {"normalized_prediction_mse": float(errors["prediction"].mean()/max(variance, settings.variance_floor)),
            "mse": {name: float(value.mean()) for name,value in errors.items()},
            "prediction_minus_persistence": difference("persistence"),
            "prediction_minus_permuted_actions": difference("permuted_prediction"),
            "scope": "logged-action association, not simulator action consequences"}, errors


def screen_retention(train_features, dev_features, train_windows, dev_windows, settings, device, n_actions):
    from .diagnostics import binary_auc, paired_auc_interval, fit_outcome_probe

    truth = dev_windows["labels"].flatten(0,1)
    valid = dev_windows["valid"].flatten(0,1)
    fit_truth = train_windows["labels"].flatten(0,1)
    fit_valid = train_windows["valid"].flatten(0,1)
    coverage = []
    for col, name in enumerate(train_windows["label_names"]):
        counts = {}
        for split, y, mask in (("train",fit_truth,fit_valid),("dev",truth,valid)):
            counts[split] = {"positive": int((y[:,col]&mask[:,col]).sum()),
                             "negative": int((~y[:,col]&mask[:,col]).sum())}
        supported = all(v["positive"] >= settings.minimum_positive and v["negative"] >= settings.minimum_negative
                        for v in counts.values())
        coverage.append({"label": name, "counts": counts, "supported": supported})
    selected = torch.tensor([row["supported"] for row in coverage])
    clusters = dev_windows["clusters"].repeat_interleave(dev_windows["actions"].shape[1])
    report = {"coverage": coverage, "probes": {}, "critical_semantic_retention": "not_evaluated",
              "scope": "short-future archive proxies conditioned on outgoing action"}
    outputs = {}
    for hidden, family in ((False,"linear"),(True,"mlp")):
        predictions = {}
        for name in ("cls","projected"):
            # One row per retained transition; a stacked window contributes each
            # of its frame-gap actions as its own one-hot block.
            def conditioned(features, windows):
                actions = windows["actions"]
                actions = actions.reshape(-1, actions.shape[-1]) if actions.ndim == 3 else actions.reshape(-1, 1)
                encoded = torch.nn.functional.one_hot(actions, n_actions).float().flatten(1)
                return torch.cat((features[name][:, :-1].flatten(0, 1), encoded), -1).to(device)
            fit_x, dev_x = [conditioned(features, windows)
                            for features, windows in ((train_features, train_windows), (dev_features, dev_windows))]
            predictions[name] = fit_outcome_probe(fit_x, fit_truth.float().to(device), fit_valid.float().to(device),
                                                  dev_x, settings, hidden=hidden)
        outputs[family] = predictions
        if bool(selected.any()):
            comparison = paired_auc_interval(predictions["projected"][:,selected], predictions["cls"][:,selected],
                                             truth[:,selected], valid[:,selected], clusters,
                                             draws=settings.bootstrap_draws, seed=settings.seed+400)
        else:
            comparison = {"status": "insufficient_coverage", "difference": None, "interval": None}
        comparison["auc"] = {name: [binary_auc(scores[valid[:,i],i], truth[valid[:,i],i])
                                    for i in range(truth.shape[1])] for name,scores in predictions.items()}
        report["probes"][family] = comparison
    report["projection_stop"] = all(row["interval"] is not None and row["interval"][1] < -settings.auc_margin
                                     for row in report["probes"].values())
    return report, {"scores": outputs, "truth": truth, "valid": valid, "clusters": clusters}


def screen_joint_pair(runs, episodes, dataset_contract, settings, output):
    """Paired G1 with checkpoint identity, mechanics and component-local stops."""
    import json
    from .config import config_from_dict, recipe_dict, recipe_digest
    from .checkpoint import read_lewm_bundle
    from .data import atomic_manifest, screen_windows, _sha256
    from .gates import contract_digest
    from .world_api import load_bundle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "screen.json"
    if report_path.exists():
        raise ValueError("screen_output: refusing to replace a sealed report")
    report = {"schema": "d4mj_joint_screen_report_v1", "settings": recipe_dict(settings),
              "settings_id": recipe_digest(settings), "dataset_id": contract_digest(dataset_contract),
              "sources": lewm_source_manifest(config_from_dict(json.loads(
                  (Path(runs["raw"])/"resolved_recipe.json").read_text()))),
              "arms": {}, "components": {}, "decision": "stop_component",
              "architecture_verdict": "not_evaluated", "m4_authorized": False}
    payloads = {}
    try:
        pair_path = Path(runs["raw"]).parent/"pair.json"
        pair = json.loads(pair_path.read_text())
        if pair["screen_settings_id"] != recipe_digest(settings) or pair["dataset_id"] != contract_digest(dataset_contract):
            raise ComponentGateError("screen_settings", "screen differs from the pre-training seal")
        report["pair_manifest"] = {"path":str(pair_path.resolve()),"sha256":_sha256(pair_path)}
        for variant in ("raw", "tc"):
            run = Path(runs[variant]); config = config_from_dict(json.loads((run/"resolved_recipe.json").read_text()))
            checkpoint = run/"joint"/f"step-{config.joint.screen_step:06d}.pt"
            payloads[variant] = p = read_lewm_bundle(checkpoint)
            initial = read_lewm_bundle(run/"joint/step-000000.pt")
            if (p["config"] != recipe_dict(config) or p["step"] != config.joint.screen_step
                    or p["dataset"] != dataset_contract or initial["step"] != 0
                    or initial["config"] != p["config"] or initial["dataset"] != dataset_contract
                    or initial["initial_identity"] != p["initial_identity"]):
                raise ComponentGateError("pair_identity", "wrong recipe, dataset, step or initial checkpoint")
            for name in ("encoder", "world"):
                if tensor_state_digest(initial["modules"][name]) != p["initial_identity"][name]:
                    raise ComponentGateError("initial_identity", "saved initialization bytes do not match training")
            report["arms"][variant] = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
                                      "recipe_id": p["recipe_id"], "initial_identity": p["initial_identity"]}
        raw, tc = payloads["raw"], payloads["tc"]
        from .lewm_config import pair_axis
        try:
            axis = pair_axis(raw["config"], tc["config"])
        except ValueError as error:
            raise ComponentGateError("pair_identity", str(error)) from error
        report["pair_axis"] = axis
        # A variant pair must sit in the slot that names it; a centering pair is
        # TC in both slots, so only the axis itself identifies the arms.
        if axis == "variant" and any(payloads[v]["config"]["variant"] != v for v in ("raw", "tc")):
            raise ComponentGateError("pair_identity", "a variant pair must name its arms raw and tc")
        if (raw["initial_identity"] != tc["initial_identity"]
                or not torch.equal(raw["sampler"]["generator"],tc["sampler"]["generator"])
                or not torch.equal(raw["projection_rng"],tc["projection_rng"])):
            raise ComponentGateError("pair_identity", "raw/TC differ beyond the declared objective")
        histories = [[json.loads(line) for line in (Path(runs[v])/"joint/metrics.jsonl").read_text().splitlines()]
                     for v in ("raw","tc")]
        length = raw["step"]
        if any([row["update"] for row in h] != list(range(1,length+1)) for h in histories):
            raise ComponentGateError("pair_history", "screen requires every update exactly once")
        if any(a["windows"] != b["windows"] or a["learning_rate"] != b["learning_rate"]
               for a,b in zip(*histories,strict=True)):
            raise ComponentGateError("pair_history", "actual windows or schedules differ")
        first_raw, first_tc = histories[0][0], histories[1][0]
        if (abs(first_raw["prediction"]-first_tc["prediction"]) > 1e-6
                or abs(first_raw["regularization"]-first_tc["regularization"]) <= 1e-6):
            raise ComponentGateError("objective_contrast", "first paired update lacks the intended isolated treatment")
        if not all(all(torch.isfinite(torch.tensor(float(value))) for key,value in row.items()
                       if isinstance(value,(int,float))) for h in histories for row in h):
            raise ComponentGateError("training_finiteness", "nonfinite training history")
        report["components"]["pair_identity"] = {"status": "pass"}
        config = config_from_dict(raw["config"])
        windows = {split: screen_windows(episodes, config, settings, split) for split in ("train","dev")}
        atomic_manifest(output/"windows.json", {split:{"episode_ids":w["episode_ids"],"starts":w["starts"].tolist()}
                                                for split,w in windows.items()})
        report["components"]["objective_contrast"] = {"status":"pass","detail":objective_audit(config)}
        for variant in ("raw","tc"):
            entry = report["arms"][variant]
            bundle, _ = load_bundle(entry["checkpoint"])
            bundle.world.requires_grad_(True)
            for name, check in (("normalization",normalization_audit),("recurrence",recurrence_audit)):
                try:
                    report["components"][f"{variant}_{name}"] = {"status":"pass","detail":check(bundle)}
                except Exception as error:
                    raise ComponentGateError(f"{variant}_{name}",str(error)) from error
            bundle.world.requires_grad_(False)
            features = {split: screen_features(bundle,w,settings) for split,w in windows.items()}
            z = features["dev"]["projected"]
            entry["spectra"] = {"raw":covariance_summary(z), "residual":covariance_summary(z-z.mean(1,keepdim=True)),
                                "persistent":covariance_summary(z.mean(1))}
            entry["temporal_power"] = torch.fft.rfft(z.double(),dim=1).abs().square().mean((0,2)).tolist()
            variance = covariance_summary(features["train"]["projected"])["coordinate_variance"]
            if variance < settings.variance_floor:
                raise ComponentGateError(f"{variant}_latent_variance", "numerical export collapse")
            entry["prediction"], errors = screen_prediction_report(features["dev"],windows["dev"]["clusters"],settings,variance=variance)
            entry["retention"], probe_rows = screen_retention(features["train"],features["dev"],windows["train"],windows["dev"],
                                                            settings,bundle.device,bundle.n_actions)
            torch.save({"features":features,"errors":errors,"probes":probe_rows},output/f"{variant}_rows.pt")
            del bundle
            initial_bundle, _ = load_bundle(Path(runs[variant])/"joint/step-000000.pt")
            initial_features = {split:screen_features(initial_bundle,w,settings) for split,w in windows.items()}
            initial_variance = covariance_summary(initial_features["train"]["projected"])["coordinate_variance"]
            entry["initial_prediction"], _ = screen_prediction_report(initial_features["dev"],windows["dev"]["clusters"],settings,variance=initial_variance)
            del initial_bundle, initial_features, features
            if entry["retention"]["projection_stop"]:
                raise ComponentGateError(f"{variant}_projection_retention", "both fixed probes exceed the loss margin with paired support")
            entry["learning_progress"] = entry["prediction"]["normalized_prediction_mse"] < entry["initial_prediction"]["normalized_prediction_mse"]
        report["decision"] = "continue_joint_budget" if all(a["learning_progress"] for a in report["arms"].values()) else "review_required"
    except Exception as error:
        component = getattr(error,"component","joint_screen_execution")
        report["components"][component] = {"status":"fail","reason":str(error)}
        report["blocked_component"] = component
    report["artifacts"] = {p.name:_sha256(p) for p in output.iterdir() if p.is_file() and p != report_path}
    report["artifact_root"] = str(output.resolve())
    report["report_id"] = contract_digest(report)
    atomic_manifest(report_path,report)
    return report


# --- G2/G3/G4 phase gates -------------------------------------------------------------------
#
# `gates.require_bridge_gate` and `require_actor_gate` own identity and the stop decision; they
# deliberately refuse to compute evidence, so this is where the measurements live.  Three sealed
# reports are needed, not one: H2 -> H16, H16 -> actor, and the actor screen -> actor budget.
#
# Every component fails closed.  A stratum without support reports `insufficient_coverage` and
# does not pass, because an absent label set must never authorize thousands of further updates.

GATE_NOT_EVALUATED = {
    "stochastic_successor_modes": "needs repeated-simulator-seed forks; not a gate component",
    "persistent_memory_utility": "needs varied earlier context with the observation held fixed",
    "decoded_tiles": "needs the M6 renderer",
}


def _paired_mean_interval(left, right, clusters, *, draws: int, seed: int):
    """Episode-clustered bootstrap of mean(left) - mean(right). Lower is better for errors."""
    import torch
    left, right, clusters = left.flatten(), right.flatten(), clusters.flatten()
    groups = [torch.where(clusters == key)[0] for key in clusters.unique(sorted=True)]
    if len(groups) < 2:
        return {"difference": None, "interval": None, "excludes_zero": False,
                "status": "insufficient_coverage"}
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(draws):
        pick = torch.cat([groups[i] for i in
                          torch.randint(len(groups), (len(groups),), generator=generator)])
        samples.append(float(left[pick].mean() - right[pick].mean()))
    low, high = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"difference": float(left.mean() - right.mean()),
            "interval": [low, high], "excludes_zero": bool(low > 0 or high < 0)}


def _cluster_ids(names, device):
    """Stable across processes. Python's `hash` for strings is salted by PYTHONHASHSEED, so using
    it moved bootstrap group membership between runs and could shift finite-draw intervals."""
    return torch.tensor([int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                         % (2**31) for name in names], device=device)


def _component(status: str, metrics: dict, evidence: list, criterion: dict | None = None) -> dict:
    """A component states the quantity it was decided on, so the boundary can recompute it.

    `require_bridge_gate` recomputes `status` from `criterion`, so the two cannot disagree and a
    hand-edited status is refused even when the content digest is recomputed over the edit.
    """
    if criterion is None:
        # A component with nothing to decide on must not read as a pass.
        criterion = {"quantity": "unmeasured", "value": 0.0, "threshold": 1.0, "direction": "greater"}
    return {"status": status, "metrics": metrics, "evidence": evidence, "criterion": criterion}


def _raw_rows(output, name: str, rows: dict) -> dict:
    """The per-row inputs a result was computed from, so it can be recomputed rather than trusted.

    A run of this cost should not leave only aggregates behind: an interval that cannot be
    recomputed is an assertion about a number nobody can check.
    """
    from .data import _sha256
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.rows.pt"
    torch.save({key: (value.cpu() if torch.is_tensor(value) else value)
                for key, value in rows.items()}, path)
    return {"path": str(path), "sha256": _sha256(path),
            "rows": int(next((len(v) for v in rows.values() if torch.is_tensor(v)), 0))}


def _evidence(output, name: str, payload: dict) -> list:
    """Write one immutable evidence file and bind it by bytes."""
    from .data import _sha256, atomic_manifest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.json"
    atomic_manifest(path, payload)
    return [{"path": str(path), "sha256": _sha256(path)}]


def _seal(body: dict) -> dict:
    from .gates import contract_digest
    return dict(body, report_id=contract_digest(body))


def _gate_traces(bundle, episodes, config, *, batches: int, seed: int):
    """A DEV-only sample drawn exactly as Phase 2 draws, reused by every component.

    This previously received the whole latent cache and sampled it unfiltered, while training
    subsets to ``split == "train"`` (train.py). The gate was therefore scoring the model partly on
    its own training data and partly on FINAL, which no pre-final gate may open. The DEV subset is
    built here, membership is asserted rather than assumed, and the exact episode/start ledger is
    returned so the selection can be recomputed and audited.
    """
    from .data import sample_bridge_batch, sample_bridge_terminals
    dev = episodes.subset(index for index, episode in enumerate(episodes)
                          if episode.split == "dev")
    if not len(dev):
        raise ComponentGateError("gate_split", "no DEV episode reached the gate corpus")
    allowed = {episode.episode_id for episode in dev}
    forbidden = {episode.episode_id for episode in episodes if episode.split == "final"}
    rng = torch.Generator().manual_seed(seed)
    traces, ledger = [], []
    for update in range(batches):
        main = sample_bridge_batch(dev, rng, config, update)
        terminal = sample_bridge_terminals(dev, rng, config, update)
        for batch in (main, terminal):
            selected = set(batch.episode_ids)
            if selected & forbidden:
                raise ComponentGateError("gate_split", "a FINAL episode reached a pre-final gate")
            if not selected <= allowed:
                raise ComponentGateError("gate_split", "a non-DEV episode reached the gate sample")
        ledger.append({"update": update,
                       "main": list(zip(main.episode_ids, main.starts.tolist())),
                       "terminal": list(zip(terminal.episode_ids, terminal.starts.tolist()))})
        traces.append((main.to(bundle.device), terminal.to(bundle.device)))
    return traces, {"split": "dev", "dev_episodes": len(dev), "batches": batches,
                    "seed": seed, "selection": ledger}


def _derangement(count: int, generator) -> torch.Tensor:
    """A permutation with no fixed point, so every row really is re-labelled."""
    while True:
        order = torch.randperm(count, generator=generator)
        if count < 2 or bool((order != torch.arange(count)).all()):
            return order


@torch.no_grad()
def _action_blind_rollout(bundle, prefix, rows: int, depth: int):
    """The conditional mean over actions at every generated step, from the observed anchor state.

    At each step all ``n_actions`` successors are predicted from the SAME state and averaged, and
    that average is accepted as the next latent. This is the control the spec means by
    "marginal-action": the best prediction available to a model that ignores which action was
    taken. A rollout told one specific wrong action is noisier, and therefore a weaker reference.

    ``prefix`` is the anchor state from the real teacher-forced prefix, so this differs from the
    true rollout only in the generated suffix -- the one thing under test.
    """
    count = bundle.n_actions
    device = prefix.latent.device
    actions = torch.arange(count, device=device).repeat(rows)[:, None]
    state, predicted = prefix, []
    for _ in range(depth):
        advanced, _ = bundle.advance(bundle.repeat_state(state, count), actions)
        latent = advanced.latent.reshape(rows, count, *advanced.latent.shape[1:]).mean(1)
        predicted.append(latent)
        # Carry the fan's own averaged recurrent state forward, so every later step is action-blind
        # too rather than only the first.
        state = type(advanced)(latent,
                               tuple(type(m)(*(v.reshape(rows, count, *v.shape[1:]).mean(1)
                                               for v in (m.conv, m.ssm)))
                                     for m in advanced.memory),
                               advanced.history.reshape(rows, count, *advanced.history.shape[1:]).mean(1),
                               advanced.step)
    return torch.cat(predicted, 1)


@torch.no_grad()
def _rollouts(bundle, traces, depth: int, *, seed: int):
    """Generated, persistence and marginal-action rollouts on one shared sample.

    Two distinct controls, which an earlier version conflated:

    ``marginal``  the ACTION-BLIND prediction: at each generated step the successor is the mean
                  over all actions of what the world predicts, which is the conditional mean and
                  therefore the best predictor available to a model that ignores the action. A
                  single deranged action is noisier than this and so is an easier baseline; using
                  it made the gate weaker than the specification asks.
    ``deranged``  the SENSITIVITY probe: the rollout told a wrong action. Beating this shows only
                  that actions are distinguished, not that they are used well, so it is reported
                  separately and never substitutes for the marginal control.
    """
    from .train import _bridge_initial_state, _outgoing_actions, bridge_rollout
    generator = torch.Generator().manual_seed(seed)
    rows = {"generated": [], "persistence": [], "marginal": [], "deranged": [], "clusters": [],
            "effect_true": [], "effect_pred": []}
    for main, _ in traces:
        batch = main.main
        z = batch.latents
        if depth >= z.shape[1]:
            continue
        actions = _outgoing_actions(batch, bundle.n_actions)
        initial = _bridge_initial_state(bundle, main)
        _, generated, _, anchor = bridge_rollout(bundle, z, actions, depth, initial)
        order = _derangement(len(z), generator).to(z.device)
        _, deranged, _, _ = bridge_rollout(bundle, z, actions[order], depth, initial)
        prefix = bundle.world.teacher(z[:, :anchor + 1], actions[:, :anchor], state=initial).state
        marginal = _action_blind_rollout(bundle, prefix, len(z), depth)
        truth = z[:, anchor + 1:anchor + 1 + depth].float()
        base = z[:, anchor:anchor + 1].float().expand_as(truth)
        per_row = lambda x: (x.float() - truth).square().mean(dim=tuple(range(1, truth.ndim)))
        rows["generated"].append(per_row(generated))
        rows["marginal"].append(per_row(marginal))
        rows["deranged"].append(per_row(deranged))
        rows["persistence"].append(per_row(base))
        rows["effect_true"].append((truth - base).flatten(1))
        rows["effect_pred"].append((generated.float() - base).flatten(1))
        rows["clusters"].append(_cluster_ids(main.episode_ids, z.device))
    if not rows["clusters"]:
        return None
    return {key: torch.cat(value).cpu() for key, value in rows.items()}


def _recursive_dynamics(bundle, traces, depths, *, draws: int, seed: int, output):
    """TC-16: the rollout must beat persistence AND the marginal-action baseline at each depth."""
    measured, passing = {}, True
    for depth in depths:
        roll = _rollouts(bundle, traces, depth, seed=seed + depth)
        if roll is None:
            measured[f"depth_{depth}"] = {"status": "insufficient_coverage"}
            passing = False
            continue
        # Errors: the rollout is better when its error is LOWER, so the contrast is baseline-minus.
        vs_persistence = _paired_mean_interval(roll["persistence"], roll["generated"],
                                               roll["clusters"], draws=draws, seed=seed + depth)
        vs_marginal = _paired_mean_interval(roll["marginal"], roll["generated"],
                                            roll["clusters"], draws=draws, seed=seed + 97 + depth)
        vs_deranged = _paired_mean_interval(roll["deranged"], roll["generated"],
                                            roll["clusters"], draws=draws, seed=seed + 61 + depth)
        beat = (vs_persistence.get("difference") or 0) > 0 and vs_persistence.get("excludes_zero") \
            and (vs_marginal.get("difference") or 0) > 0 and vs_marginal.get("excludes_zero")
        measured[f"depth_{depth}"] = {
            "rows": int(len(roll["generated"])),
            "generated_mse": float(roll["generated"].mean()),
            "persistence_mse": float(roll["persistence"].mean()),
            "marginal_action_mse": float(roll["marginal"].mean()),
            "deranged_action_mse": float(roll["deranged"].mean()),
            "vs_persistence": vs_persistence, "vs_marginal_action": vs_marginal,
            "vs_deranged_action": vs_deranged,
            "baselines": "marginal = conditional mean over all actions (the gating control); "
                         "deranged = one wrong action (sensitivity only, never substitutes)",
            "beats_both_baselines": bool(beat)}
        passing = passing and bool(beat)
    cleared = sum(1 for d in depths if measured.get(f"depth_{d}", {}).get("beats_both_baselines"))
    measured["depths_cleared"] = cleared
    measured["depths_required"] = len(depths)
    measured["failed_checks"] = len(depths) - cleared
    return _component("pass" if passing else "fail", measured,
                      _evidence(output, "recursive_dynamics", measured),
                      {"quantity": "depths_not_cleared",
                       "value": float(len(depths) - cleared), "threshold": 0.5, "direction": "less"})


def _action_effects(bundle, traces, depth: int, *, draws: int, seed: int, output,
                    heads=None, prior=None, forks=None):
    """TC-16: does the predicted CHANGE track the real one, and does re-labelling the action cost?"""
    roll = _rollouts(bundle, traces, depth, seed=seed)
    if roll is None:
        return _component("insufficient_coverage",
                          {"reason": "no row reached this depth", "failed_checks": 1},
                          _evidence(output, "action_effects", {"reason": "no coverage"}))
    true_effect, predicted = roll["effect_true"], roll["effect_pred"]
    residual = (predicted - true_effect).square().sum()
    total = (true_effect - true_effect.mean(0, keepdim=True)).square().sum()
    cosine = torch.nn.functional.cosine_similarity(predicted, true_effect, dim=-1)
    sensitivity = _paired_mean_interval(roll["deranged"], roll["generated"], roll["clusters"],
                                        draws=draws, seed=seed + 11)
    blind = _paired_mean_interval(roll["marginal"], roll["generated"], roll["clusters"],
                                  draws=draws, seed=seed + 12)
    metrics = {"depth": depth, "rows": int(len(cosine)),
               "scope": "latent effects on DEV windows; state-conditioned consequences are "
                        "measured on held-out all-action forks under `all_action`",
               "effect_r2": float(1 - residual / total.clamp(min=1e-12)),
               "effect_cosine": float(cosine.mean()),
               "action_sensitivity_vs_deranged": sensitivity,
               "vs_action_blind_mean": blind,
               "uses_the_action": bool((sensitivity.get("difference") or 0) > 0
                                       and sensitivity.get("excludes_zero")),
               "beats_action_blind": bool((blind.get("difference") or 0) > 0
                                          and blind.get("excludes_zero"))}
    # Distinguishing actions is not using them well: a positive effect R^2 can come from shared
    # state/time evolution alone, so the action-blind control gates too.
    failed = sum(0 if value else 1 for value in
                 (metrics["uses_the_action"], metrics["beats_action_blind"], metrics["effect_r2"] > 0))
    if forks is not None:
        # The decisive part: within-root decisions over all 17 actions from the same state.
        # Global trivial baselines cannot establish these, so they gate here.
        outcomes = _action_outcomes(bundle, heads, prior, forks, draws=draws,
                                    seed=seed + 300, output=output)
        metrics["all_action"] = outcomes["metrics"]
        failed += int(outcomes["metrics"]["failed_checks"])
    else:
        metrics["all_action"] = {"status": "insufficient_coverage",
                                 "reason": "no all-action fork population supplied"}
        failed += 1
    metrics["failed_checks"] = failed
    return _component("pass" if not failed else "fail", metrics,
                      _evidence(output, "action_effects", metrics),
                      {"quantity": "failed_checks", "value": float(failed),
                       "threshold": 0.5, "direction": "less"})


@torch.no_grad()
def _head_readouts(bundle, heads, traces, depth: int):
    """Head outputs on OBSERVED and on GENERATED states, at the same positions and offsets.

    The earlier version read teacher-forced observed latents only. That is exactly the blind spot
    M03 found: reward, continuation and policy can all work on observed states while the same
    heads fail on generated ones, and aggregate latent MSE does not detect it. Both paths are
    measured here, at identical positions, so the comparison is like for like.
    """
    from .agent import head_targets
    from .data import to_head_batch
    from .train import _bridge_initial_state, _outgoing_actions, bridge_rollout
    keys = ("reward_pred", "reward_true", "reward_mask", "continue_prob", "continue_true",
            "continue_mask", "policy_hit", "policy_action", "policy_mask", "clusters",
            "gen_reward_pred", "gen_continue_prob", "gen_policy_hit", "gen_policy_kl")
    rows = {key: [] for key in keys}
    centers = heads.centers
    for main, _ in traces:
        batch = to_head_batch(main)
        z = batch.latents
        if depth >= z.shape[1]:
            continue
        actions = _outgoing_actions(batch, bundle.n_actions)
        initial = _bridge_initial_state(bundle, main)
        teacher, _, generated_features, anchor = bridge_rollout(bundle, z, actions, depth, initial)
        targets = head_targets(batch, bundle.config)
        suffix = slice(anchor + 1, anchor + 1 + depth)

        def expectation(logits):
            probability = logits.softmax(-1)
            mean = (probability * centers).sum(-1)
            return mean.sign() * torch.expm1(mean.abs())

        observed = heads(teacher.features)
        # The generated readouts cover the suffix; the observed ones are sliced to match exactly.
        produced = heads(generated_features)
        rows["reward_pred"].append(expectation(observed["reward"][:, suffix, 0]).flatten())
        rows["gen_reward_pred"].append(expectation(produced["reward"][:, :, 0]).flatten())
        rows["reward_true"].append(targets["reward"][:, suffix, 0].flatten())
        rows["reward_mask"].append((targets["valid"][:, suffix, 0]
                                    * targets["reward_rows"][:, :, 0]).flatten())
        rows["continue_prob"].append(observed["continuation"][:, suffix, 0].sigmoid().flatten())
        rows["gen_continue_prob"].append(produced["continuation"][..., 0].sigmoid().flatten())
        rows["continue_true"].append(targets["continuation"][:, suffix, 0].flatten())
        rows["continue_mask"].append(targets["continuation_valid"][:, suffix, 0].flatten())
        truth = targets["action"][:, suffix, 0].long()
        rows["policy_hit"].append((observed["policy"][:, suffix, 0].argmax(-1) == truth).float().flatten())
        rows["gen_policy_hit"].append((produced["policy"][:, :, 0].argmax(-1) == truth).float().flatten())
        left = observed["policy"][:, suffix, 0].log_softmax(-1)
        right = produced["policy"][:, :, 0].log_softmax(-1)
        rows["gen_policy_kl"].append((left.exp() * (left - right)).sum(-1).flatten())
        rows["policy_action"].append(truth.flatten())
        rows["policy_mask"].append((targets["action_valid"][:, suffix, 0]
                                    * targets["policy_rows"][:, :, 0]).flatten())
        rows["clusters"].append(_cluster_ids(main.episode_ids, z.device).repeat_interleave(depth))
    if not rows["clusters"]:
        return None
    return {key: torch.cat(value).cpu() for key, value in rows.items()}


def _outcome_calibration(bundle, heads, traces, depth, *, draws: int, seed: int, output):
    """TC-17: reward must beat zero AND marginal; balanced terminal BCE must beat log(2).

    The balanced reference is the spec's: a constant 0.5 predictor scores log(2) on a balanced
    pair, so an always-continue head -- which is otherwise flattered by mostly-alive data -- fails
    the classwise comparison however good its aggregate looks.
    """
    import math
    read = _head_readouts(bundle, heads, traces, depth)
    if read is None:
        return _component("insufficient_coverage", {"reason": "no readouts", "failed_checks": 1},
                          _evidence(output, "outcome_calibration", {"reason": "no coverage"}))
    # Gated on the GENERATED path. Observed numbers are reported beside them, because a gate that
    # only ever sees observed states cannot detect the observed-to-generated transfer failure.
    metrics = {"balanced_reference_bce": math.log(2), "gated_on": "generated states"}
    ok = True

    mask = read["reward_mask"] > 0
    if int(mask.sum()) < 2:
        metrics["reward"] = {"status": "insufficient_coverage"}
        ok = False
    else:
        truth = read["reward_true"][mask]
        predicted, observed_reward = read["gen_reward_pred"][mask], read["reward_pred"][mask]
        clusters = read["clusters"][mask]
        zero = torch.zeros_like(truth)
        marginal = torch.full_like(truth, float(truth.mean()))
        square = lambda x: (x - truth).square()
        vs_zero = _paired_mean_interval(square(zero), square(predicted), clusters,
                                        draws=draws, seed=seed + 1)
        vs_marginal = _paired_mean_interval(square(marginal), square(predicted), clusters,
                                           draws=draws, seed=seed + 2)
        beat = all((c.get("difference") or 0) > 0 and c.get("excludes_zero")
                   for c in (vs_zero, vs_marginal))
        metrics["reward"] = {"generated_mse": float(square(predicted).mean()),
                             "observed_mse": float(square(observed_reward).mean()),
                             "zero_mse": float(square(zero).mean()),
                             "marginal_mse": float(square(marginal).mean()),
                             "vs_zero": vs_zero, "vs_marginal": vs_marginal,
                             "beats_both": bool(beat)}
        ok = ok and beat

    mask = read["continue_mask"] > 0
    truth = read["continue_true"][mask]
    probability = read["gen_continue_prob"][mask].clamp(1e-6, 1 - 1e-6)
    observed_probability = read["continue_prob"][mask].clamp(1e-6, 1 - 1e-6)
    alive, dead = truth > 0.5, truth <= 0.5
    if int(alive.sum()) < 2 or int(dead.sum()) < 2:
        metrics["continuation"] = {"status": "insufficient_coverage",
                                   "alive": int(alive.sum()), "dead": int(dead.sum())}
        ok = False
    else:
        bce = -(truth * probability.log() + (1 - truth) * (1 - probability).log())
        classwise = {"alive": float(bce[alive].mean()), "dead": float(bce[dead].mean())}
        balanced = (classwise["alive"] + classwise["dead"]) / 2
        observed_bce = -(truth * observed_probability.log()
                         + (1 - truth) * (1 - observed_probability).log())
        metrics["continuation"] = {
            "balanced_bce": balanced, "aggregate_bce": float(bce.mean()),
            "observed_balanced_bce": float((observed_bce[alive].mean()
                                            + observed_bce[dead].mean()) / 2),
            "classwise_bce": classwise,
            "brier": float((probability - truth).square().mean()),
            "mean_probability": {"alive": float(probability[alive].mean()),
                                 "dead": float(probability[dead].mean())},
            "beats_balanced_reference": bool(balanced < math.log(2)
                                             and max(classwise.values()) < math.log(2)
                                             and float(bce.mean()) < math.log(2))}
        ok = ok and metrics["continuation"]["beats_balanced_reference"]
    failed = 0 if ok else 1
    metrics["failed_checks"] = failed
    return _component("pass" if ok else "fail", metrics,
                      _evidence(output, "outcome_calibration", metrics),
                      {"quantity": "failed_checks", "value": float(failed),
                       "threshold": 0.5, "direction": "less"})


def _observed_bc(bundle, heads, traces, depth, *, draws: int, seed: int, output):
    """TC-17: observed-path BC on the relevant half, and the same policy on generated states.

    Binding, not merely reported: a BC that cannot beat "always take the most frequent action" has
    not established observed control, and the generated-state agreement and KL are gated too --
    that divergence is the failure M03 localized.
    """
    read = _head_readouts(bundle, heads, traces, depth)
    if read is None:
        return _component("insufficient_coverage", {"reason": "no readouts", "failed_checks": 1},
                          _evidence(output, "observed_bc", {"reason": "no coverage"}))
    mask = read["policy_mask"] > 0
    if int(mask.sum()) < 2:
        return _component("insufficient_coverage",
                          {"relevant_rows": int(mask.sum()), "failed_checks": 1},
                          _evidence(output, "observed_bc", {"relevant_rows": int(mask.sum())}))
    hit, truth, clusters = read["policy_hit"][mask], read["policy_action"][mask], read["clusters"][mask]
    generated_hit = read["gen_policy_hit"][mask]
    counts = torch.bincount(truth.long(), minlength=bundle.n_actions)
    marginal = (truth == int(counts.argmax())).float()
    contrast = _paired_mean_interval(hit, marginal, clusters, draws=draws, seed=seed + 3)
    generated_contrast = _paired_mean_interval(generated_hit, marginal, clusters,
                                               draws=draws, seed=seed + 4)
    transfer = _paired_mean_interval(hit, generated_hit, clusters, draws=draws, seed=seed + 5)
    metrics = {"rows": int(mask.sum()), "top1_agreement": float(hit.mean()),
               "generated_top1_agreement": float(generated_hit.mean()),
               "generated_policy_kl": float(read["gen_policy_kl"][mask].mean()),
               "most_frequent_action_rate": float(marginal.mean()),
               "vs_most_frequent": contrast,
               "generated_vs_most_frequent": generated_contrast,
               "observed_minus_generated": transfer,
               "beats_marginal": bool((contrast.get("difference") or 0) > 0
                                      and contrast.get("excludes_zero")),
               "generated_beats_marginal": bool((generated_contrast.get("difference") or 0) > 0
                                                and generated_contrast.get("excludes_zero"))}
    failed = sum(0 if value else 1 for value in
                 (metrics["beats_marginal"], metrics["generated_beats_marginal"]))
    metrics["failed_checks"] = failed
    return _component("pass" if not failed else "fail", metrics,
                      _evidence(output, "observed_bc", metrics),
                      {"quantity": "failed_checks", "value": float(failed),
                       "threshold": 0.5, "direction": "less"})


GATE_DEPTHS = {"h2": (1, 2), "h16": (1, 2, 4, 8, 16)}


def _source_contract(bundle, payload, cache_contract, checkpoint, output):
    """Identity the loaders already established, recorded as measurements rather than re-derived."""
    from .data import _sha256
    predictor_bn = [m for m in bundle.world.predictor_projector.modules()
                    if isinstance(m, torch.nn.BatchNorm1d)]
    metrics = {"checkpoint_sha256": _sha256(Path(checkpoint)),
               "parent_checkpoint": payload.get("parent", {}).get("checkpoint")
                                    if isinstance(payload.get("parent"), dict) else payload.get("parent"),
               "cache_path": cache_contract.get("path"),
               "cache_manifest_sha256": cache_contract.get("manifest_sha256"),
               "encoder_frozen": bool(getattr(bundle.encoder, "_frozen", False)),
               "predictor_bn_modules": len(predictor_bn),
               "predictor_bn_in_eval": all(not m.training for m in predictor_bn),
               "capabilities": dict(payload.get("capabilities", {}))}
    failed = sum(0 if value else 1 for value in
                 (metrics["encoder_frozen"], metrics["predictor_bn_in_eval"],
                  bool(metrics["cache_manifest_sha256"])))
    metrics["failed_checks"] = failed
    return _component("pass" if not failed else "fail", metrics,
                      _evidence(output, "source_contract", metrics),
                      {"quantity": "failed_checks", "value": float(failed),
                       "threshold": 0.5, "direction": "less"})


def _semantic_retention(bundle, raw_episodes, settings, output):
    """G2 retention: projected `z` must be noninferior to its own CLS within the .03 AUC margin.

    This needs raw frames, not the Phase-2 latent cache, because the cache stores projected `z`
    only and the comparison is precisely projected-versus-CLS. Without the corpus it fails closed.
    """
    from .data import screen_windows
    if raw_episodes is None:
        return _component("insufficient_coverage",
                          {"reason": "retention needs the raw corpus", "failed_checks": 1},
                          _evidence(output, "semantic_retention", {"reason": "no corpus supplied"}))
    device = bundle.config.runtime.device
    windows = {split: screen_windows(raw_episodes, bundle.config, settings, split)
               for split in ("train", "dev")}
    features = {split: screen_features(bundle, windows[split], settings) for split in windows}
    report, _ = screen_retention(features["train"], features["dev"], windows["train"], windows["dev"],
                                 settings, device, bundle.n_actions)
    # `projection_stop` alone FAILS OPEN: it is `all(interval is not None and ...)`, so when no
    # label has support every interval is None, the `all` is False, and "not projection_stop"
    # would read as a pass. Noninferiority has to be positively established, not inferred from an
    # absent measurement, so every probe family must resolve an interval and none may show
    # confident inferiority beyond the margin.
    probes = report["probes"]
    resolved = [row for row in probes.values() if row.get("interval") is not None]
    report["auc_margin"] = settings.auc_margin
    report["resolved_probe_families"] = len(resolved)
    report["probe_families"] = len(probes)
    if len(resolved) != len(probes) or not resolved:
        report["noninferior_to_cls"] = None
        report["failed_checks"] = 1
        return _component("insufficient_coverage", report,
                          _evidence(output, "semantic_retention", report))
    # Noninferiority is a statement about the LOWER bound: the loss we cannot rule out must be
    # smaller than the margin. Testing the upper bound instead asks only "could it be fine?", so
    # an interval like [-0.20, +0.01] -- which permits a 0.20 AUC loss -- read as noninferior.
    bounds = [row["interval"][0] for row in resolved]
    established = all(low > -settings.auc_margin for low in bounds)
    report["lower_bounds"] = bounds
    report["failed_checks"] = sum(1 for low in bounds if low <= -settings.auc_margin)
    report["rule"] = "noninferior iff every lower 95% bound exceeds -auc_margin"
    report["noninferior_to_cls"] = bool(established)
    return _component("pass" if established else "fail", report,
                      _evidence(output, "semantic_retention", report),
                      {"quantity": "min_lower_bound", "value": float(min(bounds)),
                       "threshold": -settings.auc_margin, "direction": "greater"})


def _paired_uncertainty(components, settings, output):
    """Every decision-bearing contrast, with its episode-clustered interval, in one place."""
    gathered = {}
    def walk(prefix, node):
        if isinstance(node, dict):
            if "interval" in node and "difference" in node:
                gathered[prefix] = {k: node.get(k) for k in ("difference", "interval", "excludes_zero")}
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
    for name, component in components.items():
        walk(name, component.get("metrics", {}))
    unresolved = sorted(name for name, row in gathered.items() if not row["interval"])
    metrics = {"draws": settings.bootstrap_draws, "clustering": "whole held-out episode",
               "contrasts": gathered, "resolved": len(gathered) - len(unresolved),
               "unresolved": unresolved,
               "failed_checks": len(unresolved) if gathered else 1,
               "rule": "every decision-bearing contrast must carry an interval; one resolved "
                       "contrast out of many is not paired uncertainty"}
    status = "pass" if gathered and not unresolved else "insufficient_coverage"
    return _component(status, metrics, _evidence(output, "paired_uncertainty", metrics),
                      {"quantity": "unresolved_contrasts",
                       "value": float(len(unresolved) if gathered else 1),
                       "threshold": 0.5, "direction": "less"})


def bridge_gate(bundle, heads, payload, episodes, cache_contract, settings, output, *,
                stage: str, checkpoint, raw_episodes=None, batches: int = 24,
                forks=None, prior=None, panel=None, reference=None):
    """The sealed G2/G3 report that `require_bridge_gate` will accept or refuse.

    Pass rule, from EVALUATION.md G3: permit the longer bridge only when source contracts,
    retention, action-effect improvement over persistence AND marginal controls, and outcome
    calibration beyond the trivial baselines all hold, with paired uncertainty. Every component
    fails closed, so missing support blocks rather than authorizes.
    """
    from .config import recipe_digest
    from .gates import contract_digest
    output = Path(output)
    if any(output.glob('*.json')):
        # Evidence is immutable per attempt: atomic_manifest replaces files, so reusing a
        # directory would silently overwrite a failed attempt's rows with a retry's.
        raise ComponentGateError('gate_output', f'{output} already holds evidence; each attempt writes a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    config = bundle.config
    depths = tuple(d for d in GATE_DEPTHS[stage] if d < config.agent.sequence)
    seed = config.seed + (700 if stage == "h2" else 800)
    traces, ledger = _gate_traces(bundle, episodes, config, batches=batches, seed=seed)

    components = {
        "source_contract": _source_contract(bundle, payload, cache_contract, checkpoint, output),
        "semantic_retention": _critical_retention(bundle, reference, panel, settings, output)
                              if panel is not None else
                              _semantic_retention(bundle, raw_episodes, settings, output),
        "recursive_dynamics": _recursive_dynamics(bundle, traces, depths, draws=settings.bootstrap_draws,
                                                  seed=seed, output=output),
        "action_effects": _action_effects(bundle, traces, max(depths),
                                          draws=settings.bootstrap_draws, seed=seed + 20,
                                          output=output, heads=heads, prior=prior, forks=forks),
        "outcome_calibration": _outcome_calibration(bundle, heads, traces, max(depths),
                                                    draws=settings.bootstrap_draws,
                                                    seed=seed + 40, output=output),
        "observed_bc": _observed_bc(bundle, heads, traces, max(depths),
                                    draws=settings.bootstrap_draws, seed=seed + 60, output=output),
    }
    components["paired_uncertainty"] = _paired_uncertainty(components, settings, output)

    measured = components["recursive_dynamics"]["metrics"]
    validated = 0
    for depth in depths:
        entry = measured.get(f"depth_{depth}", {})
        if entry.get("beats_both_baselines"):
            validated = max(validated, depth)
    body = {"schema": "d4mj_lewm_bridge_gate_v1",
            "checkpoint_sha256": _source_contract_digest(checkpoint),
            "recipe_id": recipe_digest(config),
            "cache_id": contract_digest(cache_contract),
            "stage": stage,
            "decision": "continue_h16" if stage == "h2" else "authorize_actor",
            "validated_recursive_depth": int(validated),
            "evaluated_depths": list(depths),
            "evaluation": {"screen_settings_id": recipe_digest(settings), "batches": batches,
                           "seed": seed, "sample": ledger},
            "not_evaluated": dict(GATE_NOT_EVALUATED),
            "components": components}
    report = _seal(body)
    from .data import atomic_manifest
    atomic_manifest(output / f"bridge_gate_{stage}.json", report)
    return report


def _source_contract_digest(path):
    from .data import _sha256
    return _sha256(Path(path))


@torch.no_grad()
def _actor_diagnostics(bundle, heads, prior, traces, horizon: int):
    """One imagined rollout per starting context, with the quantities G4 asks for."""
    from .actor_critic import lambda_returns
    from .agent import head_targets
    from .imagination import imagine
    from .train import _bridge_initial_state, _outgoing_actions
    from .data import to_head_batch
    values, returns, entropies, priors, chosen, realized = [], [], [], [], [], []
    start_value = []
    policy_rng = torch.Generator(device=bundle.device).manual_seed(bundle.config.seed + 909)
    for main, _ in traces:
        batch = to_head_batch(main)
        actions = _outgoing_actions(batch, bundle.n_actions)
        observed = bundle.world.teacher(batch.latents, actions,
                                        state=_bridge_initial_state(bundle, main))
        trajectory = imagine(bundle, heads, observed.state, observed.features[:, -1:],
                             None, policy_rng, bundle.config)
        target = lambda_returns(trajectory, bundle.config)
        values.append(trajectory.value[:, :-1].flatten().cpu())
        returns.append(target.flatten().cpu())
        # The recorded return from the SAME position the rollout starts at, over the same
        # horizon. The previous version reversed the accumulation and then took [:, -1], which is
        # the last recorded REWARD, and compared B*H imagined values against B recorded ones by
        # truncating to the shorter -- so the two arrays shared neither position nor root. One
        # value per row, paired with that row's own recorded return.
        targets_real = head_targets(batch, bundle.config)
        reward = targets_real["reward"][..., 0]
        alive = targets_real["continuation"][..., 0]
        anchor = reward.shape[1] - 1
        gamma, running = bundle.config.gamma, torch.zeros_like(reward[:, 0])
        for step in range(min(horizon, anchor)):
            index = anchor - step
            running = reward[:, index] + gamma * alive[:, index] * running
        realized.append(running.cpu())
        start_value.append(trajectory.value[:, 0].flatten().cpu())
        distribution = trajectory.logits.softmax(-1)
        entropies.append((-(distribution * distribution.clamp_min(1e-9).log()).sum(-1)).flatten().cpu())
        reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0].softmax(-1)
        priors.append((distribution * (distribution.clamp_min(1e-9).log()
                                       - reference.clamp_min(1e-9).log())).sum(-1).flatten().cpu())
        chosen.append(trajectory.action.flatten().cpu())
    if not values:
        return None
    packed = {k: torch.cat(v) for k, v in
              (("value", values), ("returns", returns), ("entropy", entropies),
               ("kl_to_prior", priors), ("action", chosen))}
    packed["realized_return"] = torch.cat(realized)
    packed["start_value"] = torch.cat(start_value)
    packed["clusters"] = torch.cat([_cluster_ids(main.episode_ids, torch.device("cpu"))
                                    for main, _ in traces])
    return packed


def actor_gate(bundle, heads, prior, payload, episodes, cache_contract, settings, output, *,
               checkpoint, batches: int = 16, forks=None):
    """The sealed G4 screen report: continue to the 5,000 budget, or stop.

    G4 asks whether "the learned environment remains valid and the critic/action diagnostic does
    not reveal exploitation or a broken treatment". Each of those is measured, and each fails
    closed.
    """
    from .config import recipe_digest
    from .data import atomic_manifest
    from .gates import contract_digest
    output = Path(output)
    if any(output.glob('*.json')):
        # Evidence is immutable per attempt: atomic_manifest replaces files, so reusing a
        # directory would silently overwrite a failed attempt's rows with a retry's.
        raise ComponentGateError('gate_output', f'{output} already holds evidence; each attempt writes a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    config = bundle.config
    horizon = config.agent.horizon
    seed = config.seed + 900
    traces, ledger = _gate_traces(bundle, episodes, config, batches=batches, seed=seed)

    frozen = _source_contract(bundle, payload, cache_contract, checkpoint, output)
    roll = _rollouts(bundle, traces, min(horizon, config.agent.sequence - 1), seed=seed)
    if roll is None:
        validity = _component("insufficient_coverage",
                              {"reason": "no row reached the horizon", "failed_checks": 1},
                              _evidence(output, "model_validity", {"reason": "no coverage"}))
    else:
        contrast = _paired_mean_interval(roll["persistence"], roll["generated"], roll["clusters"],
                                         draws=settings.bootstrap_draws, seed=seed + 1)
        metrics = {"horizon": horizon, "generated_mse": float(roll["generated"].mean()),
                   "persistence_mse": float(roll["persistence"].mean()),
                   "vs_persistence": contrast,
                   "frozen_world": frozen["metrics"]["predictor_bn_in_eval"],
                   "still_beats_persistence": bool((contrast.get("difference") or 0) > 0
                                                   and contrast.get("excludes_zero"))}
        failed = sum(0 if v else 1 for v in (metrics["still_beats_persistence"], metrics["frozen_world"]))
        metrics["failed_checks"] = failed
        validity = _component("pass" if not failed else "fail", metrics,
                              _evidence(output, "model_validity", metrics),
                              {"quantity": "failed_checks", "value": float(failed),
                               "threshold": 0.5, "direction": "less"})

    diagnostics = _actor_diagnostics(bundle, heads, prior, traces, horizon)
    if diagnostics is None:
        critic = _component("insufficient_coverage",
                            {"reason": "no imagined rollout", "failed_checks": 1},
                            _evidence(output, "critic_direction", {"reason": "no coverage"}))
        distribution = critic
    else:
        value, target = diagnostics["value"], diagnostics["returns"]
        real, start = diagnostics["realized_return"], diagnostics["start_value"]
        centred = lambda x: x - x.mean()
        def correlate(left, right):
            return float((centred(left) * centred(right)).mean()
                         / (left.std().clamp_min(1e-9) * right.std().clamp_min(1e-9)))
        if len(start) != len(real):
            raise ComponentGateError("critic_direction",
                                     "value and recorded-return rows are not paired")
        against_real = correlate(start, real)
        metrics = {"value_vs_real_return_correlation": against_real,
                   "value_vs_lambda_return_correlation": correlate(value, target),
                   "circularity_note": "lambda returns bootstrap from the critic, so only the "
                                       "correlation against the REAL recorded return is external",
                   "mean_start_value": float(start.mean()),
                   "mean_real_return": float(real.mean()),
                   "value_bias_vs_real": float((start - real).mean()),
                   "rows": int(len(start)),
                   "pairing": "one imagined start value per row against that row's own recorded "
                              "discounted return over the same horizon",
                   "tracks_real_returns": bool(against_real > 0),
                   "failed_checks": 0 if against_real > 0 else 1}
        critic = _component("pass" if metrics["tracks_real_returns"] else "fail", metrics,
                            _evidence(output, "critic_direction", metrics),
                            {"quantity": "value_vs_real_return_correlation",
                             "value": against_real, "threshold": 0.0, "direction": "greater"})
        counts = torch.bincount(diagnostics["action"].long(), minlength=bundle.n_actions).float()
        share = counts / counts.sum().clamp_min(1)
        metrics = {"mean_entropy": float(diagnostics["entropy"].mean()),
                   "uniform_entropy": float(torch.tensor(float(bundle.n_actions)).log()),
                   "mean_kl_to_prior": float(diagnostics["kl_to_prior"].mean()),
                   "actions_used": int((counts > 0).sum()), "n_actions": bundle.n_actions,
                   "max_action_share": float(share.max()),
                   "collapsed": bool(float(share.max()) > 0.95)}
        # Collapse onto a handful of actions is collapse too, so the share threshold is joined by
        # an explicit coverage requirement rather than left at "not 95% one action".
        metrics["min_actions_required"] = max(2, bundle.n_actions // 4)
        metrics["collapsed"] = bool(metrics["max_action_share"] > 0.95
                                    or metrics["actions_used"] < metrics["min_actions_required"])
        metrics["failed_checks"] = 1 if metrics["collapsed"] else 0
        distribution = _component("fail" if metrics["collapsed"] else "pass", metrics,
                                  _evidence(output, "action_distribution", metrics),
                                  {"quantity": "collapsed", "value": 1.0 if metrics["collapsed"] else 0.0,
                                   "threshold": 0.5, "direction": "less"})

    if forks is not None:
        # G4's safety question: does the ACTOR die more than its own immutable BC would, weighted
        # by each policy's own action distribution over real one-step outcomes?
        safety = _action_outcomes(bundle, heads, prior, forks, draws=settings.bootstrap_draws,
                                  seed=seed + 400, output=output)
        weighted = safety["metrics"].get("policy_weighted_true_death", {})
        distribution["metrics"]["policy_weighted_true_death"] = weighted
        if not weighted.get("no_safety_regression", False):
            distribution["metrics"]["failed_checks"] = \
                int(distribution["metrics"].get("failed_checks", 0)) + 1
            distribution["status"] = "fail"
    components = {"model_validity": validity, "critic_direction": critic,
                  "action_distribution": distribution}
    components["paired_uncertainty"] = _paired_uncertainty(components, settings, output)
    body = {"schema": "d4mj_lewm_actor_gate_v1",
            "checkpoint_sha256": _source_contract_digest(checkpoint),
            "recipe_id": recipe_digest(config),
            "cache_id": contract_digest(cache_contract),
            "stage": "actor_screen", "decision": "continue_actor",
            "evaluation": {"screen_settings_id": recipe_digest(settings), "batches": batches,
                           "seed": seed, "sample": ledger},
            "validated_recursive_depth": int(payload.get("capabilities", {})
                                             .get("validated_recursive_depth", 0)),
            "not_evaluated": dict(GATE_NOT_EVALUATED),
            "components": components}
    report = _seal(body)
    atomic_manifest(output / "actor_gate.json", report)
    return report


# --- all-action DEV forks: state-conditioned consequences, not global baselines ---------------
#
# Beating a global trivial baseline permits state-INDEPENDENT action knowledge: observed policy
# agreement 80%, generated 10% and a most-frequent baseline of 5% passes both paths while generated
# transfer has collapsed. The gate therefore needs within-root decisions over all 17 actions from
# the same state, which is what `broad_forks_v2` records. TC-17 keeps fork corpora out of TRAINING;
# EVALUATION.md keeps simulator forks evaluation-only, which is exactly this use. The fork seeds
# (15000-16504) are disjoint from the expert archive (0-319), support-v2 (20270731+) and the sealed
# M03 evaluation seeds, so no root here was ever trained on.

FORK_STORE = ROOT.parent / "artifacts/eda/broad_forks_v2"


def fork_population(config, *, roots: int, seed: int, store=None):
    """Held-out all-action roots: the window, every true successor, and the real outcomes."""
    import glob as _glob
    from .lewm_config import window_layout
    store = Path(store or FORK_STORE)
    offsets = window_layout(config.joint)[0]
    span = offsets[-1] + 1
    if offsets != tuple(range(span)):
        raise ComponentGateError("fork_window", "fork windows assume a consecutive encoded window")
    paths = sorted(_glob.glob(str(store / "seed-*.pt")))
    if not paths:
        raise ComponentGateError("fork_corpus", f"no all-action fork roots under {store}")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(paths), generator=generator).tolist()
    rows = []
    for index in order:
        for row in torch.load(paths[index], map_location="cpu", weights_only=False):
            if len(row["frames"]) < span:
                continue
            rows.append(row)
            if len(rows) >= roots:
                break
        if len(rows) >= roots:
            break
    pack = lambda key: torch.stack([row[key] for row in rows])
    return {"frames": torch.stack([row["frames"][-span:] for row in rows]),
            "past_actions": torch.stack([row["led_to_action"][-span + 1:] for row in rows]),
            "successors": pack("successors"), "reward": pack("reward").float(),
            "terminated": pack("terminated").bool(), "health_delta": pack("health_delta").float(),
            "achievement_delta": pack("achievement_delta").long(),
            "bc_action": torch.tensor([int(row["bc_action"]) for row in rows]),
            "seed": torch.tensor([int(row["seed"]) for row in rows]),
            "roots": len(rows)}


@torch.no_grad()
def _fork_readouts(bundle, heads, forks, *, batch: int = 8):
    """Generated and TRUE-SUCCESSOR head readouts for all 17 actions from each root."""
    device, count = bundle.device, bundle.n_actions
    out = {key: [] for key in ("generated_reward", "generated_death", "true_reward", "true_death",
                               "policy", "prior_policy", "effect_generated", "effect_true")}
    centers = heads.centers
    for start in range(0, forks["roots"], batch):
        stop = min(forks["roots"], start + batch)
        rows = stop - start
        frames = forks["frames"][start:stop].to(device)
        past = forks["past_actions"][start:stop].to(device)
        successors = forks["successors"][start:stop].to(device)
        z = bundle.encoder(frames)
        state = bundle.world.teacher(z, past).state
        actions = torch.arange(count, device=device).repeat(rows)[:, None]
        fan = bundle.repeat_state(state, count)
        advanced, features = bundle.advance(fan, actions)

        def read(head_features):
            readout = heads(head_features)
            probability = readout["reward"][:, -1, 0].softmax(-1)
            mean = (probability * centers).sum(-1)
            reward = mean.sign() * torch.expm1(mean.abs())
            death = 1.0 - readout["continuation"][:, -1, 0].sigmoid()
            return reward.reshape(rows, count), death.reshape(rows, count), readout

        generated_reward, generated_death, readout = read(features)
        # The TRUE-SUCCESSOR substitution: the same heads reading the real next state, which
        # separates a transition error from an outcome-head error.
        z_true = bundle.encoder(successors.flatten(0, 1).unsqueeze(1))
        true_state, true_features = bundle.world.observe_latent(fan, actions, z_true)
        true_reward, true_death, _ = read(true_features)
        out["generated_reward"].append(generated_reward.cpu())
        out["generated_death"].append(generated_death.cpu())
        out["true_reward"].append(true_reward.cpu())
        out["true_death"].append(true_death.cpu())
        out["policy"].append(heads(bundle.world.features(state))["policy"][:, -1, 0]
                             .softmax(-1).cpu())
        anchor = z[:, -1:, 0]
        out["effect_generated"].append((advanced.latent[:, 0, 0].reshape(rows, count, -1)
                                        - anchor).cpu())
        out["effect_true"].append((z_true[:, 0, 0].reshape(rows, count, -1) - anchor).cpu())
    return {key: torch.cat(value) for key, value in out.items() if value}


def _regret(truth, score, *, maximize: bool):
    """Within-root decision cost: what the chosen action gave up against the best available."""
    chosen = (score.argmax(1) if maximize else score.argmin(1))[:, None]
    best = truth.amax(1, keepdim=True) if maximize else truth.amin(1, keepdim=True)
    return (best - truth.gather(1, chosen)).squeeze(1).abs()


def _action_outcomes(bundle, heads, prior, forks, *, draws: int, seed: int, output):
    """G3: state-conditioned action consequences, decided WITHIN each root over all 17 actions.

    Every contrast is within-root, so a model that has learned only which actions are good on
    average cannot pass: the action-marginal control makes exactly that choice and is the
    reference. Terminal ranking, reward regret and policy-weighted death are reported against
    both the generated path and the true-successor substitution, which separates a transition
    error from an outcome-head error.
    """
    from .diagnostics import binary_auc
    read = _fork_readouts(bundle, heads, forks)
    true_reward, true_death = forks["reward"], forks["terminated"].float()
    clusters = forks["seed"]
    reward_varies = true_reward.amax(1) > true_reward.amin(1)
    death_varies = true_death.amax(1) > true_death.amin(1)
    metrics = {"roots": forks["roots"], "actions": bundle.n_actions,
               "reward_opportunity_roots": int(reward_varies.sum()),
               "terminal_opportunity_roots": int(death_varies.sum()),
               "minimum_opportunity_roots": settings_minimum(draws)}
    failed = 0

    # --- reward: within-root regret against the action-marginal choice -----------------------
    if int(reward_varies.sum()) < metrics["minimum_opportunity_roots"]:
        metrics["reward"] = {"status": "insufficient_coverage"}
        failed += 1
    else:
        truth = true_reward[reward_varies]
        marginal = truth.mean(0, keepdim=True).expand_as(truth)
        model = _regret(truth, read["generated_reward"][reward_varies], maximize=True)
        blind = _regret(truth, marginal, maximize=True)
        oracle = _regret(truth, read["true_reward"][reward_varies], maximize=True)
        contrast = _paired_mean_interval(blind, model, clusters[reward_varies],
                                         draws=draws, seed=seed + 1)
        metrics["reward"] = {"generated_regret": float(model.mean()),
                             "action_marginal_regret": float(blind.mean()),
                             "true_successor_regret": float(oracle.mean()),
                             "vs_action_marginal": contrast,
                             "beats_marginal": bool((contrast.get("difference") or 0) > 0
                                                    and contrast.get("excludes_zero"))}
        failed += 0 if metrics["reward"]["beats_marginal"] else 1

    # --- terminal: within-root safe choice and ranking ---------------------------------------
    if int(death_varies.sum()) < metrics["minimum_opportunity_roots"]:
        metrics["terminal"] = {"status": "insufficient_coverage"}
        failed += 1
    else:
        truth = true_death[death_varies]
        marginal = truth.mean(0, keepdim=True).expand_as(truth)
        safe = 1.0 - _regret(truth, read["generated_death"][death_varies], maximize=False)
        blind_safe = 1.0 - _regret(truth, marginal, maximize=False)
        oracle_safe = 1.0 - _regret(truth, read["true_death"][death_varies], maximize=False)
        contrast = _paired_mean_interval(safe, blind_safe, clusters[death_varies],
                                         draws=draws, seed=seed + 2)
        flat_truth = truth.flatten().bool()
        auc = binary_auc(read["generated_death"][death_varies].flatten(), flat_truth)
        metrics["terminal"] = {"generated_safe_choice": float(safe.mean()),
                               "action_marginal_safe_choice": float(blind_safe.mean()),
                               "true_successor_safe_choice": float(oracle_safe.mean()),
                               "generated_death_auc": auc,
                               "vs_action_marginal": contrast,
                               "beats_marginal": bool((contrast.get("difference") or 0) > 0
                                                      and contrast.get("excludes_zero"))}
        failed += 0 if metrics["terminal"]["beats_marginal"] else 1

    # --- policy-weighted true death: the actor against its own immutable BC ------------------
    with torch.no_grad():
        policy = read["policy"]
        reference = prior_policy(bundle, heads, prior, forks) if prior is not None else None
    weighted = (policy * true_death).sum(1)
    metrics["policy_weighted_true_death"] = {"actor": float(weighted.mean())}
    if reference is not None:
        bc_weighted = (reference * true_death).sum(1)
        contrast = _paired_mean_interval(bc_weighted, weighted, clusters, draws=draws, seed=seed + 3)
        metrics["policy_weighted_true_death"].update({
            "bc": float(bc_weighted.mean()), "bc_minus_actor": contrast,
            "no_safety_regression": bool((contrast.get("difference") or 0) >= 0
                                         or not contrast.get("excludes_zero"))})
        failed += 0 if metrics["policy_weighted_true_death"]["no_safety_regression"] else 1

    # --- effect-equivalence: actions whose real successors coincide -------------------------
    true_effect, generated_effect = read["effect_true"], read["effect_generated"]
    pairwise = torch.cdist(true_effect, true_effect)
    equivalent = pairwise < pairwise.amax(dim=(1, 2), keepdim=True).clamp_min(1e-9) * 0.02
    off = ~torch.eye(bundle.n_actions, dtype=torch.bool).expand_as(equivalent)
    share = float((equivalent & off).float().mean())
    generated_gap = torch.cdist(generated_effect, generated_effect)
    metrics["effect_equivalence"] = {
        "equivalent_pair_share": share,
        "mean_generated_gap_on_equivalent_pairs":
            float(generated_gap[equivalent & off].mean()) if bool((equivalent & off).any()) else None,
        "mean_generated_gap_overall": float(generated_gap[off].mean()),
        "note": "actions with coincident real successors should not be driven apart"}
    metrics["failed_checks"] = failed
    metrics["raw_rows"] = _raw_rows(output, "action_outcomes", {
        "seed": clusters, "bc_action": forks["bc_action"],
        "true_reward": true_reward, "true_terminated": forks["terminated"],
        "health_delta": forks["health_delta"], "achievement_delta": forks["achievement_delta"],
        "generated_reward": read["generated_reward"], "generated_death": read["generated_death"],
        "true_successor_reward": read["true_reward"], "true_successor_death": read["true_death"],
        "policy": read["policy"]})
    return _component("pass" if not failed else "fail", metrics,
                      _evidence(output, "action_outcomes", metrics))


def settings_minimum(draws: int) -> int:
    """Opportunity floor: a decision measured on a handful of roots is not a measurement."""
    return 24


@torch.no_grad()
def prior_policy(bundle, heads, prior, forks, *, batch: int = 8):
    """The immutable BC prior's action distribution at the same roots."""
    device = bundle.device
    out = []
    for start in range(0, forks["roots"], batch):
        stop = min(forks["roots"], start + batch)
        z = bundle.encoder(forks["frames"][start:stop].to(device))
        state = bundle.world.teacher(z, forks["past_actions"][start:stop].to(device)).state
        out.append(prior(bundle.world.features(state))["policy"][:, -1, 0].softmax(-1).cpu())
    return torch.cat(out)


# --- critical retention panel ------------------------------------------------------------------
#
# The four labels the joint screen carries (reward sign, event, termination) are not the critical
# suite. G2 asks for health, inventory/resources, local tiles and action prerequisites, against
# CLS *and* a preselected reference. The 2026-09-18 coverage audit already located and verified
# 129 exact addresses with zero pixel mismatch; the labels it recorded are reused, while every
# representation is re-encoded from the new checkpoints. Prior scores are never reused as results.

ADDRESS_BOOK = (ROOT.parent /
                "artifacts/experiments/20260918_m03_probe_coverage_audit/evidence/"
                "verified_root_index.json")
SUPPORT_STORE = ROOT.parent / "artifacts/craftax_support_v2"


def retention_addresses(path=None, store=None):
    """Verified (frame, labels, split, episode) rows for the critical suite."""
    import json as _json
    from .m03.gate import STATIC_BINARY
    path = Path(path or ADDRESS_BOOK)
    store = Path(store or SUPPORT_STORE)
    if not path.is_file():
        raise ComponentGateError("retention_panel", f"no verified address book at {path}")
    book = _json.loads(path.read_text())
    manifest = _json.loads((store / "manifest.json").read_text())
    frames, labels, splits, episodes, cache = [], [], [], [], {}
    for row in book:
        shard = row["shard"]
        if shard not in cache:
            cache[shard] = torch.load(store / manifest["shards"][shard]["file"],
                                      weights_only=False, mmap=True)
        episode = cache[shard]["episodes"][row["slot"]]
        if episode["split"] != row["split"]:
            raise ComponentGateError("retention_panel", "address split disagrees with the store")
        frames.append(episode["observations"][row["t"]].clone())
        positive = set(row["positive_labels"])
        labels.append(torch.tensor([name in positive for name in STATIC_BINARY]))
        splits.append(row["split"])
        episodes.append(row["episode_id"])
    return {"frames": torch.stack(frames), "labels": torch.stack(labels),
            "split": splits, "episode_id": episodes, "names": list(STATIC_BINARY)}


@torch.no_grad()
def _encode_panel(bundle, frames, *, batch: int = 32):
    """Projected z and unprojected CLS for one encoder, on identical frames."""
    projected, cls = [], []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].unsqueeze(1).to(bundle.device)
        z, c = bundle.encoder.projected_and_cls(chunk)
        projected.append(z[:, 0, 0].float().cpu())
        cls.append(c[:, 0].float().cpu())
    return torch.cat(projected), torch.cat(cls)


def _critical_retention(bundle, reference, panel, settings, output):
    """G2: projected `z` noninferior to its own CLS AND to the preselected reference.

    Noninferiority is a LOWER-bound statement on every probe family, over the critical labels that
    have real support. A label without support is recorded and excluded rather than counted as a
    pass, and a panel with too few supported labels fails closed.
    """
    from .diagnostics import binary_auc, fit_outcome_probe, paired_auc_interval
    if panel is None:
        return _component("insufficient_coverage",
                          {"reason": "no critical retention panel supplied", "failed_checks": 1},
                          _evidence(output, "semantic_retention", {"reason": "no panel"}))
    labels, names = panel["labels"], panel["names"]
    is_train = torch.tensor([s == "train" for s in panel["split"]])
    is_dev = torch.tensor([s == "dev" for s in panel["split"]])
    support = []
    for column, name in enumerate(names):
        counts = {split: {"positive": int((labels[mask, column]).sum()),
                          "negative": int((~labels[mask, column]).sum())}
                  for split, mask in (("train", is_train), ("dev", is_dev))}
        ok = all(v["positive"] >= settings.minimum_positive // 2
                 and v["negative"] >= settings.minimum_negative // 2 for v in counts.values())
        support.append({"label": name, "counts": counts, "supported": ok})
    selected = torch.tensor([row["supported"] for row in support])
    report = {"panel": "verified critical addresses, re-encoded from this checkpoint",
              "labels": len(names), "supported_labels": int(selected.sum()),
              "coverage": support, "auc_margin": settings.auc_margin,
              "probes": {}, "macro_auc": {}}
    if int(selected.sum()) < 8:
        report["failed_checks"] = 1
        report["reason"] = "too few critical labels have support on this panel"
        return _component("insufficient_coverage", report,
                          _evidence(output, "semantic_retention", report))

    features = {}
    z, cls = _encode_panel(bundle, panel["frames"])
    features["projected"], features["cls"] = z, cls
    if reference is not None:
        features["reference"], _ = _encode_panel(reference, panel["frames"])
    truth_train = labels[is_train][:, selected].float()
    truth_dev = labels[is_dev][:, selected]
    valid_train = torch.ones_like(truth_train)
    valid_dev = torch.ones_like(truth_dev, dtype=torch.bool)
    clusters = torch.tensor([abs(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],
                                                "little")) for name in panel["episode_id"]])[is_dev]
    device = bundle.config.runtime.device
    failed = 0
    for hidden, family in ((False, "linear"), (True, "mlp")):
        scores = {}
        for name, value in features.items():
            scores[name] = fit_outcome_probe(value[is_train].to(device), truth_train.to(device),
                                             valid_train.to(device), value[is_dev].to(device),
                                             settings, hidden=hidden)
        report["macro_auc"][family] = {
            name: float(np.mean([a for a in
                                 (binary_auc(s[:, i], truth_dev[:, i]) for i in range(truth_dev.shape[1]))
                                 if a is not None]))
            for name, s in scores.items()}
        comparisons = {}
        for against in ("cls", "reference"):
            if against not in scores:
                continue
            comparisons[against] = paired_auc_interval(
                scores["projected"], scores[against], truth_dev, valid_dev, clusters,
                draws=settings.bootstrap_draws, seed=settings.seed + 900)
        report["probes"][family] = comparisons
        for against, row in comparisons.items():
            low = (row.get("interval") or [None])[0]
            if low is None or low <= -settings.auc_margin:
                failed += 1
    report["failed_checks"] = failed
    report["rule"] = ("projected z must be noninferior to CLS and to the preselected reference: "
                      "every probe family's lower 95% bound above -auc_margin")
    report["raw_rows"] = _raw_rows(output, "semantic_retention", {
        "labels": labels, "supported": selected, "split": [s for s in panel["split"]],
        "episode_id": list(panel["episode_id"]), "names": names,
        **{f"features_{name}": value for name, value in features.items()}})
    return _component("pass" if not failed else "fail", report,
                      _evidence(output, "semantic_retention", report),
                      {"quantity": "failed_checks", "value": float(failed),
                       "threshold": 0.5, "direction": "less"})
