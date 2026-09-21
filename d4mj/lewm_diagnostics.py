"""Component-scoped M0-M3 gates. Failures never imply a verdict on the architecture."""

import copy
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


def _component(status: str, metrics: dict, evidence: list) -> dict:
    return {"status": status, "metrics": metrics, "evidence": evidence}


def _evidence(output, name: str, payload: dict) -> list:
    """Write one immutable evidence file and bind it by bytes."""
    from .data import _sha256, atomic_manifest
    path = Path(output) / f"{name}.json"
    atomic_manifest(path, payload)
    return [{"path": str(path), "sha256": _sha256(path)}]


def _seal(body: dict) -> dict:
    from .gates import contract_digest
    return dict(body, report_id=contract_digest(body))


def _gate_traces(bundle, episodes, config, *, batches: int, seed: int):
    """A fixed held-out sample drawn exactly as Phase 2 draws, reused by every component."""
    from .data import sample_bridge_batch, sample_bridge_terminals
    rng = torch.Generator().manual_seed(seed)
    traces = []
    for update in range(batches):
        traces.append((sample_bridge_batch(episodes, rng, config, update).to(bundle.device),
                       sample_bridge_terminals(episodes, rng, config, update).to(bundle.device)))
    return traces


def _derangement(count: int, generator) -> torch.Tensor:
    """A permutation with no fixed point, so every row really is re-labelled."""
    while True:
        order = torch.randperm(count, generator=generator)
        if count < 2 or bool((order != torch.arange(count)).all()):
            return order


@torch.no_grad()
def _rollouts(bundle, traces, depth: int, *, seed: int):
    """Generated, persistence and marginal-action rollouts on one shared sample.

    The marginal-action baseline re-labels each row's actions with a fixed derangement: it asks
    what the rollout predicts when it is told the wrong action, which is the prediction available
    to a world that ignores the action. The same quantity is the action-sensitivity test, so it is
    computed once and read twice rather than approximated separately.
    """
    from .train import _bridge_initial_state, _outgoing_actions, bridge_rollout
    generator = torch.Generator().manual_seed(seed)
    rows = {"generated": [], "persistence": [], "marginal": [], "clusters": [],
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
        truth = z[:, anchor + 1:anchor + 1 + depth].float()
        base = z[:, anchor:anchor + 1].float().expand_as(truth)
        per_row = lambda x: (x.float() - truth).square().mean(dim=tuple(range(1, truth.ndim)))
        rows["generated"].append(per_row(generated))
        rows["marginal"].append(per_row(deranged))
        rows["persistence"].append(per_row(base))
        rows["effect_true"].append((truth - base).flatten(1))
        rows["effect_pred"].append((generated.float() - base).flatten(1))
        rows["clusters"].append(torch.tensor(
            [abs(hash(name)) % (2**31) for name in main.episode_ids], device=z.device))
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
        beat = (vs_persistence.get("difference") or 0) > 0 and vs_persistence.get("excludes_zero") \
            and (vs_marginal.get("difference") or 0) > 0 and vs_marginal.get("excludes_zero")
        measured[f"depth_{depth}"] = {
            "rows": int(len(roll["generated"])),
            "generated_mse": float(roll["generated"].mean()),
            "persistence_mse": float(roll["persistence"].mean()),
            "marginal_action_mse": float(roll["marginal"].mean()),
            "vs_persistence": vs_persistence, "vs_marginal_action": vs_marginal,
            "beats_both_baselines": bool(beat)}
        passing = passing and bool(beat)
    return _component("pass" if passing else "fail", measured,
                      _evidence(output, "recursive_dynamics", measured))


def _action_effects(bundle, traces, depth: int, *, draws: int, seed: int, output):
    """TC-16: does the predicted CHANGE track the real one, and does re-labelling the action cost?"""
    roll = _rollouts(bundle, traces, depth, seed=seed)
    if roll is None:
        return _component("insufficient_coverage", {"reason": "no row reached this depth"},
                          _evidence(output, "action_effects", {"reason": "no coverage"}))
    true_effect, predicted = roll["effect_true"], roll["effect_pred"]
    residual = (predicted - true_effect).square().sum()
    total = (true_effect - true_effect.mean(0, keepdim=True)).square().sum()
    cosine = torch.nn.functional.cosine_similarity(predicted, true_effect, dim=-1)
    sensitivity = _paired_mean_interval(roll["marginal"], roll["generated"], roll["clusters"],
                                        draws=draws, seed=seed + 11)
    metrics = {"depth": depth, "rows": int(len(cosine)),
               "effect_r2": float(1 - residual / total.clamp(min=1e-12)),
               "effect_cosine": float(cosine.mean()),
               "action_sensitivity": sensitivity,
               "uses_the_action": bool((sensitivity.get("difference") or 0) > 0
                                       and sensitivity.get("excludes_zero"))}
    status = "pass" if metrics["uses_the_action"] and metrics["effect_r2"] > 0 else "fail"
    return _component(status, metrics, _evidence(output, "action_effects", metrics))


@torch.no_grad()
def _head_readouts(bundle, heads, traces):
    """Observed-path head outputs and their targets, on the shared held-out sample."""
    from .agent import head_targets
    from .data import to_head_batch
    from .train import _bridge_initial_state, _outgoing_actions
    rows = {"reward_pred": [], "reward_true": [], "reward_mask": [],
            "continue_prob": [], "continue_true": [], "continue_mask": [],
            "policy_hit": [], "policy_action": [], "policy_mask": [], "clusters": []}
    centers = heads.centers
    for main, _ in traces:
        batch = to_head_batch(main)
        actions = _outgoing_actions(batch, bundle.n_actions)
        teacher = bundle.world.teacher(batch.latents, actions,
                                       state=_bridge_initial_state(bundle, main))
        read = heads(teacher.features)
        targets = head_targets(batch, bundle.config)
        probabilities = read["reward"][:, :, 0].softmax(-1)
        mean = (probabilities * centers).sum(-1)
        rows["reward_pred"].append((mean.sign() * torch.expm1(mean.abs())).flatten())
        rows["reward_true"].append(targets["reward"][..., 0].flatten())
        rows["reward_mask"].append((targets["valid"][..., 0] * targets["reward_rows"][:, :, 0]).flatten())
        rows["continue_prob"].append(read["continuation"][..., 0].sigmoid().flatten())
        rows["continue_true"].append(targets["continuation"][..., 0].flatten())
        rows["continue_mask"].append(targets["continuation_valid"][..., 0].flatten())
        choice = read["policy"][:, :, 0].argmax(-1)
        truth = targets["action"][..., 0].long()
        rows["policy_hit"].append((choice == truth).float().flatten())
        rows["policy_action"].append(truth.flatten())
        rows["policy_mask"].append((targets["action_valid"][..., 0] * targets["policy_rows"][:, :, 0]).flatten())
        blocks = batch.latents.shape[1]
        rows["clusters"].append(torch.tensor(
            [abs(hash(name)) % (2**31) for name in main.episode_ids],
            device=batch.latents.device).repeat_interleave(blocks))
    if not rows["clusters"]:
        return None
    return {key: torch.cat(value).cpu() for key, value in rows.items()}


def _outcome_calibration(bundle, heads, traces, *, draws: int, seed: int, output):
    """TC-17: reward must beat zero AND marginal; balanced terminal BCE must beat log(2).

    The balanced reference is the spec's: a constant 0.5 predictor scores log(2) on a balanced
    pair, so an always-continue head -- which is otherwise flattered by mostly-alive data -- fails
    the classwise comparison however good its aggregate looks.
    """
    import math
    read = _head_readouts(bundle, heads, traces)
    if read is None:
        return _component("insufficient_coverage", {"reason": "no readouts"},
                          _evidence(output, "outcome_calibration", {"reason": "no coverage"}))
    metrics = {"balanced_reference_bce": math.log(2)}
    ok = True

    mask = read["reward_mask"] > 0
    if int(mask.sum()) < 2:
        metrics["reward"] = {"status": "insufficient_coverage"}
        ok = False
    else:
        truth, predicted = read["reward_true"][mask], read["reward_pred"][mask]
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
        metrics["reward"] = {"mse": float(square(predicted).mean()),
                             "zero_mse": float(square(zero).mean()),
                             "marginal_mse": float(square(marginal).mean()),
                             "vs_zero": vs_zero, "vs_marginal": vs_marginal,
                             "beats_both": bool(beat)}
        ok = ok and beat

    mask = read["continue_mask"] > 0
    truth, probability = read["continue_true"][mask], read["continue_prob"][mask].clamp(1e-6, 1 - 1e-6)
    alive, dead = truth > 0.5, truth <= 0.5
    if int(alive.sum()) < 2 or int(dead.sum()) < 2:
        metrics["continuation"] = {"status": "insufficient_coverage",
                                   "alive": int(alive.sum()), "dead": int(dead.sum())}
        ok = False
    else:
        bce = -(truth * probability.log() + (1 - truth) * (1 - probability).log())
        classwise = {"alive": float(bce[alive].mean()), "dead": float(bce[dead].mean())}
        balanced = (classwise["alive"] + classwise["dead"]) / 2
        metrics["continuation"] = {
            "balanced_bce": balanced, "aggregate_bce": float(bce.mean()),
            "classwise_bce": classwise,
            "brier": float((probability - truth).square().mean()),
            "mean_probability": {"alive": float(probability[alive].mean()),
                                 "dead": float(probability[dead].mean())},
            "beats_balanced_reference": bool(balanced < math.log(2)
                                             and max(classwise.values()) < math.log(2)
                                             and float(bce.mean()) < math.log(2))}
        ok = ok and metrics["continuation"]["beats_balanced_reference"]
    return _component("pass" if ok else "fail", metrics,
                      _evidence(output, "outcome_calibration", metrics))


def _observed_bc(bundle, heads, traces, *, draws: int, seed: int, output):
    """TC-17: observed-path BC on the relevant half, with its own complete recurrent state.

    Reported rather than margin-gated here: EVALUATION.md puts the `.5`-achievement noninferiority
    comparison in G4, against a real-game BC, not at the bridge.
    """
    read = _head_readouts(bundle, heads, traces)
    if read is None:
        return _component("insufficient_coverage", {"reason": "no readouts"},
                          _evidence(output, "observed_bc", {"reason": "no coverage"}))
    mask = read["policy_mask"] > 0
    if int(mask.sum()) < 2:
        return _component("insufficient_coverage", {"relevant_rows": int(mask.sum())},
                          _evidence(output, "observed_bc", {"relevant_rows": int(mask.sum())}))
    hit, truth, clusters = read["policy_hit"][mask], read["policy_action"][mask], read["clusters"][mask]
    counts = torch.bincount(truth.long(), minlength=bundle.n_actions)
    marginal = (truth == int(counts.argmax())).float()
    contrast = _paired_mean_interval(hit, marginal, clusters, draws=draws, seed=seed + 3)
    metrics = {"rows": int(mask.sum()), "top1_agreement": float(hit.mean()),
               "most_frequent_action_rate": float(marginal.mean()),
               "vs_most_frequent": contrast,
               "beats_marginal": bool((contrast.get("difference") or 0) > 0
                                      and contrast.get("excludes_zero"))}
    return _component("pass", metrics, _evidence(output, "observed_bc", metrics))


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
    ok = metrics["encoder_frozen"] and metrics["predictor_bn_in_eval"] and bool(metrics["cache_manifest_sha256"])
    return _component("pass" if ok else "fail", metrics, _evidence(output, "source_contract", metrics))


def _semantic_retention(bundle, raw_episodes, settings, output):
    """G2 retention: projected `z` must be noninferior to its own CLS within the .03 AUC margin.

    This needs raw frames, not the Phase-2 latent cache, because the cache stores projected `z`
    only and the comparison is precisely projected-versus-CLS. Without the corpus it fails closed.
    """
    from .data import screen_windows
    if raw_episodes is None:
        return _component("insufficient_coverage", {"reason": "retention needs the raw corpus"},
                          _evidence(output, "semantic_retention", {"reason": "no corpus supplied"}))
    device = bundle.config.runtime.device
    windows = {split: screen_windows(raw_episodes, bundle.config, settings, split)
               for split in ("train", "dev")}
    features = {split: screen_features(bundle, windows[split], settings) for split in windows}
    report, _ = screen_retention(features["train"], features["dev"], windows["train"], windows["dev"],
                                 settings, device, bundle.n_actions)
    # `projection_stop` is True exactly when projected is inferior to CLS beyond the margin.
    ok = not report["projection_stop"]
    report["auc_margin"] = settings.auc_margin
    report["noninferior_to_cls"] = bool(ok)
    return _component("pass" if ok else "fail", report, _evidence(output, "semantic_retention", report))


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
    metrics = {"draws": settings.bootstrap_draws, "clustering": "whole held-out episode",
               "contrasts": gathered, "resolved": sum(1 for v in gathered.values() if v["interval"])}
    status = "pass" if metrics["resolved"] else "insufficient_coverage"
    return _component(status, metrics, _evidence(output, "paired_uncertainty", metrics))


def bridge_gate(bundle, heads, payload, episodes, cache_contract, settings, output, *,
                stage: str, checkpoint, raw_episodes=None, batches: int = 24):
    """The sealed G2/G3 report that `require_bridge_gate` will accept or refuse.

    Pass rule, from EVALUATION.md G3: permit the longer bridge only when source contracts,
    retention, action-effect improvement over persistence AND marginal controls, and outcome
    calibration beyond the trivial baselines all hold, with paired uncertainty. Every component
    fails closed, so missing support blocks rather than authorizes.
    """
    from .config import recipe_digest
    from .gates import contract_digest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = bundle.config
    depths = tuple(d for d in GATE_DEPTHS[stage] if d < config.agent.sequence)
    seed = config.seed + (700 if stage == "h2" else 800)
    traces = _gate_traces(bundle, episodes, config, batches=batches, seed=seed)

    components = {
        "source_contract": _source_contract(bundle, payload, cache_contract, checkpoint, output),
        "semantic_retention": _semantic_retention(bundle, raw_episodes, settings, output),
        "recursive_dynamics": _recursive_dynamics(bundle, traces, depths, draws=settings.bootstrap_draws,
                                                  seed=seed, output=output),
        "action_effects": _action_effects(bundle, traces, max(depths), draws=settings.bootstrap_draws,
                                          seed=seed + 20, output=output),
        "outcome_calibration": _outcome_calibration(bundle, heads, traces, draws=settings.bootstrap_draws,
                                                    seed=seed + 40, output=output),
        "observed_bc": _observed_bc(bundle, heads, traces, draws=settings.bootstrap_draws,
                                    seed=seed + 60, output=output),
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
    from .imagination import imagine
    from .train import _bridge_initial_state, _outgoing_actions
    from .data import to_head_batch
    values, returns, entropies, priors, chosen = [], [], [], [], []
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
        distribution = trajectory.logits.softmax(-1)
        entropies.append((-(distribution * distribution.clamp_min(1e-9).log()).sum(-1)).flatten().cpu())
        reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0].softmax(-1)
        priors.append((distribution * (distribution.clamp_min(1e-9).log()
                                       - reference.clamp_min(1e-9).log())).sum(-1).flatten().cpu())
        chosen.append(trajectory.action.flatten().cpu())
    if not values:
        return None
    return {k: torch.cat(v) for k, v in
            (("value", values), ("returns", returns), ("entropy", entropies),
             ("kl_to_prior", priors), ("action", chosen))}


def actor_gate(bundle, heads, prior, payload, episodes, cache_contract, settings, output, *,
               checkpoint, batches: int = 16):
    """The sealed G4 screen report: continue to the 5,000 budget, or stop.

    G4 asks whether "the learned environment remains valid and the critic/action diagnostic does
    not reveal exploitation or a broken treatment". Each of those is measured, and each fails
    closed.
    """
    from .config import recipe_digest
    from .data import atomic_manifest
    from .gates import contract_digest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = bundle.config
    horizon = config.agent.horizon
    seed = config.seed + 900
    traces = _gate_traces(bundle, episodes, config, batches=batches, seed=seed)

    frozen = _source_contract(bundle, payload, cache_contract, checkpoint, output)
    roll = _rollouts(bundle, traces, min(horizon, config.agent.sequence - 1), seed=seed)
    if roll is None:
        validity = _component("insufficient_coverage", {"reason": "no row reached the horizon"},
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
        validity = _component("pass" if metrics["still_beats_persistence"] and metrics["frozen_world"]
                              else "fail", metrics, _evidence(output, "model_validity", metrics))

    diagnostics = _actor_diagnostics(bundle, heads, prior, traces, horizon)
    if diagnostics is None:
        critic = _component("insufficient_coverage", {"reason": "no imagined rollout"},
                            _evidence(output, "critic_direction", {"reason": "no coverage"}))
        distribution = critic
    else:
        value, target = diagnostics["value"], diagnostics["returns"]
        centred = lambda x: x - x.mean()
        correlation = float((centred(value) * centred(target)).mean()
                            / (value.std().clamp_min(1e-9) * target.std().clamp_min(1e-9)))
        metrics = {"value_return_correlation": correlation,
                   "mean_value": float(value.mean()), "mean_return": float(target.mean()),
                   "value_bias": float((value - target).mean()),
                   "tracks_returns": bool(correlation > 0)}
        critic = _component("pass" if metrics["tracks_returns"] else "fail", metrics,
                            _evidence(output, "critic_direction", metrics))
        counts = torch.bincount(diagnostics["action"].long(), minlength=bundle.n_actions).float()
        share = counts / counts.sum().clamp_min(1)
        metrics = {"mean_entropy": float(diagnostics["entropy"].mean()),
                   "uniform_entropy": float(torch.tensor(float(bundle.n_actions)).log()),
                   "mean_kl_to_prior": float(diagnostics["kl_to_prior"].mean()),
                   "actions_used": int((counts > 0).sum()), "n_actions": bundle.n_actions,
                   "max_action_share": float(share.max()),
                   "collapsed": bool(float(share.max()) > 0.95)}
        distribution = _component("fail" if metrics["collapsed"] else "pass", metrics,
                                  _evidence(output, "action_distribution", metrics))

    components = {"model_validity": validity, "critic_direction": critic,
                  "action_distribution": distribution}
    components["paired_uncertainty"] = _paired_uncertainty(components, settings, output)
    body = {"schema": "d4mj_lewm_actor_gate_v1",
            "checkpoint_sha256": _source_contract_digest(checkpoint),
            "recipe_id": recipe_digest(config),
            "cache_id": contract_digest(cache_contract),
            "stage": "actor_screen", "decision": "continue_actor",
            "validated_recursive_depth": int(payload.get("capabilities", {})
                                             .get("validated_recursive_depth", 0)),
            "not_evaluated": dict(GATE_NOT_EVALUATED),
            "components": components}
    report = _seal(body)
    atomic_manifest(output / "actor_gate.json", report)
    return report
