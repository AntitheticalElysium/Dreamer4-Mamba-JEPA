"""Unit tests for the non-generative T-JEPA arm (D030-D033)."""
from __future__ import annotations

from dataclasses import asdict

import torch

from d4_mamba_jepa.cartpole_baseline import cartpole_config, cartpole_jepa_config
from d4_mamba_jepa.data import SequenceBatch
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.objectives import jepa_self_prediction_loss
from d4_mamba_jepa.rollout import sample_next_packed, shortcut_schedule
from d4_mamba_jepa.training import WorldLossNormalizer, world_loss

DEVICE = torch.device("cpu")


def _batch(cfg, B=2, T=6):
    torch.manual_seed(0)
    return SequenceBatch(
        observations=torch.randint(0, 255, (B, T, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8),
        led_to_actions=torch.randint(0, cfg.n_actions, (B, T)),
        led_to_rewards=torch.rand(B, T),
        led_to_continues=torch.ones(B, T),
        outcome_valid=torch.ones(B, T, dtype=torch.bool),
    )


def test_jepa_config_and_isolation():
    base, jepa = cartpole_config(), cartpole_jepa_config()
    assert jepa.representation_objective == "jepa"
    assert jepa.arm_id == "T-JEPA"
    # Only the representation objective differs from the T-BASE control.
    b, j = asdict(base), asdict(jepa)
    diff = {k for k in b if b[k] != j.get(k)}
    assert diff == {"representation_objective"}, diff


def test_jepa_world_has_no_decoder_but_has_target_and_predictor():
    world = D4LiteWorld(cartpole_jepa_config())
    assert world.decoder is None
    assert world.target_encoder is not None
    assert world.jepa_predictor is not None
    assert world.jepa_projection is not None and world.jepa_prediction is not None
    # Base arm keeps the decoder.
    assert D4LiteWorld(cartpole_config()).decoder is not None


def test_jepa_rollout_is_deterministic_and_ignores_schedule():
    world = D4LiteWorld(cartpole_jepa_config()).eval()
    cfg = world.cfg
    obs = torch.randint(0, 255, (2, 5, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8)
    past = world.encode_frames(obs, frozen=True).packed
    led = torch.randint(0, 2, (2, 6))
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(999)
    z1, a1 = sample_next_packed(
        world, past_packed=past, led_to_actions=led,
        schedule=shortcut_schedule(cfg.k_max, cfg.k_max), use_cache=True, generator=g1,
    )
    z2, a2 = sample_next_packed(
        world, past_packed=past, led_to_actions=led,
        schedule=shortcut_schedule(cfg.k_max, 1), use_cache=False, generator=g2,
    )
    # No denoising, no noise: identical regardless of generator/schedule/cache.
    assert torch.equal(z1, z2)
    assert z1.shape == (2, cfg.n_spatial, cfg.d_spatial)


def test_jepa_target_is_stop_gradient_and_online_trains():
    world = D4LiteWorld(cartpole_jepa_config()).train()
    norm = WorldLossNormalizer()
    loss, _ = world_loss(world, _batch(world.cfg), normalizer=norm)
    loss.backward()
    assert all(p.grad is None for p in world.target_encoder.parameters())
    assert all(p.grad is None for p in world.jepa_target_projection.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in world.encoder.parameters())
    assert any(p.grad is not None for p in world.jepa_predictor.parameters())


def test_jepa_ema_update_is_exact_convex_combination():
    world = D4LiteWorld(cartpole_jepa_config())
    # perturb online so it differs from the deepcopy target
    with torch.no_grad():
        for p in world.encoder.parameters():
            p.add_(0.1)
    tau = 0.9
    tgt_old = [p.detach().clone() for p in world.target_encoder.parameters()]
    onl = [p.detach().clone() for p in world.encoder.parameters()]
    world.update_jepa_target(tau)
    for new, old, on in zip(world.target_encoder.parameters(), tgt_old, onl):
        expected = tau * old + (1.0 - tau) * on
        assert torch.allclose(new, expected, atol=1e-6)


def test_jepa_loss_is_bounded_and_target_detached():
    world = D4LiteWorld(cartpole_jepa_config()).train()
    cfg = world.cfg
    obs = torch.randint(0, 255, (2, 6, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8)
    clean = world.encode_frames(obs, frozen=False).packed
    loss, metrics = jepa_self_prediction_loss(
        world, frames=obs, clean=clean, led_to_actions=torch.randint(0, 2, (2, 6))
    )
    # normalized-MSE of unit vectors lies in [0, 4]; cosine in [-1, 1].
    assert 0.0 <= float(loss.detach()) <= 4.0
    assert -1.0 <= float(metrics["jepa_cosine"]) <= 1.0
    # not collapsed at init: online prediction dispersion is strictly positive.
    assert float(metrics["jepa_online_std"]) > 0.0
    loss.backward()
    # gradient must not reach the target encoder through the detached target.
    assert all(p.grad is None for p in world.target_encoder.parameters())


def test_sigreg_arm_drops_ema_and_heads_and_penalizes_collapse():
    from dataclasses import replace
    from d4_mamba_jepa.training import WorldLossNormalizer, world_loss

    cfg = replace(cartpole_jepa_config(), jepa_anticollapse="sigreg", jepa_sigreg_slices=64)
    world = D4LiteWorld(cfg).train()
    # LeJEPA drops the EMA target and the projection/prediction heads.
    assert world.target_encoder is None
    assert world.jepa_projection is None and world.jepa_prediction is None
    assert world.jepa_target_projection is None
    assert world.sigreg_test is not None
    assert world.jepa_predictor is not None  # predictor (the rollout) is kept
    # SIGReg is low for isotropic Gaussian, high for collapsed embeddings.
    gaussian = torch.randn(96, 128)
    collapsed = torch.zeros(96, 128) + torch.randn(1, 128)
    assert float(world.sigreg_test(gaussian)) < float(world.sigreg_test(collapsed))
    # Loss is finite and reports the SIGReg + prediction terms; encoder trains.
    loss, m = world_loss(world, _batch(cfg), normalizer=WorldLossNormalizer())
    assert bool(torch.isfinite(loss))
    assert "jepa/jepa_sigreg" in m and "jepa/jepa_prediction" in m
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in world.encoder.parameters())


def _jepa_world(backend="transformer"):
    from d4_mamba_jepa.cartpole_baseline import cartpole_jepa_config
    from d4_mamba_jepa.model import D4LiteWorld

    return D4LiteWorld(cartpole_jepa_config(backend))


def test_jepa_heads_train_on_rollout_latents_like_deployment():
    """Regression for the JEPA train/deployment head mismatch.

    Training must expose the reward/continuation heads to post-transition agent
    tokens produced from PREDICTOR-generated latents, which is what
    ``rollout._sample_next_jepa`` feeds them at deployment. Previously they were
    trained on a clean pass over real encoder latents.
    """
    import torch

    from d4_mamba_jepa.objectives import jepa_self_prediction_loss
    from d4_mamba_jepa.rollout import sample_next_packed

    world = _jepa_world()
    world.eval()
    cfg = world.cfg
    B, T = 2, cfg.sequence_length
    torch.manual_seed(0)
    frames = torch.rand(B, T, 3, cfg.image_size, cfg.image_size)
    actions = torch.randint(0, cfg.n_actions, (B, T))

    encoded = world.encode_frames(frames, frozen=True)
    _, _, rollout_agents = jepa_self_prediction_loss(
        world,
        frames=frames,
        clean=encoded.packed,
        led_to_actions=actions,
        return_rollout_agents=True,
    )
    # Shape contract: one post-transition token set per imagined jump.
    assert rollout_agents.shape == (
        B, cfg.jepa_jumps, cfg.n_agent, cfg.dynamics_d_model
    )

    # The training tokens must be numerically identical to what the deployment
    # rollout primitive produces for the first imagined step from the same
    # context, which is only true if both read predictor-generated latents.
    context = T - cfg.jepa_jumps
    with torch.no_grad():
        _, deploy_agent = sample_next_packed(
            world,
            past_packed=encoded.packed[:, :context],
            led_to_actions=actions[:, : context + 1],
            schedule=None,
            use_cache=False,
        )
    torch.testing.assert_close(
        rollout_agents[:, 0], deploy_agent[:, 0], rtol=1e-4, atol=1e-4
    )


def test_jepa_targets_stay_frozen_after_train():
    """Freezing is by requires_grad only; the mode stays training."""
    world = _jepa_world()
    world.train()
    assert not any(p.requires_grad for p in world.target_encoder.parameters())
    assert not any(
        p.requires_grad for p in world.jepa_target_projection.parameters()
    )
    assert world.target_encoder.training is True
    assert world.jepa_target_projection.training is True


def test_jepa_rollout_context_agent_is_exact_and_saves_a_pass():
    """Carrying the last context-agent token across rollout steps is bit-exact.

    pass-1 of step k (the clean context pass) recomputes exactly what step k-1's
    pass-2 produced, so reusing it must give identical latents/agents while doing
    one fewer dynamics pass per step after the first.
    """
    from d4_mamba_jepa.rollout import sample_next_packed

    world = D4LiteWorld(cartpole_jepa_config()).eval()
    cfg = world.cfg
    torch.manual_seed(0)
    B, context, horizon = 2, 4, 5
    frames = torch.randint(
        0, 255, (B, context, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8
    )
    past0 = world.encode_frames(frames, frozen=True).packed
    actions = torch.randint(0, cfg.n_actions, (B, context + horizon))
    ctx_actions, plan = actions[:, :context], actions[:, context:context + horizon]

    def rollout(carry):
        past, context_agent, outs = past0.clone(), None, []
        for k in range(horizon):
            led = torch.cat([ctx_actions, plan[:, : k + 1]], dim=1)
            nl, ag = sample_next_packed(
                world, past_packed=past, led_to_actions=led, schedule=None,
                use_cache=False, context_agent=(context_agent if carry else None),
            )
            outs.append((nl, ag))
            context_agent = ag
            past = torch.cat([past, nl[:, None]], dim=1)
        return outs

    naive, carried = rollout(False), rollout(True)
    for (nl0, ag0), (nl1, ag1) in zip(naive, carried):
        assert torch.equal(nl0, nl1)   # bit-exact latents
        assert torch.equal(ag0, ag1)   # bit-exact agent tokens

    # Count dynamics passes: naive = 2/step; carried = 2 (first) + 1/subsequent.
    calls = {"n": 0}
    original = world.forward_dynamics
    world.forward_dynamics = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), original(*a, **k))[1]
    rollout(True); carried_calls = calls["n"]
    calls["n"] = 0
    rollout(False); naive_calls = calls["n"]
    world.forward_dynamics = original
    assert carried_calls == horizon + 1
    assert naive_calls == 2 * horizon
    assert carried_calls < naive_calls


def test_jepa_target_stays_in_training_mode():
    """Regression: the EMA target must never be in eval() mode.

    A BatchNorm target in eval() mode normalizes with stale running statistics
    instead of current-batch statistics, which silently collapses the SPR
    representation while the training cosine looks high (a causally-validated
    bug). ``_build_jepa`` must freeze the target by ``requires_grad`` ONLY and
    leave it in training mode -- from construction (this is what the baseline
    got wrong) and across an ``eval() -> train()`` cycle.
    """
    world = _jepa_world()
    # 1. Never constructed in eval mode. On the pre-fix baseline the target was
    #    built with .eval(), so this assertion would have FAILED there.
    assert world.target_encoder.training is True
    assert world.jepa_target_projection.training is True
    # 2. The mode is load-bearing only because the projection has BatchNorm.
    assert any(
        isinstance(module, torch.nn.BatchNorm1d)
        for module in world.jepa_target_projection.modules()
    ), "target projection must contain BatchNorm for the mode to matter"
    # 3. Frozen by requires_grad, and it survives a diagnostic eval()->train().
    world.eval()
    world.train()
    assert world.target_encoder.training is True
    assert world.jepa_target_projection.training is True
    assert not any(p.requires_grad for p in world.target_encoder.parameters())
    assert not any(
        p.requires_grad for p in world.jepa_target_projection.parameters()
    )


def test_cfg_jepa_weight_is_honoured_not_a_dead_field():
    """`cfg.jepa_weight` must actually scale the self-prediction term.

    It was previously declared and validated in D4LiteConfig but read by no
    code, so a diagnostic that set it to 0.0 silently trained the ordinary
    baseline and reproduced it bit-for-bit. Regression: the default must leave
    the total equal to the raw term, and 0.0 must remove its contribution.
    """
    from dataclasses import replace

    from d4_mamba_jepa.training import LossWeights

    cfg = cartpole_jepa_config()
    torch.manual_seed(0)
    batch = SequenceBatch(
        observations=torch.randint(
            0, 255, (2, cfg.sequence_length, 3, cfg.image_size, cfg.image_size),
            dtype=torch.uint8),
        led_to_actions=torch.zeros((2, cfg.sequence_length), dtype=torch.long),
        led_to_rewards=torch.zeros((2, cfg.sequence_length)),
        led_to_continues=torch.ones((2, cfg.sequence_length)),
        outcome_valid=torch.ones((2, cfg.sequence_length), dtype=torch.bool),
    )

    def evaluate(weight):
        torch.manual_seed(1)
        world = D4LiteWorld(replace(cfg, jepa_weight=weight))
        total, metrics = world_loss(
            world, batch, normalizer=WorldLossNormalizer(),
            weights=LossWeights(reward=0.0, continuation=0.0),
        )
        return float(total), float(metrics["loss/jepa"])

    total_one, raw_one = evaluate(1.0)
    total_zero, raw_zero = evaluate(0.0)

    # The reported raw term is unchanged; only its contribution is scaled.
    assert abs(raw_one - raw_zero) < 1e-5
    assert abs(total_one - raw_one) < 1e-5
    assert abs(total_zero) < 1e-6
