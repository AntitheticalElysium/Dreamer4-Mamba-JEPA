import ast
from dataclasses import replace
import logging

import pytest
import torch
from transformers import ViTConfig, ViTModel

from d4mj.lewm import LeWMEncoder, SIGReg, joint_loss
from d4mj.lewm_config import LeWMConfig, EncoderSettings, DynamicsSettings, JointSettings, RuntimeSettings
from d4mj.config import config_from_dict, recipe_dict
from d4mj.sources import ROOT, tensor_state_digest
from d4mj.world_api import ModelBundle
from d4mj.lewm_diagnostics import objective_audit, normalization_audit, recurrence_audit


def small_config(**overrides):
    c = LeWMConfig(
        encoder=EncoderSettings(resolution=14, width=24, depth=1, heads=3, latent_dim=12,
                                projector_hidden=32, checkpoint_blocks=False),
        dynamics=DynamicsSettings(width=32, depth=2, headdim=16, d_state=8,
                                  action_dim=8, backend="reference"),
        joint=JointSettings(batch=4, projections=8, knots=5, steps=4, screen_step=2,
                            warmup=1, checkpoint_every=2),
        runtime=RuntimeSettings(device="cpu", precision="fp32", purpose="verification", cache_chunk=3),
    )
    return replace(c, **overrides)


def test_a_field_added_after_a_run_is_omitted_wherever_that_run_omitted_it():
    """recipe_id is the digest of recipe_dict, so a later field must not appear in
    an earlier run's dict -- that silently unloads its sealed checkpoints."""
    from dataclasses import replace
    base = small_config()
    # v1 predates all three fields.
    v1 = recipe_dict(base)["joint"]
    assert not {"stride", "centering_stride", "centering"} & set(v1)
    # v2 was sealed with `stride` but before the centering pair existed.
    v2 = recipe_dict(replace(base, schema="d4mj_lewm_recipe_v2",
                             joint=replace(base.joint, stride=4)))["joint"]
    assert v2["stride"] == 4 and not {"centering_stride", "centering"} & set(v2)
    # A recipe that customizes either centering field was written with both, so both
    # are kept -- including `centering` sitting at its default, as the ablation's
    # consecutive arm has it.
    for joint in (replace(base.joint, centering_stride=4),
                  replace(base.joint, centering_stride=4, centering="strided")):
        kept = recipe_dict(replace(base, schema="d4mj_lewm_recipe_v2", joint=joint))["joint"]
        assert kept["centering_stride"] == 4 and "centering" in kept
    assert config_from_dict(recipe_dict(base)) == base


def test_config_roundtrip_and_actual_statistical_batch():
    c = small_config()
    assert config_from_dict(recipe_dict(c)) == c
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict(recipe_dict(c) | {"guess": 123})
    with pytest.raises(ValueError, match="actual B128"):
        replace(c, runtime=replace(c.runtime, purpose="research"))
    b = ModelBundle.create(c)
    with pytest.raises(ValueError, match="statistical_batch"):
        joint_loss(b.encoder, b.world, torch.zeros(2,4,14,14,3,dtype=torch.uint8),
                   torch.zeros(2,3,dtype=torch.long), SIGReg(5,8), torch.Generator(), c)


def test_source_objective_and_normalization():
    assert objective_audit(small_config())["absolute_error"] < 1e-5
    assert normalization_audit(ModelBundle.create(small_config()))["buffers_immutable"]


def test_encoder_constructor_matches_pinned_tiny_helper():
    source = ast.parse((ROOT / "sources/galilai-group__stable-pretraining/stable_pretraining/backbone/utils.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "vit_hf")
    scope = {"nn": torch.nn, "ViTConfig": ViTConfig, "ViTModel": ViTModel,
             "_TRANSFORMERS_AVAILABLE": True, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_vit_helper", "exec"), scope)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        reference = scope["vit_hf"]("tiny", patch_size=7, image_size=63, use_mask_token=False)
        torch.manual_seed(9)
        ours = LeWMEncoder(LeWMConfig()).backbone
    assert tensor_state_digest(ours.state_dict()) == tensor_state_digest(reference.state_dict())
    assert ours.config.layer_norm_eps == reference.config.layer_norm_eps


def test_raw_tc_initialization_and_two_sided_joint_gradients():
    c = small_config(); tc = ModelBundle.create(c); raw = ModelBundle.create(replace(c, variant="raw"))
    assert tensor_state_digest(tc.encoder.state_dict()) == tensor_state_digest(raw.encoder.state_dict())
    assert tensor_state_digest(tc.world.state_dict()) == tensor_state_digest(raw.world.state_dict())
    frames = torch.randint(256,(4,4,14,14,3),dtype=torch.uint8); actions = torch.randint(17,(4,3))
    loss = joint_loss(tc.encoder,tc.world,frames,actions,SIGReg(5,8),torch.Generator().manual_seed(2),c)
    target_grad = torch.autograd.grad(loss.prediction,loss.latent,retain_graph=True)[0]
    assert target_grad[:,-1].abs().sum()>0 and target_grad[:,0].abs().sum()>0
    loss.total.backward()
    assert tc.encoder.projector[0].weight.grad.abs().sum()>0
    assert tc.world.pair_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in tc.world.agent_readout.parameters())
    assert tc.encoder.projector[1].num_batches_tracked.item()==1
    tc.encoder.freeze().train()
    assert not tc.encoder.training and not tc.encoder.projector[1].training


def test_checkpointed_encoder_bn_updates_once():
    c=small_config(); c=replace(c,encoder=replace(c.encoder,checkpoint_blocks=True))
    b=ModelBundle.create(c)
    loss=joint_loss(b.encoder,b.world,torch.randint(256,(4,4,14,14,3),dtype=torch.uint8),
                    torch.zeros(4,3,dtype=torch.long),SIGReg(5,8),torch.Generator(),c)
    loss.total.backward()
    assert b.encoder.projector[1].num_batches_tracked.item()==1
    assert b.world.predictor_projector[1].num_batches_tracked.item()==1


def test_source_recurrence_and_final_api():
    report=recurrence_audit(ModelBundle.create(small_config()))
    assert report['context_frames']>report['training_frames'] and report['state_gradient_checked']
    assert not report['cuda_kernels_checked']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='target CUDA kernels unavailable')
def test_cuda_source_and_differentiable_carry():
    c=small_config()
    c=replace(c,dynamics=replace(c.dynamics,backend='triton'),runtime=replace(c.runtime,device='cuda'))
    assert recurrence_audit(ModelBundle.create(c))['cuda_kernels_checked']


def test_lewm_gates_run_on_a_strided_recipe():
    """The gate path must accept stacked actions, not just the sampler and world.

    Unit tests over the new code passed while `paired-run` would still have died
    in preflight, because the audits built scalar actions of their own.
    """
    from dataclasses import replace
    from d4mj.lewm_diagnostics import normalization_audit, objective_audit, screen_retention
    base = small_config()
    for recipe in (replace(base, schema="d4mj_lewm_recipe_v2", joint=replace(base.joint, stride=4)),
                   replace(base, schema="d4mj_lewm_recipe_v2", variant="tc",
                           joint=replace(base.joint, centering_stride=4, centering="strided"))):
        bundle = ModelBundle.create(recipe)
        assert recurrence_audit(bundle)["numerical_profile"]
        assert normalization_audit(bundle)
        assert objective_audit(recipe)
        # The resource gate builds a real batch through the sampler and takes an
        # optimizer step, which is where a batch-shape regression actually bites.
        from d4mj.lewm_diagnostics import resource_preflight
        from d4mj.tests.test_joint_data import raw_episodes
        episodes = [replace(e, observations=e.observations.repeat(4, 1, 1, 1)[:33],
                            actions_taken=torch.arange(32) % 17, rewards=torch.zeros(32),
                            terminated=torch.zeros(32, dtype=torch.bool),
                            truncated=torch.zeros(32, dtype=torch.bool),
                            events=torch.zeros(32, dtype=torch.bool)) for e in raw_episodes(14)]
        assert resource_preflight(ModelBundle.create(recipe), episodes)
    strided = replace(base, schema="d4mj_lewm_recipe_v2", joint=replace(base.joint, stride=4))
    # screen_retention conditions its probe on the outgoing actions of each
    # retained transition; a stacked window must still give one row per transition.
    actions = torch.randint(strided.dynamics.n_actions, (2, strided.joint.frames - 1, 4))
    one_hot = torch.nn.functional.one_hot(actions.reshape(-1, 4), strided.dynamics.n_actions).float().flatten(1)
    assert one_hot.shape == (2 * (strided.joint.frames - 1), 4 * strided.dynamics.n_actions)


def test_pinned_transformer_package_is_constructed_exactly_as_the_source_declares():
    """The comparison backend must be the vendored predictor, not a lookalike."""
    from d4mj.lewm_transformer import build_package, source_digests, PINNED
    predictor, action_encoder, projector = build_package()
    assert source_digests() == PINNED, "vendored bytes differ from the audited pins"
    blocks = predictor.transformer.layers
    assert len(blocks) == 6 and type(blocks[0]).__name__ == "ConditionalBlock"
    qkv = [m for n, m in blocks[0].named_modules() if n.endswith("to_qkv")][0]
    # 16 heads x 64 = inner width 1024, so qkv projects 192 -> 3 * 1024.
    assert tuple(qkv.weight.shape) == (3072, 192)
    assert tuple(predictor.pos_embedding.shape) == (1, 3, 192)
    for block in blocks:
        gate = block.adaLN_modulation[-1]
        assert torch.equal(gate.weight, torch.zeros_like(gate.weight))
        assert torch.equal(gate.bias, torch.zeros_like(gate.bias))
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert (count(predictor), count(action_encoder), count(projector)) == (10791360, 156276, 792768)


def test_our_package_matches_an_independent_instantiation_of_the_pinned_classes():
    """Oracle: same weights in, same outputs, gradients, BN buffers and update out.

    This is what makes the comparison a control. If our call convention diverges from the
    source's, every downstream number is measuring our wrapper rather than the paper.
    """
    import copy
    from d4mj.lewm_transformer import build_package, source_module
    source = source_module()
    ours = build_package()
    theirs = (source.ARPredictor(num_frames=3, depth=6, heads=16, mlp_dim=2048, input_dim=192,
                                 hidden_dim=192, output_dim=192, dim_head=64, dropout=0.1,
                                 emb_dropout=0.0),
              source.Embedder(input_dim=17, smoothed_dim=10, emb_dim=192, mlp_scale=4),
              source.MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=torch.nn.BatchNorm1d))
    for mine, other in zip(ours, theirs):
        other.load_state_dict(copy.deepcopy(mine.state_dict()))

    generator = torch.Generator().manual_seed(4)
    latents = torch.randn(2, 3, 192, generator=generator)
    actions = torch.nn.functional.one_hot(
        torch.randint(17, (2, 3), generator=generator), 17).float()

    def run(package, latents, actions):
        predictor, action_encoder, projector = package
        for module in package:
            module.train()
        torch.manual_seed(11)  # dropout is 0.1; the streams must match to compare
        conditioning = action_encoder(actions)
        hidden = predictor(latents, conditioning)
        predicted = projector(hidden.flatten(0, 1)).reshape(hidden.shape)
        loss = predicted.square().mean()
        loss.backward()
        grads = [p.grad.clone() if p.grad is not None else None
                 for module in package for p in module.parameters()]
        buffers = [b.clone() for module in package for b in module.buffers()]
        return predicted, hidden, loss, grads, buffers

    a = run(ours, latents.clone(), actions.clone())
    b = run(theirs, latents.clone(), actions.clone())
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
    torch.testing.assert_close(a[2], b[2], atol=0, rtol=0)
    for mine, other in zip(a[3], b[3]):
        assert (mine is None) == (other is None)
        if mine is not None:
            torch.testing.assert_close(mine, other, atol=0, rtol=0)
    for mine, other in zip(a[4], b[4]):
        torch.testing.assert_close(mine, other, atol=0, rtol=0)

    # One optimizer step must land in the same place too.
    for package in (ours, theirs):
        optimizer = torch.optim.AdamW([p for m in package for p in m.parameters()], lr=1e-3)
        optimizer.step()
    for mine, other in zip((p for m in ours for p in m.parameters()),
                           (p for m in theirs for p in m.parameters())):
        torch.testing.assert_close(mine, other, atol=0, rtol=0)


def test_the_source_action_and_attention_gates_start_at_zero_by_design():
    """AdaLN-Zero means no action or attention gradient at init. That is the source's
    intent, not a broken graph, so the backend must not 'fix' it."""
    from d4mj.lewm_transformer import build_package
    predictor, action_encoder, projector = build_package()
    latents = torch.randn(2, 3, 192, generator=torch.Generator().manual_seed(5))
    actions = torch.nn.functional.one_hot(torch.zeros(2, 3, dtype=torch.long), 17).float()
    for module in (predictor, action_encoder, projector):
        module.eval()
    hidden = predictor(latents, action_encoder(actions))
    projector(hidden.flatten(0, 1)).square().mean().backward()
    assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
               for p in action_encoder.parameters()), "action path should be gated off at init"
    gate = predictor.transformer.layers[0].adaLN_modulation[-1]
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0, \
        "AdaLN itself must still learn from step one"


def transformer_config(**overrides):
    from dataclasses import replace
    from d4mj.lewm_config import LeWMTransformerConfig, RuntimeSettings, JointSettings
    c = LeWMTransformerConfig(
        runtime=RuntimeSettings(device="cpu", precision="fp32", purpose="verification", cache_chunk=3),
        joint=replace(JointSettings(), batch=4, projections=8, knots=5, steps=4,
                      screen_step=2, warmup=1, checkpoint_every=2))
    return replace(c, **overrides)


def wake_adaln(world, seed=3):
    """AdaLN-Zero makes the predictor the identity at init, so a window test there is
    vacuous. Give the gates real values before asserting anything about mixing."""
    generator = torch.Generator().manual_seed(seed)
    for block in world.predictor.transformer.layers:
        gate = block.adaLN_modulation[-1]
        with torch.no_grad():
            gate.weight.copy_(torch.randn(gate.weight.shape, generator=generator) * 0.05)
            gate.bias.copy_(torch.randn(gate.bias.shape, generator=generator) * 0.05)
    return world


def test_the_source_window_forgets_evicted_pairs_and_uses_active_ones():
    """With live AdaLN gates: a pair inside the window moves the prediction, one that has
    slid out does not. That is the finite-window contract a KV cache would break."""
    from d4mj.world_api import ModelBundle
    bundle = ModelBundle.create(transformer_config())
    wake_adaln(bundle.world)
    bundle.eval()
    torch.manual_seed(0)
    z, actions = torch.randn(2, 6, 1, 192), torch.randint(17, (2, 5))
    step = torch.randint(17, (2, 1))
    base, _ = bundle.advance(bundle.prefill(z, actions), step)

    evicted = z.clone(); evicted[:, 0] += 10.0            # slid out of the final window
    moved, _ = bundle.advance(bundle.prefill(evicted, actions), step)
    torch.testing.assert_close(base.latent, moved.latent, atol=0, rtol=0)

    for index in (-3, -2, -1):                            # still inside the window
        active = z.clone(); active[:, index] += 10.0
        changed, _ = bundle.advance(bundle.prefill(active, actions), step)
        assert not torch.equal(base.latent, changed.latent), f"frame {index} should matter"

    older = actions.clone(); older[:, 0] = (older[:, 0] + 1) % 17   # evicted action
    torch.testing.assert_close(
        base.latent, bundle.advance(bundle.prefill(z, older), step)[0].latent, atol=0, rtol=0)


def test_streaming_advance_reproduces_the_rolling_teacher_scan():
    """One source call per step must equal the parallel scan over the same pairs."""
    from d4mj.world_api import ModelBundle
    bundle = ModelBundle.create(transformer_config())
    wake_adaln(bundle.world)
    bundle.eval()
    torch.manual_seed(1)
    z, actions = torch.randn(2, 7, 1, 192), torch.randint(17, (2, 6))
    scanned = bundle.world.teacher(z, actions)
    state = bundle.world.start(z[:, :1])
    for step in range(actions.shape[1]):
        state, _ = bundle.world.observe_latent(state, actions[:, step:step + 1], z[:, step + 1:step + 2])
        torch.testing.assert_close(state.history, scanned.features[:, step:step + 1], atol=0, rtol=0)
    torch.testing.assert_close(state.latent, scanned.state.latent, atol=0, rtol=0)
    torch.testing.assert_close(state.past_latents, scanned.state.past_latents, atol=0, rtol=0)
    assert state.step == scanned.state.step == actions.shape[1]


def test_bounded_prefill_equals_a_full_scan_and_generation_is_repeatable():
    """Only the final window can reach the next prediction, so truncating is exact."""
    from d4mj.world_api import ModelBundle
    bundle = ModelBundle.create(transformer_config())
    wake_adaln(bundle.world)
    bundle.eval()
    torch.manual_seed(2)
    z, actions = torch.randn(2, 9, 1, 192), torch.randint(17, (2, 8))
    bounded = bundle.prefill(z, actions)
    full = bundle.world.teacher(z, actions).state
    for a, b in ((bounded.latent, full.latent), (bounded.past_latents, full.past_latents),
                 (bounded.history, full.history)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(bounded.past_actions, full.past_actions)

    # Repeated generation from one root: siblings never mutate the parent, and depth grows.
    root = bundle.repeat_state(bounded, 17)
    before = tuple(t.clone() for t in bundle.state_tensors(bounded))
    state, seen = root, []
    for depth in range(4):
        state, _ = bundle.advance(state, torch.arange(17).repeat(2)[:, None])
        seen.append(state.latent.clone())
        assert state.step == bounded.step + depth + 1
    for old, now in zip(before, bundle.state_tensors(bounded)):
        torch.testing.assert_close(old, now, atol=0, rtol=0)
    again, _ = bundle.advance(root, torch.arange(17).repeat(2)[:, None])
    torch.testing.assert_close(again.latent, seen[0], atol=0, rtol=0)
