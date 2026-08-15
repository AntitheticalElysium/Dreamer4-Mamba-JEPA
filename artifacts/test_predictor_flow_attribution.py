from pathlib import Path

import torch

from artifacts.predictor_flow_attribution_common import (
    PREDICTORS,
    SOURCE_FILES,
    make_world,
    parameter_report,
    shared_state_digest,
)
from artifacts.evaluate_predictor_flow_archive import predict_record, predict_records
from artifacts.summarize_predictor_flow_attribution import contrast_vector, paired_difference
from d4mj.checkpoint import save
from d4mj.config import Config


def test_capacity_control_matches_token_predictor():
    config = Config(transition="direct", time_mixer="attention")
    report = parameter_report(config)
    deep = report["deep_mlp"]["predictor"]
    token = report["token_transformer"]["predictor"]
    assert report["current"]["predictor"] < deep
    assert abs(deep - token) / token < 0.05


def test_predictors_preserve_shape_bound_and_agent_firewall():
    config = Config(transition="direct", time_mixer="attention")
    features = torch.randn(2, 3, 24, config.d_model)
    action = torch.tensor([[1, 2, 3], [4, 5, 6]])
    for predictor in PREDICTORS:
        torch.manual_seed(11)
        world = make_world(config, predictor).eval()
        predicted = world.predict(features, action)
        changed_agent = features.clone()
        changed_agent[:, :, world.agent] += 1000
        assert predicted.shape == (2, 3, config.n_spatial, config.d_spatial)
        assert float(predicted.detach().abs().max()) <= 1.0
        assert torch.equal(predicted, world.predict(changed_agent, action))
        assert not torch.equal(predicted, world.predict(features, action.roll(1, 0)))


def test_topology_cells_share_every_non_predictor_tensor():
    config = Config(transition="direct", time_mixer="attention")
    digests = []
    for predictor in PREDICTORS:
        torch.manual_seed(config.seed + 1)
        digests.append(shared_state_digest(make_world(config, predictor)))
    assert len(set(digests)) == 1


def test_pinned_predictor_sources_exist_and_lock_vjepa_commit():
    assert all(path.exists() for path in SOURCE_FILES)
    lock = Path("third_party/SOURCES.lock").read_text()
    assert "facebookresearch__vjepa2" in lock
    assert "204698b45b3712590f06245fbfba32d3be539812" in lock


def test_archive_prediction_uses_arm_appropriate_transition_path():
    record = {
        "latents": torch.randn(3, 16, 32).tanh(),
        "actions_taken": torch.tensor([2, 4]),
        "transitions": torch.tensor([1]),
    }
    for transition in ("direct", "flow"):
        config = Config(transition=transition, time_mixer="attention")
        torch.manual_seed(17)
        world = make_world(config).eval()
        predicted = predict_record(
            world, record, config, context=2, samples=2, seed=31
        )
        expected = {"reset16"} if transition == "direct" else {
            "reset16_first",
            "reset16_mean",
        }
        assert set(predicted) == expected
        assert all(value.shape == (16, 32) for value in predicted.values())


def test_archive_record_packing_keeps_group_and_pool_contracts():
    config = Config(transition="direct", time_mixer="attention")
    records = []
    for label in (0.0, 1.0):
        records.append(
            {
                "latents": torch.randn(3, 16, 32).tanh(),
                "actions_taken": torch.tensor([2, 4]),
                "transitions": torch.tensor([1]),
                "labels": torch.tensor([label]),
                "group": 9,
                "pool": "support",
            }
        )
    packed = predict_records(
        make_world(config).eval(), records, config, context=2, samples=2
    )["reset16"]
    assert packed["group"].tolist() == [9, 9]
    assert packed["pool"] == ["support", "support"]
    assert packed["predicted"].shape == (2, 16, 32)


def test_custom_predictor_checkpoint_round_trip(tmp_path):
    config = Config(transition="direct", time_mixer="attention")
    torch.manual_seed(19)
    world = make_world(config, "token_transformer")
    path = tmp_path / "world.pt"
    save(path, config, part0=world)
    from artifacts.predictor_flow_attribution_common import load_world

    restored = load_world(path, config, "token_transformer")
    for first, second in zip(world.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(first, second)


def test_paired_effect_rule_requires_the_declared_band():
    control = torch.zeros(32)
    intervention = torch.full((32,), 0.2)
    report = paired_difference(
        intervention,
        control,
        minimum_effect=0.1,
        seed=7,
        samples=100,
    )
    assert report["material_improvement"]
    assert not report["practically_equivalent"]


def test_archive_contrast_uses_successor_delta_not_successor_status():
    data = {
        "predicted": torch.tensor([[[[11.0]]], [[[13.0]]]]),
        "label": torch.tensor([False, True]),
        "group": torch.tensor([0, 0]),
    }
    current = torch.tensor([[[[10.0]]], [[[12.5]]]])
    value = contrast_vector(data, current, torch.tensor([1.0]))
    assert torch.equal(value, torch.tensor([-0.5]))
