"""Regression tests for the pre-P1 review findings; these test metric contracts, not learned performance."""
import copy

import numpy as np
import pytest
import torch
import yaml

from saccade import train as TR
from saccade.eval import gates as G, s1


def test_ap_groups_ties_and_is_invariant_to_row_order():
    y = np.array([1, 1, 0, 0])
    assert s1.ap(y, np.full(4, .5)) == .5
    assert s1.ap(y[::-1], np.full(4, .5)) == .5
    y, scores = np.array([1, 0, 1, 0]), np.array([.9, .9, .5, .1])
    expected = .5 * .5 + .5 * (2 / 3)
    for seed in range(10):
        ix = np.random.default_rng(seed).permutation(4)
        assert s1.ap(y[ix], scores[ix]) == pytest.approx(expected)
    assert s1.ap(np.zeros(4), scores) is None
    with pytest.raises(ValueError, match="finite"):
        s1.ap(y, [np.nan, 1, 2, 3])


def _encoding(n=12):
    torch.manual_seed(1)
    return {"p": torch.randn(n, 81, 16).half(), "g": torch.randn(n, 9, 64).half(),
            "v": torch.randn(n, 8).half(), "fix": torch.full((n, 2), 4, dtype=torch.long),
            "prev_action": torch.tensor([-1, *([0] * (n - 1))]), "reward": torch.zeros(n),
            "terminated": torch.zeros(n, dtype=torch.bool), "bp": torch.randn(n, 256).half()}


def test_classifier_conditions_on_history_and_beliefs_without_target_leakage():
    rec = _encoding()
    features = lambda r: s1.conditioning_features([r], np.array([0]), np.array([10]), 8)
    base = features(rec)
    for key, t in (("p", 2), ("g", 9), ("v", 3), ("reward", 4), ("terminated", 5),
                   ("prev_action", 6), ("fix", 7), ("bp", 2), ("bp", 10), ("prev_action", 10), ("fix", 10)):
        other = copy.deepcopy(rec)
        if other[key].dtype == torch.bool:
            other[key][t] = ~other[key][t]
        else:
            other[key][t] += 1
        assert not np.array_equal(base, features(other)), (key, t)
    other = copy.deepcopy(rec)
    for key in ("p", "g", "v", "reward", "terminated"):
        other[key][10] = 0
    assert np.array_equal(base, features(other))                          # the outcome is not a condition
    other = copy.deepcopy(rec)
    other["p"][0] += 10
    assert np.array_equal(base, features(other))                          # raw frames outside the window


def test_classifier_history_padding_stays_inside_its_episode():
    rec = _encoding()
    rows = s1.conditioning_features([rec], np.array([0]), np.array([1]), 8)
    width = s1.frame_features([rec]).shape[1] + rec["bp"].shape[1] + 1
    history = rows[0, :8 * width].reshape(8, width)
    assert np.count_nonzero(history[:-1]) == 0
    assert history[-1, -1] == 1


def test_classifier_uses_the_same_precision_for_real_and_generated():
    x = {"p": torch.randn(5, 81, 16), "g": torch.randn(5, 9, 64), "v": torch.randn(5, 8),
         "r": torch.tensor([0., 1., .1, -.1, .9]), "term": torch.zeros(5)}
    rounded = {k: v.half() if k in ("p", "g", "v") else v for k, v in x.items()}
    assert s1.outcome_features(x).dtype == np.float16
    assert np.array_equal(s1.outcome_features(x), s1.outcome_features(rounded))


def test_classifier_real_vs_itself_is_exactly_chance():
    rng = np.random.default_rng(4)
    context = rng.normal(size=(512, 80)).astype(np.float16)
    outcome = rng.normal(size=(512, 4)).astype(np.float16)
    ep = np.repeat(np.arange(64), 8)
    result = s1.c2st(context, outcome, outcome, ep, np.random.default_rng(3))
    assert result["accuracy"] == .5


def test_classifier_detects_dependence_with_identical_marginals():
    # Real = (x, x); synthetic = (x, -x), balanced in every episode. Each coordinate has exactly the same marginal.
    x = np.tile(np.array([-1., 1.]), 500)[:, None].astype(np.float16)
    ep = np.repeat(np.arange(100), 10)
    result = s1.c2st(x, x, -x, ep, np.random.default_rng(1))
    assert result["accuracy"] > .95


def test_classifier_requires_separate_physical_episodes():
    x = np.ones((8, 1), np.float16)
    with pytest.raises(ValueError, match="train, validation and test"):
        s1.c2st(x, x, x, np.zeros(8), np.random.default_rng(1))


def test_absence_targets_record_visible_history_and_last_seen_side():
    grid = np.zeros((67, 7, 9), np.int8)
    grid[0, 0, 4] = 1
    grid[0, 3, 8] = 2
    cls, side = s1.departed_episode(grid)
    for lo, hi, label in ((1, 4, 1), (5, 8, 2), (9, 16, 3), (17, 64, 4)):
        assert np.all(cls[lo:hi + 1, 0] == label)
        assert np.all(side[lo:hi + 1, 0] == 0)
        assert np.all(side[lo:hi + 1, 1] == 3)
    assert np.all(cls[65:] == 0) and np.all(side[65:] == -1)
    grid[30, 3, 0] = 1
    cls, side = s1.departed_episode(grid)
    assert cls[30, 0] == 0 and cls[31, 0] == 1 and side[31, 0] == 2
    assert not s1.departed_episode(np.zeros((2, 7, 9), np.int8))[0].any()
    player = np.zeros((2, 7, 9), np.int8)
    player[0, 3, 4] = 4
    assert not s1.departed_episode(player)[0].any()                        # an occluded arrow is not a sighting


def test_hidden_lifecycle_changes_do_not_erase_observed_history():
    L = {"ep_T": np.array([4]), "s_pos": np.full((5, 2), 10), "s_view": np.zeros((5, 7, 9)),
         "s_body": np.full((5, 4), 9), "t_reward": np.zeros(4), "t_terminated": np.zeros(4), "t_action": np.zeros(4)}
    for kind in G.KINDS:
        L[f"s_{kind}_p"] = np.full((5, 1, 2), 100)
        L[f"s_{kind}_m"] = np.zeros((5, 1), bool)
    L["s_zombies_p"][0, 0] = [7, 10]
    L["s_zombies_m"][:] = True
    alt = copy.deepcopy(L)
    alt["s_zombies_m"][1:] = False
    assert np.array_equal(G.episode_grids(L)[0]["grid"], G.episode_grids(alt)[0]["grid"])
    for a, b in zip(s1.departed(L), s1.departed(alt)):
        assert np.array_equal(a, b)
    assert np.all(s1.departed(L)[0][1:, 0] == 1)


def test_unlock_probability_matches_the_health_adjusted_event():
    values = torch.tensor([-.9, 0., .1, .4, 1.])
    classes = torch.tensor([2, 2, 4, 3, 0])
    logp = torch.eye(5)[classes].log()
    dh = torch.tensor([-9., 1., 0., -6., -9.])
    assert torch.equal(s1.unlock_probability(logp, values, dh), torch.tensor([1., 0., 1., 1., 0.]))
    logp = torch.tensor([[0., .7, 0., 0., .3]]).log()
    assert s1.unlock_probability(logp, values, torch.tensor([0.])).item() == pytest.approx(.3)


def test_reader_round_trip_is_frozen_and_checkpoint_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(s1, "LABELS", tmp_path)
    (tmp_path / "s0_train400_labels.npz").write_bytes(b"unit-test label identity")
    ck = tmp_path / "checkpoint.pt"
    ck.write_bytes(b"checkpoint identity")
    cfg = TR.load_cfg()
    reader = s1.MLP(24, {"kind": 5, "terr": 17}).eval()
    x = torch.randn(9, 24)
    metadata = s1.save_reader(reader, tmp_path / "reader.pt", ck, 1, cfg)
    restored = s1.load_reader(tmp_path / "reader.pt", ck, "cpu")
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    for key, value in reader(x).items():
        assert torch.equal(value, restored(x)[key])
    assert metadata["sha256"] == TR.file_sha256(tmp_path / "reader.pt")
    ck.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different encoder"):
        s1.load_reader(tmp_path / "reader.pt", ck, "cpu")


def test_perception_blocker_distinguishes_unmeasured_from_passed():
    estimate = {"est": 1., "se": 0., "n_cond": 10}
    seed = {f"{light}_r1_2": {f"{metric}_{kind}": copy.deepcopy(estimate) for metric in ("recall", "precision")
                            for kind in G.KINDS} for light in ("day", "night")}
    reference = {"0": seed, "1": copy.deepcopy(seed)}
    assert s1.perception_blocker([seed, seed], [seed, seed], reference)["status"] == "passed"
    missing = copy.deepcopy(seed)
    missing["night_r1_2"]["precision_arrows"] = None
    result = s1.perception_blocker([missing, seed], [seed, seed], reference)
    assert result["status"] == "unmeasured" and result["unmeasured"]
    broken = copy.deepcopy(seed)
    broken["night_r1_2"]["recall_arrows"]["est"] = 0.
    assert s1.perception_blocker([broken, broken], [seed, seed], reference)["status"] == "blocked"


def test_m0_yaml_has_no_duplicate_keys():
    def walk(node):
        if isinstance(node, yaml.MappingNode):
            keys = [k.value for k, _ in node.value]
            assert len(set(keys)) == len(keys), keys
            for _, value in node.value:
                walk(value)
    walk(yaml.compose((TR.ROOT / "saccade/m0.yaml").read_text()))


def test_one_step_factorised_is_the_system_sampler(monkeypatch):
    """S1.3 (decision 1's cascade): one_step's draw for a tuple equals the system's own path for it, the
    real context committed frame by frame by context_pass and then M.sample's cascade, from the same generator state."""
    from saccade import model as M
    monkeypatch.setattr(s1, "DEV", torch.device("cpu"))
    cfg = copy.deepcopy(TR.load_cfg())
    torch.manual_seed(0)
    model = M.M0(cfg).eval()
    for m in [b.mod for b in model.denoiser.blocks] + [model.denoiser.final_mod, *model.denoiser.out]:
        torch.nn.init.normal_(m.weight, std=0.02)
    pc, P = cfg["perception"], cfg["data"]["context_prefix"]
    g = torch.Generator().manual_seed(1)
    for T, seed in ((30, 3), (30, 11), (30, 5)):                         # t = 6, 24 (full context), 1 (one frame)
        enc = [{"p": torch.randn(T, M.N, pc["periphery"]["token_dim"], generator=g), "g": torch.randn(T, M.G, pc["fovea"]["token_dim"], generator=g),
                "v": torch.randn(T, pc["lighting"]["dim"], generator=g), "fix": torch.randint(1, 8, (T, 2), generator=g),
                "bp": torch.randn(T, cfg["belief"]["width"], generator=g), "prev_action": torch.randint(0, 17, (T,), generator=g)}]
        eps_, ts, out = s1.one_step(model, enc, cfg, 1, np.random.default_rng(seed))
        t, e = int(ts[0]), enc[0]
        k = min(P, t)
        bp = torch.cat([torch.zeros(P - k, e["bp"].shape[1]), e["bp"][t - k:t + 1]])[None]
        mem = model.belief.memory(bp)[:, P - k:]
        gen, cache = torch.Generator().manual_seed(0), model.denoiser.new_cache()
        with torch.no_grad():
            for i in range(k):
                f = t - k + i
                cache = M.context_pass(model.denoiser, tuple(e[key][f][None, None].float() for key in ("p", "g", "v")),
                                       e["prev_action"][f].view(1, 1), mem[:, i:i + 1], e["fix"][f].view(1, 1, 2), cache, cfg, gen)
            ref = M.sample(model.denoiser, [tuple(e[key].shape[1:]) for key in ("p", "g", "v")], e["prev_action"][t].view(1, 1),
                           mem[:, -1:], e["fix"][t].view(1, 1, 2), cache, cfg, gen)
        for key, r in zip(("p", "g", "v"), ref):
            assert torch.equal(out[key][0], r[0, 0].float()), (t, key)
