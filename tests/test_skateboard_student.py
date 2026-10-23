"""Observation/action adapter parity and immutable native routing on CPU."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from unisupermimic.interaction.multitask_scene import G1InteractionCPU
from unisupermimic.interaction.skateboard_skill import (
    OBS_SCHEMA, SOURCE_DEFAULT, SOURCE_NAMES, SOURCE_SCALE,
    RetainedSkillPolicy, SkateboardObservation, SkateboardStudent, state_digest,
)
from unisupermimic.rl.algos.supermimic.networks import Policy


def test_native_29_action_mapping_holds_unactuated_source_wrists_at_zero():
    env = G1InteractionCPU("skateboard", skateboard_foot_contacts="husky_capsules")
    encoder = SkateboardObservation(env)
    source = np.linspace(-1., 1., 23, dtype=np.float32)
    native = encoder.source_to_native(source)
    assert native.shape == (29,)
    actual_targets = env.default_q + env.action_scale * native
    np.testing.assert_allclose(actual_targets[encoder.indices], SOURCE_DEFAULT + SOURCE_SCALE * source,
                               rtol=0, atol=5e-8)
    wrist_indices = [i for i, name in enumerate(env.joint_names) if "wrist" in name]
    np.testing.assert_allclose(actual_targets[wrist_indices], 0, rtol=0, atol=5e-8)
    encoder.record_action(native)
    np.testing.assert_allclose(encoder.previous, source, rtol=0, atol=2e-7)
    qpos = encoder.initial_qpos()
    np.testing.assert_array_equal(qpos[7 + encoder.indices], SOURCE_DEFAULT)
    np.testing.assert_array_equal(qpos[7 + np.array(wrist_indices)], 0)


def test_observation_history_matches_independent_local_expert_adapter():
    husky = Path("/home/tcl/Desktop/start/humanoid_skateboarding")
    if not (husky / "ckpts/test.onnx").is_file():
        pytest.skip("Independent local HUSKY expert fixture is absent; no download attempted")
    # Existing feasibility adapter independently derives source joint order from
    # the original HUSKY XML and uses the upstream history stacking convention.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from probe_husky_transfer import HuskyNativeExpert
    env = G1InteractionCPU("skateboard", skateboard_foot_contacts="husky_capsules")
    expert = HuskyNativeExpert(env, husky)
    encoder = SkateboardObservation(env)
    np.testing.assert_array_equal(encoder.indices, expert.native_indices)
    qpos = expert.initial_qpos()
    np.testing.assert_array_equal(qpos, encoder.initial_qpos())
    expert.reset_history()
    encoder.reset()
    for _ in range(12):
        expected_action, expected_observation = expert.action_native(speed=.7, heading_target=.12)
        actual_observation = encoder.encode(speed=.7, heading_target=.12)
        assert actual_observation.shape == (395,)
        assert actual_observation.dtype == np.float32
        np.testing.assert_allclose(actual_observation, expected_observation, rtol=0, atol=2e-6)
        np.testing.assert_allclose(encoder.source_to_native(expert.action), expected_action,
                                   rtol=0, atol=2e-7)
        env.step(expected_action)
        encoder.record_action(expected_action)
    encoder.reset()
    assert encoder.step_number == 0
    np.testing.assert_array_equal(encoder.previous, np.zeros(23))
    np.testing.assert_array_equal(np.asarray(encoder.history), np.zeros((5, 79)))


def test_native_function_and_normalization_unchanged_after_skill_training():
    torch.manual_seed(932)
    base = Policy(obs_dim=5, goal_dim=7, hist=3, z_dim=2, num_joints=4, hidden=(12, 8))
    base.obs_norm.update(torch.randn(20, 5))
    base.goal_norm.update(torch.randn(20, 7))
    bundle = RetainedSkillPolicy(base, SkateboardStudent(hidden=(12, 8)))
    before = {key: value.clone() for key, value in bundle.base.state_dict().items()}
    history, goal = torch.randn(4, 3, 5), torch.randn(4, 7)
    expected = base(history, goal).detach()
    optimizer = torch.optim.Adam(bundle.skateboard.parameters(), lr=.01)
    bundle.train()
    for _ in range(3):
        optimizer.zero_grad()
        bundle(torch.randn(4, 395), task="skateboard").square().mean().backward()
        optimizer.step()
    bundle.base.obs_norm.update(torch.randn(20, 5) + 50.)
    bundle.base.goal_norm.update(torch.randn(20, 7) + 50.)
    assert not bundle.base.training
    assert all(not p.requires_grad and p.grad is None for p in bundle.base.parameters())
    assert all(torch.equal(before[key], value) for key, value in bundle.base.state_dict().items())
    assert torch.equal(bundle(history, goal, task="tracking"), expected)
    with pytest.raises(ValueError, match="Unknown|unknown"):
        bundle(history, goal, task="stairs")


def test_checkpoint_restores_policy_and_rejects_unknown_contracts(tmp_path):
    torch.manual_seed(939)
    base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    bundle = RetainedSkillPolicy(base, SkateboardStudent(hidden=(12, 8))).eval()
    snapshot = {
        "schema": "pera_retained_skills_v1",
        "model_spec": {"hidden": [12, 8], "observation_schema": OBS_SCHEMA,
                       "contact_model": "husky_capsules", "action_space": "native_normalized_29"},
        "policy": bundle.state_dict(),
    }
    checkpoint = tmp_path / "candidate.pt"
    torch.save(snapshot, checkpoint)
    loaded, _ = RetainedSkillPolicy.load(checkpoint)
    obs = torch.randn(3, 395)
    assert torch.equal(bundle(obs, task="skateboard"), loaded(obs, task="skateboard"))
    assert not loaded.training and not loaded.base.training
    for field in ("observation_schema", "contact_model", "action_space"):
        original = snapshot["model_spec"][field]
        snapshot["model_spec"][field] = "unsupported-contract"
        torch.save(snapshot, checkpoint)
        with pytest.raises(ValueError):
            RetainedSkillPolicy.load(checkpoint)
        snapshot["model_spec"][field] = original


def test_certified_checkpoint_freezes_old_skill_including_statistics(tmp_path):
    torch.manual_seed(948)
    base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    student = SkateboardStudent(hidden=(12, 8))
    student.initialize_statistics(torch.randn(8, 395), torch.randn(8, 29))
    bundle = RetainedSkillPolicy(base, student).eval()
    snapshot = {
        "schema": "pera_retained_skills_v1",
        "model_spec": {"hidden": [12, 8], "observation_schema": OBS_SCHEMA,
                       "contact_model": "husky_capsules", "action_space": "native_normalized_29"},
        "policy": bundle.state_dict(), "certified_skills": ["skateboard"],
        "certified_skill_sha256": {"skateboard": state_digest(student.state_dict())},
    }
    checkpoint = tmp_path / "certified.pt"
    torch.save(snapshot, checkpoint)
    loaded, _ = RetainedSkillPolicy.load(checkpoint)
    before = {key: value.clone() for key, value in loaded.state_dict().items()}
    assert loaded.certified_skills == {"skateboard"}
    assert all(not p.requires_grad for p in loaded.parameters())
    loaded.train()
    assert not loaded.base.training and not loaded.skateboard.training
    with pytest.raises(RuntimeError):
        loaded.skateboard.initialize_statistics(torch.randn(8, 395) + 50., torch.randn(8, 29) + 50.)
    # Training a fresh attached module must not alter either certified route.
    new_task = torch.nn.Linear(29, 4)
    optimizer = torch.optim.Adam(new_task.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        new_task(loaded(torch.randn(4, 395), task="skateboard")).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(value, before[key]) for key, value in loaded.state_dict().items())
    assert state_digest(loaded.skateboard.state_dict()) == snapshot["certified_skill_sha256"]["skateboard"]
    # Tampering with even a normalization buffer must invalidate certification.
    snapshot["policy"]["skateboard.obs_mean"][0] += .01
    torch.save(snapshot, checkpoint)
    with pytest.raises(ValueError, match="parameters/statistics changed"):
        RetainedSkillPolicy.load(checkpoint)
