"""The independent physical audit detects changed tracking outputs, not just keys."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.interaction.skateboard_scene import ROBOT_SCENE
from unisupermimic.interaction.skateboard_skill import RetainedSkillPolicy, SkateboardStudent
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

spec = importlib.util.spec_from_file_location(
    'retention_audit', Path(__file__).resolve().parents[1] / 'tools/check_retained_skill_policy.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_state_comparison_detects_missing_and_changed_buffers():
    original = {'weight': torch.ones(2), 'norm': torch.zeros(1)}
    assert audit.mismatched_tensors(original, {k: v.clone() for k, v in original.items()}) == []
    assert audit.mismatched_tensors(original, {'weight': torch.zeros(2)}) == ['norm', 'weight']


@pytest.mark.skipif(not (ROBOT_SCENE.parent / 'assets/pelvis.STL').is_file(), reason='Local assets absent')
def test_real_physics_audit_passes_exact_copy_and_rejects_modified_actions():
    torch.set_num_threads(2)
    env = audit.G1InteractionCPU('tracking')
    q = torch.tensor(env.default_qpos, dtype=torch.float32)
    clip = ReferenceClip('synthetic_static_audit_only', G1_JOINT_NAMES,
                         q[:3].repeat(21, 1), q[3:7].repeat(21, 1), q[7:].repeat(21, 1))
    base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29, hidden=(16, 8)).eval()
    bundle = RetainedSkillPolicy(base, SkateboardStudent(hidden=(8,))).eval()
    result = audit.paired_rollout(base, bundle, clip, steps=20)
    assert result['passed_equivalence']
    assert result['simulated_steps'] == 20
    assert result['maximum_absolute_errors'] == {'action': 0., 'qpos': 0., 'qvel': 0.}

    class ChangedTracking(torch.nn.Module):
        def forward(self, observations, goal, *, task):
            return bundle(observations, goal, task=task) + .001

    failed = audit.paired_rollout(base, ChangedTracking().eval(), clip, steps=20)
    assert not failed['passed_equivalence']
    assert not failed['per_step_bitwise_equal']['action']
    assert not failed['per_step_bitwise_equal']['qpos']
    assert failed['first_mismatch_frame'] == 0
    assert failed['maximum_absolute_errors']['action'] > 0
