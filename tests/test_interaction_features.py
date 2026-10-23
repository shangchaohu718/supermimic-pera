"""CPU contracts for native observations and prediction-dependent interaction features."""

import math

import torch

from unisupermimic.interaction.features import (
    context_features, native_proprio, reward_and_done, state_from_observation, terrain_height,
)


def state(batch=5):
    return (
        torch.tensor([[0., 0., 0.8]]).repeat(batch, 1), torch.eye(3).repeat(batch, 1, 1),
        torch.zeros(batch, 3), torch.zeros(batch, 3), torch.zeros(batch, 29), torch.zeros(batch, 29),
    )


def test_mujoco_local_angular_velocity_is_rotated_to_world():
    qpos = torch.zeros(36)
    qpos[2] = 0.8
    qpos[3], qpos[6] = math.sqrt(0.5), math.sqrt(0.5)
    qvel = torch.zeros(35)
    qvel[0], qvel[3] = 2, 1
    converted = state_from_observation({"robot_qpos": qpos, "robot_qvel": qvel})
    assert converted[0].shape == (1, 3)
    assert torch.equal(converted[2], torch.tensor([[2., 0., 0.]]))
    assert torch.allclose(converted[3], torch.tensor([[0., 1., 0.]]), atol=2e-7)


def test_native_99_observation_is_exact_manual_layout():
    native = list(state(2))
    native[4] = torch.arange(58).reshape(2, 29).float() / 100
    native[5] = torch.ones(2, 29) * 0.5
    native[3] = torch.tensor([[1., 2., 3.]]).repeat(2, 1)
    default = torch.ones(29) * 0.1
    previous = torch.ones(2, 29) * -0.2
    actual = native_proprio(tuple(native), previous, state(2), default)
    expected = torch.cat((
        native[4] - default, native[5], native[3],
        torch.tensor([[0., 0., -1.]]).repeat(2, 1), previous,
        torch.tensor([[1., 0., 0., 1., 0., 0.]]).repeat(2, 1),
    ), dim=-1)
    assert actual.shape == (2, 99)
    assert torch.equal(actual, expected)


def test_context_contract_and_object_presence_flags():
    robot = state()
    objects = torch.zeros(5, 27)
    objects[:, 3] = 1
    objects[:, 0] = 0.4
    targets = torch.tensor([[1., 0., 0.5]]).repeat(5, 1)
    features = context_features(robot, objects, torch.arange(5), targets, torch.full((5,), 0.8), {})
    assert features.shape == (5, 64)
    assert torch.equal(features[:, :5], torch.eye(5))
    assert torch.equal(features[:, 5], torch.tensor([0., 0., 1., 1., 1.]))
    assert torch.equal(features[:, 36], torch.tensor([0., 0.5, 0., 0.5, 0.5]))
    assert torch.equal(features[:2, 6:33], torch.zeros(2, 27))
    assert torch.equal(features[:, 33:36], targets - robot[0])
    assert torch.all(features[:, 46] == 0.8) and torch.all(features[:, 47] == 0.5)


def test_terrain_requeries_when_predicted_root_crosses_step():
    first = state(1)
    next_robot = (first[0] + torch.tensor([[0.75, 0., 0.]]), *first[1:])
    boxes = {1: torch.tensor([[0.75, 0., 0.08, 0.16, 0.6, 0.08]])}
    task = torch.tensor([1])
    obj = torch.zeros(1, 27)
    target = torch.zeros(1, 3)
    initial = context_features(first, obj, task, target, torch.tensor([0.8]), boxes)
    advanced = context_features(next_robot, obj, task, target, torch.tensor([0.8]), boxes)
    assert initial[0, 41] == 0
    assert advanced[0, 41] == 0.16
    assert terrain_height(next_robot[0][:, :2], torch.tensor([0]), boxes)[0] == 0


def test_dense_reward_gradients_and_fall_probability_are_finite():
    robot = list(state())
    robot[0] = (robot[0] + 0.01).requires_grad_(True)
    obj = torch.zeros(5, 27)
    obj[:, 3] = 1
    obj[:, 0], obj[:, 2] = 0.3, 0.18
    obj.requires_grad_(True)
    action = torch.full((5, 29), 0.1, requires_grad=True)
    reward, done = reward_and_done(tuple(robot), obj, action, torch.zeros_like(action), state(),
                                   torch.tensor([[0.5, 0., 0.3]]).repeat(5, 1), torch.arange(5), {})
    assert reward.shape == done.shape == (5,)
    assert torch.isfinite(reward).all() and torch.all((done >= 0) & (done <= 1))
    (reward.sum() - done.sum()).backward()
    assert torch.isfinite(robot[0].grad).all() and robot[0].grad.abs().sum() > 0
    assert torch.isfinite(obj.grad).all() and obj.grad[2:4].abs().sum() > 0
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0


def test_board_support_and_surface_velocity_are_relative_to_moving_board():
    robot = state(1)
    obj = torch.zeros(1, 27)
    obj[:, 3] = 1
    obj[:, 2] = 0.1
    obj[:, 14] = 0.5
    moving_robot = (robot[0], robot[1], torch.tensor([[0.5, 0., 0.]]), *robot[3:])
    args = (obj, torch.zeros(1, 29), torch.zeros(1, 29), robot, torch.zeros(1, 3), torch.tensor([4]), {})
    matched, _ = reward_and_done(moving_robot, *args)
    slipping, _ = reward_and_done(robot, *args)
    assert matched > slipping
