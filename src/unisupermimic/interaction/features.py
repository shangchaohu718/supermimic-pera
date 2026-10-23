"""Native observations and small, shared interaction features for the prototype.

Terrain is queried from predicted positions on every call, including imagination
steps. Rewards are dense learning signals, not completion metrics: box transport
requires independent real contact/lift checks and stairs/board skills require
full simulator rollouts. No contact force or task success is inferred here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, quat_to_mat, rot_angle_sq
from unisupermimic.rl.contracts.supermimic import State


def state_from_observation(observation: Mapping[str, Any], device: str | torch.device = "cpu") -> State:
    """Convert MuJoCo robot qpos36/qvel35 into the native batched state.

    MuJoCo free-joint linear velocity is world-frame and angular velocity is
    body-local. Native SuperMimic stores both velocities in the world frame.
    """
    qpos = torch.as_tensor(observation["robot_qpos"], dtype=torch.float32, device=device)
    qvel = torch.as_tensor(observation["robot_qvel"], dtype=torch.float32, device=device)
    if qpos.ndim == 1:
        qpos, qvel = qpos.unsqueeze(0), qvel.unsqueeze(0)
    if qpos.shape[-1] != 36 or qvel.shape[-1] != 35 or qpos.shape[:-1] != qvel.shape[:-1]:
        raise ValueError("Expected matching robot_qpos[...,36] and robot_qvel[...,35]")
    rotation = quat_to_mat(qpos[..., 3:7])
    return (
        qpos[..., :3], rotation, qvel[..., :3],
        (rotation @ qvel[..., 3:6].unsqueeze(-1)).squeeze(-1),
        qpos[..., 7:36], qvel[..., 6:35],
    )


def native_proprio(state: State, previous_action: torch.Tensor, reference_state: State,
                   default_q: torch.Tensor) -> torch.Tensor:
    """Exact noise-free native 99D observation with zero actuator-offset context."""
    _, rotation, _, angular, joint_position, joint_velocity = state
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat((
        joint_position - default_q,
        joint_velocity,
        (rotation_t @ angular.unsqueeze(-1)).squeeze(-1),
        -rotation[..., 2, :],
        previous_action,
        mat_to_6d(rotation_t @ reference_state[1]),
    ), dim=-1)


def terrain_height(xy: torch.Tensor, task_ids: torch.Tensor,
                   terrain_boxes_per_task: Mapping[int, Any]) -> torch.Tensor:
    """Height of world-aligned boxes (center xyz, half-size xyz), with floor z=0.

    Queries may have shape Bx2 or BxNx2. The true box-height field is piecewise
    constant: the output changes when the predicted query crosses a step edge,
    with zero position gradient inside a tread. Dynamic objects are excluded.
    """
    if xy.shape[0] != task_ids.shape[0] or xy.shape[-1] != 2:
        raise ValueError("Terrain queries require matching batch sizes and final dimension 2")
    heights = xy.new_zeros(xy.shape[:-1])
    for task_id, raw_boxes in terrain_boxes_per_task.items():
        boxes = torch.as_tensor(raw_boxes, device=xy.device, dtype=xy.dtype).reshape(-1, 6)
        if not boxes.numel():
            continue
        inside = (xy.unsqueeze(-2) - boxes[:, :2]).abs() <= boxes[:, 3:5]
        covered = inside.all(-1)
        tops = boxes[:, 2] + boxes[:, 5]
        task_heights = torch.where(covered, tops, torch.zeros_like(tops)).amax(-1)
        selected = (task_ids == int(task_id)).reshape((xy.shape[0],) + (1,) * (xy.ndim - 2))
        heights = torch.where(selected, task_heights.clamp_min(0), heights)
    return heights


def context_features(
    state: State,
    object_state: torch.Tensor,
    task_ids: torch.Tensor,
    targets: torch.Tensor,
    target_heights: torch.Tensor,
    terrain_boxes_per_task: Mapping[int, Any],
) -> torch.Tensor:
    """Return Bx64 features with a 3x3 terrain query at the current predicted root.

    Layout: task one-hot[0:5], object-present[5], relative object state[6:33],
    relative target xyz[33:36], desired x-speed[36], heights[37:46], nominal
    robot reference height[46], object target height[47], reserved zeros[48:64].
    ``target_heights`` is B (robot height) or Bx2 (robot, object height).
    """
    position = state[0]
    batch = position.shape[0]
    task_ids = task_ids.to(device=position.device, dtype=torch.long)
    if object_state.shape != (batch, 27) or targets.shape != (batch, 3):
        raise ValueError("Expected object_state Bx27 and targets Bx3")
    one_hot = F.one_hot(task_ids, num_classes=5).to(position.dtype)
    present = ((task_ids == 2) | (task_ids == 3) | (task_ids == 4)).to(position.dtype)
    relative_object = torch.cat((object_state[:, :3] - position, object_state[:, 3:]), dim=-1)
    relative_object = relative_object * present.unsqueeze(-1)
    relative_target = targets - position
    speed = ((task_ids == 1) | (task_ids == 3) | (task_ids == 4)).to(position.dtype) * 0.5
    offsets = position.new_tensor([-0.3, 0.0, 0.3])
    grid_x, grid_y = torch.meshgrid(offsets, offsets, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(9, 2)
    heights = terrain_height(position[:, None, :2] + grid, task_ids, terrain_boxes_per_task)
    target_heights = torch.as_tensor(target_heights, dtype=position.dtype, device=position.device)
    if target_heights.ndim == 2 and target_heights.shape == (batch, 2):
        robot_height, object_height = target_heights[:, 0], target_heights[:, 1]
    else:
        robot_height = torch.broadcast_to(target_heights.reshape(-1), (batch,))
        object_height = targets[:, 2]
    return torch.cat((
        one_hot, present[:, None], relative_object, relative_target, speed[:, None],
        heights, robot_height[:, None], object_height[:, None], position.new_zeros(batch, 16),
    ), dim=-1)


def reward_and_done(
    state: State,
    object_state: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    reference_state: State,
    object_target: torch.Tensor,
    task_ids: torch.Tensor,
    terrain_boxes: Mapping[int, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prototype dense shaping and differentiable fall probability (both B).

    Robot reference height is interpreted above static support on terrain
    tasks. The board terms reward root balance and matching the board's moving
    surface velocity; actual foot contacts remain simulator evaluation metrics.
    Box shaping only rewards free-object displacement toward the target; it
    cannot by itself certify grasping, lifting, or carrying.
    """
    position, rotation, velocity, _, joint_position, _ = state
    task_ids = task_ids.to(device=position.device, dtype=torch.long)
    static_height = terrain_height(position[:, :2], task_ids, terrain_boxes)
    terrain_task = (task_ids == 1) | (task_ids == 3)
    board_task = task_ids == 4
    box_task = (task_ids == 2) | (task_ids == 3)
    nominal_height = reference_state[0][:, 2]
    desired_height = nominal_height + torch.where(terrain_task, static_height, torch.zeros_like(static_height))
    joint_match = torch.exp(-(joint_position - reference_state[4]).square().mean(-1) / 0.25**2)
    orientation_match = torch.exp(-rot_angle_sq(rotation, reference_state[1]) / 0.4**2)
    height_match = torch.exp(-(position[:, 2] - desired_height).square() / 0.25**2)
    velocity_target = reference_state[2][:, :2]
    commanded_velocity = velocity_target.new_tensor([0.5, 0.0]).expand_as(velocity_target)
    velocity_target = torch.where((terrain_task | board_task)[:, None], commanded_velocity, velocity_target)
    velocity_match = torch.exp(-(velocity[:, :2] - velocity_target).square().sum(-1) / 0.7**2)
    reward = 0.8 * joint_match + 0.4 * orientation_match + 0.4 * height_match + 0.3 * velocity_match
    reward = reward - 0.005 * (action - previous_action).square().mean(-1) - 0.0001 * action.square().mean(-1)
    object_match = torch.exp(-(object_state[:, :3] - object_target).square().sum(-1) / 0.2**2)
    reward = reward + box_task.to(reward.dtype) * 0.8 * object_match

    board_rotation = quat_to_mat(object_state[:, 3:7])
    board_omega = (board_rotation @ object_state[:, 17:20].unsqueeze(-1)).squeeze(-1)
    # HUSKY tilt hinge is at the root origin with local axis (-1, 0, 0).
    # Include its velocity when querying the moving support surface.
    tilt_local = torch.stack((-object_state[:, 20], torch.zeros_like(object_state[:, 20]),
                              torch.zeros_like(object_state[:, 20])), -1)
    board_omega = board_omega + (board_rotation @ tilt_local.unsqueeze(-1)).squeeze(-1)
    board_relative = position - object_state[:, :3]
    surface_velocity = object_state[:, 14:17] + torch.cross(board_omega, board_relative, dim=-1)
    center_match = torch.exp(-board_relative[:, :2].square().sum(-1) / 0.3**2)
    surface_match = torch.exp(-(velocity[:, :2] - surface_velocity[:, :2]).square().sum(-1) / 0.7**2)
    reward = reward + board_task.to(reward.dtype) * (0.4 * center_match + 0.3 * surface_match)

    support_height = torch.where(board_task, object_state[:, 2], static_height)
    above_support = position[:, 2] - support_height
    height_failure = torch.sigmoid((0.35 - above_support) / 0.06)
    tilt_failure = torch.sigmoid((0.45 - rotation[:, 2, 2]) / 0.08)
    soft_done = 1 - (1 - height_failure) * (1 - tilt_failure)
    return reward, soft_done


__all__ = ["state_from_observation", "native_proprio", "terrain_height", "context_features", "reward_and_done"]
