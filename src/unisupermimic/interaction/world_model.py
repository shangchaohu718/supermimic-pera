"""Joint robot/object dynamics with a frozen native SuperMimic foundation.

The object state is ``qpos[14], qvel[13]``: a free root (xyz, wxyz),
seven padded articulated positions, world linear velocity, root-local angular
velocity, and seven articulated velocities. Boxes use zero articulated slots.
The shared residual predicts real transition corrections; it does not advance
objects along a reference trajectory. Callers must refresh terrain/context from
the predicted state at each imagined step.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from unisupermimic.rl.algos.supermimic.geometry import invariant_features, so3_exp, yaw_mat
from unisupermimic.rl.algos.supermimic.networks import RunningNorm, WorldModel, mlp, run_mlp
from unisupermimic.rl.contracts.supermimic import State


class JointInteractionWorldModel(nn.Module):
    """Learn one small joint dynamics correction for all interaction tasks.

    ``interaction_mask == 0`` selects the exact frozen native robot prediction
    and preserves the supplied object state, even after residual training.
    ``object_mask`` defaults to context channel 5 (object presence), independently
    of the interaction mask so stairs can change robot dynamics without an
    object. Native context is already normalized and defaults to 33 zeros.

    Input statistics belong only to the correction network. Fixed physical
    output scales deliberately have no learned mean: updating normalization
    cannot turn the initially zero correction into a nonzero one.
    """

    def __init__(
        self,
        base_model: WorldModel,
        context_dim: int = 64,
        object_dim: int = 27,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if base_model.num_joints != 29 or base_model.context_dim != 33:
            raise ValueError("The interaction contract requires the native 29-joint, 33-context WM")
        if context_dim < 6 or object_dim != 27 or not hidden:
            raise ValueError("Expected context_dim >= 6, object_dim == 27, and nonempty hidden")
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self.context_dim = context_dim
        self.object_dim = object_dim
        self.robot_dim = base_model.out_dim
        self.in_dim = 71 + 29 + context_dim + object_dim
        self.out_dim = self.robot_dim + object_dim
        self.input_norm = RunningNorm(self.in_dim, eps=1e-2)
        self.correction = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.correction[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        robot_scale = torch.tensor(
            [0.03] * 3 + [0.05] * 3 + [0.20] * 3 + [0.5] * 3 + [0.05] * 29 + [1.0] * 29
        )
        object_scale = torch.tensor(
            [0.03] * 3 + [0.05] * 4 + [0.05] * 7 + [0.20] * 3 + [0.5] * 3 + [1.0] * 7
        )
        self.register_buffer("output_scale", torch.cat((robot_scale, object_scale)))

    def train(self, mode: bool = True) -> JointInteractionWorldModel:
        super().train(mode)
        self.base_model.eval()
        return self

    def inputs(
        self, state: State, object_state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if object_state.shape[-1] != self.object_dim:
            raise ValueError(f"object_state must have {self.object_dim} features")
        if context.shape[-1] != self.context_dim or action.shape[-1] != 29:
            raise ValueError("Incorrect interaction context or action feature count")
        return torch.cat((invariant_features(state), action, context, object_state), dim=-1)

    @staticmethod
    def _mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if mask.ndim == like.ndim and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        return torch.broadcast_to(mask.to(device=like.device, dtype=torch.bool), like.shape[:-1])

    @staticmethod
    def _unit_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
        length = quaternion.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(quaternion)
        identity[..., 0] = 1
        return torch.where(length > 1e-8, quaternion / length.clamp_min(1e-8), identity)

    @staticmethod
    def _apply_robot_increment(state: State, delta: torch.Tensor) -> State:
        position, rotation, linear, angular, joint_position, joint_velocity = state
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), dim=-1)
        return (
            position + (yaw_mat(rotation) @ dp.unsqueeze(-1)).squeeze(-1),
            rotation @ so3_exp(dr),
            linear + (rotation @ dv.unsqueeze(-1)).squeeze(-1),
            angular + (rotation @ dw.unsqueeze(-1)).squeeze(-1),
            joint_position + dq,
            joint_velocity + dqd,
        )

    @torch.no_grad()
    def update_normalization(
        self,
        state: State,
        object_state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        interaction_mask: torch.Tensor | None = None,
    ) -> None:
        features = self.inputs(state, object_state, action, context)
        if interaction_mask is not None:
            features = features[self._mask(interaction_mask, object_state)]
        if features.numel():
            self.input_norm.update(features)

    def step(
        self,
        state: State,
        object_state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        interaction_mask: torch.Tensor,
        *,
        object_mask: torch.Tensor | None = None,
        base_context: torch.Tensor | None = None,
    ) -> tuple[State, torch.Tensor]:
        features = self.inputs(state, object_state, action, context)
        active = self._mask(interaction_mask, object_state)
        present = self._mask(context[..., 5] if object_mask is None else object_mask, object_state)
        if base_context is None:
            base_context = action.new_zeros((*action.shape[:-1], self.base_model.context_dim))
        # Freezing parameters must not detach this calculation: imagined policy
        # optimization needs the native derivative with respect to its action.
        native = self.base_model.step(state, action, base_context)
        delta = run_mlp(self.correction, self.input_norm(features)) * self.output_scale
        candidate = self._apply_robot_increment(native, delta[..., : self.robot_dim])
        robot = tuple(
            torch.where(active.reshape(active.shape + (1,) * (new.ndim - active.ndim)), new, old)
            for new, old in zip(candidate, native, strict=True)
        )
        obj_delta = delta[..., self.robot_dim :]
        object_candidate = object_state + obj_delta
        quaternion = self._unit_quaternion(object_candidate[..., 3:7])
        object_candidate = torch.cat(
            (object_candidate[..., :3], quaternion, object_candidate[..., 7:]), dim=-1
        )
        obj = torch.where((active & present).unsqueeze(-1), object_candidate, object_state)
        return robot, obj

    def loss(
        self,
        state: State,
        object_state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        interaction_mask: torch.Tensor,
        next_state: State,
        next_object_state: torch.Tensor,
        *,
        object_mask: torch.Tensor | None = None,
        base_context: torch.Tensor | None = None,
        update_norm: bool = False,
    ) -> torch.Tensor:
        """Dimension-scaled one-step transition loss, excluding absent objects.

        This is supervised on simulator transitions, not reference animation.
        Quaternion targets are hemisphere aligned because q and -q describe the
        same orientation. Multi-step callers can repeatedly invoke ``step`` and
        use refreshed context to accumulate this same physical state error.
        """
        if update_norm:
            self.update_normalization(state, object_state, action, context, interaction_mask)
        predicted, predicted_object = self.step(
            state, object_state, action, context, interaction_mask,
            object_mask=object_mask, base_context=base_context,
        )
        active = self._mask(interaction_mask, object_state).to(action.dtype)
        present = self._mask(context[..., 5] if object_mask is None else object_mask, object_state)
        robot_error = self.base_model.state_error(predicted, next_state)
        robot_loss = (robot_error / self.output_scale[: self.robot_dim]).square().mean(-1)
        target_quat = self._unit_quaternion(next_object_state[..., 3:7])
        sign = torch.where((predicted_object[..., 3:7] * target_quat).sum(-1, keepdim=True) < 0, -1, 1)
        object_error = torch.cat((
            predicted_object[..., :3] - next_object_state[..., :3],
            predicted_object[..., 3:7] - sign * target_quat,
            predicted_object[..., 7:] - next_object_state[..., 7:],
        ), dim=-1)
        object_loss = (object_error / self.output_scale[self.robot_dim :]).square().mean(-1)
        robot_mean = (robot_loss * active).sum() / active.sum().clamp_min(1)
        object_weights = active * present.to(active.dtype)
        object_mean = (object_loss * object_weights).sum() / object_weights.sum().clamp_min(1)
        return robot_mean + object_mean


__all__ = ["JointInteractionWorldModel"]
