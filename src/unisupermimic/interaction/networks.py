"""One shared interaction policy with an exact native tracking fallback.

The native actor (including its normalization statistics) remains frozen. A
small, zero-initialized residual learns from all interaction tasks together.
``interaction_mask == 0`` selects the native action exactly; this is an action
retention guarantee on identical inputs, not a guarantee about changed physics.
Old interaction skills require balanced replay and physical evaluation gates.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from unisupermimic.rl.algos.supermimic.networks import Policy, mlp, run_mlp


class RetainedInteractionPolicy(nn.Module):
    """Frozen native actor plus a bounded, context-conditioned shared residual.

    Args:
        base_policy: Native SuperMimic actor; copied so its owner can still use it.
        context_dim: Explicit physical interaction features, shared by every task.
        hidden: Residual network widths (the native actor widths stay unchanged).
        residual_scale: Maximum absolute residual in native normalized actions.

    Observations have shape ``(B, hist, obs_dim)``, goals ``(B, goal_dim)``,
    context ``(B, context_dim)`` and mask ``(B,)`` or ``(B, 1)``. Masks must be
    zero or one: a tracking sample cannot accidentally receive a soft residual.
    Frozen parameters still permit derivatives through observations during
    differentiable world-model rollouts.
    """

    def __init__(
        self,
        base_policy: Policy,
        context_dim: int = 64,
        hidden: Sequence[int] = (128, 128),
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or residual_scale <= 0:
            raise ValueError("context_dim and residual_scale must be positive")
        self.base = copy.deepcopy(base_policy).requires_grad_(False)
        self.base.eval()
        # RunningNorm.update is explicit, independent of train/eval mode.
        self.base.obs_norm.max_count = 0.0
        self.base.goal_norm.max_count = 0.0
        self.hist = base_policy.hist
        self.obs_dim = base_policy.obs_norm.mean.numel()
        self.goal_dim = base_policy.goal_norm.mean.numel()
        self.num_joints = base_policy.dec[-1].out_features
        self.context_dim = context_dim
        self.residual_scale = float(residual_scale)
        self.hidden = tuple(hidden)
        latent_dim = base_policy.enc[-1].out_features
        self.residual = mlp(self.obs_dim + latent_dim + context_dim, hidden, self.num_joints)
        self.residual.to(next(self.base.parameters()))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @classmethod
    def from_native_checkpoint(
        cls,
        path: str | Path,
        *,
        policy_kwargs: Mapping[str, object] | None = None,
        context_dim: int = 64,
        hidden: Sequence[int] = (128, 128),
        residual_scale: float = 0.25,
    ) -> RetainedInteractionPolicy:
        """Read a local native checkpoint on CPU; never downloads or selects CUDA."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or "pi" not in checkpoint:
            raise ValueError("Expected native checkpoint containing 'pi'")
        kwargs = dict(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
        if policy_kwargs is not None:
            kwargs.update(policy_kwargs)
        base = Policy(**kwargs)
        base.load_state_dict(checkpoint["pi"], strict=True)
        return cls(base, context_dim=context_dim, hidden=hidden, residual_scale=residual_scale)

    def train(self, mode: bool = True) -> RetainedInteractionPolicy:
        super().train(mode)
        self.base.eval()
        return self

    def interaction_parameters(self) -> Iterator[nn.Parameter]:
        """Only these parameters should be passed to the interaction optimizer."""
        return self.residual.parameters()

    def forward(
        self,
        obs_hist: torch.Tensor,
        goal: torch.Tensor,
        context: torch.Tensor,
        interaction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if obs_hist.ndim != 3 or obs_hist.shape[1:] != (self.hist, self.obs_dim):
            raise ValueError(f"obs_hist must have shape (B, {self.hist}, {self.obs_dim})")
        batch = obs_hist.shape[0]
        if goal.shape != (batch, self.goal_dim):
            raise ValueError(f"goal must have shape (B, {self.goal_dim})")
        if context.shape != (batch, self.context_dim):
            raise ValueError(f"context must have shape (B, {self.context_dim})")
        if interaction_mask.shape not in ((batch,), (batch, 1)):
            raise ValueError("interaction_mask must have shape (B,) or (B, 1)")
        if not bool(torch.all((interaction_mask == 0) | (interaction_mask == 1))):
            raise ValueError("interaction_mask must contain only zero or one")
        native_action = self.base(obs_hist, goal)
        latent = torch.tanh(run_mlp(self.base.enc, self.base.goal_norm(goal)))
        latest_obs = self.base.obs_norm(obs_hist[:, -1])
        residual_input = torch.cat((latest_obs, latent, context), dim=-1)
        delta = self.residual_scale * torch.tanh(run_mlp(self.residual, residual_input))
        # torch.where, unlike multiplying by zero, retains native values even
        # when an irrelevant context contains non-finite values.
        return torch.where(
            interaction_mask.reshape(batch, 1).bool(), native_action + delta, native_action
        )


def replay_action_loss(
    predicted_action: torch.Tensor,
    anchor_action: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Behavioral anchor loss on stored old-task actions; targets never train.

    Use alongside balanced mixed-task dynamics/reward replay, not in place of it.
    Anchors should come from an accepted earlier checkpoint on stored inputs.
    """
    if predicted_action.ndim != 2 or anchor_action.shape != predicted_action.shape:
        raise ValueError("predicted_action and anchor_action must have identical (B, A) shape")
    if predicted_action.shape[0] == 0:
        raise ValueError("replay batch must not be empty")
    error = (predicted_action - anchor_action.detach()).square().mean(-1)
    if sample_weight is None:
        return error.mean()
    if sample_weight.shape != error.shape:
        raise ValueError("sample_weight must have shape (B,)")
    if not bool(torch.all(torch.isfinite(sample_weight) & (sample_weight >= 0))):
        raise ValueError("sample_weight must be finite and nonnegative")
    total_weight = sample_weight.sum()
    if not bool(total_weight > 0):
        raise ValueError("sample_weight must contain a positive weight")
    return (error * sample_weight).sum() / total_weight


@torch.no_grad()
def evaluate_retention(
    predicted_action: torch.Tensor,
    anchor_action: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    max_action_error: float = 0.05,
) -> dict[str, object]:
    """Report a per-task action regression gate; physical success is separate.

    All old tasks must appear in the caller's evaluation batch. This gate only
    covers supplied samples; passing it alone does not prove retained skills.
    """
    if predicted_action.ndim != 2 or anchor_action.shape != predicted_action.shape:
        raise ValueError("predicted_action and anchor_action must have identical (B, A) shape")
    if task_ids.shape != (predicted_action.shape[0],) or not task_ids.numel():
        raise ValueError("task_ids must have nonempty shape (B,)")
    if max_action_error < 0:
        raise ValueError("max_action_error must be nonnegative")
    delta = (predicted_action - anchor_action).abs()
    per_task = {}
    for task_id in task_ids.unique().tolist():
        errors = delta[task_ids == task_id]
        finite = bool(torch.isfinite(errors).all())
        max_error = float(errors.max()) if finite else float("inf")
        per_task[str(task_id)] = {
            "samples": errors.shape[0],
            "max_action_error": max_error,
            "mean_action_error": float(errors.mean()) if finite else float("inf"),
            "passed": finite and max_error <= max_action_error,
        }
    return {
        "passed": all(result["passed"] for result in per_task.values()),
        "threshold": max_action_error,
        "per_task": per_task,
        "scope": "supplied action samples; physical evaluation is additionally required",
    }


__all__ = ["RetainedInteractionPolicy", "evaluate_retention", "replay_action_loss"]
