"""CPU checks for exact tracking retention and shared interaction learning."""

import pytest
import torch

from unisupermimic.interaction.networks import (
    RetainedInteractionPolicy,
    evaluate_retention,
    replay_action_loss,
)
from unisupermimic.rl.algos.supermimic.networks import Policy


def setup_policy():
    torch.manual_seed(7)
    base = Policy(obs_dim=5, goal_dim=7, hist=3, z_dim=2, num_joints=4, hidden=(12, 8))
    base.obs_norm.update(torch.randn(100, 5))
    base.goal_norm.update(torch.randn(100, 7))
    policy = RetainedInteractionPolicy(base, context_dim=6, hidden=(16, 8))
    obs, goal, context = torch.randn(4, 3, 5), torch.randn(4, 7), torch.randn(4, 6)
    return base, policy, obs, goal, context


def test_tracking_is_exact_after_shared_residual_updates_and_normalizer_calls():
    base, policy, obs, goal, context = setup_policy()
    original = base(obs, goal).detach()
    before = {key: value.clone() for key, value in policy.base.state_dict().items()}
    assert torch.equal(policy(obs, goal, context, torch.ones(4)), original)
    optimizer = torch.optim.Adam(policy.interaction_parameters(), lr=0.02)
    policy.train()
    assert not policy.base.training
    for _ in range(3):
        optimizer.zero_grad()
        loss = (policy(obs, goal, context, torch.ones(4)) - 0.5).square().mean()
        loss.backward()
        optimizer.step()
    policy.base.obs_norm.update(torch.randn(20, 5) + 100)
    policy.base.goal_norm.update(torch.randn(20, 7) + 100)
    assert torch.equal(policy(obs, goal, context, torch.zeros(4)), original)
    assert not torch.equal(policy(obs, goal, context, torch.ones(4)), original)
    assert all(torch.equal(value, before[key]) for key, value in policy.base.state_dict().items())
    assert all(parameter.grad is None for parameter in policy.base.parameters())
    assert all(parameter.grad is not None for parameter in policy.residual.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in policy.residual.parameters())
    delta = policy(obs, goal, context, torch.ones(4)) - original
    assert delta.abs().max() <= policy.residual_scale


def test_mixed_batch_mask_and_input_gradient_flow():
    base, policy, obs, goal, context = setup_policy()
    with torch.no_grad():
        policy.residual[-1].bias.fill_(1)
    obs.requires_grad_()
    actions = policy(obs, goal, context, torch.tensor([[0], [1], [0], [1]]))
    native = base(obs, goal)
    assert torch.equal(actions[[0, 2]], native[[0, 2]])
    assert not torch.equal(actions[[1, 3]], native[[1, 3]])
    actions.square().sum().backward()
    assert obs.grad is not None and obs.grad.abs().sum() > 0


@pytest.mark.parametrize("bad_part", ["obs", "goal", "context", "mask_shape", "mask_value"])
def test_rejects_wrong_input_contract(bad_part):
    _, policy, obs, goal, context = setup_policy()
    mask = torch.zeros(4)
    if bad_part == "obs":
        obs = obs.flatten(1)
    elif bad_part == "goal":
        goal = goal[:, :-1]
    elif bad_part == "context":
        context = context[:, :-1]
    elif bad_part == "mask_shape":
        mask = torch.zeros(4, 2)
    else:
        mask.fill_(0.5)
    with pytest.raises(ValueError):
        policy(obs, goal, context, mask)


def test_local_checkpoint_retains_normalization_and_native_action(tmp_path):
    base, _, obs, goal, context = setup_policy()
    path = tmp_path / "last.pt"
    torch.save({"pi": base.state_dict(), "iteration": 7000}, path)
    loaded = RetainedInteractionPolicy.from_native_checkpoint(
        path,
        policy_kwargs=dict(obs_dim=5, goal_dim=7, hist=3, z_dim=2, num_joints=4, hidden=(12, 8)),
        context_dim=6,
    )
    assert torch.equal(loaded(obs, goal, context, torch.zeros(4)), base(obs, goal))
    assert next(loaded.parameters()).device.type == "cpu"


def test_replay_targets_detached_and_per_task_gate_blocks_regression():
    predicted = torch.tensor([[0.1, 0.1], [0.0, 0.0]], requires_grad=True)
    anchor = torch.zeros_like(predicted, requires_grad=True)
    loss = replay_action_loss(predicted, anchor, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(loss, torch.tensor(0.0025))
    loss.backward()
    assert anchor.grad is None and predicted.grad is not None
    report = evaluate_retention(predicted, anchor, torch.tensor([0, 1]), max_action_error=0.05)
    assert not report["passed"]
    assert not report["per_task"]["0"]["passed"]
    assert report["per_task"]["1"]["passed"]
    assert evaluate_retention(anchor, anchor, torch.tensor([0, 1]))["passed"]


def test_replay_and_gate_reject_invalid_batches():
    with pytest.raises(ValueError):
        replay_action_loss(torch.zeros(0, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError):
        replay_action_loss(torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2))
    with pytest.raises(ValueError):
        replay_action_loss(torch.zeros(2, 2), torch.zeros(2, 2), torch.tensor([1.0, -1.0]))
    report = evaluate_retention(
        torch.full((1, 2), float("nan")), torch.zeros(1, 2), torch.zeros(1, dtype=torch.int64)
    )
    assert not report["passed"]
