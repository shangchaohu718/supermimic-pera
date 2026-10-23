"""CPU checks of dynamics gradients, frozen capabilities, and object masking."""

import torch

from unisupermimic.interaction.world_model import JointInteractionWorldModel
from unisupermimic.rl.algos.supermimic.networks import WorldModel


def fixture(batch=4):
    torch.manual_seed(21)
    base = WorldModel(hidden=(32, 32))
    # A pretrained model has nonzero dynamics; emulate that without a checkpoint.
    with torch.no_grad():
        base.net[-1].weight.normal_(std=0.01)
    model = JointInteractionWorldModel(base, hidden=(32, 32))
    state = (
        torch.randn(batch, 3), torch.eye(3).repeat(batch, 1, 1),
        torch.randn(batch, 3), torch.randn(batch, 3),
        torch.randn(batch, 29), torch.randn(batch, 29),
    )
    obj = torch.zeros(batch, 27)
    obj[:, 3] = 1
    ctx = torch.zeros(batch, 64)
    ctx[:, 5] = 1
    action = torch.randn(batch, 29)
    return model, state, obj, action, ctx


def test_zero_gate_preserves_native_dynamics_exactly_after_residual_changes():
    model, state, obj, action, ctx = fixture()
    before = {k: v.clone() for k, v in model.base_model.state_dict().items()}
    with torch.no_grad():
        model.correction[-1].weight.normal_(std=0.1)
        model.correction[-1].bias.normal_()
    model.update_normalization(state, obj, action, ctx)
    model.train()
    native = model.base_model.step(state, action, torch.zeros(4, 33))
    prediction, predicted_obj = model.step(state, obj, action, ctx, torch.zeros(4))
    assert all(torch.equal(a, b) for a, b in zip(prediction, native, strict=True))
    assert torch.equal(predicted_obj, obj)
    assert not model.base_model.training
    assert all(not p.requires_grad for p in model.base_model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.base_model.state_dict().items())


def test_zero_initialization_is_zero_after_normalization_update():
    model, state, obj, action, ctx = fixture()
    model.update_normalization(state, obj, action, ctx)
    native = model.base_model.step(state, action, torch.zeros(4, 33))
    prediction, predicted_obj = model.step(state, obj, action, ctx, torch.ones(4))
    assert all(torch.equal(a, b) for a, b in zip(prediction, native, strict=True))
    assert torch.equal(predicted_obj, obj)


def test_object_prediction_can_learn_and_is_not_reference_replay():
    model, state, obj, action, ctx = fixture()
    with torch.no_grad():
        target_robot = model.base_model.step(state, action, torch.zeros(4, 33))
    target_object = obj.clone()
    target_object[:, 0] += 0.03
    optimizer = torch.optim.Adam(model.correction.parameters(), lr=1e-2)
    loss = model.loss(state, obj, action, ctx, torch.ones(4), target_robot, target_object, update_norm=True)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert model.correction[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base_model.parameters())
    optimizer.step()
    _, learned_object = model.step(state, obj, action, ctx, torch.ones(4))
    assert not torch.equal(learned_object, obj)
    assert torch.allclose(learned_object[:, 3:7].norm(dim=-1), torch.ones(4), atol=1e-6)


def test_policy_gradients_flow_through_frozen_base_and_joint_correction():
    model, state, obj, action, ctx = fixture()
    action.requires_grad_(True)
    with torch.no_grad():
        model.correction[-1].weight.normal_(std=0.01)
    predicted, predicted_obj = model.step(state, obj, action, ctx, torch.ones(4))
    (predicted[0].square().sum() + predicted_obj.square().sum()).backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base_model.parameters())


def test_stairs_without_an_object_leave_object_slots_unchanged():
    model, state, obj, action, ctx = fixture()
    ctx[:, 5] = 0
    # Absent objects may use all-zero padding, including their quaternion.
    obj.zero_()
    with torch.no_grad():
        model.correction[-1].bias.fill_(0.1)
    native = model.base_model.step(state, action, torch.zeros(4, 33))
    prediction, predicted_obj = model.step(state, obj, action, ctx, torch.ones(4))
    assert not torch.equal(prediction[0], native[0])
    assert torch.equal(predicted_obj, obj)


def test_quaternion_sign_does_not_create_loss_and_empty_mask_is_finite():
    model, state, obj, action, ctx = fixture()
    with torch.no_grad():
        next_state, next_object = model.step(state, obj, action, ctx, torch.ones(4))
    next_object = next_object.clone()
    next_object[:, 3:7] *= -1
    same_pose_loss = model.loss(state, obj, action, ctx, torch.ones(4), next_state, next_object)
    assert same_pose_loss < 1e-10
    empty_loss = model.loss(state, obj, action, ctx, torch.zeros(4), next_state, next_object, update_norm=True)
    assert torch.isfinite(empty_loss) and empty_loss == 0


def test_explicit_object_mask_and_base_context_are_honored():
    model, state, obj, action, ctx = fixture()
    with torch.no_grad():
        model.correction[-1].bias.fill_(0.1)
    ctx[:, 5] = 0
    base_context = torch.randn(4, 33)
    prediction, predicted_obj = model.step(
        state, obj, action, ctx, torch.zeros(4, 1),
        object_mask=torch.ones(4, 1), base_context=base_context,
    )
    native = model.base_model.step(state, action, base_context)
    assert all(torch.equal(a, b) for a, b in zip(prediction, native, strict=True))
    _, predicted_obj = model.step(
        state, obj, action, ctx, torch.ones(4), object_mask=torch.ones(4),
    )
    assert not torch.equal(predicted_obj, obj)
