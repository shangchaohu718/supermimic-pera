"""Opt-in HUSKY foot geometry keeps native tracking and robot dynamics intact."""
import mujoco
import numpy as np
import pytest

from unisupermimic.interaction.multitask_scene import (
    G1InteractionCPU, TASKS, compose_interaction_scene,
)
from unisupermimic.interaction.skateboard_scene import ROBOT_SCENE

pytestmark = pytest.mark.skipif(
    not (ROBOT_SCENE.parent / 'assets/pelvis.STL').is_file(),
    reason='Local G1 assets absent; tests never download files',
)


@pytest.fixture
def capsule_env():
    return G1InteractionCPU('skateboard', skateboard_foot_contacts='husky_capsules')


def test_capsules_preserve_every_native_inertial_and_actuator(capsule_env):
    native = G1InteractionCPU('skateboard')
    assert (capsule_env.model.nq, capsule_env.model.nv, capsule_env.model.nu) == (50, 48, 36)
    assert capsule_env.joint_names == native.joint_names
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                  'dof_armature', 'jnt_range', 'actuator_gainprm',
                  'actuator_biasprm', 'actuator_forcerange', 'actuator_trnid'):
        np.testing.assert_array_equal(getattr(capsule_env.model, field),
                                      getattr(native.model, field))
    np.testing.assert_array_equal(capsule_env.action_scale, native.action_scale)
    np.testing.assert_array_equal(capsule_env.default_q, native.default_q)
    action = np.linspace(-.3, .3, 29)
    capsule_env.step(action)
    np.testing.assert_allclose(capsule_env.data.ctrl[capsule_env.robot_actuators],
                               native.default_q + native.action_scale * action)
    np.testing.assert_array_equal(capsule_env.data.ctrl[capsule_env.object_actuators], 0)


def test_exact_footprint_and_all_contact_geometries_are_accounted_for(capsule_env):
    env = capsule_env
    for side in ('left', 'right'):
        geoms = sorted(env.foot_geoms[side])
        assert len(geoms) == 7
        assert env.foot_geoms[side] == env.board_reader.foot_geoms[side]
        np.testing.assert_array_equal(env.model.geom_type[geoms], mujoco.mjtGeom.mjGEOM_CAPSULE)
        np.testing.assert_array_equal(env.model.geom_priority[geoms], 1)
        np.testing.assert_allclose(env.model.geom_size[geoms, 0], .01)
        np.testing.assert_allclose(env.model.geom_size[geoms, 1],
                                  [.025005 if side == 'left' else .025,
                                   .0835, .091, .093, .091, .0835, .025])
    # The source left-front capsule is slightly angled rather than exactly axial.
    first = env.model.geom('left_ankle_roll_link_collision_1').id
    quat = np.array([.707107, -.0141393, .706965, 0])
    np.testing.assert_allclose(env.model.geom_quat[first], quat / np.linalg.norm(quat))


def test_capsule_feet_and_passive_wheels_carry_weight_in_real_physics(capsule_env):
    env = capsule_env
    for _ in range(20):
        observation = env.step(np.zeros(29))
    contacts = observation['interaction']['contacts']
    for side in ('left', 'right'):
        assert contacts[f'{side}_foot_board']['normal_force'] > 50
        assert contacts[f'{side}_foot_board']['active_count'] > 0
        assert contacts[f'{side}_foot_ground']['active_count'] == 0
    assert contacts['wheel_ground']['normal_force'] == pytest.approx(
        sum(env.model.body_mass) * 9.81, rel=.1)
    assert observation['robot_qpos'][2] > .65
    assert np.isfinite(env.data.qpos).all()
    assert not any(w.number for w in env.data.warning)


def test_default_skateboard_still_uses_four_native_spheres():
    env = G1InteractionCPU('skateboard')
    assert env.metadata['skateboard_foot_contacts'] == 'native_spheres'
    for geoms in env.foot_geoms.values():
        assert len(geoms) == 4
        np.testing.assert_array_equal(env.model.geom_type[sorted(geoms)], mujoco.mjtGeom.mjGEOM_SPHERE)


@pytest.mark.parametrize('task', [task for task in TASKS if task != 'skateboard'])
def test_capsule_choice_cannot_modify_other_task_physics(task):
    with pytest.raises(ValueError, match='only available for the skateboard'):
        G1InteractionCPU(task, skateboard_foot_contacts='husky_capsules')
    with pytest.raises(ValueError, match='only available for the skateboard'):
        compose_interaction_scene(task, skateboard_foot_contacts='husky_capsules')


def test_unknown_contact_model_rejected():
    with pytest.raises(ValueError, match='Unknown skateboard_foot_contacts'):
        G1InteractionCPU('skateboard', skateboard_foot_contacts='unknown')


def test_tracking_still_loads_verbatim_native_xml():
    env = G1InteractionCPU('tracking')
    native = mujoco.MjModel.from_xml_path(str(ROBOT_SCENE))
    original = mujoco.MjData(native)
    mujoco.mj_resetDataKeyframe(native, original, native.key('stand').id)
    mujoco.mj_forward(native, original)
    assert env.xml is None
    assert (env.model.ngeom, env.model.njnt) == (native.ngeom, native.njnt)
    for action in (np.zeros(29), np.linspace(-.1, .1, 29)):
        env.step(action)
        original.ctrl[:] = env.default_q + env.action_scale * action
        mujoco.mj_step(native, original, nstep=4)
        mujoco.mj_forward(native, original)
        np.testing.assert_array_equal(env.data.qpos, original.qpos)
        np.testing.assert_array_equal(env.data.qvel, original.qvel)
