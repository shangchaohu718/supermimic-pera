"""Real CPU physics checks for shared G1 task scenes; no policy success claims."""
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from unisupermimic.interaction.multitask_scene import G1InteractionCPU, TASKS
from unisupermimic.interaction.skateboard_scene import ROBOT_SCENE


pytestmark = pytest.mark.skipif(
    not (ROBOT_SCENE.parent / "assets/pelvis.STL").is_file(),
    reason="Local meshes unavailable; tests never download assets",
)


@pytest.mark.parametrize("task", TASKS)
def test_all_scenes_step_native_actions_with_fixed_observation_shapes(task):
    env = G1InteractionCPU(task)
    before_mass = mujoco.MjModel.from_xml_path(str(ROBOT_SCENE)).body_mass
    np.testing.assert_array_equal(env.model.body_mass[:len(before_mass)], before_mass)
    observation = env.step(np.linspace(-.05, .05, 29))
    assert len(observation["robot_qpos"]) == 36
    assert len(observation["robot_qvel"]) == 35
    assert len(observation["object_state"]) == 27
    assert len(observation["terrain_heights"]) == 81
    assert observation["time_s"] == pytest.approx(.02)
    assert np.isfinite(env.data.qpos).all()
    assert np.isfinite(env.data.qvel).all()
    np.testing.assert_allclose(env.data.ctrl[:29], env.default_q + env.action_scale * np.linspace(-.05, .05, 29))
    assert not any(w.number for w in env.data.warning)
    if task != "tracking":
        assert env.model.geom("pera/left_palm_collision").id > 0
        assert env.model.site("pera/right_palm").id > 0


def test_tracking_uses_identical_native_model_and_trajectory():
    env = G1InteractionCPU("tracking")
    original = mujoco.MjModel.from_xml_path(str(ROBOT_SCENE))
    data = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, data, original.key("stand").id)
    mujoco.mj_forward(original, data)
    assert (env.model.nq, env.model.nv, env.model.ngeom, env.model.nsite) == (
        original.nq, original.nv, original.ngeom, original.nsite)
    np.testing.assert_array_equal(env.data.qpos, data.qpos)
    rng = np.random.default_rng(123)
    for _ in range(15):
        action = rng.normal(0, .05, 29)
        env.step(action)
        data.ctrl[:] = env.default_q + env.action_scale * action
        mujoco.mj_step(original, data, nstep=4)
        mujoco.mj_forward(original, data)
        np.testing.assert_array_equal(env.data.qpos, data.qpos)
        np.testing.assert_array_equal(env.data.qvel, data.qvel)


def test_static_terrain_query_matches_geometry_and_stair_contact():
    env = G1InteractionCPU("stairs")
    assert env.terrain_boxes.shape == (4, 6)
    np.testing.assert_allclose(env.terrain_boxes[:, 2] + env.terrain_boxes[:, 5], [.08, .16, .24, .32])
    np.testing.assert_allclose(env.query_terrain([[0, 0], [.7, 0], [1.0, 0], [1.4, 0], [1.8, 0], [1.8, 2]]),
                               [0, .08, .16, .24, .32, 0])
    state = env.default_qpos.copy()
    state[0], state[2] = .72, state[2] + .08
    env.reset(state)
    force = 0.
    for _ in range(25):
        obs = env.step(np.zeros(29))
        force = max(force, sum(obs["contacts"][f"{s}_foot_terrain"]["normal_force"]
                               for s in ("left", "right")))
    assert force > 100  # Solver-resolved feet/stair support, not visual-only geometry.
    env = G1InteractionCPU("box_obstacle")
    np.testing.assert_allclose(env.query_terrain([[1.1, 0], [.4, 0], [1.1, .7]]), [.12, 0, 0])


def test_box_is_free_and_falls_then_loads_ground():
    env = G1InteractionCPU("box")
    assert env.model.neq == 0
    joint = env.model.joint("pera/box_free").id
    assert env.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    qadr = env.model.jnt_qposadr[joint]
    env.data.qpos[qadr:qadr+3] = [2, 0, 1.0]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(10):
        env.step(np.zeros(29))
    assert .7 < env.data.qpos[qadr+2] < .9
    for _ in range(45):
        obs = env.step(np.zeros(29))
    assert env.data.qpos[qadr+2] == pytest.approx(.18, abs=.002)
    assert obs["contacts"]["object_terrain"]["force_world"][2] == pytest.approx(19.62, rel=.02)


@pytest.mark.parametrize("task", ["box", "skateboard"])
def test_explicit_object_reset_preserves_robot_and_object_state(task):
    env = G1InteractionCPU(task)
    robot = env.default_qpos.copy()
    robot[0] = -.3
    velocity = np.linspace(-.01, .01, 35)
    obj = np.zeros(27)
    obj[:7] = [.5, .3, .2, 1, 0, 0, 0]
    obj[14:20] = [.01, .02, .03, .04, .05, .06]
    obs = env.reset(robot, velocity, obj)
    np.testing.assert_array_equal(obs["robot_qpos"], robot)
    np.testing.assert_array_equal(obs["robot_qvel"], velocity)
    np.testing.assert_array_equal(obs["object_state"], obj)
    obj[0] += .3
    obs = env.set_object_state(obj)
    np.testing.assert_array_equal(obs["object_state"], obj)


def test_palms_generate_real_box_contacts():
    env = G1InteractionCPU("box")
    palm = env.model.geom("pera/left_palm_collision").id
    position = env.data.geom_xpos[palm].copy()
    joint = env.model.joint("pera/box_free").id
    qadr = env.model.jnt_qposadr[joint]
    # Place the free box just into the palm from below to exercise the solver.
    env.data.qpos[qadr:qadr+3] = position + [0, 0, -.185]
    mujoco.mj_forward(env.model, env.data)
    contact = env.observe()["contacts"]["left_palm_object"]
    assert contact["count"] > 0
    assert contact["normal_force"] > 0
    assert np.linalg.norm(contact["force_world"]) > 0


def test_board_keeps_passive_actuators_and_actual_foot_support():
    env = G1InteractionCPU("skateboard")
    for _ in range(20):
        obs = env.step(np.zeros(29))
    np.testing.assert_array_equal(env.data.ctrl[29:], 0)
    assert obs["interaction"]["contacts"]["wheel_ground"]["force_world"][2] > 300
    assert obs["interaction"]["contacts"]["left_foot_board"]["normal_force"] > 50
    assert obs["interaction"]["contacts"]["right_foot_board"]["normal_force"] > 50


def test_reset_validation_and_cpu_only_import():
    env = G1InteractionCPU("box")
    before = env.data.qpos.copy()
    with pytest.raises(ValueError):
        env.step(np.full(29, np.nan))
    with pytest.raises(ValueError):
        env.reset(np.zeros(36))
    np.testing.assert_array_equal(env.data.qpos, before)
    env.step(np.zeros(29))
    env.reset()
    assert env.data.time == 0
    np.testing.assert_array_equal(env.data.qvel, 0)
    command = ("import sys; from unisupermimic.interaction.multitask_scene import G1InteractionCPU; "
               "G1InteractionCPU('stairs'); assert 'torch' not in sys.modules; "
               "assert 'warp' not in sys.modules")
    subprocess.run([sys.executable, "-c", command], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable"})
