"""CPU integration regressions using the actual native G1 and articulated board."""
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from unisupermimic.interaction.skateboard_scene import G1SkateboardCPU, ROBOT_SCENE


@pytest.fixture
def env():
    if not (ROBOT_SCENE.parent/'assets/pelvis.STL').exists():
        pytest.skip('Local G1 mesh assets not installed; no automatic downloads')
    return G1SkateboardCPU()


def test_native_robot_contract_and_action_mapping_preserved(env):
    original = mujoco.MjModel.from_xml_path(str(ROBOT_SCENE))
    assert (env.model.nq, env.model.nv, env.model.nu) == (50, 48, 36)
    for field in ('body_mass', 'body_inertia'):
        np.testing.assert_allclose(getattr(env.model, field)[:original.nbody], getattr(original, field))
    for field in ('actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_trnid'):
        np.testing.assert_allclose(getattr(env.model, field)[:29], getattr(original, field))
    action = np.linspace(-.1,.1,29)
    env.step(action, substeps=1)
    np.testing.assert_allclose(env.data.ctrl[:29], env.default_q+env.action_scale*action)
    np.testing.assert_array_equal(env.data.ctrl[29:],0)
    assert len(env.observe()['robot_qpos']) == 36
    assert len(env.observe()['robot_qvel']) == 35
    assert len(env.observe()['interaction']['board']['qpos']) == 14
    assert len(env.observe()['interaction']['board']['qvel']) == 13


@pytest.mark.parametrize('placement,expected,absent', [
    ('board','foot_board','foot_ground'), ('ground','foot_ground','foot_board')])
def test_actual_feet_load_the_correct_support(env, placement, expected, absent):
    env.reset(placement)
    for _ in range(20):
        observation = env.step(np.zeros(29))
    groups = observation['interaction']['contacts']
    for side in ('left','right'):
        assert groups[f'{side}_{expected}']['normal_force'] > 50
        assert groups[f'{side}_{expected}']['force_world'][2] > 50
        assert groups[f'{side}_{absent}']['active_count'] == 0
    weight = sum(env.model.body_mass)*9.81 if placement == 'board' else 6.95*9.81
    assert groups['wheel_ground']['force_world'][2] == pytest.approx(weight, rel=.1)
    assert observation['robot_qpos'][2] > .65
    assert not any(w.number for w in env.data.warning)


def test_reset_clears_board_motion_and_invalid_actions_fail(env):
    env.data.qvel[:] = 1
    env.data.ctrl[:] = 7
    env.reset()
    np.testing.assert_array_equal(env.data.qvel,0)
    np.testing.assert_array_equal(env.data.ctrl[29:],0)
    before=env.data.qpos.copy()
    for action in (np.zeros(36),np.full(29,np.nan)):
        with pytest.raises(ValueError):
            env.step(action)
    np.testing.assert_array_equal(env.data.qpos,before)


def test_cpu_adapter_does_not_import_training_or_gpu_runtime(env):
    command = ('import sys; from unisupermimic.interaction.skateboard_scene import G1SkateboardCPU; '
               'G1SkateboardCPU(); assert "torch" not in sys.modules; assert "warp" not in sys.modules')
    subprocess.run([sys.executable,'-c',command],check=True,
                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'disable'})
