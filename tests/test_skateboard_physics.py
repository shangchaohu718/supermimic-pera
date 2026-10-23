"""CPU physics regressions for the vendored articulated HUSKY skateboard.

These tests use only primitive board geometry and the native ground plane, so
they need neither downloaded G1 meshes nor policy checkpoints. They check actual
force support, directional rolling, and spring/equality dynamics.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest


ASSETS = Path(__file__).resolve().parents[1] / "src/unisupermimic/assets"
PREFIX = "skateboard/"


@pytest.fixture
def board():
    root = ET.parse(ASSETS / "objects/husky_skateboard.xml").getroot()
    scene = ET.parse(ASSETS / "robots/g1/scene_supermimic.xml").getroot()
    option = scene.find("option")
    option.set("iterations", "10")
    option.set("ls_iterations", "20")
    option.set("ccd_iterations", "50")
    root.insert(0, option)
    floor = scene.find("worldbody/geom[@name='floor']")
    floor.attrib.pop("material", None)
    root.find("worldbody").insert(0, floor)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for _ in range(400):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def assert_healthy(data):
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert all(warning.number == 0 for warning in data.warning)


def test_four_wheels_support_board_weight(board):
    model, data = board
    assert (model.nq, model.nv, model.nu) == (14, 13, 7)
    assert model.body_mass.sum() == pytest.approx(6.95)
    floor = model.geom("floor").id
    wheel_contacts = set()
    vertical_force = 0.0
    for i, contact in enumerate(data.contact):
        pair = tuple(int(g) for g in contact.geom)
        if floor not in pair:
            continue
        wheel = pair[1] if pair[0] == floor else pair[0]
        assert model.geom(wheel).name.endswith("_wheel_collision")
        wheel_contacts.add(wheel)
        assert contact.dim == 6
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        vertical_force += force[2] * (1 if pair[1] == wheel else -1)
    assert len(wheel_contacts) == 4
    expected_weight = model.body_mass.sum() * -model.opt.gravity[2]
    assert vertical_force == pytest.approx(expected_weight, rel=0.02)
    assert data.qpos[2] == pytest.approx(0.1, abs=0.002)
    assert_healthy(data)


def test_free_wheels_roll_forward_and_resist_sideways_slip(board):
    model, settled = board
    displacements = {}
    for axis, name in ((0, "forward"), (1, "sideways")):
        data = mujoco.MjData(model)
        data.qpos[:] = settled.qpos
        start = data.qpos[:3].copy()
        data.qvel[axis] = 0.5
        if name == "forward":
            for joint_id in range(model.njnt):
                if model.joint(joint_id).name.endswith("_wheel_joint"):
                    data.qvel[model.jnt_dofadr[joint_id]] = 0.5 / 0.03
        for _ in range(100):
            mujoco.mj_step(model, data)
        displacements[name] = abs(data.qpos[axis] - start[axis])
        assert not data.ctrl.any()
        for actuator_id in range(model.nu):
            if model.actuator(actuator_id).name.endswith("_wheel_joint"):
                assert data.actuator_force[actuator_id] == 0.0
        assert_healthy(data)
    assert displacements["forward"] > 0.1
    assert displacements["sideways"] < 0.04
    assert displacements["forward"] > 5 * displacements["sideways"]


def test_tilt_truck_springs_restore_coupled_pose(board):
    model, data = board
    tilt = model.joint(PREFIX + "board_tilt_joint").id
    data.qpos[model.jnt_qposadr[tilt]] = 0.03
    # Upstream equality defines tilt = -0.577 * truck, not its inverse.
    for name in ("front_truck_roll_joint", "rear_truck_roll_joint"):
        truck = model.joint(PREFIX + name).id
        data.qpos[model.jnt_qposadr[truck]] = -0.03 / 0.577
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[model.jnt_qposadr[tilt]]) < 0.003
    for equality_id in range(model.neq):
        joint1, joint2 = model.eq_obj1id[equality_id], model.eq_obj2id[equality_id]
        q1 = data.qpos[model.jnt_qposadr[joint1]]
        q2 = data.qpos[model.jnt_qposadr[joint2]]
        assert abs(q1 + 0.577 * q2) < 0.001
    assert_healthy(data)
