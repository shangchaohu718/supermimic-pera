"""CPU physical checks for contact signs, grouping and moving support frames."""

import json

import mujoco
import numpy as np
import pytest

from unisupermimic.interaction.skateboard_contacts import SkateboardContactReader


def make_scene():
    """Small analytic scene with all seven articulated skateboard coordinates."""
    trucks = []
    for end, x in (("front", 0.25), ("rear", -0.25)):
        wheels = "".join(
            f'''<body name="skateboard/{end}_{side}_wheel" pos="0 {y} -.10">
              <joint name="skateboard/{end}_{side}_wheel_joint" axis="0 1 0"/>
              <geom name="skateboard/{end}_{side}_wheel_collision" type="sphere"
                    size=".05" mass=".2"/>
            </body>'''
            for side, y in (("left", 0.08), ("right", -0.08))
        )
        trucks.append(f'''<body pos="{x} 0 0">
          <joint name="skateboard/{end}_truck_roll_joint" axis="0 0 1"/>
          <inertial pos="0 0 0" mass=".5" diaginertia=".01 .01 .01"/>
          {wheels}</body>''')
    xml = f'''<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
      <default><joint damping="2"/><geom friction="1 .01 .001"/></default>
      <worldbody><geom name="floor" type="plane" size="10 10 .1"/>
        <body name="skateboard/skateboard_deck" pos="0 0 .15">
          <freejoint name="skateboard/floating_base_joint_skateboard"/>
          <inertial pos=".01 0 0" mass="20" diaginertia="1 1 1"/>
          <body name="skateboard/board_tilt_body">
            <joint name="skateboard/board_tilt_joint" axis="1 0 0" stiffness="100"/>
            <geom name="skateboard/skateboard_deck_collision" type="box"
                  size=".4 .1 .02" mass="10"/>
            <geom name="skateboard/skateboard_marker_collision" type="box"
                  size=".01 .01 .001" pos=".35 0 .02" mass=".01"/>
          </body>{''.join(trucks)}
        </body>
        <body name="left_ankle_roll_link" pos="-.15 0 .20">
          <freejoint name="left_foot_free"/>
          <geom name="left_ankle_roll_link_collision_1" type="sphere" size=".03" mass="1"/>
        </body>
        <body name="right_ankle_roll_link" pos="2 0 .03">
          <freejoint name="right_foot_free"/>
          <geom name="right_ankle_roll_link_collision_1" type="sphere" size=".03" mass="2"/>
        </body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def test_contact_forces_support_weight_with_both_geom_orders():
    model, data = make_scene()
    reader = SkateboardContactReader(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    snapshot = reader.read(data)
    groups = snapshot["contacts"]
    # A sphere-box contact puts the supported sphere in geom0. Plane-sphere
    # puts it in geom1. Both forces must point up, not just have positive norm.
    left = model.geom("left_ankle_roll_link_collision_1").id
    right = model.geom("right_ankle_roll_link_collision_1").id
    assert any(c.geom[0] == left for c in data.contact[:data.ncon])
    assert any(c.geom[1] == right for c in data.contact[:data.ncon])
    assert groups["left_foot_board"]["force_world"][2] == pytest.approx(9.81, abs=0.25)
    assert groups["right_foot_ground"]["force_world"][2] == pytest.approx(19.62, abs=0.25)
    assert groups["left_foot_ground"]["active_count"] == 0
    assert groups["right_foot_board"]["active_count"] == 0
    assert groups["wheel_ground"]["force_world"][2] > 300
    assert groups["wheel_ground"]["active_count"] == 4
    json.dumps(snapshot, allow_nan=False)


def test_surface_relative_velocity_includes_root_and_tilt_angular_motion():
    model, data = make_scene()
    reader = SkateboardContactReader(model)
    root_joint = model.joint("skateboard/floating_base_joint_skateboard")
    root_q, root_v = int(root_joint.qposadr[0]), int(root_joint.dofadr[0])
    data.qpos[root_q:root_q + 3] = [0, 0, 10]
    data.qvel[root_v:root_v + 6] = [1, 2, 3, 0, 0, 2]
    data.qvel[reader.internal_qvel[0]] = 1.0
    point = np.array([.2, .3, 10.4])
    expected = np.array([1, 2, 3]) + np.cross([1, 0, 2], point - [0, 0, 10])
    foot = model.joint("left_foot_free")
    qadr, vadr = int(foot.qposadr[0]), int(foot.dofadr[0])
    data.qpos[qadr:qadr + 3] = point
    data.qvel[vadr:vadr + 3] = expected
    mujoco.mj_forward(model, data)
    snapshot = reader.read(data)
    np.testing.assert_allclose(snapshot["feet"]["left"]["support_velocity_at_foot_world"], expected)
    np.testing.assert_allclose(snapshot["feet"]["left"]["linear_velocity_relative_world"], 0, atol=1e-12)
    np.testing.assert_allclose(snapshot["board"]["surface_angular_velocity_world"], [1, 0, 2])


def test_board_generalized_state_keeps_all_passive_joints():
    model, data = make_scene()
    reader = SkateboardContactReader(model)
    positions = np.linspace(-.04, .04, 7)
    velocities = np.linspace(-1, 1, 7)
    data.qpos[reader.internal_qpos] = positions
    data.qvel[reader.internal_qvel] = velocities
    mujoco.mj_forward(model, data)
    board = reader.read(data)["board"]
    assert len(board["qpos"]) == 14
    assert len(board["qvel"]) == 13
    np.testing.assert_allclose(board["qpos"][7:], positions)
    np.testing.assert_allclose(board["qvel"][6:], velocities)


def test_missing_board_fails_explicitly():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="skateboard/skateboard_deck"):
        SkateboardContactReader(model)
