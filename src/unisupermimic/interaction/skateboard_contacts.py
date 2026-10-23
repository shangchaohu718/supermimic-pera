"""CPU MuJoCo state and contact adapter for native G1 on the HUSKY skateboard.

The scene supplies the HUSKY articulated board; this module does not alter its
dynamics. Contact forces are forces *on the named foot/wheel*, expressed in the
world frame. Support-relative velocity accounts for rotation of the actual tilt
body rather than treating the board root as a translating rigid surface.

Only NumPy and CPU MuJoCo are imported. Call ``mj_forward`` after changing state,
or ``mj_step`` before reading. The result is JSON serializable and is a diagnostic
interface, not a trained policy observation contract.
"""

from __future__ import annotations

import mujoco
import numpy as np


BOARD_JOINT_NAMES = (
    "board_tilt_joint",
    "front_truck_roll_joint",
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_truck_roll_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
)


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Skateboard scene is missing {kind.name}: {name}")
    return result


def _twist(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    """Return world [angular, linear] velocity at the body frame origin."""
    result = np.zeros(6)
    # XBODY refers to xpos; BODY refers to the inertial frame (xipos).
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, result, 0)
    return result


def _point_velocity(data: mujoco.MjData, body: int, twist: np.ndarray,
                    point: np.ndarray) -> np.ndarray:
    return twist[3:] + np.cross(twist[:3], point - data.xpos[body])


class SkateboardContactReader:
    """Resolve names once and read independently of MuJoCo body/joint ordering."""

    def __init__(self, model: mujoco.MjModel, prefix: str = "skateboard/",
                 floor_name: str = "floor") -> None:
        self.model = model
        self.root = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "skateboard_deck")
        self.surface = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "board_tilt_body")
        self.floor = _id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
        self.foot_bodies = {
            side: _id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
            for side in ("left", "right")
        }
        self.foot_geoms = {
            side: {
                g for g in range(model.ngeom)
                if model.geom_bodyid[g] == body
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            }
            for side, body in self.foot_bodies.items()
        }
        if any(not geoms for geoms in self.foot_geoms.values()):
            raise ValueError("Both native G1 ankle-roll bodies need collision geoms")
        self.deck_geoms = {
            _id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
            for name in ("skateboard_deck_collision", "skateboard_marker_collision")
        }
        self.wheel_geoms = {
            _id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name + "_wheel_collision")
            for name in ("front_left", "front_right", "rear_left", "rear_right")
        }
        free_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT,
                         prefix + "floating_base_joint_skateboard")
        if model.jnt_type[free_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Skateboard root must have a free joint")
        self.joint_names = [prefix + name for name in BOARD_JOINT_NAMES]
        joints = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Expected the seven HUSKY skateboard hinge joints")
        self.internal_qpos = model.jnt_qposadr[joints].copy()
        self.internal_qvel = model.jnt_dofadr[joints].copy()
        qadr, vadr = int(model.jnt_qposadr[free_joint]), int(model.jnt_dofadr[free_joint])
        self.qpos_indices = np.r_[np.arange(qadr, qadr + 7), self.internal_qpos]
        self.qvel_indices = np.r_[np.arange(vadr, vadr + 6), self.internal_qvel]

    def read(self, data: mujoco.MjData) -> dict:
        """Return board state, foot-relative motion, and force-resolved contacts.

        ``qpos`` uses MuJoCo free-joint quaternion order wxyz; ``qvel`` is the
        original generalized velocity (free angular velocity is body-local).
        Separately labelled world twists avoid confusing these conventions.
        Contact ``count`` includes solver candidates, whereas ``active_count``
        requires a positive resolved normal force.
        """
        model = self.model
        surface_twist = _twist(model, data, self.surface)
        root_twist = _twist(model, data, self.root)
        rotation = data.xmat[self.surface].reshape(3, 3)
        result = {
            "time": float(data.time),
            "board": {
                "qpos": data.qpos[self.qpos_indices].tolist(),
                "qvel": data.qvel[self.qvel_indices].tolist(),
                "internal_joint_names": self.joint_names.copy(),
                "internal_qpos": data.qpos[self.internal_qpos].tolist(),
                "internal_qvel": data.qvel[self.internal_qvel].tolist(),
                "root_position_world": data.xpos[self.root].tolist(),
                "root_quaternion_wxyz": data.xquat[self.root].tolist(),
                "root_angular_velocity_world": root_twist[:3].tolist(),
                "root_linear_velocity_world": root_twist[3:].tolist(),
                "surface_position_world": data.xpos[self.surface].tolist(),
                "surface_quaternion_wxyz": data.xquat[self.surface].tolist(),
                "surface_angular_velocity_world": surface_twist[:3].tolist(),
                "surface_linear_velocity_world": surface_twist[3:].tolist(),
            },
            "feet": {},
            "contacts": {},
        }
        groups = result["contacts"]
        for name in ("left_foot_board", "right_foot_board", "left_foot_ground",
                     "right_foot_ground", "wheel_ground"):
            groups[name] = {"count": 0, "active_count": 0, "normal_force": 0.0,
                            "force_world": np.zeros(3), "points_world": [],
                            "max_tangential_speed": 0.0}

        for side, body in self.foot_bodies.items():
            twist = _twist(model, data, body)
            point = data.xpos[body]
            support_velocity = _point_velocity(data, self.surface, surface_twist, point)
            relative_velocity = twist[3:] - support_velocity
            result["feet"][side] = {
                "position_world": point.tolist(),
                "position_surface_frame": (rotation.T @ (point - data.xpos[self.surface])).tolist(),
                "support_velocity_at_foot_world": support_velocity.tolist(),
                "linear_velocity_relative_world": relative_velocity.tolist(),
                "linear_velocity_relative_surface_frame": (rotation.T @ relative_velocity).tolist(),
                "angular_velocity_relative_surface_frame":
                    (rotation.T @ (twist[:3] - surface_twist[:3])).tolist(),
            }

        wrench = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = (int(contact.geom[0]), int(contact.geom[1]))
            target = None
            group_name = None
            for side, foot_geoms in self.foot_geoms.items():
                for slot in (0, 1):
                    if pair[slot] in foot_geoms:
                        if pair[1 - slot] in self.deck_geoms:
                            group_name, target = side + "_foot_board", slot
                        elif pair[1 - slot] == self.floor:
                            group_name, target = side + "_foot_ground", slot
            for slot in (0, 1):
                if pair[slot] in self.wheel_geoms and pair[1 - slot] == self.floor:
                    group_name, target = "wheel_ground", slot
            if group_name is None:
                continue
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            # Contact rows are the contact-frame axes in world coordinates.
            # Positive solver force acts on geom[1], and its reaction on geom[0].
            force = contact.frame.reshape(3, 3).T @ wrench[:3]
            force *= 1.0 if target == 1 else -1.0
            group = groups[group_name]
            group["count"] += 1
            group["active_count"] += int(wrench[0] > 1e-8)
            group["normal_force"] += float(wrench[0])
            group["force_world"] += force
            group["points_world"].append(contact.pos.tolist())
            # Relative contact-point motion includes both bodies' rotations.
            velocities = []
            for geom in (pair[target], pair[1 - target]):
                body = int(model.geom_bodyid[geom])
                velocities.append(_point_velocity(data, body, _twist(model, data, body),
                                                  contact.pos))
            velocity = velocities[0] - velocities[1]
            normal = contact.frame[:3]
            tangential = velocity - np.dot(velocity, normal) * normal
            group["max_tangential_speed"] = max(group["max_tangential_speed"],
                                                 float(np.linalg.norm(tangential)))
        for group in groups.values():
            group["force_world"] = group["force_world"].tolist()
        return result
