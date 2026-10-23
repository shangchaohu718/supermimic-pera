"""CPU physical scenes sharing the native 29-action G1 interface.

Tracking loads the original XML verbatim. Interaction tasks add only local
primitive geometry, passive palm collision pads, or the vendored HUSKY board.
Objects remain free: no welds, grasp constraints, or external assistance forces.
This is a simulator interface, not a claim of learned task success.
"""
from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .skateboard_scene import ROBOT_SCENE, compose_skateboard_scene
from .skateboard_contacts import SkateboardContactReader, _twist

TASKS = ("tracking", "stairs", "box", "box_obstacle", "skateboard")
OBJECT_STATE_DIM = 27  # free qpos7, hinge qpos7, free qvel6, hinge qvel7
SKATEBOARD_FOOT_CONTACTS = ("native_spheres", "husky_capsules")


def _validate_foot_contacts(task: str, variant: str) -> None:
    if variant not in SKATEBOARD_FOOT_CONTACTS:
        raise ValueError(f"Unknown skateboard_foot_contacts {variant!r}")
    if variant != "native_spheres" and task != "skateboard":
        raise ValueError("HUSKY foot contacts are only available for the skateboard task")


def _husky_foot_contacts(root: ET.Element) -> None:
    """Use the source HUSKY seven-capsule footprint for skateboard transfer.

    Dimensions copied from TeleHuman/humanoid_skateboarding test_scene/mjlab_scene.xml,
    commit d93833e80deff7f927c0b80ef9c435d8b5c488fe, CC BY-NC 4.0.
    See LICENSES/HUSKY-CC-BY-NC-4.0.md. Explicit native inertials are retained;
    density=0 additionally prevents these contact pads from contributing mass.
    """
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        if body is None or body.find("inertial") is None:
            raise ValueError(f"Expected native {side} ankle body with explicit inertia")
        for geom in list(body.findall("geom")):
            if geom.get("name", "").startswith(f"{side}_ankle_roll_link_collision_"):
                body.remove(geom)
        half_lengths = [.025005 if side == "left" else .025, .0835, .091,
                        .093, .091, .0835, .025]
        positions = [( .075, -.0265 if side == "left" else -.026),
                     (.0395, -.018), (.039, -.01), (.039, 0),
                     (.039, .01), (.0395, .018), (.075, .026)]
        for i, (length, (x, y)) in enumerate(zip(half_lengths, positions), 1):
            # Preserve the slightly angled first left capsule in the export.
            quat = (".707107 -.0141393 .706965 0" if i == 1 and side == "left"
                    else ".707107 0 .707107 0" if i in (1, 7)
                    else ".707107 0 -.707107 0")
            ET.SubElement(body, "geom", name=f"{side}_ankle_roll_link_collision_{i}",
                          type="capsule", size=f".01 {length}", pos=f"{x} {y} -.025",
                          quat=quat, priority="1",
                          contype="1", conaffinity="0", group="3", density="0",
                          friction="1 .005 .0001")


def _native_tree(robot_scene: Path) -> ET.Element:
    scene = ET.parse(robot_scene).getroot()
    robot_path = robot_scene.parent / scene.find("include").attrib["file"]
    root = ET.parse(robot_path).getroot()
    compiler = root.find("compiler")
    meshdir = robot_path.parent / compiler.get("meshdir", "")
    for mesh in root.findall("./asset/mesh"):
        path = (meshdir / mesh.attrib["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local G1 mesh missing: {path}; no download attempted")
        mesh.set("file", str(path))
    compiler.attrib.pop("meshdir", None)
    for section in scene:
        if section.tag in ("include", "keyframe"):
            continue
        dest = root.find(section.tag)
        if dest is None:
            root.append(copy.deepcopy(section))
        elif section.tag in ("asset", "worldbody"):
            dest.extend(copy.deepcopy(list(section)))
        else:
            dest.attrib.update(section.attrib)
    return root


def compose_interaction_scene(task: str, robot_scene: Path = ROBOT_SCENE, *,
                              skateboard_foot_contacts: str = "native_spheres") -> str:
    """Build named physical primitives; tracking callers should load native XML."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    _validate_foot_contacts(task, skateboard_foot_contacts)
    root = (ET.fromstring(compose_skateboard_scene(robot_scene))
            if task == "skateboard" else _native_tree(robot_scene))
    root.set("model", "SuperMimic-PERA G1 " + task)
    if skateboard_foot_contacts == "husky_capsules":
        _husky_foot_contacts(root)
    if task != "tracking":
        for side in ("left", "right"):
            hand = root.find(f".//body[@name='{side}_rubber_hand']")
            # Passive collision volume on the existing fixed rubber hand;
            # density=0 preserves the original native hand inertia/mass.
            ET.SubElement(hand, "geom", name=f"pera/{side}_palm_collision", type="box",
                          pos="0.04 0 0", size="0.045 0.025 0.016", density="0",
                          contype="1", conaffinity="0", friction="1.0 0.005 0.0001",
                          rgba="0.2 0.5 0.85 0.5")
            ET.SubElement(hand, "site", name=f"pera/{side}_palm", pos="0.04 0 0",
                          size="0.008", rgba="0.1 0.7 0.9 1")
    world = root.find("worldbody")
    if task == "stairs":
        # Four ascending, contiguous 8 cm steps, 32 cm treads, 1.2 m width.
        for i in range(4):
            height = .08 * (i + 1)
            ET.SubElement(world, "geom", name=f"pera/step_{i}", type="box",
                          pos=f"{.6 + .32 * i + .16} 0 {height / 2}",
                          size=f".16 .6 {height / 2}", contype="1", conaffinity="1",
                          friction="0.8 0.005 0.0001", rgba="0.45 0.55 0.65 1")
    if task in ("box", "box_obstacle"):
        body = ET.SubElement(world, "body", name="pera/box", pos=".45 0 .181")
        ET.SubElement(body, "freejoint", name="pera/box_free")
        ET.SubElement(body, "geom", name="pera/box_collision", type="box",
                      size=".18 .16 .18", mass="2", contype="1", conaffinity="1",
                      friction="1.0 0.005 0.0001", rgba="0.7 0.45 0.15 1")
        for side, y in (("left", .16), ("right", -.16)):
            ET.SubElement(body, "site", name=f"pera/box_{side}_grasp", pos=f"0 {y} 0",
                          size=".01", rgba="0.1 0.9 0.2 1")
    if task == "box_obstacle":
        ET.SubElement(world, "geom", name="pera/obstacle", type="box", pos="1.1 0 .06",
                      size=".12 .6 .06", contype="1", conaffinity="1",
                      friction="0.8 0.005 0.0001", rgba="0.7 0.3 0.2 1")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


class G1InteractionCPU:
    """One MuJoCo CPU environment; task identity is fixed for its lifetime.

    ``step`` accepts native normalized actions (29), not joint positions.
    ``object_state`` is always 27: root qpos7, internal qpos7, root qvel6,
    internal qvel7; box/absent slots are zero padded. Root quaternion is wxyz,
    free-joint angular velocity is body local, as in MuJoCo qvel.
    ``query_terrain`` returns static support heights and excludes free objects;
    board support motion is reported separately in ``interaction``.
    """

    def __init__(self, task: str = "tracking", *, seed: int = 0,
                 robot_scene: Path = ROBOT_SCENE,
                 skateboard_foot_contacts: str = "native_spheres"):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}")
        _validate_foot_contacts(task, skateboard_foot_contacts)
        self.skateboard_foot_contacts = skateboard_foot_contacts
        self.task, self.task_id = task, TASKS.index(task)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        native = ET.parse(robot_scene).getroot()
        key = native.find("./keyframe/key[@name='stand']")
        self.default_q = np.fromstring(key.attrib["ctrl"], sep=" ")
        self.default_qpos = np.fromstring(key.attrib["qpos"], sep=" ")
        robot_file = robot_scene.parent / native.find("include").attrib["file"]
        self.joint_names = [a.attrib["joint"] for a in ET.parse(robot_file).getroot().find("actuator")]
        self.xml = (None if task == "tracking" else compose_interaction_scene(
            task, robot_scene, skateboard_foot_contacts=skateboard_foot_contacts))
        self.model = (mujoco.MjModel.from_xml_path(str(robot_scene)) if self.xml is None
                      else mujoco.MjModel.from_xml_string(self.xml))
        self.data = mujoco.MjData(self.model)
        self.robot_actuators = np.array([self.model.actuator(n).id for n in self.joint_names])
        self.robot_qpos = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names])
        self.robot_qvel = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in self.joint_names])
        self.object_actuators = np.setdiff1d(np.arange(self.model.nu), self.robot_actuators)
        self.action_scale = (.25 * self.model.actuator_forcerange[self.robot_actuators, 1]
                             / self.model.actuator_gainprm[self.robot_actuators, 0])
        if not np.array_equal(self.robot_qpos, np.arange(7, 36)):
            raise ValueError("Native robot qpos layout changed")
        if not np.array_equal(self.robot_qvel, np.arange(6, 35)):
            raise ValueError("Native robot qvel layout changed")
        self.foot_geoms = {
            side: {g for g in range(self.model.ngeom)
                   if (self.model.geom(g).name or "").startswith(f"{side}_ankle_roll_link_collision_")}
            for side in ("left", "right")
        }
        self.terrain_geoms = {self.model.geom("floor").id}
        self.terrain_geoms.update(g for g in range(self.model.ngeom)
                                 if (self.model.geom(g).name or "").startswith(("pera/step_", "pera/obstacle")))
        # World-axis-aligned static boxes, exported for differentiable height
        # queries at each imagined robot position. Floor is implicitly z=0.
        self.terrain_boxes = np.array([
            np.r_[self.model.geom_pos[g], self.model.geom_size[g]]
            for g in sorted(self.terrain_geoms)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        ], dtype=float).reshape(-1, 6)
        self.box_body = self.model.body("pera/box").id if task in ("box", "box_obstacle") else None
        self.box_geom = self.model.geom("pera/box_collision").id if self.box_body is not None else None
        self.board_reader = SkateboardContactReader(self.model) if task == "skateboard" else None
        self.palm_geoms = ({side: self.model.geom(f"pera/{side}_palm_collision").id
                            for side in ("left", "right")} if task != "tracking" else {})
        self.metadata = {"task": task, "task_id": self.task_id, "seed": seed,
                         "robot_nq": 36, "robot_nv": 35, "action_dim": 29,
                         "object_state_dim": OBJECT_STATE_DIM, "control_dt_s": .02,
                         "physics_backend": "mujoco_cpu", "assisted_grasp": False,
                         "skateboard_foot_contacts": skateboard_foot_contacts,
                         "box_mass_kg": 2 if self.box_body is not None else None,
                         "stairs_rise_m": .08 if task == "stairs" else None,
                         "obstacle_height_m": .12 if task == "box_obstacle" else None}
        self.reset()

    def reset(self, robot_qpos: np.ndarray | None = None,
              robot_qvel: np.ndarray | None = None,
              object_state: np.ndarray | None = None) -> dict:
        """Reset physics; explicit robot state is preserved without terrain offsets."""
        if robot_qpos is not None:
            robot_qpos = np.asarray(robot_qpos, dtype=float)
            if robot_qpos.shape != (36,) or not np.isfinite(robot_qpos).all():
                raise ValueError("robot_qpos must contain 36 finite values")
            if np.linalg.norm(robot_qpos[3:7]) < 1e-8:
                raise ValueError("robot root quaternion must be nonzero")
        if robot_qvel is not None:
            robot_qvel = np.asarray(robot_qvel, dtype=float)
            if robot_qvel.shape != (35,) or not np.isfinite(robot_qvel).all():
                raise ValueError("robot_qvel must contain 35 finite values")
        if object_state is not None:
            object_state = self._validate_object_state(object_state)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:36] = self.default_qpos if robot_qpos is None else robot_qpos
        self.data.qvel[:35] = 0 if robot_qvel is None else robot_qvel
        if self.board_reader is not None:
            self.data.qpos[self.board_reader.qpos_indices[:7]] = [0, 0, .1, 1, 0, 0, 0]
            if robot_qpos is None:
                self.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
                mujoco.mj_forward(self.model, self.data)
                feet = sorted(set.union(*self.foot_geoms.values()))
                self.data.qpos[:2] -= self.data.geom_xpos[feet, :2].mean(axis=0)
                bottoms = self.data.geom_xpos[feet, 2] - self.model.geom_size[feet, 0]
                capsules = self.model.geom_type[feet] == mujoco.mjtGeom.mjGEOM_CAPSULE
                bottoms -= (capsules * self.model.geom_size[feet, 1]
                            * np.abs(self.data.geom_xmat[feet, 8]))
                bottom = np.min(bottoms)
                self.data.qpos[2] += .111 - bottom
        if object_state is not None:
            self._assign_object_state(object_state)
        self.data.ctrl[self.robot_actuators] = self.default_q
        self.data.ctrl[self.object_actuators] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def _validate_object_state(self, object_state: np.ndarray) -> np.ndarray:
        value = np.asarray(object_state, dtype=float)
        if value.shape != (OBJECT_STATE_DIM,) or not np.isfinite(value).all():
            raise ValueError("object_state must contain 27 finite values")
        if self.box_body is None and self.board_reader is None:
            if np.any(value):
                raise ValueError("This task has no free object")
        elif np.linalg.norm(value[3:7]) < 1e-8:
            raise ValueError("Object root quaternion must be nonzero")
        return value

    def _assign_object_state(self, value: np.ndarray) -> None:
        if self.board_reader is not None:
            self.data.qpos[self.board_reader.qpos_indices] = value[:14]
            self.data.qvel[self.board_reader.qvel_indices] = value[14:]
        elif self.box_body is not None:
            joint = self.model.joint("pera/box_free").id
            qa, va = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
            self.data.qpos[qa:qa+7] = value[:7]
            self.data.qvel[va:va+6] = value[14:20]

    def set_object_state(self, object_state: np.ndarray) -> dict:
        """Assign a free-object initial state; does not constrain it thereafter."""
        self._assign_object_state(self._validate_object_state(object_state))
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, action: np.ndarray, *, substeps: int = 4) -> dict:
        action = np.asarray(action, dtype=float)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("action must contain 29 finite native normalized values")
        if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps <= 0:
            raise ValueError("substeps must be a positive integer")
        self.data.ctrl[self.robot_actuators] = self.default_q + self.action_scale * action
        self.data.ctrl[self.object_actuators] = 0
        mujoco.mj_step(self.model, self.data, nstep=substeps)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite CPU physics state")
        return self.observe()

    def query_terrain(self, xy: np.ndarray) -> np.ndarray:
        """Static height at arbitrary world XY; query again at every predicted state."""
        xy = np.asarray(xy, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("xy must have finite (..., 2) shape")
        height = np.zeros(xy.shape[:-1])
        for gid in self.terrain_geoms:
            if self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            center, size = self.model.geom_pos[gid], self.model.geom_size[gid]
            inside = np.all(np.abs(xy - center[:2]) <= size[:2] + 1e-9, axis=-1)
            height = np.where(inside, np.maximum(height, center[2] + size[2]), height)
        return height

    def terrain_patch(self, center_xy: np.ndarray | None = None, *, nside: int = 9,
                      extent: float = .8) -> np.ndarray:
        """9x9 static heights (81) in an axis-aligned world grid by default."""
        center_xy = self.data.qpos[:2] if center_xy is None else np.asarray(center_xy)
        grid = np.linspace(-extent, extent, nside)
        dx, dy = np.meshgrid(grid, grid, indexing="ij")
        return self.query_terrain(np.stack([dx, dy], axis=-1) + center_xy)

    def _contacts(self) -> dict:
        groups = {name: {"count": 0, "active_count": 0, "normal_force": 0.,
                         "force_world": np.zeros(3)}
                  for name in ("left_foot_terrain", "right_foot_terrain",
                               "left_palm_object", "right_palm_object", "object_terrain")}
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = tuple(map(int, contact.geom))
            group, target = None, None
            for slot in (0, 1):
                for side, ids in self.foot_geoms.items():
                    if pair[slot] in ids and pair[1-slot] in self.terrain_geoms:
                        group, target = f"{side}_foot_terrain", slot
                for side, gid in self.palm_geoms.items():
                    if pair[slot] == gid and pair[1-slot] == self.box_geom:
                        group, target = f"{side}_palm_object", slot
                if pair[slot] == self.box_geom and pair[1-slot] in self.terrain_geoms:
                    group, target = "object_terrain", slot
            if group is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            result = groups[group]
            result["count"] += 1
            result["active_count"] += int(force[0] > 1e-8)
            result["normal_force"] += float(force[0])
            result["force_world"] += ((1 if target == 1 else -1)
                                      * (contact.frame.reshape(3, 3).T @ force[:3]))
        for group in groups.values():
            group["force_world"] = group["force_world"].tolist()
        return groups

    def observe(self) -> dict:
        object_state = np.zeros(OBJECT_STATE_DIM)
        interaction = self.board_reader.read(self.data) if self.board_reader is not None else {}
        if self.board_reader is not None:
            object_state[:14] = self.data.qpos[self.board_reader.qpos_indices]
            object_state[14:] = self.data.qvel[self.board_reader.qvel_indices]
        elif self.box_body is not None:
            j = self.model.joint("pera/box_free").id
            qa, va = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
            object_state[:7] = self.data.qpos[qa:qa+7]
            object_state[14:20] = self.data.qvel[va:va+6]
            twist = _twist(self.model, self.data, self.box_body)
            interaction["box"] = {"position_world": self.data.xpos[self.box_body].tolist(),
                                   "quaternion_wxyz": self.data.xquat[self.box_body].tolist(),
                                   "linear_velocity_world": twist[3:].tolist(),
                                   "angular_velocity_world": twist[:3].tolist()}
        palms = {}
        for side in ("left", "right"):
            if self.task == "tracking":
                body = self.model.body(f"{side}_rubber_hand").id
                point = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ [.04, 0, 0]
            else:
                point = self.data.site_xpos[self.model.site(f"pera/{side}_palm").id]
            palms[side] = point.tolist()
        return {"time_s": float(self.data.time), "task": self.task, "task_id": self.task_id,
                "robot_qpos": self.data.qpos[:36].tolist(),
                "robot_qvel": self.data.qvel[:35].tolist(),
                "object_state": object_state.tolist(),
                "object_mask": float(self.box_body is not None or self.board_reader is not None),
                "terrain_heights": self.terrain_patch().reshape(-1).tolist(),
                "palm_positions_world": palms, "contacts": self._contacts(),
                "interaction": interaction}
