"""Native 29-DoF SuperMimic G1 plus the locally vendored HUSKY skateboard.

The HUSKY XML is CC BY-NC 4.0; see its source manifest and LICENSES directory.
This module composes assets without importing Torch, Warp, or a GPU backend.
"""
from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
ROBOT_SCENE = ASSETS / 'robots/g1/scene_supermimic.xml'
BOARD_XML = ASSETS / 'objects/husky_skateboard.xml'


def compose_skateboard_scene(robot_scene: Path = ROBOT_SCENE) -> str:
    """Compose native scene and board; resolve local mesh paths without copying meshes."""
    scene = ET.parse(robot_scene).getroot()
    include = scene.find('include')
    if include is None:
        raise ValueError('Expected native robot include')
    robot_path = robot_scene.parent / include.attrib['file']
    root = ET.parse(robot_path).getroot()
    root.set('model', 'SuperMimic-PERA G1 29DoF + HUSKY skateboard (CPU integration)')
    compiler = root.find('compiler')
    meshdir = robot_path.parent / compiler.get('meshdir', '')
    for mesh in root.findall('./asset/mesh'):
        path = (meshdir / mesh.attrib['file']).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Local G1 mesh missing: {path}; no download attempted')
        mesh.set('file', str(path))
    compiler.attrib.pop('meshdir', None)
    board = ET.parse(BOARD_XML).getroot()
    for source in (scene, board):
        for section in source:
            if section.tag in ('include', 'keyframe') or not isinstance(section.tag, str):
                continue
            dest = root.find(section.tag)
            if dest is None:
                root.append(copy.deepcopy(section))
            elif section.tag in ('asset', 'worldbody', 'actuator', 'default', 'equality'):
                dest.extend(copy.deepcopy(list(section)))
            else:
                dest.attrib.update(section.attrib)
    # Native tracking disables robot self-collision (1/0). Board must receive bit1.
    for geom in root.findall("./worldbody/body[@name='skateboard/skateboard_deck']//geom"):
        geom.set('contype', '1')
        geom.set('conaffinity', '1')
    ET.indent(root)
    return ET.tostring(root, encoding='unicode')


class G1SkateboardCPU:
    """Single CPU scene with 29 robot actions and separately observed board state.

    Not registered as G1SuperMimic: existing tracking WM/replay cannot represent
    articulated objects. No rewards, training, or checkpoint conversion is implied.
    """
    def __init__(self, robot_scene: Path = ROBOT_SCENE):
        # Read the same native model/keyframe without importing the training
        # package, whose __init__ imports Torch/UniLab even for robot constants.
        native = ET.parse(robot_scene).getroot()
        robot_file = robot_scene.parent / native.find('include').attrib['file']
        self.joint_names = [a.attrib['joint'] for a in ET.parse(robot_file).getroot().find('actuator')]
        self.default_q = np.fromstring(native.find("./keyframe/key[@name='stand']").attrib['ctrl'], sep=' ')
        self.xml = compose_skateboard_scene(robot_scene)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (50, 48, 36):
            raise ValueError('Unexpected robot/board dimensions')
        self.robot_actuators = np.array([self.model.actuator(n).id for n in self.joint_names])
        self.robot_qpos = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names])
        self.robot_qvel = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in self.joint_names])
        self.board_actuators = np.array([i for i in range(self.model.nu) if i not in self.robot_actuators])
        # Native SuperMimic normalization: 0.25 * effort_limit / kp.
        self.action_scale = (.25 * self.model.actuator_forcerange[self.robot_actuators, 1]
                             / self.model.actuator_gainprm[self.robot_actuators, 0])
        if len(self.joint_names) != 29 or self.default_q.shape != (29,):
            raise ValueError('Expected native 29-DoF standing keyframe')
        if not np.array_equal(self.robot_qpos, np.arange(7, 36)) or not np.array_equal(self.robot_qvel, np.arange(6, 35)):
            raise ValueError('Native robot state order changed')
        self.reset()

    def reset(self, placement: str = 'board') -> dict:
        """Geometry placement for CPU tests, not a learned balance initialization.

        Rotate native stance 90 degrees so feet fit along the board length.
        Place the lowest foot sphere 1 mm above the 0.11 m board top.
        Ground reset moves the free board away to independently test foot-floor.
        """
        if placement not in ('board', 'ground'):
            raise ValueError('placement must be board or ground')
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot_qpos] = self.default_q
        self.data.qpos[:3] = [0, 0, .9]
        self.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        free = self.model.joint('skateboard/floating_base_joint_skateboard').id
        ba = self.model.jnt_qposadr[free]
        self.data.qpos[ba:ba+3] = [0 if placement == 'board' else 2, 0, .1]
        self.data.qpos[ba+3:ba+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        feet = [self.model.geom(f'{s}_ankle_roll_link_collision_{i}').id
                for s in ('left', 'right') for i in range(1, 5)]
        self.data.qpos[:2] -= self.data.geom_xpos[feet, :2].mean(axis=0)
        bottom = np.min(self.data.geom_xpos[feet, 2]-self.model.geom_size[feet, 0])
        target = .111 if placement == 'board' else .001
        self.data.qpos[2] += target-bottom
        self.data.ctrl[self.robot_actuators] = self.default_q
        self.data.ctrl[self.board_actuators] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, action: np.ndarray, *, substeps: int = 4) -> dict:
        """Native normalized action -> default + action_scale * action at 50 Hz.

        Only robot actuators are exposed. Original board springs retain zero
        targets; wheel actuators have zero gain and receive no propulsion command.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('action must be 29 finite values in native G1 joint order')
        if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps <= 0:
            raise ValueError('substeps must be a positive integer')
        self.data.ctrl[self.robot_actuators] = self.default_q + self.action_scale*action
        self.data.ctrl[self.board_actuators] = 0
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        # Contacts/forces and body transforms must describe the returned state.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Non-finite CPU simulation state')
        return self.observe()

    def observe(self) -> dict:
        from .skateboard_contacts import SkateboardContactReader
        if not hasattr(self, '_contacts'):
            self._contacts = SkateboardContactReader(self.model)
        return {
            'time_s': float(self.data.time),
            'robot_qpos': self.data.qpos[:36].tolist(),
            'robot_qvel': self.data.qvel[:35].tolist(),
            'interaction': self._contacts.read(self.data),
        }
