"""Small CPU reference clips with the native SuperMimic goal convention.

Only root/joint state and small object state arrays are retained. There is no
tracked-body FK allocation, simulator import, download, or CUDA initialization.
CSV interpolation and finite differences intentionally match native MotionLib.
VAIC poses are wxyz, as specified by its official loader's slerp conversion.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    flatten_state,
    mat_to_6d,
    quat_slerp,
    quat_to_mat,
    so3_log,
    yaw_mat,
)

CONTROL_DT = 0.02


def _gradient(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    result[1:-1] = (value[2:] - value[:-2]) / (2 * CONTROL_DT)
    result[0] = (value[1] - value[0]) / CONTROL_DT
    result[-1] = (value[-1] - value[-2]) / CONTROL_DT
    return result


def _angular_velocity(rotation: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((len(rotation), 3), dtype=rotation.dtype)
    result[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    result[0] = so3_log(rotation[1] @ rotation[0].transpose(-1, -2)) / CONTROL_DT
    result[-1] = so3_log(rotation[-1] @ rotation[-2].transpose(-1, -2)) / CONTROL_DT
    return result


def _sampling(source_frames: int, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if source_frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Reference requires at least two frames and positive finite FPS")
    num_frames = math.floor(((source_frames - 1) / fps) / CONTROL_DT) + 1
    if num_frames < 2:
        raise ValueError("Reference duration must span at least two control frames")
    sample = torch.arange(num_frames, dtype=torch.float32) * CONTROL_DT * fps
    lower = sample.floor().long().clamp(max=source_frames - 2)
    return lower, (sample - lower.float()).clamp(0, 1)


def _lerp(value: torch.Tensor, lower: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    blend = alpha.reshape((-1,) + (1,) * (value.ndim - 1))
    return value[lower] * (1 - blend) + value[lower + 1] * blend


def _validate_joints(joint_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(joint_names)
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("The native G1 reference requires 29 unique joint names")
    return names


class ReferenceClip:
    """One 50 Hz clip, kept on CPU and indexed without allocating body FK.

    ``state`` is native flat state ``[p3, quat_wxyz4, v_world3, w_world3,
    joint_q29, joint_qd29]``; ``qpos`` and ``qvel`` are NumPy arrays suitable for
    MuJoCo's free joint (angular velocity is in the root/body frame).
    ``objects`` has shape ``(N, 27)``: free qpos7, internal joint qpos7,
    free qvel6, internal joint qvel7. Box internal joints are zero. Absent
    objects are all-zero and must be masked using ``object_mask``. Static
    terrain poses, when provided, are separate in ``terrain_objects``.
    """

    def __init__(
        self,
        name: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        quaternion: torch.Tensor,
        joints: torch.Tensor,
        *,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        joint_velocity: torch.Tensor | None = None,
        objects: np.ndarray | None = None,
        object_names: Sequence[str] = (),
        terrain_objects: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self.joint_names = _validate_joints(joint_names)
        self.num_frames = len(position)
        expected = ((self.num_frames, 3), (self.num_frames, 4), (self.num_frames, 29))
        if self.num_frames < 2 or any(
            tensor.shape != shape for tensor, shape in zip((position, quaternion, joints), expected)
        ):
            raise ValueError("Invalid root/joint reference dimensions")
        if any(tensor.device.type != "cpu" for tensor in (position, quaternion, joints)):
            raise ValueError("ReferenceClip is constructed from CPU tensors")
        if not all(torch.isfinite(tensor).all() for tensor in (position, quaternion, joints)):
            raise ValueError("Reference contains non-finite values")
        if bool((quaternion.norm(dim=-1) < 1e-8).any()):
            raise ValueError("Reference contains zero quaternions")
        self.position = position
        self.quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
        # quat_to_mat normalizes internally; use the source tensor so CSV
        # arithmetic stays bit-identical to MotionLib (no extra normalization).
        self.rotation = quat_to_mat(quaternion)
        self.linear_velocity = _gradient(position) if linear_velocity is None else linear_velocity
        self.angular_velocity = (
            _angular_velocity(self.rotation) if angular_velocity is None else angular_velocity
        )
        self.joint_pos = joints
        self.joint_vel = _gradient(joints) if joint_velocity is None else joint_velocity
        for velocity, shape in (
            (self.linear_velocity, (self.num_frames, 3)),
            (self.angular_velocity, (self.num_frames, 3)),
            (self.joint_vel, (self.num_frames, 29)),
        ):
            if velocity.shape != shape or not bool(torch.isfinite(velocity).all()):
                raise ValueError("Invalid reference velocities")
        inverse = self.rotation.transpose(-1, -2)
        self.linear_velocity_local = (inverse @ self.linear_velocity.unsqueeze(-1)).squeeze(-1)
        self.angular_velocity_local = (inverse @ self.angular_velocity.unsqueeze(-1)).squeeze(-1)
        self.state = flatten_state((
            self.position, self.rotation, self.linear_velocity,
            self.angular_velocity, self.joint_pos, self.joint_vel,
        ))
        self.qpos = torch.cat((position, self.quaternion, joints), dim=-1).numpy().copy()
        self.qvel = torch.cat((
            self.linear_velocity, self.angular_velocity_local, self.joint_vel,
        ), dim=-1).numpy().copy()
        if objects is not None and (objects.shape != (self.num_frames, 27) or not np.isfinite(objects).all()):
            raise ValueError("objects must contain finite (N, 27) packed state")
        self.objects = np.zeros((self.num_frames, 27), dtype=np.float32) if objects is None else objects
        self.object_names = tuple(object_names)
        self.object_mask = float(bool(self.object_names))
        self.terrain_objects = terrain_objects or {}

    @classmethod
    def from_csv(cls, path: str | Path, native_joint_names: Sequence[str]) -> ReferenceClip:
        """Load native 36-column LAFAN/G1 CSV, with xyzw root, from 30 to 50 Hz."""
        path = Path(path)
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [
                [float(value) for value in row]
                for row in csv.reader(stream)
                if row and not row[0].lstrip().startswith("#")
            ]
        raw = torch.tensor(rows, dtype=torch.float32)
        if raw.ndim != 2 or raw.shape[1] != 36 or raw.shape[0] < 2:
            raise ValueError(f"{path} must have at least two rows and 36 columns")
        lower, alpha = _sampling(len(raw), 30.0)
        quaternion = raw[:, [6, 3, 4, 5]]
        return cls(
            path.stem, native_joint_names,
            _lerp(raw[:, :3], lower, alpha),
            quat_slerp(quaternion[lower], quaternion[lower + 1], alpha),
            _lerp(raw[:, 7:36], lower, alpha),
        )

    @classmethod
    def from_vaic(cls, directory: str | Path, joint_names: Sequence[str]) -> ReferenceClip:
        """Load local VAIC motion.npz/meta.json with explicit joint-name remapping.

        Supplied world velocities are preserved (and linearly interpolated if
        needed). No pickle loading or import of VAIC/IsaacLab is required.
        """
        directory = Path(directory)
        names = _validate_joints(joint_names)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        source_joints = meta["joint_names"]
        if len(set(source_joints)) != len(source_joints) or any(name not in source_joints for name in names):
            raise ValueError("VAIC metadata is missing native joints or has duplicate joint names")
        body_names = meta["body_names"]
        if "pelvis" not in body_names or len(set(body_names)) != len(body_names):
            raise ValueError("VAIC metadata requires a unique pelvis body")
        root = body_names.index("pelvis")
        mapping = [source_joints.index(name) for name in names]
        with np.load(directory / "motion.npz", allow_pickle=False) as data:
            tensors = {
                key: torch.as_tensor(np.array(data[key], dtype=np.float32, copy=True))
                for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel")
            }
        frames = len(tensors["joint_pos"])
        if any(len(value) != frames for value in tensors.values()):
            raise ValueError("VAIC arrays have inconsistent frame counts")
        lower, alpha = _sampling(frames, float(meta["fps"]))
        # At native 50 Hz preserve samples exactly, avoiding floating-index drift.
        same_rate = float(meta["fps"]) == 50.0
        def linear(value):
            return value if same_rate else _lerp(value, lower, alpha)
        def orientation(value):
            return value if same_rate else quat_slerp(value[lower], value[lower + 1], alpha)
        robot_bodies = {"pelvis", "torso_link", *(name.replace("_joint", "_link") for name in names)}
        extra_names = tuple(name for name in body_names if name not in robot_bodies)
        if any(name in extra_names for name in ("skateboard", "cart")):
            raise ValueError("VAIC board/cart geometry cannot initialize the HUSKY/box scene")
        object_names = ("mybox",) if "mybox" in extra_names else ()
        objects = None
        terrain_objects = {}
        count = frames if same_rate else len(lower)
        if object_names:
            objects = np.zeros((count, 27), dtype=np.float32)
            body = body_names.index("mybox")
            object_quat = orientation(tensors["body_quat_w"][:, body])
            objects[:, :3] = linear(tensors["body_pos_w"][:, body]).numpy()
            objects[:, 3:7] = object_quat.numpy()
            objects[:, 14:17] = linear(tensors["body_lin_vel_w"][:, body]).numpy()
            angular_world = linear(tensors["body_ang_vel_w"][:, body])
            objects[:, 17:20] = (
                quat_to_mat(object_quat).transpose(-1, -2) @ angular_world.unsqueeze(-1)
            ).squeeze(-1).numpy()
        for name in extra_names:
            if name == "mybox":
                continue
            body = body_names.index(name)
            terrain_objects[name] = torch.cat((
                linear(tensors["body_pos_w"][:, body]),
                orientation(tensors["body_quat_w"][:, body]),
            ), dim=-1).numpy()
        return cls(
            directory.name, names,
            linear(tensors["body_pos_w"][:, root]),
            orientation(tensors["body_quat_w"][:, root]),
            linear(tensors["joint_pos"][:, mapping]),
            linear_velocity=linear(tensors["body_lin_vel_w"][:, root]),
            angular_velocity=linear(tensors["body_ang_vel_w"][:, root]),
            joint_velocity=linear(tensors["joint_vel"][:, mapping]),
            objects=objects, object_names=object_names, terrain_objects=terrain_objects,
        )

    def goal(self, indices: torch.Tensor, num_frames: int = 10) -> torch.Tensor:
        """Native 73-feature future frames in current reference heading, nearest first.

        Keeps clip storage on CPU, computes only the selected batch, and returns
        the small goal batch on the indices' device. End-of-clip frames clamp.
        """
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be a 1-D integer tensor")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        frames = indices.detach().cpu().long()
        if bool(((frames < 0) | (frames >= self.num_frames)).any()):
            raise IndexError("reference frame out of range")
        future = (frames[:, None] + torch.arange(1, num_frames + 1)).clamp(max=self.num_frames - 1)
        heading_t = yaw_mat(self.rotation[frames]).transpose(-1, -2).unsqueeze(1)
        position_delta = (
            heading_t @ (self.position[future] - self.position[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        rotation = mat_to_6d(heading_t @ self.rotation[future])
        features = torch.cat((
            position_delta, rotation, self.linear_velocity_local[future],
            self.angular_velocity_local[future], self.joint_pos[future], self.joint_vel[future] * 0.1,
        ), dim=-1)
        return features.reshape(len(frames), num_frames * 73).to(indices.device)


__all__ = ["ReferenceClip"]
