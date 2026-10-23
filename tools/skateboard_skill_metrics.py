"""Independent physical certification for autonomous push-and-skate.

This CPU-only module knows nothing about rewards, training loss, or the actor.
The left foot may push on the ground; the right foot is the riding support.
Held-out resets and thresholds are fixed before policy selection. A pass covers
only these scenarios, and does not imply command tracking or general mastery.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


HELD_OUT_RESETS = (
    {"name": "nominal", "root_xy_m": [0., 0.], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "offset_front_left", "root_xy_m": [.01, .01], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "offset_front_right", "root_xy_m": [.01, -.01], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "offset_rear_left", "root_xy_m": [-.01, .01], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "offset_rear_right", "root_xy_m": [-.01, -.01], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "tilt_positive", "root_xy_m": [0., 0.], "root_roll_pitch_rad": [.015, .015], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "tilt_negative", "root_xy_m": [0., 0.], "root_roll_pitch_rad": [-.015, -.015], "root_xy_velocity_m_s": [0., 0.]},
    {"name": "velocity_forward_left", "root_xy_m": [0., 0.], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [.03, .03]},
    {"name": "velocity_backward_right", "root_xy_m": [0., 0.], "root_roll_pitch_rad": [0., 0.], "root_xy_velocity_m_s": [-.03, -.03]},
)

CRITERIA = {
    "minimum_duration_s": 30.,
    "settling_grace_s": .2,
    "tail_duration_s": 5.,
    "active_normal_force_n": 5.,
    "minimum_pelvis_above_board_m": .45,
    "minimum_pelvis_up_dot_world_up": .65,
    "minimum_right_foot_board_fraction": .80,
    "minimum_right_foot_board_tail_fraction": .80,
    "maximum_no_foot_board_gap_s": .30,
    "maximum_right_foot_ground_gap_s": .06,
    "maximum_robot_board_xy_distance_m": .65,
    "minimum_board_forward_progress_m": 5.,
    "maximum_unintended_support_frames": 0,
    "maximum_physics_warning_count": 0,
}


def apply_reset_perturbation(env: Any, specification: dict) -> dict:
    """Apply one fixed initial-state perturbation; never modify ongoing dynamics."""
    data = env.data
    data.qpos[:2] += np.asarray(specification["root_xy_m"], dtype=float)
    roll, pitch = specification["root_roll_pitch_rad"]
    qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0., 0.])
    qp = np.array([np.cos(pitch / 2), 0., np.sin(pitch / 2), 0.])
    offset, updated = np.zeros(4), np.zeros(4)
    mujoco.mju_mulQuat(offset, qp, qr)
    mujoco.mju_mulQuat(updated, data.qpos[3:7].copy(), offset)
    data.qpos[3:7] = updated / np.linalg.norm(updated)
    data.qvel[:2] += np.asarray(specification["root_xy_velocity_m_s"], dtype=float)
    mujoco.mj_forward(env.model, data)
    return {"specification": copy.deepcopy(specification),
            "initial_qpos": data.qpos.tolist(), "initial_qvel": data.qvel.tolist()}


def _max_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


@dataclass(frozen=True)
class PhysicalFrame:
    time_s: float
    pelvis_above_board_m: float
    pelvis_up_dot: float
    robot_board_xy_distance_m: float
    board_forward_progress_m: float
    right_board: bool
    left_board: bool
    right_ground: bool
    left_ground: bool
    wheel_ground: bool
    unintended_support: bool


class SkateboardSkillMetrics:
    """Accumulate actual MuJoCo contacts after each control step.

    Construct after resetting the environment, then call ``update(obs)`` after
    every 0.02 s step. ``summarize`` never alters the policy or simulator.
    A shortened rollout always fails the final 30-second certification.
    """
    def __init__(self, env: Any, requested_seconds: float = 30.):
        if requested_seconds <= 0 or not np.isfinite(requested_seconds):
            raise ValueError("requested_seconds must be positive and finite")
        self.env = env
        self.requested_seconds = float(requested_seconds)
        self.start_time = float(env.data.time)
        reader = env.board_reader
        if reader is None:
            raise ValueError("A skateboard contact reader is required")
        self.board_initial = env.data.xpos[reader.root].copy()
        self.board_forward = env.data.xmat[reader.root].reshape(3, 3)[:2, 0].copy()
        self.board_forward /= np.linalg.norm(self.board_forward)
        self.floor = env.model.geom("floor").id
        self.robot_geoms = {
            g for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] != 0
            and not (env.model.body(int(env.model.geom_bodyid[g])).name or "").startswith("skateboard/")
        }
        self.foot_geoms = set.union(*reader.foot_geoms.values())
        # Any robot nonfoot geometry using the floor or skateboard for support
        # is a failure; this catches hands, knees, torso, and forearms.
        self.board_geoms = {
            g for g in range(env.model.ngeom)
            if (env.model.body(int(env.model.geom_bodyid[g])).name or "").startswith("skateboard/")
        }
        self.support_geoms = self.board_geoms | {self.floor}
        self.pelvis = int(env.model.jnt_bodyid[0])
        self.frames: list[PhysicalFrame] = []
        self.unintended_contact_names: set[str] = set()
        self.nonfinite = False

    def update(self, observation: dict | None = None) -> PhysicalFrame:
        e, data = self.env, self.env.data
        if observation is None:
            observation = e.observe()
        contacts = observation["interaction"]["contacts"]
        threshold = CRITERIA["active_normal_force_n"]
        active = lambda name: contacts[name]["normal_force"] > threshold
        root = data.qpos[:3]
        board = data.xpos[e.board_reader.root]
        wrench = np.zeros(6)
        unintended = False
        for i in range(data.ncon):
            pair = tuple(map(int, data.contact[i].geom))
            for robot, support in (pair, pair[::-1]):
                if robot in self.robot_geoms - self.foot_geoms and support in self.support_geoms:
                    mujoco.mj_contactForce(e.model, data, i, wrench)
                    if wrench[0] > threshold:
                        unintended = True
                        self.unintended_contact_names.add(e.model.geom(robot).name or str(robot))
        frame = PhysicalFrame(
            time_s=float(data.time) - self.start_time,
            pelvis_above_board_m=float(root[2] - board[2]),
            pelvis_up_dot=float(data.xmat[self.pelvis].reshape(3, 3)[2, 2]),
            robot_board_xy_distance_m=float(np.linalg.norm(root[:2] - board[:2])),
            board_forward_progress_m=float(np.dot(board[:2] - self.board_initial[:2], self.board_forward)),
            right_board=active("right_foot_board"), left_board=active("left_foot_board"),
            right_ground=active("right_foot_ground"), left_ground=active("left_foot_ground"),
            wheel_ground=active("wheel_ground"), unintended_support=unintended,
        )
        if self.frames and frame.time_s <= self.frames[-1].time_s:
            raise ValueError("Metrics updates must follow increasing simulator time")
        self.nonfinite |= not bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        self.frames.append(frame)
        return frame

    def summarize(self) -> dict:
        return summarize_frames(
            self.frames, requested_seconds=self.requested_seconds,
            warning_counts=[int(w.number) for w in self.env.data.warning],
            nonfinite=self.nonfinite,
            unintended_contact_names=sorted(self.unintended_contact_names),
        )


def summarize_frames(frames: list[PhysicalFrame], *, requested_seconds: float,
                     warning_counts: list[int], nonfinite: bool = False,
                     unintended_contact_names: list[str] | None = None) -> dict:
    """Pure metric reduction, also usable to audit a stored per-frame trace."""
    criteria = copy.deepcopy(CRITERIA)
    failures = []
    if not frames:
        return {"success": False, "failures": ["empty_rollout"], "criteria": criteria,
                "duration_s": 0., "frames": 0}
    times = np.array([f.time_s for f in frames])
    deltas = np.diff(np.r_[0., times])
    dt = float(np.median(deltas))
    if not np.allclose(deltas, .02, rtol=0, atol=1e-7):
        failures.append("control_steps_missing_or_not_50_hz")
    measured = [f for f in frames if f.time_s > criteria["settling_grace_s"] + 1e-8]
    tail_start = max(criteria["settling_grace_s"], times[-1] - criteria["tail_duration_s"])
    tail = [f for f in measured if f.time_s > tail_start + 1e-8]
    fraction = lambda rows, attr: float(np.mean([getattr(f, attr) for f in rows])) if rows else 0.
    no_board_gap = _max_run([not (f.right_board or f.left_board) for f in measured]) * dt
    right_ground_gap = _max_run([f.right_ground for f in measured]) * dt
    min_height = min(f.pelvis_above_board_m for f in frames)
    min_up = min(f.pelvis_up_dot for f in frames)
    max_distance = max(f.robot_board_xy_distance_m for f in frames)
    unwanted = sum(f.unintended_support for f in measured)
    checks = {
        "complete_30s_horizon": times[-1] + 1e-7 >= max(requested_seconds, criteria["minimum_duration_s"]),
        "finite_state": not nonfinite,
        "pelvis_height": min_height >= criteria["minimum_pelvis_above_board_m"],
        "pelvis_orientation": min_up >= criteria["minimum_pelvis_up_dot_world_up"],
        "right_foot_board_support": fraction(measured, "right_board") >= criteria["minimum_right_foot_board_fraction"],
        "right_foot_board_support_tail": fraction(tail, "right_board") >= criteria["minimum_right_foot_board_tail_fraction"],
        "no_prolonged_board_contact_loss": no_board_gap <= criteria["maximum_no_foot_board_gap_s"] + 1e-7,
        "no_right_foot_ground_support": right_ground_gap <= criteria["maximum_right_foot_ground_gap_s"] + 1e-7,
        "stay_with_board": max_distance <= criteria["maximum_robot_board_xy_distance_m"],
        "forward_board_progress": frames[-1].board_forward_progress_m >= criteria["minimum_board_forward_progress_m"],
        "no_unintended_body_support": unwanted <= criteria["maximum_unintended_support_frames"],
        "no_physics_warnings": sum(warning_counts) <= criteria["maximum_physics_warning_count"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema": 1, "task": "autonomous_push_and_skate", "success": not failures,
        "failures": failures, "checks": checks, "criteria": criteria,
        "scope": "Fixed physical scenario; no speed-command tracking or general mastery claim",
        "duration_s": float(times[-1]), "requested_seconds": float(requested_seconds),
        "frames": len(frames), "scored_frames_after_settling": len(measured),
        "minimum_pelvis_above_board_m": min_height, "minimum_pelvis_up_dot": min_up,
        "maximum_robot_board_xy_distance_m": max_distance,
        "board_forward_progress_m": frames[-1].board_forward_progress_m,
        "right_foot_board_fraction": fraction(measured, "right_board"),
        "right_foot_board_tail_fraction": fraction(tail, "right_board"),
        "left_foot_ground_fraction": fraction(measured, "left_ground"),
        "wheel_ground_fraction": fraction(measured, "wheel_ground"),
        "maximum_no_board_contact_gap_s": no_board_gap,
        "maximum_right_foot_ground_gap_s": right_ground_gap,
        "unintended_support_frames": unwanted,
        "unintended_contact_names": unintended_contact_names or [],
        "physics_warning_counts": warning_counts,
    }


__all__ = ["HELD_OUT_RESETS", "CRITERIA", "PhysicalFrame", "SkateboardSkillMetrics",
           "apply_reset_perturbation", "summarize_frames"]
