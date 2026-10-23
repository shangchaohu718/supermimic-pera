"""A learned first interaction skill with an immutable SuperMimic foundation.

The skateboard student is trained from scratch on local expert demonstrations
and on-policy DAgger labels. Inference needs neither ONNX nor an expert. Routing
preserves the original native tracking call exactly. This is the first module
of a progressive policy, not evidence of a universal interaction policy.

Observation conventions/constants adapted from TeleHuman/humanoid_skateboarding
commit d93833e80deff7f927c0b80ef9c435d8b5c488fe (CC BY-NC 4.0; see NOTICE).
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES

SOURCE_NAMES = tuple(n for n in G1_JOINT_NAMES if "wrist" not in n)
SOURCE_DEFAULT = np.array([0, 0, 0, .23, -.20, 0, -.7, 0, 0, 1.17, -.45, 0,
                           0, 0, 0, -.03, .45, -.21, 1.32, -.7, -.845, .83, 1.19])
SOURCE_SCALE = np.array([.5475, .3507, .5475, .3507, .4386, .4386,
                         .5475, .3507, .5475, .3507, .4386, .4386,
                         .5475, .4386, .4386] + [.4386] * 8)
OBS_DIMS = (2, 1, 3, 3, 23, 23, 23, 1)
OBS_STARTS = np.cumsum((0,) + OBS_DIMS[:-1])
OBS_SCHEMA = "husky_native29_proprio79_block_history5_v1"


def state_digest(state):
    """Stable digest of every parameter and buffer, independent of checkpoint ZIP bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class SkateboardObservation:
    """Convert actual native robot state/actions into the saved observation schema."""
    def __init__(self, env):
        self.env = env
        self.indices = np.array([env.joint_names.index(n) for n in SOURCE_NAMES])
        self.reset()

    def reset(self):
        self.history = deque([np.zeros(79, np.float32) for _ in range(5)], maxlen=5)
        self.previous = np.zeros(23, np.float32)
        self.step_number = 0

    def initial_qpos(self):
        q = self.env.default_qpos.copy()
        q[:7] = [-.03, .1, .78, 1, 0, 0, 0]
        q[7:] = 0
        q[7 + self.indices] = SOURCE_DEFAULT
        return q

    def encode(self, speed=1., heading_target=0.):
        d = self.env.data
        w, x, y, z = d.qpos[3:7]
        gravity = [2*(-z*x+w*y), -2*(z*y+w*x), 1-2*(w*w+z*z)]
        heading = np.arctan2(2*(x*y+w*z), 1-2*(y*y+z*z))
        self.step_number += 1
        phase = (self.step_number * .02 / 6) % 1
        obs = np.r_[2*speed, heading_target, heading/np.pi, d.qvel[3:6]*.25,
                    gravity, d.qpos[7+self.indices]-SOURCE_DEFAULT,
                    d.qvel[6+self.indices]*.05, self.previous, phase]
        self.history.append(obs.astype(np.float32))
        hist = np.asarray(self.history)
        return np.concatenate([hist[:, start:start+dim].ravel()
                               for start, dim in zip(OBS_STARTS, OBS_DIMS)]).astype(np.float32)

    def record_action(self, native_action):
        target = self.env.default_q + self.env.action_scale * native_action
        self.previous = ((target[self.indices]-SOURCE_DEFAULT)/SOURCE_SCALE).astype(np.float32)

    def source_to_native(self, action):
        target = np.zeros(29)
        target[self.indices] = SOURCE_DEFAULT + SOURCE_SCALE * action
        return ((target-self.env.default_q)/self.env.action_scale).astype(np.float32)


class SkateboardStudent(nn.Module):
    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        if not hidden or any(type(width) is not int or width <= 0 for width in hidden):
            raise ValueError("hidden must contain positive integer widths")
        self.hidden = tuple(hidden)
        self.statistics_frozen = False
        self.register_buffer("obs_mean", torch.zeros(395))
        self.register_buffer("obs_scale", torch.ones(395))
        self.register_buffer("action_mean", torch.zeros(29))
        self.register_buffer("action_scale", torch.ones(29))
        layers = []
        widths = (395, *self.hidden, 29)
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths)-2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def initialize_statistics(self, observations, actions):
        if self.statistics_frozen:
            raise RuntimeError("Certified skill normalization statistics are frozen")
        self.obs_mean.copy_(observations.mean(0))
        self.obs_scale.copy_(observations.std(0).clamp_min(.1))
        self.action_mean.copy_(actions.mean(0))
        self.action_scale.copy_(actions.std(0).clamp_min(.1))

    def forward(self, observations):
        return self.action_mean + self.action_scale * self.net(
            (observations-self.obs_mean)/self.obs_scale)


class RetainedSkillPolicy(nn.Module):
    """One checkpoint, explicit skill routing, frozen foundation and normalization.

New tasks may add trainable modules; certified modules must be frozen and pass
their saved physical regression suite before a new bundle is accepted. No
automatic switch between tracking and skateboard is claimed or performed.
"""
    def __init__(self, base, skateboard):
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False).eval()
        self.base.obs_norm.max_count = self.base.goal_norm.max_count = 0.
        self.skateboard = skateboard
        self.certified_skills = frozenset()

    def freeze_certified_skill(self, name):
        if name != "skateboard":
            raise ValueError(f"Unknown skill: {name}")
        self.skateboard.requires_grad_(False).eval()
        self.skateboard.statistics_frozen = True
        self.certified_skills = self.certified_skills | {name}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        if "skateboard" in self.certified_skills:
            self.skateboard.eval()
        return self

    def forward(self, observations, goal=None, *, task="tracking"):
        if task == "tracking":
            return self.base(observations, goal)
        if task == "skateboard":
            return self.skateboard(observations)
        raise ValueError(f"Uncertified/unknown skill: {task}")

    @classmethod
    def load(cls, path: str | Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("schema") != "pera_retained_skills_v1":
            raise ValueError("Expected pera_retained_skills_v1 checkpoint")
        spec = ckpt["model_spec"]
        if spec["observation_schema"] != OBS_SCHEMA or spec["contact_model"] != "husky_capsules":
            raise ValueError("Unknown observation/contact schema")
        if spec.get("action_space") != "native_normalized_29":
            raise ValueError("Unknown action space; expected native_normalized_29")
        base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
        student = SkateboardStudent(tuple(spec["hidden"]))
        result = cls(base, student)
        result.load_state_dict(ckpt["policy"], strict=True)
        for skill in ckpt.get("certified_skills", []):
            if skill != "skateboard":
                raise ValueError(f"Unknown certified skill: {skill}")
            expected = ckpt.get("certified_skill_sha256", {}).get(skill)
            if expected != state_digest(result.skateboard.state_dict()):
                raise ValueError("Certified skateboard parameters/statistics changed")
            result.freeze_certified_skill(skill)
        return result.eval(), ckpt
