#!/usr/bin/env python3
"""Independent CPU audit of a progressive policy's native tracking preservation.

Checks the native checkpoint SHA, every actor parameter/normalization buffer,
all local motions at four reference histories, and paired real MuJoCo rollouts.
The baseline actor and the bundle each receive their own physical state/history.
Passing means preservation equivalence; it does not mean tracking mastery or
certification of the new skateboard task. No training, downloads, or CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable', UV_OFFLINE='1',
                  HF_HUB_OFFLINE='1', WANDB_MODE='offline', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import mujoco
import numpy as np
import torch

from unisupermimic.interaction.features import native_proprio, state_from_observation
from unisupermimic.interaction.multitask_scene import G1InteractionCPU
from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.interaction.skateboard_scene import ROBOT_SCENE
from unisupermimic.interaction.skateboard_skill import RetainedSkillPolicy
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list


def snapshot(path: Path, loader):
    """Hash and load through one open descriptor across atomic path replacement."""
    with path.open('rb') as stream:
        value = loader(stream)
        stream.seek(0)
        digest = hashlib.sha256()
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return value, digest.hexdigest()


def mismatched_tensors(expected: dict, actual: dict) -> list[str]:
    return sorted(key for key in set(expected) | set(actual)
                  if key not in expected or key not in actual
                  or not torch.equal(expected[key], actual[key]))


def physics_matches(model: mujoco.MjModel, candidate: G1InteractionCPU) -> bool:
    other = candidate.model
    dimensions = ('nq', 'nv', 'nu', 'nbody', 'njnt', 'ngeom', 'npair', 'neq')
    fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'dof_armature',
              'jnt_range', 'geom_type', 'geom_size', 'geom_pos', 'geom_quat',
              'geom_friction', 'geom_solref', 'geom_solimp', 'geom_contype',
              'geom_conaffinity', 'actuator_gainprm', 'actuator_biasprm',
              'actuator_forcerange', 'actuator_trnid')
    options = ('timestep', 'integrator', 'solver', 'iterations', 'cone')
    return (candidate.xml is None
            and all(getattr(model, k) == getattr(other, k) for k in dimensions)
            and all(np.array_equal(getattr(model, k), getattr(other, k)) for k in fields)
            and all(getattr(model.opt, k) == getattr(other.opt, k) for k in options)
            and np.array_equal(model.opt.gravity, other.opt.gravity))


def observation_from_data(data: mujoco.MjData):
    return {'robot_qpos': data.qpos.copy(), 'robot_qvel': data.qvel.copy()}


@torch.inference_mode()
def paired_rollout(native, bundle, clip: ReferenceClip, steps: int) -> dict:
    # The baseline loads the original XML directly, independently of the adapter.
    model = mujoco.MjModel.from_xml_path(str(ROBOT_SCENE))
    data = mujoco.MjData(model)
    candidate = G1InteractionCPU('tracking')
    same_physics = physics_matches(model, candidate)
    if not same_physics:
        raise RuntimeError('Candidate tracking physics differs from original native XML')
    mujoco.mj_resetData(model, data)
    data.qpos[:] = clip.qpos[0]
    data.qvel[:] = clip.qvel[0]
    data.ctrl[:] = candidate.default_q
    mujoco.mj_forward(model, data)
    candidate.reset(clip.qpos[0], clip.qvel[0])
    histories = [None, None]
    previous = [torch.zeros(1, 29), torch.zeros(1, 29)]
    default_q = torch.tensor(default_joint_pos_list(), dtype=torch.float32)
    error = {'action': 0., 'qpos': 0., 'qvel': 0.}
    equality = {'action': True, 'qpos': True, 'qvel': True}
    first_mismatch = None
    first_below = [None, None]
    min_heights = [float(data.qpos[2]), float(candidate.data.qpos[2])]
    finite = True
    target_steps = min(steps, clip.num_frames - 1)
    for frame in range(target_steps):
        reference = unflatten_state(clip.state[frame:frame+1])
        goal = clip.goal(torch.tensor([frame]))
        actions = []
        for branch, simdata in enumerate((data, candidate.data)):
            state = state_from_observation(observation_from_data(simdata))
            current = native_proprio(state, previous[branch], reference, default_q)
            histories[branch] = (current[:, None].repeat(1, 10, 1) if histories[branch] is None
                                 else torch.cat((histories[branch][:, 1:], current[:, None]), 1))
            action = (native(histories[branch], goal) if branch == 0
                      else bundle(histories[branch], goal, task='tracking'))
            actions.append(action)
        if not all(bool(torch.isfinite(a).all()) for a in actions):
            finite = False
            first_mismatch = frame if first_mismatch is None else first_mismatch
            break
        same_action = torch.equal(actions[0], actions[1])
        equality['action'] &= same_action
        error['action'] = max(error['action'], float((actions[0]-actions[1]).abs().max()))
        # Same normalized native controls, separately evaluated and integrated.
        data.ctrl[:] = candidate.default_q + candidate.action_scale * actions[0][0].numpy()
        mujoco.mj_step(model, data, nstep=4)
        mujoco.mj_forward(model, data)
        candidate.step(actions[1][0].numpy())
        for key in ('qpos', 'qvel'):
            left, right = getattr(data, key), getattr(candidate.data, key)
            finite &= bool(np.isfinite(left).all() and np.isfinite(right).all())
            equality[key] &= np.array_equal(left, right)
            error[key] = max(error[key], float(np.max(np.abs(left-right))))
        if not all(equality.values()) and first_mismatch is None:
            first_mismatch = frame
        for branch, simdata in enumerate((data, candidate.data)):
            min_heights[branch] = min(min_heights[branch], float(simdata.qpos[2]))
            if simdata.qpos[2] < .45 and first_below[branch] is None:
                first_below[branch] = frame
        previous = actions
        if not finite:
            break
    warnings = [[int(w.number) for w in d.warning] for d in (data, candidate.data)]
    return {
        'clip': clip.name, 'clip_total_frames': clip.num_frames,
        'requested_window_steps': target_steps, 'simulated_steps': frame + 1,
        'simulated_seconds': float(data.time), 'full_clip_evaluation': target_steps == clip.num_frames - 1,
        'native_physics_unchanged': same_physics, 'finite': finite,
        'per_step_bitwise_equal': equality, 'maximum_absolute_errors': error,
        'first_mismatch_frame': first_mismatch,
        'numerical_warning_counts_native_and_candidate': warnings,
        'pelvis_min_height_native_and_candidate': min_heights,
        'first_pelvis_below_045m_frame_native_and_candidate': first_below,
        'passed_equivalence': bool(same_physics and finite and all(equality.values())
                                   and not any(any(counts) for counts in warnings)),
        'interpretation': 'Paired preservation audit; height diagnostics do not certify task success',
    }


@torch.inference_mode()
def audit(native_path: Path, candidate_path: Path, motions: Path,
          rollout_clips: int = 3, rollout_steps: int = 500) -> dict:
    if rollout_clips < 1 or not 1 <= rollout_steps <= 500:
        raise ValueError('Use at least one physical clip and 1..500 steps per clip')
    torch.set_num_threads(2)
    original, native_sha = snapshot(native_path,
        lambda stream: torch.load(stream, map_location='cpu', weights_only=True))
    (bundle, metadata), candidate_sha = snapshot(candidate_path, RetainedSkillPolicy.load)
    if 'pi' not in original:
        raise ValueError("Native checkpoint requires independently saved 'pi'")
    native = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    native.load_state_dict(original['pi'], strict=True)
    native.eval().requires_grad_(False)
    bundle.eval()
    original_base = original['pi']
    mismatched = mismatched_tensors(original_base, bundle.base.state_dict())
    norm_keys = sorted(k for k in original_base if '_norm.' in k)
    norm_exact = all(k not in mismatched for k in norm_keys)
    frozen = all(not p.requires_grad for p in bundle.base.parameters())
    norm_frozen = bundle.base.obs_norm.max_count == 0 and bundle.base.goal_norm.max_count == 0
    foundation_matches = metadata.get('foundation_sha256') == native_sha
    files = sorted(motions.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No local CSV motions: {motions}')
    default_q = torch.tensor(default_joint_pos_list(), dtype=torch.float32)
    sampled = []
    clip_lengths = []
    for path in files:
        clip = ReferenceClip.from_csv(path, G1_JOINT_NAMES)
        clip_lengths.append((clip.num_frames, path))
        frames = torch.linspace(0, clip.num_frames-1, 4).round().long()
        history_frames = (frames[:, None] + torch.arange(-9, 1)).clamp(min=0)
        state = unflatten_state(clip.state[history_frames.flatten()])
        history = native_proprio(state, torch.zeros(40, 29), state, default_q).reshape(4, 10, 99)
        goal = clip.goal(frames)
        expected, actual = native(history, goal), bundle(history, goal, task='tracking')
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        sampled.append({'clip': clip.name, 'frames': frames.tolist(), 'samples': 4,
                        'actions_bitwise_equal': finite and torch.equal(expected, actual),
                        'finite': finite,
                        'max_abs_action_error': float((expected-actual).abs().max()) if finite else None})
    physical = [paired_rollout(native, bundle, ReferenceClip.from_csv(path, G1_JOINT_NAMES), rollout_steps)
                for _, path in sorted(clip_lengths)[:rollout_clips]]
    # Running statistics must also remain identical after both sampled and physical calls.
    mismatched_after = mismatched_tensors(original_base, bundle.base.state_dict())
    sampled_exact = all(row['actions_bitwise_equal'] for row in sampled)
    physical_exact = all(row['passed_equivalence'] for row in physical)
    passed = (foundation_matches and not mismatched and not mismatched_after and norm_exact
              and frozen and norm_frozen and sampled_exact and physical_exact
              and not torch.cuda.is_initialized())
    return {
        'schema': 'pera_retained_skill_audit_v1', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(), 'new_download_bytes': 0,
        'native_checkpoint': str(native_path.resolve()), 'native_sha256': native_sha,
        'native_iteration': original.get('iteration'),
        'candidate_checkpoint': str(candidate_path.resolve()), 'candidate_sha256': candidate_sha,
        'candidate_schema': metadata.get('schema'),
        'recorded_foundation_matches': foundation_matches,
        'native_base_tensors_checked': len(original_base), 'native_base_tensors_exact': not mismatched,
        'mismatched_native_tensors': mismatched, 'mismatched_after_audit': mismatched_after,
        'normalization_tensors_checked': norm_keys, 'normalization_exact': norm_exact,
        'foundation_gradients_disabled': frozen, 'normalization_updates_disabled': norm_frozen,
        'clip_count': len(sampled), 'sample_count': sum(row['samples'] for row in sampled),
        'sampled_actions_bitwise_equal': sampled_exact,
        'max_sampled_abs_action_error': max(row['max_abs_action_error'] for row in sampled) if all(row['finite'] for row in sampled) else None,
        'physical_clip_count': len(physical), 'physical_total_steps': sum(row['simulated_steps'] for row in physical),
        'physical_rollouts_bitwise_equal': physical_exact, 'passed': bool(passed),
        'sample_protocol': 'Four evenly spaced frames per local clip, ten real reference states per history, zero prior actions',
        'physical_protocol': 'Three shortest local clips by default; up to500frames from each first frame; original XML/direct MuJoCo vs bundled tracking adapter; no resets after falling',
        'scope': 'Preservation equivalence of original tracking; not baseline mastery, complete40clip evaluation, skateboard certification, or automatic task switching',
        'samples': sampled, 'physical_rollouts': physical,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'candidate', 'motions', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--rollout-clips', type=int, default=3)
    p.add_argument('--rollout-steps', type=int, default=500)
    args = p.parse_args()
    result = audit(args.native, args.candidate, args.motions, args.rollout_clips, args.rollout_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix+'.tmp')
    temp.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    temp.replace(args.output)
    print(json.dumps({k: result[k] for k in ('passed', 'native_base_tensors_checked',
          'native_base_tensors_exact', 'sample_count', 'sampled_actions_bitwise_equal',
          'physical_total_steps', 'physical_rollouts_bitwise_equal', 'cuda_initialized')}, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
