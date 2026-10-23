#!/usr/bin/env python3
"""Independent, download-free CPU physical evaluation of an interaction actor.

Example: .venv-local/bin/python tools/evaluate_interaction_cpu.py \
  --native /local/native/last.pt --checkpoint /local/interaction/accepted.pt \
  --motions /local/lafan --vaic-root /local/vaic/g1/mirobotA \
  --output outputs/interaction_eval.json --steps 512 --save-trajectories

This checks a single deterministic initial condition per task. Passing is not
evidence of broad robustness, generalization, or mastery of all local motions.
No trainer is constructed, no parameters change, and CUDA is disabled before
Torch is imported. Optional saved qpos/XML can be rendered later on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", UV_OFFLINE="1",
                  HF_HUB_OFFLINE="1", WANDB_MODE="offline", OMP_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from unisupermimic.interaction.features import (
    context_features, native_proprio, reward_and_done, state_from_observation,
)
from unisupermimic.interaction.multitask_scene import G1InteractionCPU, TASKS
from unisupermimic.interaction.networks import RetainedInteractionPolicy
from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state


CRITERIA = {
    "fall": {"pelvis_above_support_below_m": .35, "pelvis_up_dot_world_up_below": .45},
    "palm_contact_active_normal_force_above_n": 5.,
    "box": {"box_center_lift_above_initial_m": .10,
            "simultaneous_both_palm_contact_and_lift_consecutive_frames": 10,
            "no_fall_entire_rollout": True},
    "box_obstacle": {"robot_and_box_final_world_x_above_m": 1.22,
                     "box_final_center_z_above_m": .28,
                     "simultaneous_both_palm_contact_and_lift_consecutive_frames": 10,
                     "no_fall_entire_rollout": True},
    "stairs": {"robot_final_world_x_above_m": 1.88,
               "robot_final_world_z_above_m": .95, "no_fall_entire_rollout": True},
    "skateboard": {"board_final_xy_displacement_above_m": .5,
                   "both_feet_on_board_fraction_at_least": .8,
                   "each_foot_active_normal_force_above_n": 5.,
                   "no_fall_entire_rollout": True},
}


def read_checkpoint_once(path: Path) -> tuple[dict, str]:
    """Snapshot one published file, so concurrent atomic replacement cannot mix versions."""
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    value = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected checkpoint dictionary: {path}")
    return value, digest


def load_policy(native_path: Path, checkpoint_path: Path) -> tuple[RetainedInteractionPolicy, dict]:
    native, native_sha = read_checkpoint_once(native_path)
    candidate, candidate_sha = read_checkpoint_once(checkpoint_path)
    if "policy" not in candidate or "pi" not in native:
        raise ValueError("Expected native pi and interaction policy checkpoint entries")
    if candidate.get("foundation_sha256") != native_sha:
        raise ValueError("Interaction checkpoint does not match the supplied native foundation SHA256")
    base = networks.Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    base.load_state_dict(native["pi"], strict=True)
    policy = RetainedInteractionPolicy(base)
    policy.load_state_dict(candidate["policy"], strict=True)
    for key, value in base.state_dict().items():
        if not torch.equal(policy.base.state_dict()[key], value):
            raise ValueError(f"Frozen native foundation differs at {key}")
    if "stable_actor" in candidate:
        for key, value in candidate["stable_actor"].items():
            if not torch.equal(policy.residual.state_dict()[key], value):
                raise ValueError("Checkpoint policy is an unaccepted candidate; use accepted.pt")
    policy.eval().requires_grad_(False)
    provenance = {"native_checkpoint": str(native_path.resolve()), "native_sha256": native_sha,
                  "native_iteration": int(native.get("iteration", -1)),
                  "interaction_checkpoint": str(checkpoint_path.resolve()),
                  "interaction_sha256": candidate_sha,
                  "interaction_iteration": int(candidate.get("iteration", -1)),
                  "checkpoint_schema": candidate.get("schema"),
                  "frozen_foundation_identical": True}
    return policy, provenance


def reference_at(clip: ReferenceClip, frame: int):
    index = min(frame, clip.num_frames - 1)
    return (unflatten_state(clip.state[index:index+1]), clip.goal(torch.tensor([index])),
            torch.as_tensor(clip.objects[index:index+1], dtype=torch.float32))


def task_reference(task: str, env: G1InteractionCPU, motions: Path, vaic_root: Path,
                   steps: int) -> ReferenceClip:
    if task == "stairs":
        paths = sorted(motions.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No local native motion CSVs in {motions}")
        source = next((path for path in paths if path.stem.startswith("walk")), paths[0])
        return ReferenceClip.from_csv(source, env.joint_names)
    if task in ("box", "box_obstacle"):
        return ReferenceClip.from_vaic(vaic_root / "box_1", env.joint_names)
    qpos = torch.tensor(env.observe()["robot_qpos"], dtype=torch.float32)
    length = max(1000, steps + 11)
    return ReferenceClip("board_stance", env.joint_names, qpos[:3].repeat(length, 1),
                         qpos[3:7].repeat(length, 1), qpos[7:].repeat(length, 1))


@torch.inference_mode()
def evaluate_task(policy: RetainedInteractionPolicy, task: str, motions: Path,
                  vaic_root: Path, steps: int, trajectory_dir: Path | None = None) -> dict:
    env = G1InteractionCPU(task, seed=123)
    clip = task_reference(task, env, motions, vaic_root, steps)
    initial_object = clip.objects[0].copy() if task in ("box", "box_obstacle") else None
    if initial_object is not None:
        initial_object[2] = max(.181, initial_object[2])
    obs = env.reset(clip.qpos[0], clip.qvel[0], initial_object)
    initial_root = np.array(obs["robot_qpos"][:3])
    initial_obj = np.array(obs["object_state"][:3])
    default = torch.tensor(env.default_q, dtype=torch.float32)
    task_id = torch.tensor([TASKS.index(task)])
    terrain = {TASKS.index(task): torch.tensor(env.terrain_boxes, dtype=torch.float32)}
    history = None
    previous = torch.zeros(1, 29)
    qpos_trace = [env.data.qpos.copy()]
    time_trace = [0.]
    root_trace = [initial_root]
    object_trace = [initial_obj]
    palm_frames = {"left": 0, "right": 0}
    palm_peak_n = {"left": 0., "right": 0.}
    foot_board_frames = {"left": 0, "right": 0}
    both_board_frames = 0
    both_palm_run = both_palm_max = lift_contact_run = lift_contact_max = 0
    max_lift = 0.
    dense_return = 0.
    fallen = False
    fall_reason = None
    for step in range(steps):
        state = state_from_observation(obs, "cpu")
        obj = torch.tensor(obs["object_state"], dtype=torch.float32)[None]
        reference, goal, ref_object = reference_at(clip, step)
        current = native_proprio(state, previous, reference, default)
        history = (current[:, None].repeat(1, 10, 1) if history is None
                   else torch.cat((history[:, 1:], current[:, None]), dim=1))
        target = ref_object[:, :3] if task in ("box", "box_obstacle") else reference[0]
        heights = torch.stack((reference[0][:, 2], target[:, 2]), dim=-1)
        context = context_features(state, obj, task_id, target, heights, terrain)
        action = policy(history, goal, context, torch.ones(1))
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError(f"Nonfinite actor output for {task} at step {step}")
        obs = env.step(action[0].numpy())
        next_state = state_from_observation(obs, "cpu")
        next_obj = torch.tensor(obs["object_state"], dtype=torch.float32)[None]
        next_reference, _, next_ref_obj = reference_at(clip, step + 1)
        target = next_ref_obj[:, :3] if task in ("box", "box_obstacle") else next_reference[0]
        reward, _ = reward_and_done(next_state, next_obj, action, previous, next_reference,
                                    target, task_id, terrain)
        dense_return += float(reward[0])
        previous = action
        root = np.array(obs["robot_qpos"][:3])
        obj_xyz = np.array(obs["object_state"][:3])
        qpos_trace.append(env.data.qpos.copy())
        time_trace.append(obs["time_s"])
        root_trace.append(root)
        object_trace.append(obj_xyz)
        if task in ("box", "box_obstacle"):
            active = []
            for side in ("left", "right"):
                force = obs["contacts"][f"{side}_palm_object"]["normal_force"]
                palm_peak_n[side] = max(palm_peak_n[side], force)
                active.append(force > CRITERIA["palm_contact_active_normal_force_above_n"])
                palm_frames[side] += int(active[-1])
            both_palm_run = both_palm_run + 1 if all(active) else 0
            both_palm_max = max(both_palm_max, both_palm_run)
            lift = float(obj_xyz[2] - initial_obj[2])
            max_lift = max(max_lift, lift)
            lift_contact_run = (lift_contact_run + 1
                                if all(active) and lift > .1 else 0)
            lift_contact_max = max(lift_contact_max, lift_contact_run)
        if task == "skateboard":
            active = []
            for side in ("left", "right"):
                force = obs["interaction"]["contacts"][f"{side}_foot_board"]["normal_force"]
                active.append(force > 5.)
                foot_board_frames[side] += int(active[-1])
            both_board_frames += int(all(active))
        support = obj_xyz[2] if task == "skateboard" else float(env.query_terrain(root[:2]))
        above_support = float(root[2] - support)
        upright = float(next_state[1][0, 2, 2])
        if above_support < .35 or upright < .45:
            fallen = True
            fall_reason = ("pelvis_below_support_threshold" if above_support < .35
                           else "pelvis_tilt_exceeds_threshold")
            break

    count = len(root_trace) - 1
    root_array, object_array = np.array(root_trace), np.array(object_trace)
    final_root, final_obj = root_array[-1], object_array[-1]
    root_delta = final_root - initial_root
    object_delta = final_obj - initial_obj
    completed_horizon = count == steps
    stable = completed_horizon and not fallen
    both_board_fraction = both_board_frames / count
    if task == "box":
        success = stable and max_lift > .1 and lift_contact_max >= 10
    elif task == "box_obstacle":
        success = (stable and final_root[0] > 1.22 and final_obj[0] > 1.22
                   and final_obj[2] > .28 and lift_contact_max >= 10)
    elif task == "stairs":
        success = stable and final_root[0] > 1.88 and final_root[2] > .95
    else:
        success = stable and np.linalg.norm(object_delta[:2]) > .5 and both_board_fraction >= .8
    result = {
        "task": task, "reference": clip.name, "success": bool(success),
        "scope": "One deterministic real CPU physics rollout; criteria do not establish general mastery",
        "requested_steps": steps, "completed_steps": count, "duration_s": count * .02,
        "completed_horizon": completed_horizon, "fallen": fallen, "fall_reason": fall_reason,
        "dense_return": dense_return, "root_initial_xyz": initial_root.tolist(),
        "root_final_xyz": final_root.tolist(), "root_displacement_xyz": root_delta.tolist(),
        "root_xy_displacement_m": float(np.linalg.norm(root_delta[:2])),
        "root_min_height_m": float(root_array[:, 2].min()),
        "root_max_height_m": float(root_array[:, 2].max()),
        "physics_warning_counts": [int(w.number) for w in env.data.warning],
        "scene_metadata": env.metadata,
    }
    if task in ("box", "box_obstacle", "skateboard"):
        result.update(object_initial_xyz=initial_obj.tolist(), object_final_xyz=final_obj.tolist(),
                      object_displacement_xyz=object_delta.tolist(),
                      object_xy_displacement_m=float(np.linalg.norm(object_delta[:2])))
    if task in ("box", "box_obstacle"):
        result.update(box_max_center_lift_above_initial_m=max_lift,
                      box_final_center_lift_above_initial_m=float(object_delta[2]),
                      palm_contact_frames=palm_frames, palm_peak_normal_force_n=palm_peak_n,
                      both_palm_contact_max_consecutive_frames=both_palm_max,
                      lifted_both_palm_contact_max_consecutive_frames=lift_contact_max,
                      robot_past_obstacle=bool(final_root[0] > 1.22),
                      box_past_obstacle=bool(final_obj[0] > 1.22))
    if task == "skateboard":
        result.update(foot_board_contact_frames=foot_board_frames,
                      both_feet_board_contact_fraction=both_board_fraction)
    if trajectory_dir is not None:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory_dir / f"{task}.npz", qpos=np.array(qpos_trace),
                            time_s=np.array(time_trace), root_xyz=root_array,
                            object_xyz=object_array)
        (trajectory_dir / f"{task}.xml").write_text(env.xml, encoding="utf-8")
        result["trajectory"] = str((trajectory_dir / f"{task}.npz").resolve())
        result["model_xml"] = str((trajectory_dir / f"{task}.xml").resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("native", "checkpoint", "motions", "vaic-root", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.set_num_threads(2)
    networks.AMP_DTYPE = None
    started = time.monotonic()
    policy, provenance = load_policy(args.native, args.checkpoint)
    trajectories = (args.output.parent / f"{args.output.stem}_trajectories"
                    if args.save_trajectories else None)
    results = {}
    for task in TASKS[1:]:
        results[task] = evaluate_task(policy, task, args.motions, args.vaic_root,
                                      args.steps, trajectories)
        print(json.dumps({"task": task, "success": results[task]["success"],
                          "steps": results[task]["completed_steps"],
                          "fallen": results[task]["fallen"]}), flush=True)
    report = {"schema": 1, "backend": "mujoco_cpu", "policy_device": "cpu",
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "amp_dtype": None, "new_download_bytes": 0,
              "elapsed_seconds": time.monotonic() - started, "provenance": provenance,
              "criteria": CRITERIA, "results": results,
              "passed_task_count": sum(result["success"] for result in results.values()),
              "task_count": len(results),
              "limitations": ["One deterministic start per task; no domain randomization",
                              "Flat box_1 prior is reused on the simple obstacle scene",
                              "The native 2 kg primitive box differs from VAIC's source box",
                              "Dense return is not a task-success criterion",
                              "This is inference-only evaluation; no training or parameter updates"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {args.output}; passed {report['passed_task_count']}/{report['task_count']}", flush=True)


if __name__ == "__main__":
    main()
