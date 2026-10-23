#!/usr/bin/env python3
"""Independent CPU certification of a learned retained skateboard skill.

Uses only the saved PyTorch policy and real MuJoCo physics. Fixed held-out
resets/criteria come from skateboard_skill_metrics; neither training nor an
expert is imported. The report does not change checkpoint certification status.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", UV_OFFLINE="1",
                  HF_HUB_OFFLINE="1", WANDB_MODE="offline", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np
import torch

from unisupermimic.interaction.multitask_scene import G1InteractionCPU
from unisupermimic.interaction.skateboard_skill import RetainedSkillPolicy, SkateboardObservation
from unisupermimic.rl.algos.supermimic import networks
from skateboard_skill_metrics import (
    CRITERIA, HELD_OUT_RESETS, SkateboardSkillMetrics, apply_reset_perturbation,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_attribution() -> dict:
    manifest = ROOT / "src/unisupermimic/assets/objects/husky_skateboard.source.json"
    return {
        "source": json.loads(manifest.read_text(encoding="utf-8")),
        "adaptations": "HUSKY board geometry, capsule foot contacts, observation conventions; learned PyTorch student inference",
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "local_license": str(ROOT / "LICENSES/HUSKY-CC-BY-NC-4.0.md"),
        "notice": str(ROOT / "NOTICE"),
        "inference_uses_expert": False,
    }


@torch.inference_mode()
def evaluate_reset(policy, specification: dict, seconds: float,
                   trajectory_dir: Path | None = None) -> dict:
    env = G1InteractionCPU("skateboard", skateboard_foot_contacts="husky_capsules")
    encoder = SkateboardObservation(env)
    env.reset(encoder.initial_qpos(), np.zeros(35))
    reset = apply_reset_perturbation(env, specification)
    encoder.reset()
    metrics = SkateboardSkillMetrics(env, requested_seconds=seconds)
    qpos = [env.data.qpos.copy()]
    qvel = [env.data.qvel.copy()]
    times = [float(env.data.time)]
    actions = []
    runtime_error = None
    early_stop = None
    for _ in range(round(seconds * 50)):
        encoded = encoder.encode(speed=1., heading_target=0.)
        action = policy(torch.from_numpy(encoded)[None], task="skateboard")[0].numpy()
        if action.shape != (29,) or not np.isfinite(action).all():
            runtime_error = "nonfinite_or_invalid_student_action"
            metrics.nonfinite = True
            break
        try:
            observation = env.step(action)
        except RuntimeError as error:
            runtime_error = str(error)
            metrics.nonfinite = not bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
            break
        encoder.record_action(action)
        frame = metrics.update(observation)
        actions.append(action.copy())
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        times.append(float(env.data.time))
        if frame.pelvis_above_board_m < .35 or frame.robot_board_xy_distance_m > 1.:
            early_stop = "fallen_or_separated_from_board"
            break
    result = metrics.summarize()
    result.update(name=specification["name"], reset=reset, runtime_error=runtime_error,
                  early_stop=early_stop, source_policy="learned_pytorch_student_only",
                  command={"speed": 1., "heading_target": 0.},
                  scene_metadata=env.metadata)
    if runtime_error is not None:
        result["success"] = False
        result["failures"].append("runtime_error")
    if trajectory_dir is not None:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        name = specification["name"]
        frames = [asdict(frame) for frame in metrics.frames]
        columns = list(frames[0]) if frames else []
        physical = np.array([[frame[column] for column in columns] for frame in frames], dtype=float)
        trajectory = trajectory_dir / f"{name}.npz"
        np.savez_compressed(trajectory, qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            action=np.asarray(actions).reshape(-1, 29), time_s=np.asarray(times),
                            physical_metrics=physical, physical_metric_columns=np.asarray(columns),
                            control_hz=50, physics_hz=200)
        model_path = trajectory_dir / f"{name}.xml"
        model_path.write_text(env.xml, encoding="utf-8")
        result["trajectory"] = str(trajectory.resolve())
        result["model_xml"] = str(model_path.resolve())
        result["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        result["model_xml_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Report JSON path")
    parser.add_argument("--seconds", type=float, choices=(30., 60.), default=30.)
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    networks.AMP_DTYPE = None
    # Loading the same bytes that are hashed avoids a race with atomic candidate publication.
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    policy, metadata = RetainedSkillPolicy.load(io.BytesIO(checkpoint_bytes))
    policy.cpu().eval().requires_grad_(False)
    del checkpoint_bytes
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "schema": metadata["schema"], "round": metadata.get("round"),
        "foundation_sha256": metadata.get("foundation_sha256"),
        "expert_sha256_training_provenance_only": metadata.get("expert_sha256"),
        "model_spec": metadata["model_spec"], "checkpoint_status": metadata.get("status"),
        "checkpoint_certified_skills": metadata.get("certified_skills", []),
    }
    del metadata
    trajectories = (args.output.parent / f"{args.output.stem}_trajectories"
                    if args.save_trajectories else None)
    results = []
    started = time.monotonic()
    for specification in HELD_OUT_RESETS:
        result = evaluate_reset(policy, specification, args.seconds, trajectories)
        results.append(result)
        print(json.dumps({"scenario": result["name"], "success": result["success"],
                          "duration_s": result["duration_s"], "failures": result["failures"]}), flush=True)
    expert_imported = any(name == "onnxruntime" or name.startswith("onnxruntime.") for name in sys.modules)
    trainer_imported = "train_skateboard_skill" in sys.modules
    if expert_imported or trainer_imported:
        raise RuntimeError("Independent evaluation must not import ONNX runtime or training")
    attribution = source_attribution()
    report = {
        "schema": "pera_skateboard_independent_certification_v1",
        "task": "autonomous_push_and_skate", "policy_device": "cpu", "physics": "mujoco_cpu",
        "mujoco_version": mujoco.__version__, "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "amp_dtype": None,
        "expert_imported": expert_imported, "training_imported": trainer_imported,
        "new_download_bytes": 0, "elapsed_seconds": time.monotonic() - started,
        "seconds_per_reset": args.seconds, "provenance": provenance,
        "criteria": CRITERIA, "held_out_resets": list(HELD_OUT_RESETS),
        "scenario_count": len(results), "passed_scenarios": sum(row["success"] for row in results),
        "success": all(row["success"] for row in results), "results": results,
        "source_attribution": attribution,
        "limitations": ["Fixed nine small initial-state perturbations at one command",
                        "This pass does not establish speed-command tracking or general mastery",
                        "Evaluation does not update checkpoint status or select further training candidates",
                        "The learned skill uses native 29-joint actions with HUSKY capsule foot contacts"],
    }
    atomic_json(args.output, report)
    if trajectories is not None:
        atomic_json(trajectories / "attribution.json", attribution)
        (trajectories / "HUSKY-CC-BY-NC-4.0.md").write_text(
            (ROOT / "LICENSES/HUSKY-CC-BY-NC-4.0.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved {args.output}: {report['passed_scenarios']}/{report['scenario_count']} passed", flush=True)


if __name__ == "__main__":
    main()
