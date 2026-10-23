#!/usr/bin/env python3
"""Offline native29 skateboard behavior cloning + real-physics DAgger.

Fresh student weights, frozen native SuperMimic foundation, local CPU expert.
No downloads. GPU is used only for small supervised updates with a hard Torch
allocator cap and sampled process-memory checks. All rollouts run on CPU.
Held-out certification is a separate command, never used to select candidates.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", WANDB_MODE="offline",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import onnxruntime as ort
import torch
from unisupermimic.interaction.multitask_scene import G1InteractionCPU
from unisupermimic.interaction.skateboard_skill import (
    OBS_SCHEMA, RetainedSkillPolicy, SkateboardObservation, SkateboardStudent,
)
from unisupermimic.rl.algos.supermimic.networks import Policy
from skateboard_skill_metrics import SkateboardSkillMetrics, apply_reset_perturbation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LocalExpert:
    def __init__(self, path):
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opt,
                                            providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name
        assert self.session.get_inputs()[0].shape == [1, 395]

    def __call__(self, obs, encoder):
        action = self.session.run(None, {self.name: obs[None]})[0].reshape(23)
        return encoder.source_to_native(action)


def environment():
    env = G1InteractionCPU("skateboard", skateboard_foot_contacts="husky_capsules")
    return env, SkateboardObservation(env)


def training_reset(env, encoder, rng, perturb=True):
    q = encoder.initial_qpos()
    if perturb:
        q[:2] += rng.uniform(-.012, .012, 2)
        q[7:] += rng.normal(0, .008, 29)
        q[2] += rng.uniform(0, .004)
    env.reset(q, np.zeros(35))
    if perturb:
        apply_reset_perturbation(env, {
            "name": "training_random", "root_xy_m": [0., 0.],
            "root_roll_pitch_rad": rng.uniform(-.018, .018, 2).tolist(),
            "root_xy_velocity_m_s": rng.uniform(-.04, .04, 2).tolist(),
        })
    encoder.reset()


@torch.inference_mode()
def collect(expert, student, *, episodes, seconds, beta, seed):
    """Labels at the student's visited states; previous-action input is actual action."""
    rng = np.random.default_rng(seed)
    env, encoder = environment()
    observations, actions, lengths = [], [], []
    for episode in range(episodes):
        training_reset(env, encoder, rng, perturb=episode > 0)
        for step in range(round(seconds*50)):
            obs = encoder.encode()
            label = expert(obs, encoder)
            observations.append(obs)
            actions.append(label)
            if student is None:
                action = label
            else:
                prediction = student(torch.from_numpy(obs)[None])[0].numpy()
                action = beta*label + (1-beta)*prediction
            result = env.step(action)
            encoder.record_action(action)
            # Retain failure-state labels, reset before collecting unusable fallen states.
            if (env.data.qpos[2] < .50 or np.linalg.norm(env.data.qpos[:2]-env.data.qpos[36:38]) > .8
                    or not np.isfinite(env.data.qpos).all()):
                break
        lengths.append(step+1)
    return np.asarray(observations), np.asarray(actions), lengths


@torch.inference_mode()
def evaluate_student(student, *, seeds=(7101, 7102, 7103), seconds=30.):
    results = []
    for seed in seeds:
        env, encoder = environment()
        training_reset(env, encoder, np.random.default_rng(seed), perturb=seed != 7101)
        metrics = SkateboardSkillMetrics(env, requested_seconds=seconds)
        for step in range(round(seconds*50)):
            obs = encoder.encode()
            action = student(torch.from_numpy(obs)[None])[0].numpy()
            result = env.step(action)
            encoder.record_action(action)
            frame = metrics.update(result)
            if frame.pelvis_above_board_m < .35 or frame.robot_board_xy_distance_m > 1.:
                break
        summary = metrics.summarize()
        summary["development_seed"] = seed
        results.append(summary)
    return results


def memory_mib():
    value = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
    ], text=True)
    return max([int(row.split(",")[1]) for row in value.splitlines()
                if row.split(",")[0].strip() == str(os.getpid())] or [0])


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        # At most 2 GiB of Torch allocations leaves room for the CUDA context.
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(2048*1024**2/total, 0)
    native = torch.load(args.native, map_location="cpu", weights_only=True)
    base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    base.load_state_dict(native["pi"], strict=True)
    del native
    base.requires_grad_(False).eval()
    expert = LocalExpert(args.expert)
    student = SkateboardStudent().to(device)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(initialization="random_student_weights_no_expert_weight_copy",
                  algorithm="behavior_cloning_then_real_physics_DAgger",
                  expert_sha256=sha256(args.expert), foundation_sha256=sha256(args.native),
                  task="autonomous_push_and_skate", command_speed=1., command_heading=0.,
                  certification_used_for_selection=False, new_download_bytes=0,
                  allocator_cap_mib=2048, process_guard_mib=3800)
    (args.output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    started = time.monotonic()
    x, y, lengths = collect(expert, None, episodes=args.initial_episodes,
                            seconds=18., beta=1., seed=args.seed+1000)
    print(json.dumps({"event":"expert_dataset", "samples":len(x), "lengths":lengths}), flush=True)
    student.initialize_statistics(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    peak = 0
    logs = []
    stable_passes = 0
    for round_index in range(args.rounds):
        if round_index:
            cpu_student = copy.deepcopy(student).cpu().eval()
            beta = [0., .5, .25, .1][round_index] if round_index < 4 else 0.
            xx, yy, lengths = collect(expert, cpu_student, episodes=args.dagger_episodes,
                                      seconds=30., beta=beta, seed=args.seed+2000+round_index*53)
            x = np.concatenate((x,xx)); y = np.concatenate((y,yy))
            print(json.dumps({"event":"dagger_dataset", "round":round_index, "beta":beta,
                              "new_samples":len(xx), "samples":len(x), "lengths":lengths}), flush=True)
        tx, ty = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        student.train()
        updates = args.initial_updates if round_index == 0 else args.updates
        losses = []
        for update in range(updates):
            indices = torch.randint(len(tx), (args.batch,), device=device)
            prediction = student(tx[indices])
            loss = ((prediction-ty[indices])/student.action_scale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if device.type == "cuda" and update % 100 == 0:
                peak = max(peak, memory_mib())
                if peak > 3800:
                    raise RuntimeError(f"Own process VRAM guard exceeded: {peak} MiB")
        cpu_student = copy.deepcopy(student).cpu().eval()
        development = evaluate_student(cpu_student)
        passed = sum(r["success"] for r in development)
        stable_passes = stable_passes+1 if passed == len(development) else 0
        row = {"round":round_index, "dataset_samples":len(x), "updates":updates,
               "loss_last100":float(np.mean(losses[-100:])), "development_passed":passed,
               "development_total":len(development), "development":development,
               "observed_process_peak_mib":peak, "elapsed_s":time.monotonic()-started}
        logs.append(row)
        bundle = RetainedSkillPolicy(base, cpu_student)
        snapshot = {"schema":"pera_retained_skills_v1", "round":round_index,
                    "foundation_sha256":config["foundation_sha256"],
                    "expert_sha256":config["expert_sha256"],
                    "model_spec":{"hidden":list(student.hidden), "observation_schema":OBS_SCHEMA,
                                  "contact_model":"husky_capsules", "action_space":"native_normalized_29"},
                    "policy":bundle.state_dict(), "optimizer":optimizer.state_dict(),
                    "training":config, "development":row,
                    "certified_skills":[], "status":"candidate_requires_independent_certification"}
        temp = args.output/"candidate.tmp"
        torch.save(snapshot,temp)
        temp.replace(args.output/"candidate.pt")
        if passed == len(development):
            # Keep every passing candidate, even if a later DAgger round regresses.
            torch.save(snapshot, args.output/f"development_pass_round{round_index}.pt")
        (args.output/"training.json").write_text(json.dumps(logs, indent=2)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k != "development"}), flush=True)
        print(json.dumps({"development_details":[{"seed":r["development_seed"],"duration":r["duration_s"],
                 "progress":r["board_forward_progress_m"],"failures":r["failures"]} for r in development]}),flush=True)
        if stable_passes >= 2:
            break
    np.savez_compressed(args.output/"replay.npz", observations=x, actions=y)
    print(json.dumps({"event":"finished", "candidate":str(args.output/"candidate.pt"),
                      "observed_process_peak_mib":peak,"development_passed":passed,
                      "new_download_bytes":0}), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--expert",type=Path,default=Path("/home/tcl/Desktop/start/humanoid_skateboarding/ckpts/test.onnx"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--initial-episodes",type=int,default=20)
    p.add_argument("--dagger-episodes",type=int,default=10)
    p.add_argument("--rounds",type=int,default=14)
    p.add_argument("--initial-updates",type=int,default=3000)
    p.add_argument("--updates",type=int,default=1500)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--lr",type=float,default=3e-4)
    train(p.parse_args())
