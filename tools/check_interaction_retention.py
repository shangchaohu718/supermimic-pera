#!/usr/bin/env python3
"""Compare an interaction checkpoint against its independently loaded foundation.

Offline CPU audit: checks every native actor parameter/normalization buffer, then
evaluates four reference-derived observation histories from every local motion.
This verifies the unchanged native tracking action map, not physical completion
in interaction scenes and not retention of newly learned interaction tasks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

# Set before importing torch; this tool must never allocate device memory.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", WANDB_MODE="offline")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from unisupermimic.interaction.features import native_proprio
from unisupermimic.interaction.networks import RetainedInteractionPolicy
from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
    G1_JOINT_NAMES,
    default_joint_pos_list,
)


def load_snapshot(path: Path):
    """Read state and digest through one descriptor, even if the path is replaced."""
    with path.open("rb") as stream:
        snapshot = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        digest = hashlib.sha256()
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return snapshot, digest.hexdigest()


@torch.no_grad()
def audit(native_path: Path, candidate_path: Path, motions: Path) -> dict:
    torch.set_num_threads(2)
    original, native_sha = load_snapshot(native_path)
    candidate, candidate_sha = load_snapshot(candidate_path)
    if candidate.get("schema") != 2 or "policy" not in candidate:
        raise ValueError("Candidate must contain a schema=2 interaction policy checkpoint")
    if "pi" not in original:
        raise ValueError("Native checkpoint must contain its original 'pi' state dictionary")
    native_iteration = original.get("iteration")
    iteration = candidate.get("iteration")
    foundation_matches = candidate.get("foundation_sha256") == native_sha

    native = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    native.load_state_dict(original["pi"], strict=True)
    native.eval()
    interaction = RetainedInteractionPolicy.from_native_checkpoint(native_path)
    interaction.load_state_dict(candidate["policy"], strict=True)
    interaction.eval()
    candidate_base = interaction.base.state_dict()
    original_base = original["pi"]
    keys_match = candidate_base.keys() == original_base.keys()
    mismatched = sorted(
        key for key in set(candidate_base) | set(original_base)
        if key not in candidate_base or key not in original_base
        or not torch.equal(candidate_base[key], original_base[key])
    )
    norm_keys = sorted(key for key in original_base if "_norm." in key)
    norm_exact = all(key not in mismatched for key in norm_keys)
    del original, candidate

    files = sorted(motions.glob("*.csv"))
    if not files:
        raise ValueError(f"No local motion CSV files in {motions}")
    default_q = torch.tensor(default_joint_pos_list(), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(20261023)
    details = []
    for path in files:
        clip = ReferenceClip.from_csv(path, G1_JOINT_NAMES)
        frames = torch.linspace(0, clip.num_frames - 1, 4).round().long()
        # Actual ten reference states ending at each representative frame;
        # reset padding duplicates frame zero, previous-action inputs are zero.
        history_frames = (frames[:, None] + torch.arange(-9, 1)).clamp(min=0)
        history_state = unflatten_state(clip.state[history_frames.flatten()])
        obs = native_proprio(
            history_state, torch.zeros(40, 29), history_state, default_q,
        ).reshape(4, 10, 99)
        goal = clip.goal(frames)
        context = torch.randn(4, 64, generator=generator)
        expected = native(obs, goal)
        actual = interaction(obs, goal, context, torch.zeros(4))
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        error = float((actual - expected).abs().max()) if finite else None
        details.append({
            "clip": clip.name,
            "frames": frames.tolist(),
            "samples": 4,
            "finite": finite,
            "actions_bitwise_equal": finite and torch.equal(actual, expected),
            "max_abs_action_error": error,
        })

    base_exact = keys_match and not mismatched
    actions_exact = all(item["actions_bitwise_equal"] for item in details)
    return {
        "schema": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "cuda_initialized": torch.cuda.is_initialized(),
        "new_download_bytes": 0,
        "native_checkpoint": str(native_path.resolve()),
        "native_sha256": native_sha,
        "native_iteration": native_iteration,
        "candidate_checkpoint": str(candidate_path.resolve()),
        "candidate_sha256": candidate_sha,
        "candidate_iteration": iteration,
        "recorded_foundation_matches": foundation_matches,
        "native_base_tensors_checked": len(original_base),
        "native_base_tensors_exact": base_exact,
        "normalization_tensors_checked": norm_keys,
        "normalization_exact": norm_exact,
        "mismatched_native_tensors": mismatched,
        "clip_count": len(details),
        "sample_count": sum(item["samples"] for item in details),
        "actions_bitwise_equal": actions_exact,
        "max_abs_action_error": max(item["max_abs_action_error"] for item in details) if all(item["finite"] for item in details) else None,
        "passed": base_exact and norm_exact and actions_exact and foundation_matches,
        "sample_protocol": "Four evenly spaced frames per clip; ten real reference states per observation history; zero previous actions; random finite interaction context; interaction_mask=0",
        "scope": "Native tracking parameters/statistics and sampled action-map retention only; no full physical rollouts or new interaction skill retention claim",
        "clips": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.native, args.candidate, args.motions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "passed", "candidate_iteration", "native_base_tensors_exact", "normalization_exact",
        "clip_count", "sample_count", "actions_bitwise_equal", "max_abs_action_error", "cuda_initialized",
    )}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
