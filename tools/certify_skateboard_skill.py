#!/usr/bin/env python3
"""Publish a local frozen skill only after matching independent physical audits.

No network, no training, no overwrite. Reports must describe the exact input
checkpoint and the fixed full certification suite. Future skill training must
keep this module frozen and rerun its saved suite before accepting a new bundle.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", UV_OFFLINE="1", HF_HUB_OFFLINE="1", WANDB_MODE="offline")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import torch
from unisupermimic.interaction.skateboard_skill import RetainedSkillPolicy, state_digest
from skateboard_skill_metrics import CRITERIA, HELD_OUT_RESETS


def validate_reports(digest, physical, retention):
    def full_duration(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 30-1e-7
    if (physical.get("schema") != "pera_skateboard_independent_certification_v1"
            or physical.get("provenance", {}).get("checkpoint_sha256") != digest
            or retention.get("schema") != "pera_retained_skill_audit_v1"
            or retention.get("candidate_sha256") != digest):
        raise ValueError("Reports must audit this exact candidate checkpoint")
    if not physical.get("success") or not retention.get("passed"):
        raise ValueError("Physical certification and native retention must both pass")
    if (physical.get("criteria") != CRITERIA
            or physical.get("held_out_resets") != list(HELD_OUT_RESETS)
            or physical.get("scenario_count") != len(HELD_OUT_RESETS)
            or physical.get("passed_scenarios") != len(HELD_OUT_RESETS)
            or not full_duration(physical.get("seconds_per_reset"))):
        raise ValueError("Full fixed certification suite required")
    results = physical.get("results", [])
    if len(results) != len(HELD_OUT_RESETS) or any(
        not row.get("success") or not full_duration(row.get("duration_s"))
        or row.get("name") != reset["name"] or row.get("failures")
        or row.get("reset", {}).get("specification") != reset
        or row.get("source_policy") != "learned_pytorch_student_only"
        or row.get("runtime_error") is not None or row.get("early_stop") is not None
        for row, reset in zip(results, HELD_OUT_RESETS)
    ):
        raise ValueError("Individual physical results must pass the full horizon")
    if (physical.get("expert_imported") is not False or physical.get("training_imported") is not False
            or physical.get("policy_device") != "cpu"
            or not retention.get("native_base_tensors_exact")
            or not retention.get("normalization_exact")
            or not retention.get("physical_rollouts_bitwise_equal")
            or retention.get("physical_total_steps", 0) < 1500
            or retention.get("physical_clip_count", 0) < 3
            or retention.get("clip_count", 0) < 40 or retention.get("sample_count", 0) < 160
            or not retention.get("recorded_foundation_matches")
            or physical["provenance"].get("foundation_sha256") != retention.get("native_sha256")
            or not retention.get("sampled_actions_bitwise_equal")):
        raise ValueError("Independent learned-policy inference and exact native retention required")


def certify(candidate, physical_path, retention_path, output):
    if output.exists():
        raise FileExistsError(f"Will not overwrite a certified version: {output}")
    raw = candidate.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    physical_bytes, retention_bytes = physical_path.read_bytes(), retention_path.read_bytes()
    physical, retention = json.loads(physical_bytes), json.loads(retention_bytes)
    validate_reports(digest, physical, retention)
    policy, snapshot = RetainedSkillPolicy.load(io.BytesIO(raw))
    policy.freeze_certified_skill("skateboard")
    snapshot["certified_skills"] = ["skateboard"]
    snapshot["certified_skill_sha256"] = {"skateboard": state_digest(policy.skateboard.state_dict())}
    snapshot["status"] = "certified_frozen_skateboard_v1"
    snapshot["certification"] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_candidate_sha256": digest,
        "physical_report": str(physical_path.resolve()),
        "physical_report_sha256": hashlib.sha256(physical_bytes).hexdigest(),
        "retention_report": str(retention_path.resolve()),
        "retention_report_sha256": hashlib.sha256(retention_bytes).hexdigest(),
        "criteria": CRITERIA, "resets": list(HELD_OUT_RESETS),
        "regression_policy": "Keep foundation and certified skill parameters/statistics frozen; require the full saved physical suite before accepting any new skill bundle",
    }
    # A deployment artifact does not carry mutable optimizer state.
    snapshot.pop("optimizer", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(snapshot, stream)
    loaded, _ = RetainedSkillPolicy.load(output)
    assert state_digest(loaded.state_dict()) == state_digest(policy.state_dict())
    assert not any(p.requires_grad for p in loaded.parameters())
    manifest = {"checkpoint":str(output.resolve()), "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                "foundation_sha256":snapshot["foundation_sha256"],
                "certified_skill_sha256":snapshot["certified_skill_sha256"],
                "certification":snapshot["certification"], "new_download_bytes":0}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--physical",type=Path,required=True)
    p.add_argument("--retention",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(certify(a.candidate,a.physical,a.retention,a.output),indent=2))
