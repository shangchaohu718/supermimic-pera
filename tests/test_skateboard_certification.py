"""Certification cannot publish shortened, mismatched, or failed policy audits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
_SPEC = importlib.util.spec_from_file_location("skateboard_certification_test", _TOOLS / "certify_skateboard_skill.py")
certification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(certification)
from skateboard_skill_metrics import PhysicalFrame, summarize_frames
from unisupermimic.interaction.skateboard_skill import (
    OBS_SCHEMA, RetainedSkillPolicy, SkateboardStudent,
)
from unisupermimic.rl.algos.supermimic.networks import Policy


@pytest.fixture
def reports():
    frames = [PhysicalFrame(
        time_s=(i + 1) * .02, pelvis_above_board_m=.65, pelvis_up_dot=.99,
        robot_board_xy_distance_m=.1, board_forward_progress_m=(i + 1) * .02,
        right_board=True, left_board=False, right_ground=False,
        left_ground=i % 300 < 100, wheel_ground=True, unintended_support=False,
    ) for i in range(1500)]
    row = summarize_frames(frames, requested_seconds=30., warning_counts=[0] * 7)
    rows = []
    for reset in certification.HELD_OUT_RESETS:
        result = copy.deepcopy(row)
        result.update(name=reset["name"], reset={"specification": copy.deepcopy(reset)},
                      runtime_error=None, early_stop=None,
                      source_policy="learned_pytorch_student_only")
        rows.append(result)
    physical = {
        "schema": "pera_skateboard_independent_certification_v1",
        "provenance": {"checkpoint_sha256": "candidate-sha", "foundation_sha256": "native-sha"},
        "criteria": copy.deepcopy(certification.CRITERIA),
        "held_out_resets": copy.deepcopy(list(certification.HELD_OUT_RESETS)),
        "scenario_count": 9, "passed_scenarios": 9, "seconds_per_reset": 30.,
        "success": True, "results": rows, "expert_imported": False,
        "training_imported": False, "policy_device": "cpu", "physics": "mujoco_cpu",
    }
    retention = {
        "schema": "pera_retained_skill_audit_v1", "candidate_sha256": "candidate-sha",
        "native_sha256": "native-sha", "recorded_foundation_matches": True,
        "passed": True, "native_base_tensors_exact": True, "normalization_exact": True,
        "physical_rollouts_bitwise_equal": True, "physical_total_steps": 1500,
        "sampled_actions_bitwise_equal": True, "clip_count": 40, "sample_count": 160,
        "physical_clip_count": 3, "foundation_gradients_disabled": True,
        "normalization_updates_disabled": True,
    }
    return physical, retention


def test_exact_complete_matching_reports_pass(reports):
    physical, retention = reports
    certification.validate_reports("candidate-sha", physical, retention)


@pytest.mark.parametrize("report_name", ["physical", "retention"])
def test_other_checkpoint_report_cannot_certify_candidate(reports, report_name):
    physical, retention = reports
    if report_name == "physical":
        physical["provenance"]["checkpoint_sha256"] = "other-candidate"
    else:
        retention["candidate_sha256"] = "other-candidate"
    with pytest.raises(ValueError):
        certification.validate_reports("candidate-sha", physical, retention)


@pytest.mark.parametrize("case", [
    "physical_failed", "retention_failed", "failed_individual", "missing_case",
    "duplicate_case", "short_suite", "short_individual", "nan_suite", "nan_individual",
    "changed_criterion", "changed_actual_reset", "expert_inference", "trainer_imported",
    "wrong_source_policy", "runtime_error", "partial_retention", "foundation_mismatch",
])
def test_invalid_certification_evidence_rejected(reports, case):
    physical, retention = reports
    if case == "physical_failed":
        physical["success"] = False
    elif case == "retention_failed":
        retention["passed"] = False
    elif case == "failed_individual":
        physical["results"][0]["success"] = False
        physical["results"][0]["failures"] = ["right_foot_board_support"]
    elif case == "missing_case":
        physical["results"].pop()
    elif case == "duplicate_case":
        physical["results"][-1] = copy.deepcopy(physical["results"][0])
    elif case == "short_suite":
        physical["seconds_per_reset"] = 10.
    elif case == "short_individual":
        physical["results"][0]["duration_s"] = 29.9
    elif case == "nan_suite":
        physical["seconds_per_reset"] = float("nan")
    elif case == "nan_individual":
        physical["results"][0]["duration_s"] = float("nan")
    elif case == "changed_criterion":
        physical["criteria"]["minimum_board_forward_progress_m"] = .01
    elif case == "changed_actual_reset":
        physical["results"][1]["reset"]["specification"]["root_xy_m"] = [0., 0.]
    elif case == "expert_inference":
        physical["expert_imported"] = True
    elif case == "trainer_imported":
        physical["training_imported"] = True
    elif case == "wrong_source_policy":
        physical["results"][0]["source_policy"] = "expert"
    elif case == "runtime_error":
        physical["results"][0]["runtime_error"] = "physics_failure"
    elif case == "partial_retention":
        retention["clip_count"] = 1
        retention["sample_count"] = 4
        retention["physical_clip_count"] = 1
    elif case == "foundation_mismatch":
        retention["recorded_foundation_matches"] = False
    with pytest.raises(ValueError):
        certification.validate_reports("candidate-sha", physical, retention)


def test_publish_freezes_model_and_keeps_validated_provenance(tmp_path, reports):
    torch.manual_seed(956)
    base = Policy(obs_dim=99, goal_dim=730, hist=10, num_joints=29)
    bundle = RetainedSkillPolicy(base, SkateboardStudent(hidden=(8,))).eval()
    snapshot = {
        "schema": "pera_retained_skills_v1", "foundation_sha256": "native-sha",
        "model_spec": {"hidden": [8], "observation_schema": OBS_SCHEMA,
                       "contact_model": "husky_capsules", "action_space": "native_normalized_29"},
        "policy": bundle.state_dict(), "optimizer": {"mutable": True},
        "certified_skills": [], "status": "candidate_requires_independent_certification",
    }
    candidate = tmp_path / "candidate.pt"
    torch.save(snapshot, candidate)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    physical, retention = reports
    physical["provenance"]["checkpoint_sha256"] = retention["candidate_sha256"] = digest
    physical_path, retention_path = tmp_path / "physical.json", tmp_path / "retention.json"
    physical_path.write_text(json.dumps(physical))
    retention_path.write_text(json.dumps(retention))
    output = tmp_path / "accepted.pt"
    manifest = certification.certify(candidate, physical_path, retention_path, output)
    loaded, saved = RetainedSkillPolicy.load(output)
    assert saved["status"] == "certified_frozen_skateboard_v1"
    assert loaded.certified_skills == {"skateboard"}
    assert not any(p.requires_grad for p in loaded.parameters())
    assert "optimizer" not in saved
    assert manifest["certification"]["evaluated_candidate_sha256"] == digest
    assert manifest["certification"]["physical_report_sha256"] == hashlib.sha256(physical_path.read_bytes()).hexdigest()
    obs = torch.randn(4, 395)
    assert torch.equal(loaded(obs, task="skateboard"), bundle(obs, task="skateboard"))
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        certification.certify(candidate, physical_path, retention_path, output)
    assert output.read_bytes() == before
