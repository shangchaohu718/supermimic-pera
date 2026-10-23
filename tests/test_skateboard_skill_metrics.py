"""Physical acceptance must reject stepping off, brief success, and reward gaming."""
import importlib.util
from pathlib import Path
import sys
import json
from dataclasses import replace

_PATH = Path(__file__).resolve().parents[1] / "tools/skateboard_skill_metrics.py"
_SPEC = importlib.util.spec_from_file_location("skateboard_skill_metrics_test", _PATH)
metrics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = metrics
_SPEC.loader.exec_module(metrics)


def good_frames():
    return [metrics.PhysicalFrame(
        time_s=(i + 1) * .02, pelvis_above_board_m=.65, pelvis_up_dot=.99,
        robot_board_xy_distance_m=.1, board_forward_progress_m=(i + 1) * .02,
        right_board=True, left_board=False, right_ground=False,
        left_ground=i % 300 < 100, wheel_ground=True, unintended_support=False,
    ) for i in range(1500)]


def result(frames, **kwargs):
    return metrics.summarize_frames(frames, requested_seconds=30., warning_counts=[0] * 7, **kwargs)


def test_legitimate_left_foot_ground_pushes_pass():
    summary = result(good_frames())
    assert summary["success"]
    assert summary["left_foot_ground_fraction"] > 0
    json.dumps(summary)


def test_early_high_progress_and_no_fall_are_not_complete_skill():
    frames = good_frames()[:500]
    assert not result(frames)["success"]
    assert "complete_30s_horizon" in result(frames)["failures"]


def test_stepping_off_near_end_fails_even_high_overall_contact_fraction():
    frames = good_frames()
    frames[-60:] = [replace(f, right_board=False, right_ground=True) for f in frames[-60:]]
    summary = result(frames)
    assert summary["right_foot_board_fraction"] > .9
    assert not summary["success"]
    assert "right_foot_board_support_tail" in summary["failures"]
    assert "no_prolonged_board_contact_loss" in summary["failures"]


def test_hand_or_knee_support_is_not_success():
    frames = good_frames()
    frames[500] = replace(frames[500], unintended_support=True)
    assert "no_unintended_body_support" in result(frames)["failures"]


def test_contact_gaps_cannot_hide_inside_high_fraction():
    frames = good_frames()
    frames[100:120] = [replace(f, right_board=False) for f in frames[100:120]]
    assert "no_prolonged_board_contact_loss" in result(frames)["failures"]


def test_missing_frames_warnings_and_nonfinite_each_reject():
    frames = good_frames()
    assert "control_steps_missing_or_not_50_hz" in result(frames[::2])["failures"]
    assert not result(frames, nonfinite=True)["success"]
    report = metrics.summarize_frames(frames, requested_seconds=30., warning_counts=[1])
    assert "no_physics_warnings" in report["failures"]


def test_fixed_holdout_resets_are_nine_unique_cases():
    assert len(metrics.HELD_OUT_RESETS) == 9
    assert len({case["name"] for case in metrics.HELD_OUT_RESETS}) == 9
