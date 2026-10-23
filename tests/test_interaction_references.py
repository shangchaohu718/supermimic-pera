"""Reference contract checks with tiny local fixtures, no downloaded assets."""

import json

import numpy as np
import pytest
import torch

from unisupermimic.interaction.references import ReferenceClip
from unisupermimic.rl.algos.supermimic.geometry import flatten_state, quat_to_mat
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

JOINTS = tuple(f"joint_{index}" for index in range(29))


class _NoBodyKinematics:
    def forward(self, position, rotation, velocity, angular_velocity, joints, joint_velocity):
        return position[:, None], rotation[:, None], velocity[:, None], angular_velocity[:, None]


def _csv(tmp_path):
    time = np.arange(7, dtype=np.float32) / 30.0
    raw = np.zeros((len(time), 36), dtype=np.float32)
    raw[:, :3] = np.stack((time, time**2, 0.8 + time * 0.1), axis=-1)
    raw[:, 3:7] = np.stack((time * 0, time * 0, np.sin(0.5 + time), np.cos(0.5 + time)), axis=-1)
    raw[:, 7:] = time[:, None] * np.arange(29, dtype=np.float32)[None] * 0.1
    path = tmp_path / "tiny.csv"
    np.savetxt(path, raw, delimiter=",")
    return path


def test_csv_resampling_state_and_goals_exactly_match_native_motionlib(tmp_path):
    path = _csv(tmp_path)
    lightweight = ReferenceClip.from_csv(path, JOINTS)
    native = MotionLib([str(path)], _NoBodyKinematics(), "cpu")
    assert lightweight.num_frames == native.num_frames
    indices = torch.arange(native.num_frames)
    torch.testing.assert_close(lightweight.state, flatten_state(native.ref_state(indices)), atol=0, rtol=0)
    torch.testing.assert_close(lightweight.goal(indices), native.goal(indices, 10), atol=0, rtol=0)
    assert lightweight.qpos.shape == (native.num_frames, 36)
    assert lightweight.qvel.shape == (native.num_frames, 35)
    assert lightweight.objects.shape == (native.num_frames, 27)
    assert not lightweight.objects.any() and lightweight.object_mask == 0
    np.testing.assert_allclose(
        lightweight.qvel[:, 3:6], native.root_w_local.numpy(), atol=0, rtol=0,
    )


def _vaic(tmp_path, fps=50.0):
    directory = tmp_path / "box_fixture"
    directory.mkdir()
    frames = 5
    source_names = list(reversed(JOINTS))
    (directory / "meta.json").write_text(json.dumps({
        "joint_names": source_names, "body_names": ["mybox", "pelvis", "stairslope"], "fps": fps,
    }))
    position = np.zeros((frames, 3, 3), dtype=np.float32)
    position[:, 0] = [1, 2, 0.6]
    position[:, 1] = [0, 0, 0.8]
    position[:, 2] = [3, 4, 0.2]
    quaternion = np.zeros((frames, 3, 4), dtype=np.float32)
    quaternion[:, :, 0] = 1
    quaternion[:, 0] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    linear = np.zeros((frames, 3, 3), dtype=np.float32)
    linear[:, 0] = [1, 2, 3]
    linear[:, 1] = [4, 5, 6]
    angular = np.zeros_like(linear)
    angular[:, 0] = [1, 0, 0]
    joints = np.tile(np.arange(29, dtype=np.float32)[::-1], (frames, 1))
    np.savez(directory / "motion.npz", body_pos_w=position, body_quat_w=quaternion,
             body_lin_vel_w=linear, body_ang_vel_w=angular, joint_pos=joints, joint_vel=joints * 2)
    return directory


def test_vaic_joint_mapping_root_convention_and_object_packing(tmp_path):
    reference = ReferenceClip.from_vaic(_vaic(tmp_path), JOINTS)
    assert reference.num_frames == 5
    np.testing.assert_array_equal(reference.qpos[:, 7:], np.tile(np.arange(29), (5, 1)))
    np.testing.assert_array_equal(reference.qvel[:, 6:], np.tile(np.arange(29) * 2, (5, 1)))
    np.testing.assert_array_equal(reference.qpos[:, 3:7], np.tile([1, 0, 0, 0], (5, 1)))
    np.testing.assert_array_equal(reference.state[:, 7:10], np.tile([4, 5, 6], (5, 1)))
    assert reference.object_names == ("mybox",) and reference.object_mask == 1
    assert reference.objects.shape == (5, 27)
    np.testing.assert_array_equal(reference.objects[:, 14:17], np.tile([1, 2, 3], (5, 1)))
    np.testing.assert_allclose(reference.objects[:, 17:20], np.tile([0, -1, 0], (5, 1)), atol=2e-7)
    assert not reference.objects[:, 7:14].any()
    assert not reference.objects[:, 20:27].any()
    np.testing.assert_array_equal(reference.terrain_objects["stairslope"][:, :3], np.tile([3, 4, .2], (5, 1)).astype(np.float32))
    assert reference.goal(torch.tensor([0, 4])).shape == (2, 730)


def test_vaic_resamples_non50hz_and_rejects_incompatible_board(tmp_path):
    directory = _vaic(tmp_path, fps=25.0)
    reference = ReferenceClip.from_vaic(directory, JOINTS)
    assert reference.num_frames == 9 and reference.objects.shape == (9, 27)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["body_names"][0] = "skateboard"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="geometry"):
        ReferenceClip.from_vaic(directory, JOINTS)


def test_reference_input_validation(tmp_path):
    path = _csv(tmp_path)
    with pytest.raises(ValueError, match="29 unique"):
        ReferenceClip.from_csv(path, JOINTS[:-1])
    reference = ReferenceClip.from_csv(path, JOINTS)
    with pytest.raises(IndexError):
        reference.goal(torch.tensor([-1]))
    with pytest.raises(ValueError):
        reference.goal(torch.tensor([1.5]))
    with pytest.raises(ValueError):
        reference.goal(torch.tensor([1]), num_frames=0)
