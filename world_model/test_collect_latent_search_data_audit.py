import unittest
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_original_argv = sys.argv[:]
sys.argv = [
    "collect_latent_search_data.py",
    "--dataset", "unused.npz",
    "--ae-checkpoint", "unused.pt",
    "--flow-checkpoint", "unused.pt",
]
from world_model.collect_latent_search_data import (
    _IterationRecorder,
    _audit_point_cloud,
    _audit_pose_array,
)
sys.argv = _original_argv


class TrajectoryAuditTests(unittest.TestCase):
    def test_accepts_finite_cloud_with_plausible_motion(self):
        points = np.zeros((3, 4, 3), dtype=np.float32)
        points[1:, :, 0] = 0.1
        self.assertEqual(_audit_point_cloud(points, "object", 2.0, 1.0), [])

    def test_rejects_cloud_outside_base_frame_limit(self):
        points = np.zeros((2, 4, 3), dtype=np.float32)
        points[1, :, 0] = 2.1
        reasons = _audit_point_cloud(points, "object", 2.0, 3.0)
        self.assertIn("object_abs>2", reasons)

    def test_rejects_implausible_frame_jump(self):
        points = np.zeros((2, 4, 3), dtype=np.float32)
        points[1, :, 0] = 1.1
        reasons = _audit_point_cloud(points, "object", 2.0, 1.0)
        self.assertIn("object_step>1", reasons)

    def test_rejects_nonfinite_cloud_and_invalid_pose(self):
        points = np.zeros((2, 4, 3), dtype=np.float32)
        points[1, 0, 0] = np.nan
        self.assertIn("object_nonfinite", _audit_point_cloud(points, "object", 2.0, 1.0))

        pose = np.zeros((2, 7), dtype=np.float32)
        self.assertIn("object_invalid_quaternion", _audit_pose_array(pose, "object"))

    def test_recorder_writes_only_audited_trajectories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = SimpleNamespace(
                num_envs=2,
                camera_cfg={"camera_1": {"height": 2, "width": 2}},
                robot_dof_pos=np.zeros((2, 2), dtype=np.float32),
                num_actions=2,
            )
            recorder = _IterationRecorder(
                temp_dir, 1, env, 3, 4, np.eye(3, dtype=np.float32),
                0.1, 5.0, 1, [], 0, "pelvis", 2.0, 2.0, 1.0,
            )
            recorder.next_step = recorder.frame_count
            recorder.arrays["object_pose_world"][:] = np.tile(
                np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (2, 3, 1)
            )
            recorder.arrays["object_pose_world"][0, :, 0] = [0.0, 0.05, 0.1]
            recorder.arrays["object_pose_world"][1, :, 0] = [0.0, 0.1, 4.0]
            recorder.arrays["robot_base_pose_world"][:] = np.tile(
                np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (2, 3, 1)
            )
            recorder.arrays["hand_points_base"][:] = 0.1

            metric_names = (
                "fitness", "max_lift", "final_lift", "lift_score", "success_seen",
                "lifted", "collision", "arm_joint_motion", "arm_target_motion",
                "arm_target_error", "eef_target_lift", "eef_actual_lift", "valid_success",
            )
            metrics = {name: torch.zeros(2) for name in metric_names}
            for name in ("success_seen", "lifted", "collision", "valid_success"):
                metrics[name] = torch.zeros(2, dtype=torch.bool)
            metrics.update({
                "latent": torch.zeros((2, 2)),
                "world_targets": torch.zeros((2, 3, 7)),
                "hand_targets": torch.zeros((2, 3, 2)),
            })
            audit = recorder.save(
                torch.zeros((2, 2)), metrics, 1, ["joint_1", "joint_2"],
                np.zeros((4, 3), dtype=np.float32),
                torch.zeros((2, 3, 2)), torch.zeros((2, 3)),
            )

            self.assertEqual(audit["valid_candidates"].tolist(), [True, False])
            iteration_dir = Path(temp_dir) / "iteration_001"
            self.assertTrue((iteration_dir / "candidate_0000.npz").is_file())
            self.assertFalse((iteration_dir / "candidate_0001.npz").exists())
            report = json.loads((iteration_dir / "trajectory_audit.json").read_text())
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["rejected"][0]["candidate_index"], 1)


if __name__ == "__main__":
    unittest.main()
