"""Train a grasp evaluator and refine trajectory AE/Flow from Isaac Gym data.

The evaluator predicts simulator fitness from a robot-base-frame object cloud
and the complete 20-frame robot-base-frame action. Each object scene has an independent
CEM population in latent space; evaluator-ranked elites update only that
scene's search distribution. World-frame trajectory mutations are encoded
back into latent space. Isaac Gym labels each scene's final elite and one
randomly retained search trajectory. Successful actions update AE and Flow.

Example: python -m memory.train_simulation_guided_grasp --dataset
datasets/inspire_grasp.npz --ae-checkpoint runs/trajectory_ae/best.pt
--flow-checkpoint runs/latent_flow/best.pt --population-size 64 --object-count 64
--iterations 10 --cem-iterations 5 --action-noise-std 0.01
--action-noise-fraction 0.25 task=grasp
hand=g1_upper_right_wuji num_envs=64 headless=True task.env.armController=pose
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

# Isaac Gym requires its extensions to load before PyTorch when this file is
# used as the training entry point. Keep utility imports usable if a caller
# already imported torch.
if "torch" not in sys.modules:
    try:
        import isaacgym  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "isaacgym":
            raise

import numpy as np
import torch
from torch import Tensor, nn

from memory.train_latent_flow import RealNVP, _make_flow
from memory.train_trajectory_ae import _build_model
from memory.train_trajectory_vqvae import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
    rotation6d_to_matrix,
)


class GraspEvaluator(nn.Module):
    """Predict fitness from a robot-base-frame point cloud and action sequence."""

    def __init__(
        self, action_dim: int = 27, point_latent_dim: int = 64,
        reconstruction_points: int = 128,
    ) -> None:
        super().__init__()
        self.point_latent_dim = int(point_latent_dim)
        self.reconstruction_points = int(reconstruction_points)
        self.register_buffer("fitness_mean", torch.zeros(()))
        self.register_buffer("fitness_std", torch.ones(()))
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.point_projection = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, self.point_latent_dim),
        )
        self.point_decoder = nn.Sequential(
            nn.Linear(self.point_latent_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, self.reconstruction_points * 3),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(5),
        )
        self.head = nn.Sequential(
            nn.Linear(self.point_latent_dim + 128 * 5, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, pointcloud: Tensor, trajectory: Tensor,
        return_reconstruction: bool = False,
    ) -> Any:
        point_features = self.point_encoder(pointcloud)
        point_features = torch.cat(
            (point_features.amax(dim=1), point_features.mean(dim=1)), dim=-1
        )
        point_latent = self.point_projection(point_features)
        action_features = self.action_encoder(trajectory.transpose(1, 2)).flatten(1)
        normalized_fitness = self.head(torch.cat((point_latent, action_features), dim=-1)).squeeze(-1)
        fitness = normalized_fitness * self.fitness_std + self.fitness_mean
        if not return_reconstruction:
            return fitness
        reconstruction = self.point_decoder(point_latent).reshape(
            pointcloud.shape[0], self.reconstruction_points, 3
        )
        return fitness, reconstruction

    def set_fitness_stats(self, mean: float, std: float) -> None:
        self.fitness_mean.fill_(float(mean))
        self.fitness_std.fill_(max(float(std), 1e-3))


def _pointcloud_chamfer_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Symmetric Chamfer loss for unordered point sets."""
    distances = torch.cdist(prediction, target, p=2).square()
    return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


class ReplayBuffer:
    """Bounded buffer for (base-frame point cloud, action, fitness)."""

    def __init__(self, capacity: int, point_count: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.pointcloud = np.empty((capacity, point_count, 3), dtype=np.float32)
        self.trajectory = np.empty((capacity, 20, action_dim), dtype=np.float32)
        self.fitness = np.empty((capacity,), dtype=np.float32)
        self.object_id = np.empty((capacity,), dtype=np.int64)
        self.scenario_id = np.empty((capacity,), dtype=np.int64)
        self.object_pose = np.empty((capacity, 7), dtype=np.float32)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def fitness_stats(self) -> tuple[float, float]:
        values = self.fitness[self._indices()]
        return float(values.mean()), float(values.std() + 1e-3)

    def add(
        self,
        pointcloud: Tensor,
        trajectory: Tensor,
        fitness: Tensor,
        object_id: Tensor,
        scenario_id: Tensor,
        object_pose: Tensor,
        keep: Tensor | None = None,
    ) -> None:
        values = [
            value.detach().cpu().numpy()
            for value in (
                pointcloud, trajectory, fitness, object_id, scenario_id, object_pose
            )
        ]
        pc, action, score, ids, scenarios, poses = values
        if keep is not None:
            mask = keep.detach().cpu().numpy().astype(bool)
            pc, action, score, ids, scenarios, poses = (
                value[mask]
                for value in (pc, action, score, ids, scenarios, poses)
            )
        if len(score) == 0:
            return
        if len(score) > self.capacity:
            pc, action, score, ids, scenarios, poses = (
                value[-self.capacity:]
                for value in (pc, action, score, ids, scenarios, poses)
            )
        indices = (np.arange(len(score)) + self.position) % self.capacity
        self.pointcloud[indices] = pc
        self.trajectory[indices] = action
        self.fitness[indices] = score
        self.object_id[indices] = ids
        self.scenario_id[indices] = scenarios
        self.object_pose[indices] = poses
        self.position = (self.position + len(score)) % self.capacity
        self.size = min(self.capacity, self.size + len(score))

    def _indices(self) -> np.ndarray:
        return (
            np.arange(self.size)
            if self.size < self.capacity
            else np.arange(self.capacity)
        )

    def _fitness_bins(self, lift_weight: float, success_bonus: float) -> np.ndarray:
        # Fitness is in [0, lift_weight] for failures without collision and
        # receives success_bonus for simulator-validated successes.
        boundaries = np.asarray(
            [
                0.0, 0.25 * lift_weight, 0.5 * lift_weight,
                0.75 * lift_weight, success_bonus,
            ],
            dtype=np.float32,
        )
        return np.digitize(self.fitness[self._indices()], boundaries)

    def sample_stratified(
        self,
        count: int,
        device: torch.device,
        lift_weight: float,
        success_bonus: float,
    ) -> tuple[Tensor, ...]:
        bins = self._fitness_bins(lift_weight, success_bonus)
        available = [np.flatnonzero(bins == index) for index in np.unique(bins)]
        available = [indices for indices in available if len(indices)]
        quotas = np.full(len(available), count // len(available), dtype=np.int64)
        if count % len(available):
            remainder = np.random.permutation(len(available))[:count % len(available)]
            quotas[remainder] += 1
        sampled = []
        for indices, quota in zip(available, quotas):
            if quota == 0:
                continue
            unique_count = min(int(quota), len(indices))
            selected = np.random.permutation(indices)[:unique_count]
            if quota > unique_count:
                selected = np.concatenate((
                    selected,
                    np.random.choice(indices, int(quota) - unique_count, replace=True),
                ))
            sampled.append(selected)
        indices = np.concatenate(sampled)
        np.random.shuffle(indices)
        return tuple(
            torch.as_tensor(array[indices].copy(), device=device)
            for array in (
                self.pointcloud, self.trajectory, self.fitness,
                self.object_id, self.object_pose,
            )
        )

    def sample_ranking_pairs(
        self, count: int, device: torch.device
    ) -> tuple[Tensor, ...] | None:
        indices = self._indices()
        scenario_ids = self.scenario_id[indices]
        fitness = self.fitness[indices]
        groups: dict[int, list[int]] = {}
        for index, scenario_id in enumerate(scenario_ids):
            groups.setdefault(int(scenario_id), []).append(index)
        pairs = []
        for members in groups.values():
            if len(members) < 2:
                continue
            member_fitness = fitness[members]
            low, high = int(np.argmin(member_fitness)), int(np.argmax(member_fitness))
            if member_fitness[low] < member_fitness[high]:
                pairs.append((indices[members[low]], indices[members[high]]))
        if not pairs:
            return None
        pair_indices = np.asarray(pairs, dtype=np.int64)
        selected = np.random.choice(
            len(pair_indices), size=count, replace=len(pair_indices) < count
        )
        low_indices, high_indices = pair_indices[selected].T
        return tuple(
            torch.as_tensor(array[index].copy(), device=device)
            for array, index in (
                (self.pointcloud, low_indices),
                (self.trajectory, low_indices),
                (self.trajectory, high_indices),
            )
        )

    def fitness_bin_counts(
        self, lift_weight: float, success_bonus: float
    ) -> list[int]:
        bins = self._fitness_bins(lift_weight, success_bonus)
        return [int(np.count_nonzero(bins == index)) for index in range(6)]

    def arrays(self) -> dict[str, np.ndarray]:
        indices = (
            np.arange(self.size)
            if self.size < self.capacity
            else (np.arange(self.size) + self.position) % self.capacity
        )
        return {
            "pointcloud_base": self.pointcloud[indices].copy(),
            "trajectory_base": self.trajectory[indices].copy(),
            "fitness": self.fitness[indices].copy(),
            "object_id": self.object_id[indices].copy(),
            "scenario_id": self.scenario_id[indices].copy(),
            "object_pose_world": self.object_pose[indices].copy(),
        }

    def save(self, path: Path, object_labels: list[str]) -> None:
        arrays = self.arrays()
        arrays["object_labels"] = np.asarray(object_labels, dtype=np.str_)
        arrays["coordinate_frame"] = np.asarray("robot_base")
        np.savez_compressed(path, **arrays)


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to xyzw quaternions, including near-pi poses."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    # Keep the unused quaternion branches differentiable at exact zero; their
    # sqrt derivative otherwise turns zero upstream gradients into NaNs.
    q_abs = torch.sqrt(torch.stack((
        (1 + m00 + m11 + m22).clamp_min(1e-8),
        (1 + m00 - m11 - m22).clamp_min(1e-8),
        (1 - m00 + m11 - m22).clamp_min(1e-8),
        (1 - m00 - m11 + m22).clamp_min(1e-8),
    ), dim=-1))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
        torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
    ), dim=-2)
    candidates = candidates / (2 * q_abs.amax(dim=-1).clamp_min(0.1))[..., None, None]
    choice = q_abs.argmax(dim=-1)
    gather_index = choice[..., None, None].expand(*choice.shape, 1, 4)
    quaternion_wxyz = candidates.gather(-2, gather_index).squeeze(-2)
    return quaternion_normalize(quaternion_wxyz[..., [1, 2, 3, 0]])


def _world_trajectory(
    decoded: Tensor, object_pose: Tensor, eef_offset: Tensor | None
) -> Tensor:
    """Convert [xyz, rotation6d, joints] to world EEF pose plus joints."""
    object_rotation = quaternion_to_matrix(object_pose[:, 3:7])
    position = torch.einsum("bij,btj->bti", object_rotation, decoded[..., :3])
    position = position + object_pose[:, None, :3]
    relative_rotation = rotation6d_to_matrix(decoded[..., 3:9])
    relative_quaternion = _matrix_to_quaternion(relative_rotation)
    object_quaternion = object_pose[:, None, 3:7].expand_as(relative_quaternion)
    quaternion = quaternion_normalize(
        quaternion_multiply(object_quaternion, relative_quaternion)
    )
    if eef_offset is not None:
        source_rotation = quaternion_to_matrix(quaternion)
        position = position + torch.einsum(
            "btij,j->bti", source_rotation, eef_offset[:3]
        )
        offset_quaternion = eef_offset[3:7].expand_as(quaternion)
        quaternion = quaternion_normalize(
            quaternion_multiply(quaternion, offset_quaternion)
        )
    return torch.cat((position, quaternion, decoded[..., 9:]), dim=-1)


def _object_trajectory(
    world_trajectory: Tensor,
    object_pose: Tensor,
    eef_offset: Tensor | None,
) -> Tensor:
    """Reverse world pose and palm-to-EEF offset into the AE object frame."""
    position = world_trajectory[..., :3]
    quaternion = quaternion_normalize(world_trajectory[..., 3:7])
    if eef_offset is not None:
        inverse_offset = quaternion_conjugate(
            quaternion_normalize(eef_offset[3:7])
        ).expand_as(quaternion)
        quaternion = quaternion_normalize(
            quaternion_multiply(quaternion, inverse_offset)
        )
        palm_rotation = quaternion_to_matrix(quaternion)
        position = position - torch.einsum(
            "btij,j->bti", palm_rotation, eef_offset[:3]
        )
    object_rotation = quaternion_to_matrix(object_pose[:, 3:7])
    object_inverse = quaternion_conjugate(
        quaternion_normalize(object_pose[:, 3:7])
    )[:, None, :].expand_as(quaternion)
    relative_quaternion = quaternion_normalize(
        quaternion_multiply(object_inverse, quaternion)
    )
    relative_position = torch.einsum(
        "bij,btj->bti",
        object_rotation.transpose(-1, -2),
        position - object_pose[:, None, :3],
    )
    relative_rotation = quaternion_to_matrix(relative_quaternion)
    rotation6d = relative_rotation[..., :, :2].transpose(-1, -2).reshape(
        *relative_rotation.shape[:-2], 6
    )
    return torch.cat(
        (relative_position, rotation6d, world_trajectory[..., 7:]), dim=-1
    )


def _world_pointcloud(pointcloud: Tensor, object_pose: Tensor) -> Tensor:
    rotation = quaternion_to_matrix(object_pose[:, 3:7])
    return (
        torch.einsum("bij,bpj->bpi", rotation, pointcloud)
        + object_pose[:, None, :3]
    )


def _world_to_base_pointcloud(pointcloud_world: Tensor, base_pose_world: Tensor) -> Tensor:
    """Express [batch, points, xyz] in the corresponding robot-base frame."""
    base_rotation = quaternion_to_matrix(base_pose_world[:, 3:7])
    relative_position = pointcloud_world - base_pose_world[:, None, :3]
    return torch.einsum(
        "bij,bpj->bpi", base_rotation.transpose(-1, -2), relative_position
    )


def _world_to_base_trajectory(
    trajectory_world: Tensor, base_pose_world: Tensor
) -> Tensor:
    """Express [xyz, xyzw, joints] actions in the corresponding base frame."""
    base_rotation = quaternion_to_matrix(base_pose_world[:, 3:7])
    relative_position = trajectory_world[..., :3] - base_pose_world[:, None, :3]
    position_base = torch.einsum(
        "bij,btj->bti", base_rotation.transpose(-1, -2), relative_position
    )
    base_inverse = quaternion_conjugate(
        quaternion_normalize(base_pose_world[:, 3:7])
    )[:, None, :].expand_as(trajectory_world[..., 3:7])
    orientation_base = quaternion_normalize(
        quaternion_multiply(base_inverse, trajectory_world[..., 3:7])
    )
    return torch.cat(
        (position_base, orientation_base, trajectory_world[..., 7:]), dim=-1
    )


def _smooth_trajectory_noise(trajectory: Tensor, std: float) -> Tensor:
    count, length, channels = trajectory.shape
    control_steps = min(length, max(2, int(math.ceil(length / 4))))
    noise = torch.randn(
        count, control_steps, channels, dtype=trajectory.dtype,
        device=trajectory.device,
    )
    noise = torch.nn.functional.interpolate(
        noise.transpose(1, 2), size=length, mode="linear", align_corners=True
    ).transpose(1, 2)
    return trajectory + float(std) * noise


def _load_object_data(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = (
            "object_label", "object_pose_world", "object_pointcloud",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"dataset is missing required arrays: {missing}")
        result = {key: np.asarray(data[key]) for key in required}
    result["object_label"] = result["object_label"].astype(str)
    result["object_pose_world"] = result["object_pose_world"].astype(np.float32)
    result["object_pointcloud"] = result["object_pointcloud"].astype(np.float32)
    if result["object_pose_world"].shape != (len(result["object_label"]), 7):
        raise ValueError("object_pose_world must have shape [N, 7]")
    if (
        result["object_pointcloud"].ndim != 3
        or result["object_pointcloud"].shape[0] != len(result["object_label"])
        or result["object_pointcloud"].shape[-1] != 3
    ):
        raise ValueError("object_pointcloud must have shape [N, P, 3]")
    return result


def _write_object_pool(labels: list[str]) -> Path:
    import yaml

    parts = [Path(label).parts for label in labels]
    groups = {item[0] for item in parts if len(item) >= 3}
    if len(groups) != 1 or any(len(item) < 3 for item in parts):
        raise ValueError("all object_label entries must use GROUP/urdf/NAME.urdf")
    group = next(iter(groups))
    if any(item[1] != "urdf" for item in parts):
        raise ValueError("object_label entries must point into a urdf directory")
    filenames = sorted({item[-1] for item in parts})
    digest = hashlib.sha1("\n".join(filenames).encode("utf-8")).hexdigest()[:10]
    asset_list_path = (
        Path("assets") / group / f"simulation_guided_objects_{digest}.yaml"
    )
    asset_list_path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(filenames, sort_keys=False)
    if asset_list_path.exists() and asset_list_path.read_text() != content:
        raise RuntimeError(f"object-pool file has unexpected contents: {asset_list_path}")
    asset_list_path.write_text(content)
    return asset_list_path


def _sample_scenes(
    dataset: dict[str, np.ndarray],
    object_ids: np.ndarray,
    object_labels: list[str],
    all_object_labels: list[str],
    position_range: np.ndarray,
    table_height: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    row_by_label = {
        label: np.flatnonzero(dataset["object_label"] == label)
        for label in object_labels
    }
    sampled_labels = [object_labels[int(object_id)] for object_id in object_ids]
    rows = np.asarray([
        rng.choice(row_by_label[label]) for label in sampled_labels
    ])
    stable_object_ids = {label: index for index, label in enumerate(all_object_labels)}
    poses = dataset["object_pose_world"][rows].copy()
    bounds = np.asarray(position_range, dtype=np.float32)
    if bounds.shape != (3, 2):
        raise ValueError("position_range must have shape [3, 2]")
    poses[:, :3] = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(len(rows), 3)
    )
    poses[:, 2] += float(table_height)
    norms = np.linalg.norm(poses[:, 3:7], axis=-1, keepdims=True)
    invalid_quaternions = norms[:, 0] < 1e-8
    poses[:, 3:7] /= np.maximum(norms, 1e-8)
    poses[invalid_quaternions, 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0])
    return (
        torch.as_tensor(dataset["object_pointcloud"][rows].copy(), device=device),
        torch.as_tensor(poses, device=device),
        torch.as_tensor(
            [stable_object_ids[label] for label in sampled_labels],
            dtype=torch.long,
            device=device,
        ),
    )


def _environment_object_ids(env: Any, object_labels: list[str]) -> np.ndarray:
    simulator_labels = [Path(str(path)).name for path in env.object_fns]
    expected_labels = [Path(label).name for label in object_labels]
    if simulator_labels != expected_labels:
        raise RuntimeError("simulator object pool and object label mapping disagree")
    return np.asarray(env.object_asset_ids, dtype=np.int64)


def _hand_action(env: Any, hand_qpos: Tensor) -> Tensor:
    hand_cfg = getattr(env, "hand_specific_cfg", {})
    source_indices = hand_cfg.get("reference_hand_source_indices")
    if source_indices is not None and hand_qpos.shape[-1] != len(env.active_hand_dof_indices):
        if max(source_indices) >= hand_qpos.shape[-1]:
            raise ValueError("AE hand channels do not match reference_hand_source_indices")
        indices = torch.as_tensor(source_indices, dtype=torch.long, device=hand_qpos.device)
        hand_qpos = hand_qpos.index_select(-1, indices)
    if getattr(env, "dual_arm", hand_cfg.get("dualArmInspire", False)):
        left = torch.as_tensor(env.left_active_hand_dof_indices, device=env.device)
        right = torch.as_tensor(env.right_active_hand_dof_indices, device=env.device)
        use_left = env.active_arm_ids[:, None]
        lower = torch.where(
            use_left, env.robot_dof_lower_limits[left][None],
            env.robot_dof_lower_limits[right][None],
        )
        upper = torch.where(
            use_left, env.robot_dof_upper_limits[left][None],
            env.robot_dof_upper_limits[right][None],
        )
    else:
        lower = env.robot_dof_lower_limits[env.active_hand_dof_indices]
        upper = env.robot_dof_upper_limits[env.active_hand_dof_indices]
    hand_qpos = torch.maximum(torch.minimum(hand_qpos, upper), lower)
    return 2.0 * (hand_qpos - lower) / (upper - lower).clamp_min(1e-8) - 1.0


def _append_lift(world_trajectory: Tensor, steps: int, height: float) -> Tensor:
    if steps <= 0 or height == 0:
        return world_trajectory
    alpha = torch.arange(
        1, steps + 1, device=world_trajectory.device, dtype=world_trajectory.dtype
    ).view(1, steps, 1) / float(steps)
    lift = world_trajectory[:, -1:, :7].expand(-1, steps, -1).clone()
    lift[..., 2:3] += alpha * float(height)
    hand = world_trajectory[:, -1:, 7:].expand(-1, steps, -1)
    return torch.cat(
        (world_trajectory, torch.cat((lift, hand), dim=-1)), dim=1
    )


def _classify_simulation_success(
    success_seen: Tensor,
    collision_seen: Tensor,
    state_invalid: Tensor,
    max_lift: Tensor,
    final_lift: Tensor,
    lift_height: float,
    max_lift_height: float,
    min_final_lift_ratio: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reject transient, unstable, non-finite, and runaway simulator states."""
    threshold = max(float(lift_height), 1e-6)
    lifted = max_lift >= threshold
    stable_lift = final_lift >= threshold * float(min_final_lift_ratio)
    runaway = max_lift > float(max_lift_height)
    valid_success = (
        success_seen
        & lifted
        & stable_lift
        & ~collision_seen
        & ~state_invalid
        & ~runaway
    )
    return valid_success, lifted, stable_lift, runaway


def _simulate_wave(
    env: Any,
    decoded: Tensor,
    local_pointcloud: Tensor,
    object_pose: Tensor,
    object_ids: Tensor,
    eef_offset: Tensor | None,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    count = env.num_envs
    if decoded.shape[0] != count:
        raise ValueError(f"each simulation wave must contain {count} trajectories")
    env_ids = torch.arange(count, dtype=torch.long, device=env.device)
    if hasattr(env, "active_arm_ids"):
        env.active_arm_ids.fill_(False)
    env.reset_idx(env_ids, object_init_pose=object_pose.detach().cpu().numpy())
    actual_pose = env.object_init_states[:, :7].clone()
    initial_z = actual_pose[:, 2].clone()
    world_trajectory = _world_trajectory(decoded, actual_pose, eef_offset)
    world_cloud = _world_pointcloud(local_pointcloud, actual_pose)
    simulation_trajectory = _append_lift(
        world_trajectory, args.post_lift_steps, args.post_lift_height
    )
    max_lift = torch.zeros(count, dtype=torch.float32, device=env.device)
    final_lift = torch.zeros_like(max_lift)
    success_seen = torch.zeros(count, dtype=torch.bool, device=env.device)
    collision_seen = torch.zeros_like(success_seen)
    state_invalid = torch.zeros_like(success_seen)
    with torch.no_grad():
        for step in range(simulation_trajectory.shape[1]):
            action = torch.zeros(
                (count, env.num_actions), dtype=torch.float32, device=env.device
            )
            action[:, :7] = simulation_trajectory[:, step, :7]
            action[:, env.hand_dof_start_idx:] = _hand_action(
                env, simulation_trajectory[:, step, 7:]
            )
            env.step(action)
            object_state = env.root_state_tensor[env.object_indices]
            finite_state = torch.isfinite(object_state).all(dim=-1)
            lift = object_state[:, 2] - initial_z
            finite_lift = torch.isfinite(lift)
            state_invalid |= ~finite_state | ~finite_lift
            safe_lift = torch.where(finite_lift, lift, torch.zeros_like(lift))
            max_lift = torch.maximum(max_lift, safe_lift)
            final_lift = safe_lift
            success_seen |= env.successes.to(torch.bool)
            collision_seen |= env.has_hit_table.to(torch.bool)
        for _ in range(args.hold_steps):
            env.step(action)
            object_state = env.root_state_tensor[env.object_indices]
            finite_state = torch.isfinite(object_state).all(dim=-1)
            lift = object_state[:, 2] - initial_z
            finite_lift = torch.isfinite(lift)
            state_invalid |= ~finite_state | ~finite_lift
            safe_lift = torch.where(finite_lift, lift, torch.zeros_like(lift))
            max_lift = torch.maximum(max_lift, safe_lift)
            final_lift = safe_lift
            success_seen |= env.successes.to(torch.bool)
            collision_seen |= env.has_hit_table.to(torch.bool)
    valid_success, lifted, stable_lift, runaway = _classify_simulation_success(
        success_seen, collision_seen, state_invalid, max_lift, final_lift,
        args.lift_height, args.max_lift_height, args.min_final_lift_ratio,
    )
    threshold = max(float(args.lift_height), 1e-6)
    peak_score = (max_lift / threshold).clamp(0.0, 1.0)
    final_score = (final_lift / threshold).clamp(0.0, 1.0)
    lift_score = 0.5 * (peak_score + final_score)
    unstable = collision_seen | runaway | state_invalid
    fitness = (
        args.lift_weight * lift_score
        + args.success_bonus * valid_success.float()
        - args.collision_penalty * unstable.float()
    )
    return {
        "pointcloud": world_cloud,
        "trajectory": world_trajectory,
        "object_pose": actual_pose,
        "object_id": object_ids,
        "fitness": fitness,
        "max_lift": max_lift,
        "final_lift": final_lift,
        "success_seen": success_seen,
        "lifted": lifted,
        "stable_lift": stable_lift,
        "unstable_lift": lifted & ~stable_lift,
        "runaway": runaway,
        "state_invalid": state_invalid,
        "collision": collision_seen,
        "valid_success": valid_success,
    }


def _decode(flow: RealNVP, ae: nn.Module, base: Tensor, latent_shape: tuple[int, ...]) -> Tensor:
    latent = flow.inverse(base).reshape(base.shape[0], *latent_shape)
    return ae.decode(latent)


def _score_cem_population(
    evaluator: GraspEvaluator,
    cloud_world: Tensor,
    world_trajectories: Tensor,
    batch_size: int = 128,
) -> Tensor:
    """Score [scene, candidate] populations without mixing object scenes."""
    scene_count, population_size = world_trajectories.shape[:2]
    flat_trajectories = world_trajectories.reshape(
        scene_count * population_size, *world_trajectories.shape[2:]
    )
    scene_indices = torch.arange(
        scene_count, device=world_trajectories.device
    ).repeat_interleave(population_size)
    scores = torch.empty(
        scene_count * population_size,
        dtype=world_trajectories.dtype,
        device=world_trajectories.device,
    )
    for start in range(0, flat_trajectories.shape[0], batch_size):
        end = min(start + batch_size, flat_trajectories.shape[0])
        scores[start:end] = evaluator(
            cloud_world[scene_indices[start:end]], flat_trajectories[start:end]
        ).reshape(-1)
    return scores.reshape(scene_count, population_size)


def _train_evaluator(
    evaluator: GraspEvaluator,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, float]:
    evaluator.train()
    losses = []
    regression_losses = []
    ranking_losses = []
    reconstruction_losses = []
    for _ in range(args.critic_updates):
        pointcloud, trajectory, fitness, _, _ = buffer.sample_stratified(
            args.critic_batch_size, device, args.lift_weight, args.success_bonus
        )
        fitness_mean, fitness_std = buffer.fitness_stats()
        with torch.no_grad():
            evaluator.set_fitness_stats(fitness_mean, fitness_std)
        prediction, reconstructed_cloud = evaluator(
            pointcloud, trajectory, return_reconstruction=True
        )
        regression_loss = torch.nn.functional.smooth_l1_loss(
            (prediction - fitness_mean) / fitness_std,
            (fitness - fitness_mean) / fitness_std,
        )
        target_indices = torch.randint(
            pointcloud.shape[1],
            (pointcloud.shape[0], evaluator.reconstruction_points),
            device=device,
        )
        reconstruction_target = pointcloud.gather(
            1, target_indices[..., None].expand(-1, -1, 3)
        )
        reconstruction_loss = _pointcloud_chamfer_loss(
            reconstructed_cloud, reconstruction_target
        )
        ranking_pair = buffer.sample_ranking_pairs(
            max(1, args.critic_batch_size // 2), device
        )
        if ranking_pair is None:
            ranking_loss = prediction.new_zeros(())
        else:
            pair_cloud, lower_trajectory, higher_trajectory = ranking_pair
            lower_prediction = evaluator(pair_cloud, lower_trajectory)
            higher_prediction = evaluator(pair_cloud, higher_trajectory)
            ranking_loss = torch.nn.functional.softplus(
                lower_prediction - higher_prediction
            ).mean()
        loss = (
            regression_loss
            + args.ranking_loss_weight * ranking_loss
            + args.pointcloud_recon_weight * reconstruction_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(evaluator.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        regression_losses.append(float(regression_loss.detach().cpu()))
        ranking_losses.append(float(ranking_loss.detach().cpu()))
        reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
    return tuple(
        float(np.mean(values)) if values else 0.0
        for values in (losses, regression_losses, ranking_losses, reconstruction_losses)
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _candidate_quality_metrics(
    predictions: list[np.ndarray],
    fitnesses: list[np.ndarray],
    successes: list[np.ndarray],
) -> dict[str, float]:
    """Measure online evaluator ranking and end-to-end proposal success."""
    predicted = np.concatenate(predictions).astype(np.float64)
    fitness = np.concatenate(fitnesses).astype(np.float64)
    success = np.concatenate(successes).astype(bool)
    success_rate = float(success.mean())
    top_count = max(1, int(math.ceil(0.1 * len(predicted))))
    top_indices = np.argpartition(predicted, -top_count)[-top_count:]
    true_top_indices = np.argpartition(fitness, -top_count)[-top_count:]
    top10_overlap = len(set(top_indices.tolist()) & set(true_top_indices.tolist())) / top_count
    if len(predicted) > 1 and np.std(predicted) > 0 and np.std(fitness) > 0:
        correlation = float(np.corrcoef(
            _average_ranks(predicted), _average_ranks(fitness)
        )[0, 1])
    else:
        correlation = float("nan")
    return {
        "success_rate": success_rate,
        "top_success_rate": float(success[top_indices].mean()),
        "fitness_mae": float(np.abs(predicted - fitness).mean()),
        "spearman": correlation,
        "top10_overlap": float(top10_overlap),
    }


def _fine_tune_successes(
    successes: list[Tensor],
    ae: nn.Module,
    flow: RealNVP,
    ae_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    trajectories = torch.cat(successes, dim=0).to(device)
    for parameter in ae.parameters():
        parameter.requires_grad_(True)
    ae.train()
    ae_losses = []
    for _ in range(args.ae_update_epochs):
        order = torch.randperm(len(trajectories), device=device)
        for start in range(0, len(order), args.ae_batch_size):
            batch = trajectories[order[start:start + args.ae_batch_size]]
            reconstruction, _ = ae(batch)
            feature_loss = (reconstruction - batch).square().mean()
            rotation_loss = (
                rotation6d_to_matrix(reconstruction[..., 3:9])
                - rotation6d_to_matrix(batch[..., 3:9])
            ).square().mean()
            loss = feature_loss + args.rotation_loss_weight * rotation_loss
            ae_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_optimizer.step()
            ae_losses.append(float(loss.detach().cpu()))

    ae.eval()
    with torch.no_grad():
        latent = ae.encode(trajectories).reshape(len(trajectories), -1)
    for parameter in flow.parameters():
        parameter.requires_grad_(True)
    flow.train()
    flow_losses = []
    for _ in range(args.flow_update_steps):
        indices = torch.randint(
            len(latent), (min(args.ae_batch_size, len(latent)),), device=device
        )
        base, log_det = flow(latent[indices])
        log_prob = -0.5 * (
            base.square() + math.log(2.0 * math.pi)
        ).sum(dim=-1) + log_det
        loss = -log_prob.mean()
        flow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        flow_optimizer.step()
        flow_losses.append(float(loss.detach().cpu()))
    flow.eval()
    return (
        float(np.mean(ae_losses)) if ae_losses else 0.0,
        float(np.mean(flow_losses)) if flow_losses else 0.0,
    )


def _save_checkpoints(
    output_dir: Path,
    ae: nn.Module,
    flow: RealNVP,
    evaluator: GraspEvaluator,
    ae_checkpoint: dict[str, Any],
    flow_checkpoint: dict[str, Any],
    buffer: ReplayBuffer,
    object_labels: list[str],
    iteration: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {**ae_checkpoint, "model": ae.state_dict(), "simulation_guided_iteration": iteration},
        output_dir / "ae_simulation_guided.pt",
    )
    torch.save(
        {**flow_checkpoint, "model": flow.state_dict(), "simulation_guided_iteration": iteration},
        output_dir / "flow_simulation_guided.pt",
    )
    torch.save(
        {
            "model": evaluator.state_dict(),
            "config": {
                "point_dim": 3,
                "action_dim": 27,
                "sequence_length": 20,
                "coordinate_frame": "robot_base",
                "architecture": "pointnet_ae_fitness_v1",
                "point_latent_dim": args.pointcloud_latent_dim,
                "reconstruction_points": args.pointcloud_recon_points,
                "pointcloud_recon_weight": args.pointcloud_recon_weight,
                "fitness_mean": float(evaluator.fitness_mean.detach().cpu()),
                "fitness_std": float(evaluator.fitness_std.detach().cpu()),
            },
            "iteration": iteration,
        },
        output_dir / "evaluator.pt",
    )
    buffer.save(output_dir / "buffer.npz", object_labels)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="object metadata/cloud dataset; Inspire is supported")
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/simulation_guided_grasp")
    parser.add_argument(
        "--population-size", type=int, default=64,
        help="number of CEM candidates generated independently for each scene",
    )
    parser.add_argument(
        "--object-count", type=int, default=64,
        help="number of persistent object scenes/environments",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--cem-iterations", type=int, default=5,
        help="number of CEM generations per training round",
    )
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--update-rate", type=float, default=0.7)
    parser.add_argument("--max-base-abs", type=float, default=4.0)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--action-noise-fraction", type=float, default=0.25)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument(
        "--max-lift-height", type=float, default=0.50,
        help="reject trajectories whose object rises beyond this height",
    )
    parser.add_argument(
        "--min-final-lift-ratio", type=float, default=0.50,
        help="require final object lift to retain this fraction of lift-height",
    )
    parser.add_argument("--lift-weight", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=5.0)
    parser.add_argument("--post-lift-height", type=float, default=0.10)
    parser.add_argument("--post-lift-steps", type=int, default=10)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--critic-updates", type=int, default=100)
    parser.add_argument("--critic-batch-size", type=int, default=256)
    parser.add_argument(
        "--ranking-loss-weight", type=float, default=0.1,
        help="weight for within-scene pairwise fitness ranking loss",
    )
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--pointcloud-latent-dim", type=int, default=64)
    parser.add_argument("--pointcloud-recon-points", type=int, default=128)
    parser.add_argument("--pointcloud-recon-weight", type=float, default=0.1)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--min-warmup-successes", type=int, default=100)
    parser.add_argument("--min-warmup-intermediate", type=int, default=100)
    parser.add_argument("--min-evaluator-spearman", type=float, default=0.3)
    parser.add_argument("--max-evaluator-mae", type=float, default=3.0)
    parser.add_argument("--evaluator-ready-rounds", type=int, default=2)
    parser.add_argument("--cem-random-fraction", type=float, default=0.25)
    parser.add_argument("--ae-update-epochs", type=int, default=20)
    parser.add_argument("--ae-batch-size", type=int, default=64)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-5)
    parser.add_argument("--flow-update-steps", type=int, default=200)
    parser.add_argument("--flow-learning-rate", type=float, default=1e-5)
    parser.add_argument("--rotation-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="defaults to the Isaac Gym task device")
    parser.add_argument("--seed", type=int, default=42)
    args, hydra_args = parser.parse_known_args()
    return args, hydra_args


def _validate_args(args: argparse.Namespace) -> None:
    if args.population_size < 1 or args.object_count < 1 or args.iterations < 1:
        raise ValueError("population-size, object-count, and iterations must be positive")
    if args.cem_iterations < 1 or args.critic_updates < 0:
        raise ValueError("cem-iterations must be positive and critic-updates non-negative")
    if args.critic_batch_size < 1 or args.ranking_loss_weight < 0:
        raise ValueError("critic-batch-size must be positive and ranking-loss-weight non-negative")
    if args.pointcloud_latent_dim < 1 or args.pointcloud_recon_points < 1:
        raise ValueError("pointcloud latent dimension and reconstruction points must be positive")
    if args.pointcloud_recon_weight < 0:
        raise ValueError("pointcloud-recon-weight must be non-negative")
    if args.warmup_rounds < 0 or args.min_warmup_successes < 0 or args.min_warmup_intermediate < 0:
        raise ValueError("warmup thresholds must be non-negative")
    if args.evaluator_ready_rounds < 1 or args.max_evaluator_mae <= 0:
        raise ValueError("invalid evaluator readiness settings")
    if not 0.0 <= args.cem_random_fraction < 1.0:
        raise ValueError("cem-random-fraction must be in [0, 1)")
    if not 0.0 < args.elite_fraction <= 1.0:
        raise ValueError("elite-fraction must be in (0, 1]")
    if args.initial_std <= 0 or args.min_std <= 0 or args.max_std < args.min_std:
        raise ValueError("invalid CEM standard-deviation bounds")
    if not 0.0 < args.update_rate <= 1.0:
        raise ValueError("update-rate must be in (0, 1]")
    if args.action_noise_std < 0 or not 0.0 <= args.action_noise_fraction <= 1.0:
        raise ValueError("invalid action noise settings")
    if args.lift_height <= 0 or args.post_lift_steps < 0 or args.hold_steps < 0:
        raise ValueError("invalid lift/hold settings")
    if args.max_lift_height <= args.lift_height:
        raise ValueError("max-lift-height must exceed lift-height")
    if not 0.0 <= args.min_final_lift_ratio <= 1.0:
        raise ValueError("min-final-lift-ratio must be in [0, 1]")
    if args.lift_weight <= 0 or args.success_bonus <= args.lift_weight:
        raise ValueError("success-bonus must exceed lift-weight for a separate success fitness bin")
    if args.buffer_capacity < 1:
        raise ValueError("buffer-capacity must be positive")


def _run(cfg: Any, args: argparse.Namespace) -> None:
    from isaacgym import gymapi  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_seed
    from isaacgymenvs.tasks.base import vec_task
    import tasks  # noqa: F401

    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    object_data = _load_object_data(args.dataset)
    unique_labels = sorted(np.unique(object_data["object_label"]).tolist())
    object_count = args.object_count
    asset_count = min(object_count, len(unique_labels))
    object_labels = random.sample(unique_labels, asset_count)
    object_labels.sort()
    asset_list_path = _write_object_pool(object_labels)

    ae_checkpoint = torch.load(args.ae_checkpoint, map_location="cpu")
    flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
    ae_config = dict(ae_checkpoint["config"])
    flow_config = dict(flow_checkpoint["config"])
    sequence_length = int(ae_config["sequence_length"])
    input_dim = int(ae_config["input_dim"])
    hand_dim = input_dim - 9
    if sequence_length != 20:
        raise ValueError(f"this trainer expects a 20-step AE, got {sequence_length}")
    if hand_dim != 20:
        raise ValueError(
            f"the supplied AE has {hand_dim} hand channels; use the Wuji 29-D AE "
            "and Flow checkpoints for g1_upper_right_wuji"
        )
    latent_shape = tuple(flow_config["latent_shape"])
    if latent_shape != (int(ae_config["num_tokens"]), int(ae_config["latent_dim"])):
        raise ValueError(
            f"Flow latent shape {latent_shape} does not match AE "
            f"({ae_config['num_tokens']}, {ae_config['latent_dim']})"
        )
    if int(flow_config["latent_dim"]) != int(np.prod(latent_shape)):
        raise ValueError("Flow latent_dim does not match latent_shape")

    cfg.task.env.asset.multiObject = True
    cfg.task.env.asset.multiObjectList = str(asset_list_path.relative_to("assets"))
    cfg.task.env.asset.randomizeObjectAssets = False
    cfg.task.env.numEnvs = object_count
    cfg.num_envs = object_count
    cfg.task.env.episodeLength = (
        sequence_length + args.post_lift_steps + args.hold_steps + 1
    )
    cfg.task.env.randomizeTrackingReference = False
    cfg.task.env.randomizeGraspPose = False
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.task.env.render.enable = False
    cfg.force_render = False
    cfg.headless = True
    if cfg.task.env.armController != "pose":
        raise ValueError("simulation-guided grasp training requires task.env.armController=pose")
    dual_arm = bool(cfg.task.hand_config.get(
        "dualArm", cfg.task.hand_config.get("dualArmInspire", False)
    ))
    if dual_arm:
        cfg.task.env.activeArm = 1

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    def create_env() -> Any:
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, object_count,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
            True, cfg.multi_gpu, False, False, cfg,
        )

    def destroy_env(current_env: Any) -> None:
        if getattr(current_env, "viewer", None) is not None:
            current_env.gym.destroy_viewer(current_env.viewer)
        current_env.gym.destroy_sim(current_env.sim)
        if str(current_env.device).startswith("cuda"):
            torch.cuda.synchronize(current_env.device)
        vec_task.EXISTING_SIM = None

    env = create_env()
    device = torch.device(args.device or env.device)
    if str(device) != str(env.device):
        raise ValueError("--device must match the Isaac Gym environment device")

    ae = _build_model(ae_config).to(device)
    ae.load_state_dict(ae_checkpoint["model"])
    flow = _make_flow(flow_config, device)
    flow.load_state_dict(flow_checkpoint["model"])
    ae.eval()
    flow.eval()
    for model in (ae, flow):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    if dual_arm:
        eef_offset_value = cfg.task.hand_config.trajectoryEefOffsetByArm.right
    else:
        eef_offset_value = cfg.task.hand_config.get("trajectoryEefOffset", None)
    eef_offset = (
        None if eef_offset_value is None
        else torch.as_tensor(eef_offset_value, dtype=torch.float32, device=device)
    )
    if eef_offset is not None and eef_offset.shape != (7,):
        raise ValueError("trajectoryEefOffset must contain xyz+xyzw")

    object_ids = _environment_object_ids(env, object_labels)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    robot_base_pose = env.root_state_tensor[env.robot_indices, :7].clone()
    position_range = env.reset_position_range.detach().cpu().numpy()
    table_height = float(np.mean(cfg.task.env.tableHeightRange))
    buffer = ReplayBuffer(
        args.buffer_capacity, object_data["object_pointcloud"].shape[1], 27
    )
    evaluator = GraspEvaluator(
        action_dim=7 + hand_dim,
        point_latent_dim=args.pointcloud_latent_dim,
        reconstruction_points=args.pointcloud_recon_points,
    ).to(device)
    critic_optimizer = torch.optim.AdamW(evaluator.parameters(), lr=1e-3)
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.ae_learning_rate)
    flow_optimizer = torch.optim.Adam(flow.parameters(), lr=args.flow_learning_rate)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_star = args.lift_weight + args.success_bonus
    success_data: list[Tensor] = []
    total_simulation_rollouts = 0
    total_unique_candidates = 0
    total_valid_successes = 0
    first_success_rollout_budget: int | None = None
    evaluator_ready = False
    evaluator_ready_streak = 0
    best_evaluator_mae = float("inf")

    print(
        f"Initialized {env.num_envs} persistent object scenes over "
        f"{len(object_labels)} unique object assets; repeated assets use "
        "independent scene poses. Evaluator inputs use robot-base coordinates. "
        f"CEM population={args.population_size} per scene, "
        f"{args.cem_iterations} generations per training round. "
        f"Simulator success requires success_seen, peak_lift in "
        f"[{args.lift_height:.3f}, {args.max_lift_height:.3f}]m, final_lift>="
        f"{args.lift_height * args.min_final_lift_ratio:.3f}m, finite state, "
        "and no table collision."
    )
    try:
        local_cloud, scene_pose, batch_object_ids = _sample_scenes(
            object_data, object_ids, object_labels, unique_labels,
            position_range, table_height, rng, device
        )
        for iteration in range(1, args.iterations + 1):
            requested_pose = scene_pose.clone()
            requested_pose[:, :3] = torch.as_tensor(
                rng.uniform(
                    position_range[:, 0], position_range[:, 1],
                    size=(env.num_envs, 3),
                ),
                dtype=requested_pose.dtype,
                device=device,
            )
            requested_pose[:, 2] += table_height
            cloud_world = _world_pointcloud(local_cloud, requested_pose)
            cloud_base = _world_to_base_pointcloud(cloud_world, robot_base_pose)
            if len(buffer):
                evaluator.set_fitness_stats(*buffer.fitness_stats())

            for parameter in evaluator.parameters():
                parameter.requires_grad_(False)
            evaluator.eval()
            scene_count = env.num_envs
            population_size = args.population_size
            latent_dim = int(flow_config["latent_dim"])
            mean = torch.zeros(scene_count, latent_dim, device=device)
            std = torch.full_like(mean, float(args.initial_std))
            elite_count = max(1, int(np.ceil(population_size * args.elite_fraction)))
            noise_count = 0
            if (
                args.action_noise_std > 0
                and args.action_noise_fraction > 0
                and population_size > 1
            ):
                noise_count = min(
                    population_size - 1,
                    int(round(population_size * args.action_noise_fraction)),
                )
                noise_count = max(1, noise_count)
            cem_random_count = (
                min(population_size - 1, int(round(population_size * args.cem_random_fraction)))
                if population_size > 1 else 0
            )

            evaluator_guided = evaluator_ready
            cem_generations = args.cem_iterations if evaluator_guided else 1
            best_base = best_decoded = best_world = None
            best_fitness = torch.full(
                (scene_count,), float("-inf"), dtype=torch.float32, device=device
            )
            best_prediction = torch.zeros(scene_count, dtype=torch.float32, device=device)
            random_decoded = random_base = random_prediction = None
            scene_indices = torch.arange(scene_count, device=device)

            for cem_iteration in range(1, cem_generations + 1):
                candidate_base = mean[:, None, :] + std[:, None, :] * torch.randn(
                    scene_count, population_size, latent_dim, device=device
                )
                candidate_base.clamp_(-args.max_base_abs, args.max_base_abs)
                candidate_base[:, 0] = mean if best_base is None else best_base

                flat_base = candidate_base.reshape(scene_count * population_size, latent_dim)
                with torch.no_grad():
                    flat_decoded = _decode(flow, ae, flat_base, latent_shape)
                    candidate_decoded = flat_decoded.reshape(
                        scene_count, population_size, *flat_decoded.shape[1:]
                    )
                    pose_batch = requested_pose[:, None, :].expand(
                        scene_count, population_size, 7
                    ).reshape(scene_count * population_size, 7)
                    flat_world = _world_trajectory(
                        flat_decoded, pose_batch, eef_offset
                    )
                    candidate_world = flat_world.reshape(
                        scene_count, population_size, *flat_world.shape[1:]
                    )
                    if best_decoded is not None:
                        candidate_decoded[:, 0] = best_decoded
                        candidate_world[:, 0] = best_world

                    if noise_count:
                        parent_world = candidate_world[:, 0]
                        noise_parent = parent_world[:, None].expand(
                            scene_count, noise_count, *parent_world.shape[1:]
                        ).reshape(scene_count * noise_count, *parent_world.shape[1:])
                        noisy_world = _smooth_trajectory_noise(
                            noise_parent, args.action_noise_std
                        )
                        noisy_world[..., 3:7] = quaternion_normalize(
                            noisy_world[..., 3:7]
                        )
                        noise_pose = requested_pose[:, None, :].expand(
                            scene_count, noise_count, 7
                        ).reshape(scene_count * noise_count, 7)
                        noisy_decoded = _object_trajectory(
                            noisy_world, noise_pose, eef_offset
                        )
                        noisy_latent = ae.encode(noisy_decoded)
                        noisy_base, _ = flow(noisy_latent.reshape(scene_count * noise_count, -1))
                        candidate_base[:, -noise_count:] = noisy_base.reshape(
                            scene_count, noise_count, latent_dim
                        ).clamp(-args.max_base_abs, args.max_base_abs)
                        candidate_decoded[:, -noise_count:] = noisy_decoded.reshape(
                            scene_count, noise_count, *noisy_decoded.shape[1:]
                        )
                        candidate_world[:, -noise_count:] = noisy_world.reshape(
                            scene_count, noise_count, *noisy_world.shape[1:]
                        )

                    candidate_base_pose = robot_base_pose[:, None, :].expand(
                        scene_count, population_size, 7
                    ).reshape(scene_count * population_size, 7)
                    candidate_eval_trajectory = _world_to_base_trajectory(
                        candidate_world.reshape(
                            scene_count * population_size,
                            *candidate_world.shape[2:],
                        ),
                        candidate_base_pose,
                    ).reshape(
                        scene_count, population_size, *candidate_world.shape[2:]
                    )

                    predicted = _score_cem_population(
                        evaluator, cloud_base, candidate_eval_trajectory
                    )
                    if not torch.isfinite(predicted).all():
                        raise FloatingPointError("evaluator produced a non-finite score")
                    if evaluator_guided:
                        fitness = predicted.clamp(
                            0.0, args.lift_weight + args.success_bonus
                        )
                        if cem_random_count:
                            fitness[:, -cem_random_count:] = torch.rand(
                                scene_count, cem_random_count, device=device
                            ) * (args.lift_weight + args.success_bonus)
                    else:
                        # The first round seeds the replay buffer before the
                        # newly initialized evaluator is trusted for selection.
                        fitness = torch.rand_like(predicted)

                    elite_indices = torch.topk(
                        fitness, elite_count, dim=1, largest=True
                    ).indices
                    elite_base = candidate_base.gather(
                        1, elite_indices.unsqueeze(-1).expand(-1, -1, latent_dim)
                    )
                    new_mean = elite_base.mean(dim=1)
                    new_std = elite_base.std(dim=1, unbiased=False)
                    mean = (1.0 - args.update_rate) * mean + args.update_rate * new_mean
                    std = (1.0 - args.update_rate) * std + args.update_rate * new_std
                    std.clamp_(args.min_std, args.max_std)

                    top_indices = fitness.argmax(dim=1)
                    top_fitness = fitness[scene_indices, top_indices]
                    improved = top_fitness > best_fitness
                    top_base = candidate_base[scene_indices, top_indices]
                    top_decoded = candidate_decoded[scene_indices, top_indices]
                    top_world = candidate_world[scene_indices, top_indices]
                    top_prediction = predicted[scene_indices, top_indices]
                    if best_base is None:
                        best_base = top_base.clone()
                        best_decoded = top_decoded.clone()
                        best_world = top_world.clone()
                        best_fitness = top_fitness.clone()
                        best_prediction = top_prediction.clone()
                    else:
                        best_base[improved] = top_base[improved]
                        best_decoded[improved] = top_decoded[improved]
                        best_world[improved] = top_world[improved]
                        best_fitness[improved] = top_fitness[improved]
                        best_prediction[improved] = top_prediction[improved]

                    random_indices = torch.randint(
                        population_size, (scene_count,), device=device
                    )
                    sampled_decoded = candidate_decoded[scene_indices, random_indices]
                    sampled_base = candidate_base[scene_indices, random_indices]
                    sampled_prediction = predicted[scene_indices, random_indices]
                    if random_decoded is None:
                        random_decoded = sampled_decoded.clone()
                        random_base = sampled_base.clone()
                        random_prediction = sampled_prediction.clone()
                    else:
                        replace = torch.rand(scene_count, device=device) < (1.0 / cem_iteration)
                        random_decoded[replace] = sampled_decoded[replace]
                        random_base[replace] = sampled_base[replace]
                        random_prediction[replace] = sampled_prediction[replace]

                prediction_status = (
                    f"predicted_success={(best_prediction >= p_star).sum().item()}/{scene_count}"
                    if evaluator_guided else "evaluator=warming"
                )
                print(
                    f"iter {iteration:03d}/{args.iterations} CEM "
                    f"{cem_iteration:02d}/{cem_generations} "
                    f"mode={'CEM' if evaluator_guided else 'random_warmup'}: "
                    f"selected_pred_mean={float(best_prediction.mean()):.4f} "
                    f"{prediction_status} "
                    f"std={float(std.mean()):.4f} noise={noise_count}/{population_size}"
                )

            simulation_waves: list[tuple[Tensor, Tensor, Tensor]] = [
                (best_decoded, best_base, best_prediction),
                (random_decoded, random_base, random_prediction),
            ]
            simulation_masks: list[Tensor | None] = [None, None]
            simulation_wave_names = [
                "cem_best" if evaluator_guided else "random_warmup",
                "random_intermediate",
            ]

            all_successes = 0
            iteration_simulation_rollouts = 0
            iteration_unique_candidates = 0
            scenario_ids = (iteration - 1) * scene_count + scene_indices
            candidate_predictions: list[np.ndarray] = []
            candidate_fitnesses: list[np.ndarray] = []
            candidate_successes: list[np.ndarray] = []
            for wave_index, ((wave_decoded, _, wave_prediction), keep, wave_name) in enumerate(
                zip(simulation_waves, simulation_masks, simulation_wave_names)
            ):
                metrics = _simulate_wave(
                    env, wave_decoded, local_cloud, requested_pose,
                    batch_object_ids, eef_offset, args,
                )
                candidate_mask = (
                    keep if keep is not None
                    else torch.ones(env.num_envs, dtype=torch.bool, device=device)
                )
                buffer.add(
                    _world_to_base_pointcloud(
                        metrics["pointcloud"], robot_base_pose
                    ),
                    _world_to_base_trajectory(
                        metrics["trajectory"], robot_base_pose
                    ),
                    metrics["fitness"], metrics["object_id"],
                    scenario_ids, metrics["object_pose"], keep=keep,
                )
                valid = metrics["valid_success"] & candidate_mask
                success_count = int(valid.sum().item())
                all_successes += success_count
                wave_candidate_count = int(candidate_mask.sum().item())
                iteration_unique_candidates += wave_candidate_count
                iteration_simulation_rollouts += env.num_envs
                candidate_predictions.append(
                    wave_prediction[candidate_mask].detach().cpu().numpy()
                )
                candidate_fitnesses.append(
                    metrics["fitness"][candidate_mask].detach().cpu().numpy()
                )
                candidate_successes.append(
                    metrics["valid_success"][candidate_mask].detach().cpu().numpy()
                )
                if success_count and first_success_rollout_budget is None:
                    first_success_rollout_budget = (
                        total_simulation_rollouts
                        + iteration_simulation_rollouts
                    )
                if success_count:
                    object_frame = _object_trajectory(
                        metrics["trajectory"][valid],
                        metrics["object_pose"][valid],
                        eef_offset,
                    )
                    success_data.append(object_frame.detach().cpu())
                max_fitness = float(metrics["fitness"][candidate_mask].max().item())
                print(
                    f"iter {iteration:03d}/{args.iterations} "
                    f"wave {wave_index + 1}/{len(simulation_waves)} "
                    f"({wave_name}): "
                    f"best_fitness={max_fitness:.4f} "
                    f"max_lift={float(metrics['max_lift'][candidate_mask].max()):.4f}m "
                    f"successes={success_count} "
                    f"collisions={int(metrics['collision'][candidate_mask].sum())} "
                    f"unstable_lifts={int(metrics['unstable_lift'][candidate_mask].sum())} "
                    f"runaways={int(metrics['runaway'][candidate_mask].sum())} "
                    f"invalid_states={int(metrics['state_invalid'][candidate_mask].sum())} "
                    f"buffer={len(buffer)}"
                )
            for parameter in evaluator.parameters():
                parameter.requires_grad_(True)
            (
                critic_loss, critic_regression_loss, critic_ranking_loss,
                critic_reconstruction_loss,
            ) = _train_evaluator(
                evaluator, critic_optimizer, buffer, args, device
            )
            fitness_bin_counts = buffer.fitness_bin_counts(
                args.lift_weight, args.success_bonus
            )
            quality = _candidate_quality_metrics(
                candidate_predictions, candidate_fitnesses, candidate_successes
            )
            predicted_array = np.concatenate(candidate_predictions)
            fitness_array = np.concatenate(candidate_fitnesses)
            predicted_top = np.argpartition(predicted_array, -max(1, int(math.ceil(0.1 * len(predicted_array)))))[-max(1, int(math.ceil(0.1 * len(predicted_array)))):]
            true_top = np.argpartition(fitness_array, -max(1, int(math.ceil(0.1 * len(fitness_array)))))[-max(1, int(math.ceil(0.1 * len(fitness_array)))):]
            quality_path = output_dir / "evaluator_quality.csv"
            write_header = not quality_path.exists()
            with quality_path.open("a", newline="") as quality_file:
                quality_writer = csv.writer(quality_file)
                if write_header:
                    quality_writer.writerow([
                        "iteration", "spearman", "mae", "top10_overlap",
                        "predicted_top10_mean", "true_top10_mean",
                    ])
                quality_writer.writerow([
                    iteration, quality["spearman"], quality["fitness_mae"],
                    quality["top10_overlap"], predicted_array[predicted_top].mean(),
                    fitness_array[true_top].mean(),
                ])
            bin_successes = int(fitness_bin_counts[5])
            bin_intermediate = int(sum(fitness_bin_counts[2:5]))
            mae_improving = quality["fitness_mae"] < best_evaluator_mae
            if mae_improving:
                best_evaluator_mae = quality["fitness_mae"]
            ready_conditions = (
                iteration >= args.warmup_rounds
                and len(buffer) >= args.critic_batch_size
                and bin_successes >= args.min_warmup_successes
                and bin_intermediate >= args.min_warmup_intermediate
                and np.isfinite(quality["spearman"])
                and quality["spearman"] >= args.min_evaluator_spearman
                and quality["fitness_mae"] <= args.max_evaluator_mae
                and mae_improving
            )
            evaluator_ready_streak = evaluator_ready_streak + 1 if ready_conditions else 0
            evaluator_ready = evaluator_ready_streak >= args.evaluator_ready_rounds
            ae_loss = flow_loss = 0.0
            if success_data:
                ae_loss, flow_loss = _fine_tune_successes(
                    success_data, ae, flow, ae_optimizer, flow_optimizer,
                    args, device,
                )
            _save_checkpoints(
                output_dir, ae, flow, evaluator, ae_checkpoint,
                flow_checkpoint, buffer, unique_labels, iteration, args,
            )
            total_simulation_rollouts += iteration_simulation_rollouts
            total_unique_candidates += iteration_unique_candidates
            total_valid_successes += all_successes
            cumulative_success_rate = (
                total_valid_successes / max(total_unique_candidates, 1)
            )
            simulator_rollouts_per_success = (
                total_simulation_rollouts / total_valid_successes
                if total_valid_successes else float("inf")
            )
            first_success_text = (
                str(first_success_rollout_budget)
                if first_success_rollout_budget is not None else "pending"
            )
            spearman_text = (
                f"{quality['spearman']:.3f}"
                if np.isfinite(quality["spearman"]) else "n/a"
            )
            print(
                f"round {iteration:03d}/{args.iterations}: "
                f"success={all_successes}/{iteration_unique_candidates} "
                f"({quality['success_rate']:.2%}; cumulative={cumulative_success_rate:.2%}), "
                f"sim_rollouts={iteration_simulation_rollouts} "
                f"(total={total_simulation_rollouts}), "
                f"rollouts/success={simulator_rollouts_per_success:.1f}, "
                f"first_success_rollouts={first_success_text}, "
                f"evaluator_mae={quality['fitness_mae']:.3f} "
                f"Spearman={spearman_text} "
                f"top10_success={quality['top_success_rate']:.2%} "
                f"top10_overlap={quality['top10_overlap']:.2%} "
                f"(overall={quality['success_rate']:.2%}), "
                f"train_loss={critic_loss:.5f} "
                f"(reg={critic_regression_loss:.5f}, rank={critic_ranking_loss:.5f}, "
                f"pc_recon={critic_reconstruction_loss:.5f}), "
                f"fitness_bins=neg:{fitness_bin_counts[0]} "
                f"low:{fitness_bin_counts[1]} mid1:{fitness_bin_counts[2]} "
                f"mid2:{fitness_bin_counts[3]} high:{fitness_bin_counts[4]} "
                f"success:{fitness_bin_counts[5]}, AE_recon={ae_loss:.5f}, "
                f"Flow_NLL={flow_loss:.5f}, buffer={len(buffer)}"
                f", evaluator_ready={evaluator_ready}"
            )
            if all_successes:
                print(
                    f"Found {all_successes} simulator-validated successful "
                    "trajectory(s); saved buffer and updated AE/Flow."
                )
    finally:
        if env is not None:
            destroy_env(env)
            del env
            import gc
            gc.collect()
        vec_task.EXISTING_SIM = None


def main() -> None:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    import hydra

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def hydra_entry(cfg: Any) -> None:
        _run(cfg, args)

    hydra_entry()


if __name__ == "__main__":
    main()
