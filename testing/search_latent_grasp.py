"""Search a grasp trajectory in the RealNVP Gaussian base space.

Each CEM population member is evaluated by one Isaac Gym environment.  A
stage-1 candidate is represented by a standard-normal base vector ``u`` and
decoded as ``u -> flow.inverse(u | c(s)) -> AE.decode(z)`` when the Flow is
conditional. The condition is the selected object's point cloud encoded by
the point-cloud AE recorded in the Flow checkpoint. When enabled, stage 2
encodes the best trajectory's final 10 frames with a separate 10-step AE and
Flow, then searches that Flow-base neighborhood while keeping the first 10
frames fixed. Both stages also mutate a portion of the current parent in
trajectory space, encode those mutations back to Flow-base space, and stop
early when the simulator reports a valid successful grasp.

Example
-------
python -m testing.search_latent_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --pointcloud-ae-checkpoint runs/object_pointcloud_ae/best.pt \
  --stage2-ae-checkpoint runs/trajectory_ae_10/ae_10.pt \
  --stage2-flow-checkpoint runs/latent_flow_10/flow_10.pt \
  --object-index 0 \
  --population-size 64 \
  --iterations 20 \
  --twostage_iterations 10 \
  --output runs/latent_flow/best_grasp.npz \
  task=grasp hand=fr3_inspire_tac num_envs=64 headless=True \
  task.env.armController=pose

For a dual-arm G1, this search always uses the right arm. The selected URDF is
randomized on each task reset using the task's ``resetPositionRange`` and
``resetRandomRot``. Add
``--no-randomize-object-pose`` to replay at the dataset pose, or provide
``--object-position``/``--object-quaternion`` for an explicit fixed pose.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument(
        "--pointcloud-ae-checkpoint", default="",
        help="point-cloud AE for a conditional Flow; defaults to its recorded checkpoint path",
    )
    parser.add_argument(
        "--stage2-ae-checkpoint", "--stage2_ae_checkpoint", "--ae-checkpoint-stage2",
        default="runs/trajectory_ae_10/ae_10.pt",
        help="10-step AE checkpoint used by the optional second-stage suffix search",
    )
    parser.add_argument(
        "--stage2-flow-checkpoint", "--stage2_flow_checkpoint", "--flow-checkpoint-stage2",
        default="runs/latent_flow_10/flow_10.pt",
        help="10-step Flow checkpoint used by the optional second-stage suffix search",
    )
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--object-position", type=float, nargs=3, default=None)
    parser.add_argument("--object-quaternion", type=float, nargs=4, default=None)
    parser.add_argument(
        "--estimate_xyzrpy", "--estimate-xyzrpy",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        default=False,
        help="estimate the object origin and XYZ frame from the initial camera depth/mask",
    )
    parser.add_argument(
        "--no-randomize-object-pose",
        action="store_true",
        help="use the dataset/override object pose instead of task reset randomization",
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--twostage_iterations", "--twostage-iterations", type=int, default=0,
        help="search the final 10 frames with the stage-2 AE/Flow if stage 1 finds no success",
    )
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--update-rate", type=float, default=0.7)
    parser.add_argument(
        "--action-noise-std", type=float, default=0.01,
        help=(
            "trajectory-space Gaussian noise used for local exploration; "
            "XYZ is in meters, rotation-6D and hand joints use their native units"
        ),
    )
    parser.add_argument(
        "--action-noise-fraction", type=float, default=0.25,
        help="fraction of each population reserved for trajectory-space exploration",
    )
    parser.add_argument("--max-base-abs", type=float, default=4.0)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--post-lift-height", type=float, default=0.10)
    parser.add_argument("--post-lift-steps", type=int, default=10)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--lift-weight", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=5.0)
    parser.add_argument("--prior-penalty", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args


ARGS = _parse_args()


def _estimate_frame_from_points(points):
    """Estimate a gravity-aligned XYZ frame from points in world coordinates."""
    origin = points.mean(axis=0)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    centered = points.astype(np.float64) - origin
    centered -= np.outer(centered @ z_axis, z_axis)
    covariance = centered.T @ centered / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    x_axis = eigenvectors[:, -1]
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    norm = np.linalg.norm(x_axis)
    if norm < 1e-8:
        raise ValueError("masked object cloud does not define a horizontal axis")
    x_axis /= norm
    if x_axis[0] < 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return origin.astype(np.float32), np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def _estimate_object_pose_from_camera(env, env_index, fallback_pose):
    """Estimate a world-frame, gravity-aligned object pose from masked depth."""
    camera_id = int(env.camera_ids[0])
    camera_cfg = env.camera_cfg[f"camera_{camera_id}"]
    depth = env.depth_tensors[0][env_index].detach().cpu().numpy()
    segmentation = env.seg_tensors[0][env_index].detach().cpu().numpy()
    valid = (segmentation == 2) & np.isfinite(depth) & (depth > 1e-6)
    v, u = np.nonzero(valid)
    if len(u) < 4:
        raise ValueError(f"camera mask contains only {len(u)} valid depth pixels")
    z = depth[v, u].astype(np.float32)
    intrinsics = np.asarray(camera_cfg["intrinsics"], dtype=np.float32)
    points = np.stack(
        ((u - intrinsics[0, 2]) * z / intrinsics[0, 0],
         (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z), axis=1
    )
    transform = env.gym.get_camera_transform(env.sim, env.envs[env_index], env.camera_handles[0])
    camera_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
    camera_rotation = Rotation.from_quat([transform.r.x, transform.r.y, transform.r.z, transform.r.w])
    basis = Rotation.from_matrix(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32))
    points_world = camera_position + camera_rotation.apply(basis.inv().apply(points))
    world_origin, world_rotation_matrix = _estimate_frame_from_points(points_world)
    world_rotation = Rotation.from_matrix(world_rotation_matrix)
    return np.concatenate((world_origin, world_rotation.as_quat().astype(np.float32)))


def _quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_to_matrix(q):
    q = _quat_normalize(q)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quaternion(matrix):
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-8)) * 0.5
    qx = (m21 - m12) / (4 * qw).clamp_min(1e-8)
    qy = (m02 - m20) / (4 * qw).clamp_min(1e-8)
    qz = (m10 - m01) / (4 * qw).clamp_min(1e-8)
    return _quat_normalize(torch.stack((qx, qy, qz, qw), dim=-1))


def _world_wrist(object_pose, decoded, rotation6d_to_matrix):
    """Convert [population, time, 15] object-frame features to world pose."""
    if object_pose.ndim == 1:
        object_pose_batch = object_pose.unsqueeze(0).expand(decoded.shape[0], -1)
    elif object_pose.ndim == 2 and object_pose.shape[0] == decoded.shape[0]:
        object_pose_batch = object_pose
    else:
        raise ValueError(
            "object_pose must have shape [7] or [population, 7], "
            f"got {tuple(object_pose.shape)} for decoded shape {tuple(decoded.shape)}"
        )
    object_rotation = _quat_to_matrix(object_pose_batch[:, 3:7])
    position = torch.einsum("bti,bji->btj", decoded[..., :3], object_rotation)
    position = position + object_pose_batch[:, None, :3]
    relative_quaternion = _matrix_to_quaternion(
        rotation6d_to_matrix(decoded[..., 3:9])
    )
    object_quaternion = object_pose_batch[:, None, 3:7].expand_as(relative_quaternion)
    orientation = _quat_multiply(object_quaternion, relative_quaternion)
    return torch.cat((position, _quat_normalize(orientation)), dim=-1)


def _rpy_to_matrix(rpy):
    """Convert XYZ roll-pitch-yaw angles to local-frame rotation matrices."""
    roll, pitch, yaw = rpy.unbind(-1)
    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)
    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)
    rx = torch.stack(
        (torch.stack((ones, zeros, zeros), -1),
         torch.stack((zeros, cx, -sx), -1),
         torch.stack((zeros, sx, cx), -1)), -2
    )
    ry = torch.stack(
        (torch.stack((cy, zeros, sy), -1),
         torch.stack((zeros, ones, zeros), -1),
         torch.stack((-sy, zeros, cy), -1)), -2
    )
    rz = torch.stack(
        (torch.stack((cz, -sz, zeros), -1),
         torch.stack((sz, cz, zeros), -1),
         torch.stack((zeros, zeros, ones), -1)), -2
    )
    return torch.matmul(torch.matmul(rx, ry), rz)


def _rotate_object_frame_trajectory(decoded, rpy, rotation6d_to_matrix):
    """Apply one object-frame RPY rotation to every pose in each trajectory."""
    rpy = torch.as_tensor(rpy, dtype=decoded.dtype, device=decoded.device)
    if rpy.ndim == 1:
        rpy = rpy.unsqueeze(0).expand(decoded.shape[0], -1)
    if rpy.shape != (decoded.shape[0], 3):
        raise ValueError(
            f"object-frame RPY must have shape [{decoded.shape[0]}, 3], got {tuple(rpy.shape)}"
        )
    rotation = _rpy_to_matrix(rpy)
    rotated = decoded.clone()
    rotated[..., :3] = torch.einsum("bij,btj->bti", rotation, decoded[..., :3])
    relative_rotation = rotation6d_to_matrix(decoded[..., 3:9])
    rotated_rotation = torch.matmul(rotation.unsqueeze(1), relative_rotation)
    rotated[..., 3:9] = rotated_rotation[..., :, :2].transpose(-1, -2).reshape(
        decoded.shape[:-1] + (6,)
    )
    return rotated


def _append_world_post_lift(world_targets, hand_targets):
    """Append lift after palm-to-eef conversion, in the actual world frame."""
    if ARGS.post_lift_steps == 0 or ARGS.post_lift_height == 0.0:
        return world_targets, hand_targets
    steps = ARGS.post_lift_steps
    alpha = torch.arange(
        1, steps + 1, dtype=world_targets.dtype, device=world_targets.device
    ).view(1, steps, 1) / float(steps)
    lift = world_targets[:, -1:, :].expand(-1, steps, -1).clone()
    lift[..., 2:3] += alpha * float(ARGS.post_lift_height)
    hand = hand_targets[:, -1:, :].expand(-1, steps, -1)
    return torch.cat((world_targets, lift), 1), torch.cat((hand_targets, hand), 1)


def _action_targets(
    object_pose, decoded, rotation6d_to_matrix, object_frame_rpy,
):
    decoded = _rotate_object_frame_trajectory(
        decoded, object_frame_rpy, rotation6d_to_matrix
    )
    # AE trajectories are expressed in an object-relative frame. Convert both
    # position and orientation with the sampled object's world pose.
    world_targets = _world_wrist(object_pose, decoded, rotation6d_to_matrix)
    hand_targets = decoded[..., 9:]
    # The actual lift is added after the robot eef offset is applied by the
    # caller, so its Z displacement is unambiguously in world coordinates.
    return world_targets, hand_targets


def _apply_eef_offset(world_targets, eef_offset):
    """Convert source eef poses to the configured robot eef frame.

    ``eef_offset`` is [dx, dy, dz, qx, qy, qz, qw] expressed in the source
    palm frame, so the target transform is ``T_palm @ T_offset``.
    """
    if eef_offset is None:
        return world_targets
    eef_offset = eef_offset.to(world_targets)
    offset_position = eef_offset[..., :3]
    offset_quaternion = _quat_normalize(eef_offset[..., 3:7])
    source_rotation = _quat_to_matrix(world_targets[..., 3:7])
    if offset_position.ndim == 1:
        position_delta = torch.matmul(
            offset_position.view(1, 1, 3), source_rotation.transpose(-1, -2)
        ).squeeze(-2)
    else:
        # Per-candidate offsets: [B, 3] applied to [B, T, 3, 3].
        position_delta = torch.einsum(
            "bi,btji->btj", offset_position, source_rotation
        )
    position = world_targets[..., :3] + position_delta
    if offset_quaternion.ndim == 1:
        orientation_offset = offset_quaternion.view(1, 1, 4)
    else:
        orientation_offset = offset_quaternion[:, None, :]
    orientation = _quat_multiply(world_targets[..., 3:7], orientation_offset)
    return torch.cat((position, _quat_normalize(orientation)), dim=-1)


def _hand_action(env, hand_qpos):
    hand_cfg = getattr(env, "hand_specific_cfg", {})
    source_indices = hand_cfg.get("reference_hand_source_indices")
    if source_indices is not None and hand_qpos.shape[-1] != len(env.active_hand_dof_indices):
        if max(source_indices) >= hand_qpos.shape[-1]:
            raise ValueError(
                f"Decoded hand has {hand_qpos.shape[-1]} DoFs, but mapping needs source index {max(source_indices)}"
            )
        source_indices = torch.as_tensor(source_indices, dtype=torch.long, device=hand_qpos.device)
        hand_qpos = hand_qpos.index_select(-1, source_indices)
    if getattr(env, "dual_arm", hand_cfg.get("dualArmInspire", False)):
        left_idx = torch.as_tensor(env.left_active_hand_dof_indices, device=env.device)
        right_idx = torch.as_tensor(env.right_active_hand_dof_indices, device=env.device)
        lower_l, upper_l = env.robot_dof_lower_limits[left_idx], env.robot_dof_upper_limits[left_idx]
        lower_r, upper_r = env.robot_dof_lower_limits[right_idx], env.robot_dof_upper_limits[right_idx]
        use_left = env.active_arm_ids[:, None]
        lower = torch.where(use_left, lower_l[None], lower_r[None])
        upper = torch.where(use_left, upper_l[None], upper_r[None])
        hand_qpos = torch.maximum(torch.minimum(hand_qpos, upper), lower)
        return 2.0 * (hand_qpos - lower) / (upper - lower).clamp_min(1e-8) - 1.0
    lower = env.robot_dof_lower_limits[env.active_hand_dof_indices]
    upper = env.robot_dof_upper_limits[env.active_hand_dof_indices]
    hand_qpos = torch.maximum(torch.minimum(hand_qpos, upper), lower)
    return 2.0 * (hand_qpos - lower) / (upper - lower).clamp_min(1e-8) - 1.0


def _configure_object(cfg: DictConfig, label: str) -> None:
    label_path = Path(label)
    if label_path.suffix != ".urdf":
        raise ValueError(f"Expected an object URDF label, got {label!r}")
    list_path = Path("assets") / label_path.parent.parent / "search_grasp_object.yaml"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    list_path.write_text(yaml.safe_dump([label_path.name]))
    cfg.task.env.asset.multiObjectList = str(label_path.parent.parent / list_path.name)


def _load_object_pose():
    with np.load(ARGS.dataset, allow_pickle=False) as data:
        labels = data["object_label"]
        poses = np.asarray(data["object_pose_world"], dtype=np.float32)
    if not 0 <= ARGS.object_index < len(labels):
        raise IndexError(f"--object-index must be in [0, {len(labels)})")
    pose = poses[ARGS.object_index].copy()
    if ARGS.object_position is not None:
        pose[:3] = np.asarray(ARGS.object_position, dtype=np.float32)
    if ARGS.object_quaternion is not None:
        pose[3:7] = np.asarray(ARGS.object_quaternion, dtype=np.float32)
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm < 1e-8:
        pose[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        pose[3:7] /= quaternion_norm
    return str(labels[ARGS.object_index]), pose


def _load_flow_condition(dataset, object_index, flow_configs, checkpoint_override):
    conditional_configs = [
        config for config in flow_configs
        if config is not None and int(config.get("condition_dim", 0)) > 0
    ]
    if not conditional_configs:
        return None
    condition_dims = {int(config["condition_dim"]) for config in conditional_configs}
    if len(condition_dims) != 1:
        raise ValueError("stage-1 and stage-2 Flow condition dimensions must match")
    recorded_paths = {
        str(config.get("pointcloud_ae_checkpoint", ""))
        for config in conditional_configs
        if config.get("pointcloud_ae_checkpoint")
    }
    if checkpoint_override:
        checkpoint_path = checkpoint_override
    elif len(recorded_paths) == 1:
        checkpoint_path = next(iter(recorded_paths))
    elif not recorded_paths:
        raise ValueError(
            "conditional Flow requires --pointcloud-ae-checkpoint or a recorded point-cloud AE path"
        )
    else:
        raise ValueError(
            "stage-1 and stage-2 Flow checkpoints record different point-cloud AEs; "
            "pass --pointcloud-ae-checkpoint to select one"
        )

    from memory.train_pointcloud_ae import (
        _build_model as _build_pointcloud_model,
        _deterministic_points,
        _load_dataset as _load_pointcloud_dataset,
    )

    point_data, pointclouds = _load_pointcloud_dataset(dataset)
    for key in ("object_label", "object_label_id"):
        if key in point_data and len(point_data[key]) != len(pointclouds):
            raise ValueError(f"object_pointcloud rows do not align with {key} rows")
    if not 0 <= object_index < len(pointclouds):
        raise IndexError(
            f"object-index must be in [0, {len(pointclouds) - 1}], got {object_index}"
        )
    point_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    point_config = point_checkpoint["config"]
    if point_config.get("coordinate_frame") != "object":
        raise ValueError("point-cloud AE must use object-frame point clouds")
    expected_architectures = {
        config.get("pointcloud_ae_architecture")
        for config in conditional_configs
        if config.get("pointcloud_ae_architecture")
    }
    if expected_architectures and point_config.get("architecture") not in expected_architectures:
        raise ValueError(
            "point-cloud AE architecture does not match the conditional Flow checkpoint"
        )
    if int(point_config["latent_dim"]) != next(iter(condition_dims)):
        raise ValueError(
            "point-cloud AE feature dimension does not match conditional Flow: "
            f"{point_config['latent_dim']} vs {next(iter(condition_dims))}"
        )
    encoder = _build_pointcloud_model(point_config)
    encoder.load_state_dict(point_checkpoint["model"])
    encoder.eval()
    cloud = pointclouds[object_index:object_index + 1]
    cloud = _deterministic_points(cloud, int(point_config["input_points"]))
    with torch.no_grad():
        condition = encoder.encode(cloud)
    print(
        f"Conditioning Flow on object_index={object_index} point-cloud feature "
        f"dim={condition.shape[-1]} using {checkpoint_path}"
    )
    return condition


def _decode_candidates(
    flow, ae, search_vector, latent_shape, latent_dim, condition=None
):
    latent_base = search_vector[..., :latent_dim]
    # Object-frame orientation is fixed; only the learned latent is searched.
    object_frame_rpy = torch.zeros(
        (*search_vector.shape[:-1], 3), dtype=search_vector.dtype,
        device=search_vector.device,
    )
    with torch.no_grad():
        expanded_condition = (
            None if condition is None
            else condition.expand(latent_base.shape[0], -1)
        )
        latent_flat = flow.inverse(latent_base, expanded_condition)
        latent = latent_flat.reshape(search_vector.shape[0], *latent_shape)
        decoded = ae.decode(latent)
    return latent, decoded, object_frame_rpy


def _encode_candidates(flow, ae, trajectories, condition=None):
    """Project action-space exploration samples back into Flow-base space."""
    with torch.no_grad():
        latent = ae.encode(trajectories)
        expanded_condition = (
            None if condition is None
            else condition.expand(trajectories.shape[0], -1)
        )
        base, _ = flow(latent.reshape(trajectories.shape[0], -1), expanded_condition)
    return latent, base


def _select_candidate_index(fitness, valid_success):
    """Prefer a successful candidate whenever this population contains one."""
    success_indices = torch.nonzero(valid_success, as_tuple=False).flatten()
    if success_indices.numel() > 0:
        return int(success_indices[torch.argmax(fitness[success_indices])])
    return int(torch.argmax(fitness))


def _perturb_trajectory(parent, count):
    """Create local action-space mutations around one parent trajectory."""
    if count < 1:
        return parent.new_empty((0, *parent.shape))
    control_steps = min(parent.shape[0], max(2, int(math.ceil(parent.shape[0] / 4))))
    noise = torch.randn(
        (count, control_steps, parent.shape[-1]),
        dtype=parent.dtype,
        device=parent.device,
    )
    noise = torch.nn.functional.interpolate(
        noise.transpose(1, 2),
        size=parent.shape[0],
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    return parent.unsqueeze(0) + float(ARGS.action_noise_std) * noise


def _evaluate_population(
    env,
    flow,
    ae,
    base,
    latent_shape,
    object_pose,
    rotation6d_to_matrix,
    randomize_object_pose,
    eef_offset,
    latent_dim,
    reset_object_pose=None,
    frame_callback=None,
    decoded_override=None,
    latent_override=None,
    condition=None,
):
    if decoded_override is None:
        latent, decoded, object_frame_rpy = _decode_candidates(
            flow, ae, base, latent_shape, latent_dim, condition
        )
    else:
        decoded = decoded_override
        latent = (
            latent_override
            if latent_override is not None
            else torch.zeros(
                (base.shape[0], latent_dim), dtype=base.dtype, device=base.device
            )
        )
        object_frame_rpy = torch.zeros(
            (base.shape[0], 3), dtype=base.dtype, device=base.device
        )
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if randomize_object_pose:
        # Let Grasp.reset_idx sample resetPositionRange/resetRandomRot, just as
        # the checkpoint-backed collection path does.
        env.reset_idx(env_ids)
    else:
        reset_pose = object_pose if reset_object_pose is None else reset_object_pose
        env.reset_idx(env_ids, object_init_pose=reset_pose.detach().cpu().numpy())
    # Use the simulator's post-reset pose for both fixed and randomized
    # evaluation.  In particular, the estimated pose can differ from the
    # pose that reset_idx actually installs, so converting object-relative
    # trajectories with the requested pose would evaluate a shifted target.
    object_poses = env.object_init_states[:, :7].clone()

    # object_init_states is stored per environment, while object_indices are
    # global actor indices into root_state_tensor.
    if env.object_init_states.shape[0] != env.num_envs:
        raise RuntimeError(
            "Expected object_init_states to have one row per environment, "
            f"got shape {tuple(env.object_init_states.shape)} for {env.num_envs} envs"
        )
    initial_z = env.object_init_states[:, 2].clone()
    arm_qpos_at_reset = env.robot_dof_pos[:, env.arm_dof_indices].clone()
    if getattr(env, "hand_specific_cfg", {}).get("dualArmInspire", False):
        selected_eef_indices = torch.where(
            env.active_arm_ids, env.eef_idx_left, env.eef_idx_right
        )
    else:
        selected_eef_indices = env.eef_idx
    rigid_states = env.rigid_body_states.view(-1, 13)
    eef_z_before_lift = rigid_states[selected_eef_indices, 2].clone()
    eef_z_at_lift_start = eef_z_before_lift.clone()
    max_lift = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    final_lift = torch.zeros_like(max_lift)
    success_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    collision_seen = torch.zeros_like(success_seen)

    world_targets, hand_targets = _action_targets(
        object_poses,
        decoded,
        rotation6d_to_matrix,
        object_frame_rpy,
    )
    world_targets = _apply_eef_offset(world_targets, eef_offset)
    world_targets, hand_targets = _append_world_post_lift(world_targets, hand_targets)
    with torch.no_grad():
        for step in range(world_targets.shape[1]):
            action = torch.zeros(
                (env.num_envs, env.num_actions),
                dtype=torch.float32,
                device=env.device,
            )
            action[:, :7] = world_targets[:, step].to(env.device)
            action[:, env.hand_dof_start_idx:] = _hand_action(
                env, hand_targets[:, step].to(env.device)
            )
            env.step(action)
            if frame_callback is not None:
                frame_callback(step, action, "trajectory")
            if step == decoded.shape[1] - 1:
                eef_z_at_lift_start = env.rigid_body_states.view(-1, 13)[
                    selected_eef_indices, 2
                ].clone()
            lift = env.root_state_tensor[env.object_indices, 2] - initial_z
            max_lift = torch.maximum(max_lift, lift)
            final_lift = lift
            success_seen |= env.successes.to(torch.bool)
            collision_seen |= env.has_hit_table

        for hold_step in range(max(0, ARGS.hold_steps)):
            env.step(action)
            if frame_callback is not None:
                frame_callback(world_targets.shape[1] + hold_step, action, "hold")
            lift = env.root_state_tensor[env.object_indices, 2] - initial_z
            max_lift = torch.maximum(max_lift, lift)
            final_lift = lift
            success_seen |= env.successes.to(torch.bool)
            collision_seen |= env.has_hit_table

    lift_height = max(float(ARGS.lift_height), 1e-6)
    peak_score = (max_lift / lift_height).clamp(0.0, 1.0)
    final_score = (final_lift / lift_height).clamp(0.0, 1.0)
    lift_score = 0.5 * (peak_score + final_score)
    arm_joint_motion = (
        env.robot_dof_pos[:, env.arm_dof_indices] - arm_qpos_at_reset
    ).abs().amax(dim=-1)
    arm_target_motion = (
        env.cur_targets[:, env.arm_dof_indices] - arm_qpos_at_reset
    ).abs().amax(dim=-1)
    arm_target_error = (
        env.cur_targets[:, env.arm_dof_indices]
        - env.robot_dof_pos[:, env.arm_dof_indices]
    ).abs().amax(dim=-1)
    eef_z_after_lift = env.rigid_body_states.view(-1, 13)[selected_eef_indices, 2]
    eef_target_lift = (
        world_targets[:, -1, 2] - world_targets[:, decoded.shape[1] - 1, 2]
    )
    eef_actual_lift = eef_z_after_lift - eef_z_at_lift_start
    lifted = max_lift >= lift_height
    valid_success = success_seen & lifted & ~collision_seen
    fitness = (
        ARGS.lift_weight * lift_score
        + ARGS.success_bonus * valid_success.to(torch.float32)
        - ARGS.collision_penalty * collision_seen.to(torch.float32)
        - ARGS.prior_penalty * base.square().mean(dim=-1)
    )
    return {
        "fitness": fitness,
        "latent": latent,
        "decoded": decoded,
        "object_frame_rpy": object_frame_rpy,
        "object_pose": object_poses,
        "world_targets": world_targets,
        "hand_targets": hand_targets,
        "max_lift": max_lift,
        "final_lift": final_lift,
        "lift_score": lift_score,
        "success_seen": success_seen,
        "lifted": lifted,
        "collision": collision_seen,
        "arm_joint_motion": arm_joint_motion,
        "arm_target_motion": arm_target_motion,
        "arm_target_error": arm_target_error,
        "eef_target_lift": eef_target_lift,
        "eef_actual_lift": eef_actual_lift,
        "valid_success": valid_success,
    }


def _render_viewer(env) -> bool:
    """Render one frame and return whether the viewer is still open."""
    gym = env.gym
    if env.viewer is None or gym.query_viewer_has_closed(env.viewer):
        return False
    for event in gym.query_viewer_action_events(env.viewer):
        if event.action == "QUIT" and event.value > 0:
            return False
        # Some Isaac Gym builds already bind SPACE to record_frames and emit
        # that action name even when a second SPACE subscription is added.
        if event.action in {"replay_trajectory", "record_frames", "replay"} and event.value > 0.5:
            env.replay_trajectory_requested = True
        if event.action == "toggle_viewer_sync" and event.value > 0:
            env.enable_viewer_sync = not env.enable_viewer_sync
    if env.device != "cpu":
        gym.fetch_results(env.sim, True)
    if env.enable_viewer_sync:
        gym.step_graphics(env.sim)
        gym.draw_viewer(env.viewer, env.sim, True)
        gym.sync_frame_time(env.sim)
    else:
        gym.poll_viewer_events(env.viewer)
    return not gym.query_viewer_has_closed(env.viewer)


def _replay_best_trajectory(
    env,
    decoded,
    object_pose,
    rotation6d_to_matrix,
    gymapi,
    eef_offset,
    object_frame_rpy,
    prepared_world_targets=None,
    prepared_hand_targets=None,
    reset_object_pose=None,
):
    if env.viewer is None:
        raise RuntimeError("The replay environment was created without a viewer")

    gym = env.gym
    gym.subscribe_viewer_keyboard_event(
        env.viewer, gymapi.KEY_SPACE, "replay"
    )
    env.replay_trajectory_requested = False
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    decoded = decoded.to(env.device)
    object_pose = object_pose.to(env.device)
    reset_object_pose = object_pose if reset_object_pose is None else reset_object_pose.to(env.device)

    if prepared_world_targets is None or prepared_hand_targets is None:
        world_targets, hand_targets = _action_targets(
            object_pose,
            decoded.unsqueeze(0),
            rotation6d_to_matrix,
            object_frame_rpy,
        )
        world_targets = _apply_eef_offset(world_targets, eef_offset)
        world_targets, hand_targets = _append_world_post_lift(world_targets, hand_targets)
        world_targets = world_targets[0]
        hand_targets = hand_targets[0]
    else:
        world_targets = prepared_world_targets.to(env.device)
        hand_targets = prepared_hand_targets.to(env.device)

    object_xyz = object_pose[:3].detach().cpu().tolist()
    camera_position = gymapi.Vec3(
        object_xyz[0] + 0.65,
        object_xyz[1] - 0.65,
        object_xyz[2] + 0.40,
    )
    camera_target = gymapi.Vec3(*object_xyz)
    gym.viewer_camera_look_at(
        env.viewer, env.envs[0], camera_position, camera_target
    )
    print(
        "Best trajectory replay started. Focus the Isaac Gym viewer and press SPACE to replay; "
        "close the Isaac Gym viewer to exit."
    )
    print(
        "Replay object pose: "
        f"target={object_pose.detach().cpu().tolist()}, "
        f"reset={reset_object_pose.detach().cpu().tolist()}"
    )
    if world_targets.shape[0] > 1:
        trajectory_end = max(0, world_targets.shape[0] - ARGS.post_lift_steps - 1)
        print(
            "Replay target Z: "
            f"start={float(world_targets[0, 2]):.4f}, "
            f"decoded_end={float(world_targets[trajectory_end, 2]):.4f}, "
            f"lift_end={float(world_targets[-1, 2]):.4f}, "
            f"lift_delta={float(world_targets[-1, 2] - world_targets[trajectory_end, 2]):.4f} m"
        )

    with torch.no_grad():
        final_action = env.no_op_action.clone()
        final_action[:, :7] = world_targets[-1].unsqueeze(0)
        final_action[:, env.hand_dof_start_idx:] = _hand_action(
            env, hand_targets[-1].unsqueeze(0)
        )
        while True:
            if env.viewer is None or gym.query_viewer_has_closed(env.viewer):
                return
            env.reset_idx(
                env_ids, object_init_pose=reset_object_pose.detach().cpu().numpy()
            )
            env.replay_trajectory_requested = False
            actual_reset_pose = env.object_init_states[0, :7].detach()
            reset_pose_error = (actual_reset_pose - reset_object_pose).abs().max()
            print(
                "Replay actual object pose: "
                f"{actual_reset_pose.cpu().tolist()}, "
                f"max_abs_error_vs_requested={float(reset_pose_error):.6f}"
            )
            if float(reset_pose_error) > 1e-4:
                # A second settling simulation can move or rotate an object
                # slightly. Rebuild the object-relative targets in the pose
                # that is actually present before sending the first action.
                world_targets, hand_targets = _action_targets(
                    actual_reset_pose,
                    decoded.unsqueeze(0),
                    rotation6d_to_matrix,
                    object_frame_rpy,
                )
                world_targets = _apply_eef_offset(world_targets, eef_offset)
                world_targets, hand_targets = _append_world_post_lift(
                    world_targets, hand_targets
                )
                world_targets = world_targets[0]
                hand_targets = hand_targets[0]
                final_action[:, :7] = world_targets[-1].unsqueeze(0)
                final_action[:, env.hand_dof_start_idx:] = _hand_action(
                    env, hand_targets[-1].unsqueeze(0)
                )
                print("Replay targets rebuilt from the actual post-reset object pose.")
            initial_object_z = env.object_init_states[0, 2].clone()
            restart = False
            lift_start = max(0, world_targets.shape[0] - ARGS.post_lift_steps)
            lift_end = world_targets.shape[0] - 1
            diagnostic_steps = {decoded.shape[0] - 1, lift_start, lift_end}
            for step in range(world_targets.shape[0]):
                if not _render_viewer(env):
                    return
                if env.replay_trajectory_requested:
                    print("SPACE detected; restarting best trajectory.")
                    restart = True
                    break
                action = env.no_op_action.clone()
                action[:, :7] = world_targets[step].unsqueeze(0)
                action[:, env.hand_dof_start_idx:] = _hand_action(
                    env, hand_targets[step].unsqueeze(0)
                )
                env.step(action)
                if step in diagnostic_steps:
                    if getattr(env, "hand_specific_cfg", {}).get("dualArmInspire", False):
                        eef_indices = env.eef_idx_left if bool(env.active_arm_ids[0]) else env.eef_idx_right
                    else:
                        eef_indices = env.eef_idx
                    actual_pose = env.rigid_body_states.view(-1, 13)[eef_indices[0], :7]
                    target_pose = world_targets[step]
                    position_error = target_pose[:3] - actual_pose[:3]
                    quaternion_dot = torch.abs(
                        torch.sum(target_pose[3:7] * actual_pose[3:7])
                    ).clamp(0.0, 1.0)
                    orientation_error = (
                        2.0 * torch.acos(quaternion_dot) * (180.0 / math.pi)
                    )
                    object_z = env.root_state_tensor[env.object_indices[0], 2]
                    if step == decoded.shape[0] - 1:
                        phase = "decoded_end"
                    elif step == lift_start:
                        phase = "lift_start"
                    else:
                        phase = "lift_end"
                    print(
                        f"Replay {phase} step {step + 1}/{world_targets.shape[0]}: "
                        f"target_xyz={target_pose[:3].detach().cpu().tolist()}, "
                        f"actual_xyz={actual_pose[:3].detach().cpu().tolist()}, "
                        f"position_error={position_error.detach().cpu().tolist()}, "
                        f"orientation_error={float(orientation_error):.2f} deg, "
                        f"object_lift={float(object_z - initial_object_z):.4f} m"
                    )
            if restart:
                continue

            while _render_viewer(env):
                if env.replay_trajectory_requested:
                    print("SPACE detected; restarting best trajectory.")
                    env.replay_trajectory_requested = False
                    break
                env.step(final_action)


def _validate_args():
    if ARGS.population_size < 1:
        raise ValueError("--population-size must be positive")
    if ARGS.iterations < 1:
        raise ValueError("--iterations must be positive")
    if ARGS.twostage_iterations < 0:
        raise ValueError("--twostage_iterations must be non-negative")
    if not 0.0 < ARGS.elite_fraction <= 1.0:
        raise ValueError("--elite-fraction must be in (0, 1]")
    if ARGS.initial_std <= 0 or ARGS.min_std <= 0 or ARGS.max_std < ARGS.min_std:
        raise ValueError("invalid CEM standard-deviation bounds")
    if not 0.0 < ARGS.update_rate <= 1.0:
        raise ValueError("--update-rate must be in (0, 1]")
    if ARGS.action_noise_std < 0.0:
        raise ValueError("--action-noise-std must be non-negative")
    if not 0.0 <= ARGS.action_noise_fraction <= 1.0:
        raise ValueError("--action-noise-fraction must be in [0, 1]")
    if ARGS.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative")
    if ARGS.post_lift_steps < 0:
        raise ValueError("--post-lift-steps must be non-negative")
    if ARGS.post_lift_height < 0:
        raise ValueError("--post-lift-height must be non-negative")


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    from isaacgym import gymapi  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_seed
    from isaacgymenvs.tasks.base import vec_task
    import tasks  # noqa: F401

    global torch
    import torch
    from memory.train_latent_flow import _make_flow
    from memory.train_trajectory_ae import _build_model
    from memory.train_trajectory_vqvae import rotation6d_to_matrix

    _validate_args()
    label, object_pose_np = _load_object_pose()
    object_reset_pose_np = object_pose_np.copy()
    randomize_object_pose = not ARGS.no_randomize_object_pose
    if ARGS.object_position is not None or ARGS.object_quaternion is not None:
        # An explicit pose override is intended to be reproducible.
        randomize_object_pose = False
    ae_checkpoint = torch.load(ARGS.ae_checkpoint, map_location="cpu")
    flow_checkpoint = torch.load(ARGS.flow_checkpoint, map_location="cpu")
    sequence_length = int(ae_checkpoint["config"]["sequence_length"])
    stage2_ae_checkpoint = None
    stage2_flow_checkpoint = None
    if ARGS.twostage_iterations > 0:
        if not ARGS.stage2_ae_checkpoint or not ARGS.stage2_flow_checkpoint:
            raise ValueError(
                "Stage 2 requires --stage2-ae-checkpoint and --stage2-flow-checkpoint"
            )
        stage2_ae_checkpoint = torch.load(ARGS.stage2_ae_checkpoint, map_location="cpu")
        stage2_flow_checkpoint = torch.load(ARGS.stage2_flow_checkpoint, map_location="cpu")
        stage2_sequence_length = int(stage2_ae_checkpoint["config"]["sequence_length"])
        if sequence_length != 20 or stage2_sequence_length != 10:
            raise ValueError(
                "Stage 2 suffix search requires a 20-step stage-1 AE and a 10-step stage-2 AE; "
                f"got {sequence_length} and {stage2_sequence_length}"
            )
        flow_sequence_length = stage2_flow_checkpoint["config"].get("sequence_length")
        if flow_sequence_length is not None and int(flow_sequence_length) != stage2_sequence_length:
            raise ValueError(
                "Stage-2 Flow and AE sequence lengths do not match: "
                f"{flow_sequence_length} vs {stage2_sequence_length}"
            )
        if int(stage2_ae_checkpoint["config"]["input_dim"]) != int(
            ae_checkpoint["config"]["input_dim"]
        ):
            raise ValueError("Stage-2 AE input_dim must match the stage-1 AE input_dim")
        stage1_feature_order = ae_checkpoint["config"].get("feature_order")
        stage2_feature_order = stage2_ae_checkpoint["config"].get("feature_order")
        if (
            stage1_feature_order is not None
            and stage2_feature_order is not None
            and stage1_feature_order != stage2_feature_order
        ):
            raise ValueError("Stage-2 AE feature_order must match the stage-1 AE feature_order")
        if stage2_ae_checkpoint["config"].get("coordinate_frame", "initial_object") != ae_checkpoint[
            "config"
        ].get("coordinate_frame", "initial_object"):
            raise ValueError("Stage-2 AE coordinate_frame must match the stage-1 AE coordinate_frame")
    flow_condition = _load_flow_condition(
        ARGS.dataset,
        ARGS.object_index,
        [
            flow_checkpoint["config"],
            None if stage2_flow_checkpoint is None else stage2_flow_checkpoint["config"],
        ],
        ARGS.pointcloud_ae_checkpoint,
    )
    if cfg.task.env.armController != "pose":
        raise ValueError("Latent trajectory search requires task.env.armController=pose")
    _configure_object(cfg, label)
    cfg.num_envs = ARGS.population_size
    cfg.task.env.numEnvs = ARGS.population_size
    cfg.task.env.episodeLength = (
        sequence_length + ARGS.post_lift_steps + ARGS.hold_steps + 1
    )
    cfg.task.env.randomizeTrackingReference = False
    cfg.task.env.randomizeGraspPose = False
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.force_render = False
    # The search itself stays headless.  A separate one-environment viewer is
    # created after CEM finishes so only the best trajectory is shown.
    cfg.headless = True
    if ARGS.estimate_xyzrpy:
        cfg.task.env.render.enable = True
        cfg.task.env.render.data_type = "rgb_depth_seg"
        cfg.task.env.render.randomize = False
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    def make_search_env(env_count=ARGS.population_size):
        env_cfg = copy.deepcopy(cfg)
        env_cfg.num_envs = env_count
        env_cfg.task.env.numEnvs = env_count
        return isaacgymenvs.make(
            env_cfg.seed, env_cfg.task_name, env_count,
            env_cfg.sim_device, env_cfg.rl_device, env_cfg.graphics_device_id,
            env_cfg.headless, env_cfg.multi_gpu, False, env_cfg.force_render,
            env_cfg,
        )

    dual_arm = bool(cfg.task.hand_config.get(
        "dualArm", cfg.task.hand_config.get("dualArmInspire", False)
    ))
    # Search only the right arm. Keep the environment's dual-arm model, but
    # fix every population member to activeArm=1 instead of searching a side.
    if dual_arm:
        cfg.task.env.activeArm = 1
    if ARGS.estimate_xyzrpy:
        estimate_cfg = copy.deepcopy(cfg)
        estimate_cfg.num_envs = 1
        estimate_cfg.task.env.numEnvs = 1
        estimate_cfg.task.env.render.enable = True
        estimate_cfg.task.env.render.camera_ids = [1]
        estimate_cfg.task.env.render.data_type = "rgb,depth,seg"
        estimate_cfg.task.env.render.randomize = False
        estimate_cfg.task.env.enableCameraSensors = True
        estimate_cfg.task.env.enablePointCloud = True
        # The task's camera_1 is fixed by default. Use the same head-mounted
        # pose as the RGB-D collector so the camera handle attaches to the robot.
        estimate_cfg.task.env.camera_config.camera_1.mount = "head_link"
        estimate_cfg.task.env.camera_config.camera_1.position = [0.1, 0.0, 0.45]
        estimate_cfg.task.env.camera_config.camera_1.quaternion = [
            0.0, 0.38268343, 0.0, 0.92387953
        ]
        estimate_env = isaacgymenvs.make(
            estimate_cfg.seed,
            estimate_cfg.task_name,
            1,
            estimate_cfg.sim_device,
            estimate_cfg.rl_device,
            estimate_cfg.graphics_device_id,
            estimate_cfg.headless,
            estimate_cfg.multi_gpu,
            False,
            estimate_cfg.force_render,
            estimate_cfg,
        )
        estimate_device = torch.device(estimate_env.device)
        estimate_ids = torch.zeros(1, dtype=torch.long, device=estimate_env.device)
        if randomize_object_pose:
            estimate_env.reset_idx(estimate_ids)
        else:
            estimate_env.reset_idx(estimate_ids, object_init_pose=object_pose_np)
        object_pose_np = _estimate_object_pose_from_camera(estimate_env, 0, object_pose_np)
        object_reset_pose_np = (
            estimate_env.object_init_states[0, :7].detach().cpu().numpy().copy()
        )
        estimate_gym, estimate_sim = estimate_env.gym, estimate_env.sim
        if getattr(estimate_env, "viewer", None) is not None:
            estimate_gym.destroy_viewer(estimate_env.viewer)
        if str(estimate_env.device).startswith("cuda"):
            torch.cuda.synchronize(estimate_device)
        estimate_gym.destroy_sim(estimate_sim)
        del estimate_env
        import gc
        gc.collect()
        vec_task.EXISTING_SIM = None
        randomize_object_pose = False
        print("Estimated object pose from initial masked depth: {}".format(
            np.array2string(object_pose_np, precision=5)))
        print("Simulator object-root pose used for reset: {}".format(
            np.array2string(object_reset_pose_np, precision=5)))
        cfg.task.env.render.enable = False
    env = make_search_env()
    device = torch.device(env.device)
    if flow_condition is not None:
        flow_condition = flow_condition.to(device)
    flow = _make_flow(flow_checkpoint["config"], device)
    flow.load_state_dict(flow_checkpoint["model"])
    flow.eval()
    ae = _build_model(ae_checkpoint["config"]).to(device)
    ae.load_state_dict(ae_checkpoint["model"])
    ae.eval()
    latent_shape = tuple(flow_checkpoint["config"]["latent_shape"])
    stage2_flow = None
    stage2_ae = None
    stage2_latent_shape = None
    stage2_latent_dim = None
    if stage2_ae_checkpoint is not None and stage2_flow_checkpoint is not None:
        stage2_flow = _make_flow(stage2_flow_checkpoint["config"], device)
        stage2_flow.load_state_dict(stage2_flow_checkpoint["model"])
        stage2_flow.eval()
        stage2_ae = _build_model(stage2_ae_checkpoint["config"]).to(device)
        stage2_ae.load_state_dict(stage2_ae_checkpoint["model"])
        stage2_ae.eval()
        stage2_latent_shape = tuple(stage2_flow_checkpoint["config"]["latent_shape"])
        stage2_latent_dim = int(stage2_flow_checkpoint["config"]["latent_dim"])
        expected_stage2_shape = (
            int(stage2_ae_checkpoint["config"]["num_tokens"]),
            int(stage2_ae_checkpoint["config"]["latent_dim"]),
        )
        if stage2_latent_shape != expected_stage2_shape:
            raise ValueError(
                "Stage-2 Flow latent_shape does not match the stage-2 AE: "
                f"{stage2_latent_shape} vs {expected_stage2_shape}"
            )
        if stage2_latent_dim != int(np.prod(stage2_latent_shape)):
            raise ValueError(
                "Stage-2 Flow latent_dim does not match latent_shape: "
                f"{stage2_latent_dim} vs {stage2_latent_shape}"
            )
    object_pose = torch.as_tensor(object_pose_np, dtype=torch.float32, device=device)
    trajectory_eef_offset = cfg.task.hand_config.get("trajectoryEefOffset", None)
    if dual_arm:
        offsets = cfg.task.hand_config.trajectoryEefOffsetByArm
        trajectory_eef_offset_left = torch.as_tensor(offsets.left, dtype=torch.float32, device=device)
        trajectory_eef_offset_right = torch.as_tensor(offsets.right, dtype=torch.float32, device=device)
    else:
        trajectory_eef_offset_left = None if trajectory_eef_offset is None else torch.as_tensor(
            trajectory_eef_offset, dtype=torch.float32, device=device
        )
        trajectory_eef_offset_right = None
    for offset in (trajectory_eef_offset_left, trajectory_eef_offset_right):
        if offset is not None and offset.shape != (7,):
            raise ValueError("trajectoryEefOffset must contain 7 values")
    if randomize_object_pose:
        print(
            "Object reset: randomized by task.env.resetPositionRange and "
            "task.env.resetRandomRot"
        )
    else:
        print("Object reset: fixed to the dataset/command-line pose")
    latent_dim = int(flow_checkpoint["config"]["latent_dim"])
    # Search only the learned latent variables. Arm selection and object-frame
    # rotations are fixed (right arm, zero rotation).
    search_dim = latent_dim
    mean = torch.zeros(search_dim, dtype=torch.float32, device=device)
    std = torch.full_like(mean, float(ARGS.initial_std))
    best = None
    stage_one_success = False
    elite_count = max(1, int(np.ceil(ARGS.population_size * ARGS.elite_fraction)))
    exploration_count = min(
        max(0, ARGS.population_size - 1),
        int(round(ARGS.population_size * ARGS.action_noise_fraction)),
    )
    if (
        ARGS.action_noise_std > 0.0
        and ARGS.action_noise_fraction > 0.0
        and ARGS.population_size > 1
    ):
        exploration_count = max(1, exploration_count)
    parent_base = None
    parent_latent = None
    parent_decoded = None
    parent_fitness = float("-inf")

    for iteration in range(1, ARGS.iterations + 1):
        base = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            ARGS.population_size, search_dim, device=device
        )
        base = base.clamp(-ARGS.max_base_abs, ARGS.max_base_abs)
        base[0] = (
            mean if parent_base is None else parent_base
        ).clamp(-ARGS.max_base_abs, ARGS.max_base_abs)
        latent, decoded, _ = _decode_candidates(
            flow, ae, base, latent_shape, latent_dim, flow_condition
        )
        candidate_base = base.clone()
        candidate_latent = latent.clone()
        candidate_decoded = decoded.clone()
        # Keep the current parent exactly in the population.  For a noisy
        # parent this avoids losing the good action-space sample to the
        # AE->Flow->AE projection used by the other candidates.
        if parent_decoded is not None:
            candidate_decoded[0] = parent_decoded
            candidate_latent[0] = parent_latent
        if exploration_count > 0:
            exploration_parent = (
                parent_decoded if parent_decoded is not None else decoded[0]
            )
            noisy_trajectories = _perturb_trajectory(
                exploration_parent, exploration_count
            )
            noisy_latent, noisy_base = _encode_candidates(
                flow, ae, noisy_trajectories, flow_condition
            )
            candidate_base[-exploration_count:] = noisy_base.clamp(
                -ARGS.max_base_abs, ARGS.max_base_abs
            )
            candidate_latent[-exploration_count:] = noisy_latent
            candidate_decoded[-exploration_count:] = noisy_trajectories
        use_left = None
        if dual_arm:
            env.active_arm_ids.fill_(False)
            trajectory_eef_offset_batch = trajectory_eef_offset_right
        else:
            trajectory_eef_offset_batch = trajectory_eef_offset_left
        metrics = _evaluate_population(
            env,
            flow,
            ae,
            candidate_base,
            latent_shape,
            object_pose,
            rotation6d_to_matrix,
            randomize_object_pose,
            trajectory_eef_offset_batch,
            latent_dim,
            reset_object_pose=torch.as_tensor(
                object_reset_pose_np, dtype=torch.float32, device=device
            ),
            decoded_override=candidate_decoded,
            latent_override=candidate_latent,
            condition=flow_condition,
        )
        fitness = metrics["fitness"]
        found_success = bool(metrics["valid_success"].any().item())
        stage_one_success |= found_success
        elite_indices = torch.topk(fitness, elite_count, largest=True).indices
        elite_base = candidate_base[elite_indices]
        new_mean = elite_base.mean(dim=0)
        new_std = elite_base.std(dim=0, unbiased=False)
        mean = (1.0 - ARGS.update_rate) * mean + ARGS.update_rate * new_mean
        std = (1.0 - ARGS.update_rate) * std + ARGS.update_rate * new_std
        std = std.clamp(ARGS.min_std, ARGS.max_std)

        top_index = _select_candidate_index(fitness, metrics["valid_success"])
        top_fitness = float(fitness[top_index].detach().cpu())
        if top_fitness >= parent_fitness:
            parent_fitness = top_fitness
            parent_base = candidate_base[top_index].detach().clone()
            parent_latent = candidate_latent[top_index].detach().clone()
            parent_decoded = candidate_decoded[top_index].detach().clone()
        if (
            best is None
            or top_fitness > best["fitness"]
            or bool(metrics["valid_success"][top_index])
        ):
            best = {
                "fitness": top_fitness,
                "base": candidate_base[top_index].detach().clone(),
                "latent": candidate_latent[top_index].detach().clone(),
                "decoded": candidate_decoded[top_index].detach().clone(),
                "object_frame_rpy": metrics["object_frame_rpy"][top_index].detach().clone(),
                "object_pose": metrics["object_pose"][top_index].detach().clone(),
                "world_targets": metrics["world_targets"][top_index].detach().clone(),
                "hand_targets": metrics["hand_targets"][top_index].detach().clone(),
                "max_lift": float(metrics["max_lift"][top_index].detach().cpu()),
                "final_lift": float(metrics["final_lift"][top_index].detach().cpu()),
                "lift_score": float(metrics["lift_score"][top_index].detach().cpu()),
                "success": bool(metrics["valid_success"][top_index].detach().cpu()),
                "collision": bool(metrics["collision"][top_index].detach().cpu()),
                "eef_target_lift": float(metrics["eef_target_lift"][top_index].detach().cpu()),
                "eef_actual_lift": float(metrics["eef_actual_lift"][top_index].detach().cpu()),
                "iteration": iteration,
                "stage": 1,
                "active_arm": 1 if dual_arm else 0,
            }
        print(
            f"iter {iteration:03d}/{ARGS.iterations}: "
            f"best={top_fitness:.4f} global={best['fitness']:.4f} "
            f"arm={'R' if dual_arm else 'single'} "
            f"peak_lift={float(metrics['max_lift'][top_index]):.4f} "
            f"final_lift={float(metrics['final_lift'][top_index]):.4f} "
            f"success={bool(metrics['valid_success'][top_index])} "
            f"collision={bool(metrics['collision'][top_index])} "
            f"arm_target_delta={float(metrics['arm_target_motion'][top_index]):.4f}rad "
            f"arm_motion={float(metrics['arm_joint_motion'][top_index]):.4f}rad "
            f"arm_target_err={float(metrics['arm_target_error'][top_index]):.4f}rad "
            f"eef_lift_target={float(metrics['eef_target_lift'][top_index]):.4f}m "
            f"eef_lift_actual={float(metrics['eef_actual_lift'][top_index]):.4f}m "
            f"std={float(std.mean()):.4f} "
            f"explore={exploration_count}"
        )
        if found_success:
            print("Stage 1 found a successful trajectory; stopping search early.")
            break

    if best is None:
        raise RuntimeError("CEM did not produce a candidate")
    stage_two_mean = None
    stage_two_std = None
    stage_two_seed_base = None
    stage_two_seed_latent = None
    if ARGS.twostage_iterations > 0 and not stage_one_success:
        if stage2_flow is None or stage2_ae is None:
            raise RuntimeError("Stage-2 AE/Flow checkpoints were not loaded")
        print(
            "Stage 1 found no successful trajectory; starting a stage-2 latent "
            f"suffix search for {ARGS.twostage_iterations} iterations."
        )
        suffix_steps = int(stage2_ae_checkpoint["config"]["sequence_length"])
        prefix_steps = best["decoded"].shape[0] - suffix_steps
        if prefix_steps < 1:
            raise ValueError(
                f"Stage-2 suffix length {suffix_steps} must be shorter than the "
                f"stage-1 trajectory length {best['decoded'].shape[0]}"
            )
        stage_two_prefix = best["decoded"][:prefix_steps].detach().clone()
        stage_two_seed = best["decoded"][-suffix_steps:].detach().clone()
        stage_two_object_pose = best["object_pose"].detach().clone()
        with torch.no_grad():
            stage_two_seed_latent_tensor = stage2_ae.encode(stage_two_seed.unsqueeze(0))
            stage_two_seed_base_tensor, _ = stage2_flow(
                stage_two_seed_latent_tensor.reshape(1, -1), flow_condition
            )
        stage_two_seed_latent = stage_two_seed_latent_tensor[0].detach().clone()
        stage_two_seed_base = stage_two_seed_base_tensor[0].detach().clone()
        stage_two_dim = stage2_latent_dim
        stage_mean = stage_two_seed_base.clone()
        stage_std = torch.full_like(stage_mean, float(ARGS.initial_std))
        stage_elite_count = elite_count
        stage_parent_base = stage_two_seed_base.detach().clone()
        stage_parent_latent = stage_two_seed_latent.detach().clone()
        stage_parent_suffix = stage_two_seed.detach().clone()
        stage_parent_fitness = float("-inf")
        if dual_arm:
            env.active_arm_ids.fill_(False)
            stage_eef_offset = trajectory_eef_offset_right
        else:
            stage_eef_offset = trajectory_eef_offset_left

        for iteration in range(1, ARGS.twostage_iterations + 1):
            normalized = stage_mean.unsqueeze(0) + stage_std.unsqueeze(0) * torch.randn(
                ARGS.population_size, stage_two_dim, device=device
            )
            normalized = normalized.clamp(-ARGS.max_base_abs, ARGS.max_base_abs)
            normalized[0] = stage_parent_base.clamp(
                -ARGS.max_base_abs, ARGS.max_base_abs
            )
            with torch.no_grad():
                stage_two_latent, stage_two_suffix, _ = _decode_candidates(
                    stage2_flow,
                    stage2_ae,
                    normalized,
                    stage2_latent_shape,
                    stage2_latent_dim,
                    flow_condition,
                )
            stage_two_candidate_base = normalized.clone()
            stage_two_candidate_latent = stage_two_latent.clone()
            stage_two_candidate_suffix = stage_two_suffix.clone()
            stage_two_candidate_suffix[0] = stage_parent_suffix
            stage_two_candidate_latent[0] = stage_parent_latent
            if exploration_count > 0:
                noisy_suffix = _perturb_trajectory(
                    stage_parent_suffix, exploration_count
                )
                noisy_latent, noisy_base = _encode_candidates(
                    stage2_flow, stage2_ae, noisy_suffix, flow_condition
                )
                stage_two_candidate_base[-exploration_count:] = noisy_base.clamp(
                    -ARGS.max_base_abs, ARGS.max_base_abs
                )
                stage_two_candidate_latent[-exploration_count:] = noisy_latent
                stage_two_candidate_suffix[-exploration_count:] = noisy_suffix
            prefix = stage_two_prefix.unsqueeze(0).expand(
                ARGS.population_size, -1, -1
            )
            decoded_candidates = torch.cat((prefix, stage_two_candidate_suffix), dim=1)
            stage_two_latent_flat = stage_two_candidate_latent.reshape(
                ARGS.population_size, -1
            )
            metrics = _evaluate_population(
                env, flow, ae, stage_two_candidate_base, latent_shape, stage_two_object_pose,
                rotation6d_to_matrix, False, stage_eef_offset, latent_dim,
                reset_object_pose=torch.as_tensor(
                    stage_two_object_pose, dtype=torch.float32, device=device
                ),
                decoded_override=decoded_candidates,
                latent_override=stage_two_latent_flat,
                condition=flow_condition,
            )
            fitness = metrics["fitness"]
            found_success = bool(metrics["valid_success"].any().item())
            elites = torch.topk(fitness, stage_elite_count, largest=True).indices
            new_mean = stage_two_candidate_base[elites].mean(dim=0)
            new_std = stage_two_candidate_base[elites].std(dim=0, unbiased=False)
            stage_mean = (1.0 - ARGS.update_rate) * stage_mean + ARGS.update_rate * new_mean
            stage_std = (1.0 - ARGS.update_rate) * stage_std + ARGS.update_rate * new_std
            stage_std = stage_std.clamp(ARGS.min_std, ARGS.max_std)
            stage_two_mean = stage_mean.detach().clone()
            stage_two_std = stage_std.detach().clone()

            top_index = _select_candidate_index(fitness, metrics["valid_success"])
            top_fitness = float(fitness[top_index].detach().cpu())
            if top_fitness >= stage_parent_fitness:
                stage_parent_fitness = top_fitness
                stage_parent_base = stage_two_candidate_base[top_index].detach().clone()
                stage_parent_latent = stage_two_candidate_latent[top_index].detach().clone()
                stage_parent_suffix = stage_two_candidate_suffix[top_index].detach().clone()
            if top_fitness > best["fitness"] or bool(metrics["valid_success"][top_index]):
                refined_trajectory = metrics["decoded"][top_index]
                with torch.no_grad():
                    refined_latent = ae.encode(refined_trajectory.unsqueeze(0))[0]
                    refined_base, _ = flow(
                        refined_latent.reshape(1, -1), flow_condition
                    )
                best.update({
                    "fitness": top_fitness,
                    "base": refined_base[0].detach().clone(),
                    "latent": refined_latent.detach().clone(),
                    "decoded": refined_trajectory.detach().clone(),
                    "object_frame_rpy": metrics["object_frame_rpy"][top_index].detach().clone(),
                    "object_pose": metrics["object_pose"][top_index].detach().clone(),
                    "world_targets": metrics["world_targets"][top_index].detach().clone(),
                    "hand_targets": metrics["hand_targets"][top_index].detach().clone(),
                    "max_lift": float(metrics["max_lift"][top_index].detach().cpu()),
                    "final_lift": float(metrics["final_lift"][top_index].detach().cpu()),
                    "lift_score": float(metrics["lift_score"][top_index].detach().cpu()),
                    "success": bool(metrics["valid_success"][top_index].detach().cpu()),
                    "collision": bool(metrics["collision"][top_index].detach().cpu()),
                    "eef_target_lift": float(metrics["eef_target_lift"][top_index].detach().cpu()),
                    "eef_actual_lift": float(metrics["eef_actual_lift"][top_index].detach().cpu()),
                    "iteration": ARGS.iterations + iteration,
                    "stage": 2,
                    "active_arm": 1 if dual_arm else 0,
                    "stage_two_base": stage_two_candidate_base[top_index].detach().clone(),
                    "stage_two_latent": stage_two_candidate_latent[top_index].detach().clone(),
                })
            print(
                f"stage2 iter {iteration:03d}/{ARGS.twostage_iterations}: "
                f"best={top_fitness:.4f} global={best['fitness']:.4f} "
                f"success={bool(metrics['valid_success'][top_index])} "
                f"collision={bool(metrics['collision'][top_index])} "
                f"peak_lift={float(metrics['max_lift'][top_index]):.4f} "
                f"final_lift={float(metrics['final_lift'][top_index]):.4f} "
                f"std={float(stage_std.mean()):.4f} "
                f"explore={exploration_count}"
            )
            if found_success:
                print("Stage 2 found a successful trajectory; stopping search early.")
                break

    print(
        "Best solution: "
        f"fitness={best['fitness']:.4f}, success={best['success']}, "
        f"collision={best['collision']}, peak_lift={best['max_lift']:.4f}, "
        f"final_lift={best['final_lift']:.4f}, iteration={best['iteration']}, "
        f"stage={best['stage']}, "
        f"active_arm={best['active_arm']}, "
        f"eef_lift_target={best['eef_target_lift']:.4f}m, "
        f"eef_lift_actual={best['eef_actual_lift']:.4f}m"
    )
    print("Object-frame RPY search disabled; using [0.0, 0.0, 0.0] degrees.")
    print(
        f"Post-lift: +Z {ARGS.post_lift_height:.3f} m over "
        f"{ARGS.post_lift_steps} steps; requested/actual target delta="
        f"{ARGS.post_lift_height:.3f}/"
        f"{float(best['world_targets'][-1, 2] - best['world_targets'][-ARGS.post_lift_steps - 1, 2]) if ARGS.post_lift_steps else 0.0:.3f} m; "
        f"replay length={best['world_targets'].shape[0]}"
    )
    if ARGS.output:
        output = Path(ARGS.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            object_label=np.asarray(label),
            object_index=np.asarray(ARGS.object_index, dtype=np.int64),
            pointcloud_condition=(
                flow_condition[0].cpu().numpy()
                if flow_condition is not None
                else np.zeros(0, dtype=np.float32)
            ),
            object_pose_world=best["object_pose"].cpu().numpy(),
            base_u=best["base"][:latent_dim].cpu().numpy(),
            rotation_search_variables=best["base"][latent_dim:].cpu().numpy(),
            stage_two_base_u=best.get(
                "stage_two_base",
                torch.zeros(
                    int(stage2_latent_dim or 0),
                    device=device,
                ),
            ).cpu().numpy(),
            stage_two_latent_z=best.get(
                "stage_two_latent",
                torch.zeros(
                    *(stage2_latent_shape or (0,)),
                    device=device,
                ),
            ).cpu().numpy(),
            stage_two_seed_base_u=(
                stage_two_seed_base.cpu().numpy()
                if stage_two_seed_base is not None
                else np.zeros(int(stage2_latent_dim or 0), dtype=np.float32)
            ),
            stage_two_seed_latent_z=(
                stage_two_seed_latent.cpu().numpy()
                if stage_two_seed_latent is not None
                else np.zeros(stage2_latent_shape or (0,), dtype=np.float32)
            ),
            active_arm=np.asarray(best["active_arm"], dtype=np.int64),
            latent_z=best["latent"].cpu().numpy(),
            object_frame_rpy=best["object_frame_rpy"].cpu().numpy(),
            object_frame_rpy_degrees=torch.rad2deg(
                best["object_frame_rpy"]
            ).cpu().numpy(),
            object_frame_z_rotation_degrees=np.asarray(0.0, dtype=np.float32),
            trajectory=best["decoded"].cpu().numpy(),
            world_trajectory=best["world_targets"].cpu().numpy(),
            hand_trajectory=best["hand_targets"].cpu().numpy(),
            fitness=np.asarray(best["fitness"], dtype=np.float32),
            max_lift=np.asarray(best["max_lift"], dtype=np.float32),
            final_lift=np.asarray(best["final_lift"], dtype=np.float32),
            lift_score=np.asarray(best["lift_score"], dtype=np.float32),
            success=np.asarray(best["success"], dtype=np.bool_),
            collision=np.asarray(best["collision"], dtype=np.bool_),
            iteration=np.asarray(best["iteration"], dtype=np.int64),
            stage=np.asarray(best["stage"], dtype=np.int64),
            post_lift_height=np.asarray(ARGS.post_lift_height, dtype=np.float32),
            post_lift_steps=np.asarray(ARGS.post_lift_steps, dtype=np.int64),
            search_mean=mean.cpu().numpy(),
            search_std=std.cpu().numpy(),
            stage_two_search_mean=(
                stage_two_mean.cpu().numpy()
                if stage_two_mean is not None
                else np.zeros(int(stage2_latent_dim or 0), dtype=np.float32)
            ),
            stage_two_search_std=(
                stage_two_std.cpu().numpy()
                if stage_two_std is not None
                else np.zeros(int(stage2_latent_dim or 0), dtype=np.float32)
            ),
        )
        print(f"Saved best grasp solution to {output}")

    search_gym = env.gym
    search_sim = env.sim
    if getattr(env, "viewer", None) is not None:
        search_gym.destroy_viewer(env.viewer)
    if str(env.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(env.device))
    search_gym.destroy_sim(search_sim)
    del env
    import gc

    gc.collect()
    vec_task.EXISTING_SIM = None

    cfg.num_envs = 1
    cfg.task.env.numEnvs = 1
    cfg.task.env.episodeLength = max(
        100000,
        sequence_length + ARGS.post_lift_steps + ARGS.hold_steps + 1,
    )
    cfg.headless = False
    replay_env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        1,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        False,
        cfg.multi_gpu,
        False,
        cfg.force_render,
        cfg,
    )
    if dual_arm:
        replay_env.active_arm_ids.fill_(False)
    # The selected candidate already contains the exact pose used to build its
    # prepared world targets. Reuse it for replay in both randomized and fixed
    # pose modes so a second reset cannot move the object-relative trajectory.
    replay_object_pose = best["object_pose"].detach().clone()
    _replay_best_trajectory(
        replay_env,
        best["decoded"],
        best["object_pose"],
        rotation6d_to_matrix,
        gymapi,
        trajectory_eef_offset_left if best["active_arm"] == 0 else trajectory_eef_offset_right,
        best["object_frame_rpy"],
        prepared_world_targets=best["world_targets"],
        prepared_hand_targets=best["hand_targets"],
        reset_object_pose=replay_object_pose,
    )


if __name__ == "__main__":
    main()
