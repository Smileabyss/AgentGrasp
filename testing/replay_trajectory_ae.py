"""Replay a continuous-AE trajectory in an Isaac Gym FR3 + Inspire scene.

The first useful test is reconstruction replay: a dataset trajectory is encoded
by the continuous AutoEncoder, decoded, transformed from the initial object
frame to the current simulation world frame, and sent to the existing ``pose``
controller.  This avoids sampling an arbitrary latent outside the training
distribution.

Example (viewer enabled)::

    python -m testing.replay_trajectory_ae \
        --dataset datasets/inspire_grasp.npz \
        --checkpoint runs/trajectory_ae/best.pt \
        --sample-index 0 \
        task=grasp hand=fr3_inspire_tac num_envs=1 headless=False \
        task.env.armController=pose \
        task.env.episodeLength=20

The optional ``--start-position`` and ``--end-position`` values are expressed
in the object frame, in meters.  They apply an endpoint correction to the
decoded wrist-position trajectory.  This is useful for a visual test, but it
does not make the AutoEncoder endpoint-conditioned; use a conditional model if
the endpoints must be learned rather than post-processed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--active-arm", type=int, choices=(0, 1), default=1,
                        help="G1 side to activate: 0=left, 1=right")
    parser.add_argument("--start-position", type=float, nargs=3, default=None)
    parser.add_argument("--end-position", type=float, nargs=3, default=None)
    parser.add_argument("--output", default="runs/trajectory_ae/replay.npz")
    parser.add_argument("--hold-steps", type=int, default=30)
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args


ARGS = _parse_args()


def _quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_conjugate(q):
    out = q.clone()
    out[..., :3] *= -1
    return out


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
    """Convert rotation matrices to Isaac Gym's xyzw quaternion convention."""

    # This branch-free version is stable for the small rotations encountered
    # in the replay and normalizes the result before it is sent to Isaac Gym.
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-8)) * 0.5
    qx = (m21 - m12) / (4 * qw).clamp_min(1e-8)
    qy = (m02 - m20) / (4 * qw).clamp_min(1e-8)
    qz = (m10 - m01) / (4 * qw).clamp_min(1e-8)
    return _quat_normalize(torch.stack((qx, qy, qz, qw), dim=-1))


def _endpoint_correct(trajectory, start_position, end_position):
    if start_position is None and end_position is None:
        return trajectory
    corrected = trajectory.clone()
    alpha = torch.linspace(0.0, 1.0, corrected.shape[0], device=corrected.device)
    if start_position is not None:
        start = torch.as_tensor(start_position, dtype=corrected.dtype, device=corrected.device)
        corrected[:, :3] += (1.0 - alpha[:, None]) * (start - corrected[0, :3])
    if end_position is not None:
        end = torch.as_tensor(end_position, dtype=corrected.dtype, device=corrected.device)
        corrected[:, :3] += alpha[:, None] * (end - corrected[-1, :3])
    return corrected


def _world_targets(object_pose, trajectory):
    """Transform an object-frame trajectory with the sampled object pose."""
    from memory.train_trajectory_vqvae import rotation6d_to_matrix

    object_pose = object_pose[:7]
    object_position = object_pose[:3]
    object_rotation = _quat_to_matrix(object_pose[3:7])
    wrist_position_world = torch.matmul(trajectory[:, :3], object_rotation.transpose(0, 1)) + object_position
    relative_quaternion = _matrix_to_quaternion(
        rotation6d_to_matrix(trajectory[:, 3:9])
    )
    wrist_quaternion_world = _quat_multiply(object_pose[3:7].expand_as(relative_quaternion), relative_quaternion)
    return torch.cat((wrist_position_world, _quat_normalize(wrist_quaternion_world)), dim=-1)


def _apply_eef_offset(world_targets, eef_offset):
    if eef_offset is None:
        return world_targets
    offset = torch.as_tensor(eef_offset, dtype=world_targets.dtype, device=world_targets.device)
    rotation = _quat_to_matrix(world_targets[..., 3:7])
    position = world_targets[..., :3] + torch.matmul(
        offset[:3].expand(world_targets.shape[:-1] + (3,)).unsqueeze(-2),
        rotation.transpose(-1, -2),
    ).squeeze(-2)
    orientation = _quat_multiply(world_targets[..., 3:7], offset[3:7].expand_as(world_targets[..., 3:7]))
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
    indices = env.active_hand_dof_indices
    if getattr(env, "dual_arm", hand_cfg.get("dualArmInspire", False)):
        indices = env.left_active_hand_dof_indices if bool(env.active_arm_ids[0]) else env.right_active_hand_dof_indices
    lower = env.robot_dof_lower_limits[indices]
    upper = env.robot_dof_upper_limits[indices]
    return 2.0 * (hand_qpos - lower) / (upper - lower).clamp_min(1e-8) - 1.0


def _load_decoded_trajectory(device):
    from memory.train_trajectory_ae import _build_model, _load_data

    trajectories = _load_data(ARGS.dataset)
    if not 0 <= ARGS.sample_index < trajectories.shape[0]:
        raise IndexError(
            f"--sample-index {ARGS.sample_index} is outside [0, {trajectories.shape[0]})"
        )
    with np.load(ARGS.dataset, allow_pickle=False) as loaded:
        sampled_object_pose = torch.from_numpy(
            np.asarray(loaded["object_pose_world"][ARGS.sample_index], dtype=np.float32)
        )
    if sampled_object_pose.shape != (7,):
        raise ValueError("object_pose_world must contain [N, 7] poses")
    checkpoint = torch.load(ARGS.checkpoint, map_location=device)
    model = _build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    source = trajectories[ARGS.sample_index].to(device)
    with torch.no_grad():
        decoded, latent = model(source.unsqueeze(0))
    decoded = _endpoint_correct(decoded[0], ARGS.start_position, ARGS.end_position)
    return source, decoded, latent[0], sampled_object_pose.to(device)


def _dataset_sequence_length() -> int:
    """Read the stored temporal length before constructing the simulator."""
    with np.load(ARGS.dataset, allow_pickle=False) as loaded:
        if "wrist_trajectory" not in loaded:
            raise ValueError("Dataset must contain wrist_trajectory")
        trajectory = np.asarray(loaded["wrist_trajectory"])
    if trajectory.ndim != 3 or trajectory.shape[1] <= 0:
        raise ValueError("wrist_trajectory must have shape [N, T, 7]")
    return int(trajectory.shape[1])


def _configure_sampled_object(cfg: DictConfig) -> None:
    """Make the one-env replay load the same object as the sampled episode."""
    with np.load(ARGS.dataset, allow_pickle=False) as loaded:
        label = str(loaded["object_label"][ARGS.sample_index])
    # Grasp builds paths as <yaml parent>/urdf/<yaml entry>, so keep the
    # original dataset parent and write only the selected URDF basename.
    label_path = Path(label)
    if label_path.suffix != ".urdf":
        raise ValueError(f"Expected an object URDF label, got {label!r}")
    import yaml

    list_path = Path("assets") / label_path.parent.parent / "replay_object.yaml"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(yaml.safe_dump([label_path.name]))
    cfg.task.env.asset.multiObjectList = str(label_path.parent.parent / list_path.name)


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    from isaacgym import gymapi  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import tasks  # noqa: F401

    global torch
    import torch

    set_np_formatting()
    dual_arm = bool(cfg.task.hand_config.get(
        "dualArm", cfg.task.hand_config.get("dualArmInspire", False)
    ))
    if dual_arm:
        cfg.task.env.activeArm = ARGS.active_arm
        side = "left" if ARGS.active_arm == 0 else "right"
        eef_offset = cfg.task.hand_config.trajectoryEefOffsetByArm[side]
    else:
        eef_offset = cfg.task.hand_config.get("trajectoryEefOffset", None)
    cfg.num_envs = 1
    cfg.task.env.numEnvs = 1
    # The replay horizon belongs to the saved trajectory. This also handles
    # datasets resampled to 20 frames when the Hydra config still defaults to 50.
    cfg.task.env.episodeLength = _dataset_sequence_length()
    _configure_sampled_object(cfg)
    cfg.headless = bool(cfg.headless)
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        1,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        False,
        cfg.force_render,
        cfg,
    )

    device = env.device
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    # The sampled object pose defines the object frame used by the dataset.
    # Put the simulator object at that exact pose before replaying the decoded
    # trajectory; otherwise object-frame to world-frame conversion is wrong.
    source, decoded, latent, sampled_object_pose = _load_decoded_trajectory(device)
    env.reset_idx(env_ids, object_init_pose=sampled_object_pose.detach().cpu().numpy())
    if decoded.shape[0] != env.max_episode_length:
        raise ValueError(
            f"AE trajectory has {decoded.shape[0]} steps but environment has "
            f"episodeLength={env.max_episode_length}"
        )
    world_targets = _apply_eef_offset(_world_targets(sampled_object_pose, decoded), eef_offset)
    hand_targets = decoded[:, 9:]

    actual_wrist = []
    actual_hand = []
    for step in range(env.max_episode_length):
        action = torch.zeros((1, env.num_actions), dtype=torch.float32, device=device)
        action[:, :7] = world_targets[step]
        action[:, env.hand_dof_start_idx:] = _hand_action(env, hand_targets[step])
        env.step(action)
        actual_wrist.append(env.rigid_body_states.view(-1, 13)[env.eef_idx[0], :7].clone())
        actual_hand.append(env.robot_dof_pos[0, env.active_hand_dof_indices].clone())

    # Hold the final target so the viewer shows the final grasp configuration.
    for _ in range(max(0, ARGS.hold_steps)):
        action = torch.zeros((1, env.num_actions), dtype=torch.float32, device=device)
        action[:, :7] = world_targets[-1]
        action[:, env.hand_dof_start_idx:] = _hand_action(env, hand_targets[-1])
        env.step(action)

    output = Path(ARGS.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_trajectory=source.cpu().numpy(),
        decoded_trajectory=decoded.cpu().numpy(),
        latent=latent.cpu().numpy(),
        sampled_object_pose_world=sampled_object_pose.cpu().numpy(),
        actual_object_pose_world=env.object_init_states[0, :7].cpu().numpy(),
        target_wrist_world=world_targets.cpu().numpy(),
        actual_wrist_world=torch.stack(actual_wrist).cpu().numpy(),
        target_hand_qpos=hand_targets.cpu().numpy(),
        actual_hand_qpos=torch.stack(actual_hand).cpu().numpy(),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sample_index": ARGS.sample_index,
                "checkpoint": os.path.abspath(ARGS.checkpoint),
                "headless": bool(cfg.headless),
                "episode_length": int(env.max_episode_length),
                "object_frame": "initial object frame",
                "world_frame": "simulation world frame",
                "note": "Replay places the simulator object at the sampled object_pose_world and uses that same pose for object-to-world conversion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved replay targets and actual states to {output}")


if __name__ == "__main__":
    main()
