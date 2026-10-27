from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Deque, Dict

import hydra
from isaacgym import gymapi  # noqa: F401
import isaacgymenvs
import tasks
import torch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_unit,
    scale,
    tensor_clamp,
    unscale,
)
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from omegaconf import DictConfig, OmegaConf, open_dict

from world_model.PPModel.kinematics import compute_link_jacobians
from world_model.policy_runtime import _apply_checkpoint_compat_overrides, _build_policy_runner
from tasks.grasp import orientation_error
from tasks.utils import batch_linear_interpolate_poses
from world_model.test_world_model import _build_ppmodel_and_mapping, _get_wm_cfg, _resolve_world_model_setup
from world_model.train_world_model import _select_contact_points_from_hand_points


PROJECT_ROOT = Path(__file__).resolve().parent


def _get_pso_cfg(cfg: DictConfig):
    defaults = OmegaConf.create(
        {
            "env_index": 0,
            "num_steps": None,
            "warmup_steps": None,
            "plan_source": "random",
            "search_steps": None,
            "execution_mode": "trajectory",
            "execute_steps": None,
            "replan_after_execution": False,
            "num_particles": 96,
            "particle_eval_batch_size": 64,
            "num_iterations": 12,
            "log_iteration_best": True,
            "log_execution_steps": False,
            "log_final_solution": True,
            "inertia": 0.6,
            "cognitive": 1.5,
            "social": 1.5,
            "focus_resample_ratio": 0.35,
            "focus_start_ratio": 0.5,
            "focus_position_std": 0.01,
            "focus_rotation_std": 0.08,
            "focus_hand_std": 0.08,
            "focus_min_scale": 0.2,
            "approach_steps": 6,
            "close_hand_steps": 4,
            "lift_steps": 5,
            "wm_lift_steps": None,
            "reference_grasp_index": None,
            "lift_step_height": 0.02,
            "ik_iterations": 12,
            "position_margin": [0.04, 0.04, 0.08],
            "position_surface_noise": [0.02, 0.02, 0.03],
            "goal_mode": "lift_xy",
            "goal_center": None,
            "goal_center_offset": [0.0, 0.0, 0.12],
            "surface_offset_distance": 0.03,
            "contact_reward_weight": 10.0,
            "contact_distance_weight": 10.0,
            "contact_reward_scale": 0.01,
            "sdf_reward_scale": 0.01,
            "contact_distribution_weight": 4.0,
            "grasp_contact_weight": 6.0,
            "lift_contact_weight": 6.0,
            "hand_contact_radius": 0.01,
            "contact_margin": 0.015,
            "inside_penalty_weight": 20.0,
            "ground_collision_weight": 12.0,
            "ground_penetration_tolerance": 0.002,
            "transport_mode": "target_center",
            "transport_offset": [0.0, 0.0, 0.0],
            "goal_center_weight": 25.0,
            "lift_weight": 4.0,
            "xy_penalty_weight": 6.0,
            "pose_regularization_weight": 0.5,
            "hand_regularization_weight": 0.1,
            "action_smoothness_weight": 0.05,
            "init_pos_std": 0.02,
            "init_rot_std": 0.12,
            "init_hand_std": 0.15,
            "max_pos_delta": [0.08, 0.08, 0.08],
            "max_rot_delta": [0.5, 0.5, 0.5],
            "max_hand_delta": 0.8,
            "log_interval": 1,
        }
    )
    if "pso" not in cfg or cfg.pso is None:
        return defaults
    return OmegaConf.merge(defaults, cfg.pso)


def _apply_pso_env_defaults(cfg: DictConfig) -> None:
    with open_dict(cfg):
        if "train" not in cfg or cfg.train is None:
            cfg.train = OmegaConf.load(PROJECT_ROOT / "tasks" / "train" / "PPOOneStep.yaml")
        cfg.train.params.is_vision = True
        cfg.task.env.observationType = "eefpose+objinitpose+objpcl"
        cfg.task.env.armController = "pose"
        cfg.task.env.enablePointCloud = True
        cfg.task.env.randomizeTrackingReference = True
        cfg.task.env.randomizeGraspPose = True
        cfg.test = True


def _as_long_tensor(values, *, device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.long, device=device)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _build_goal_center(current_object_points: torch.Tensor, pso_cfg) -> torch.Tensor:
    current_center = current_object_points[..., :3].mean(dim=0)
    raw_goal_center = pso_cfg.get("goal_center", None)
    if raw_goal_center is not None:
        return torch.tensor(raw_goal_center, dtype=current_center.dtype, device=current_center.device)

    goal_center_offset = torch.tensor(
        list(pso_cfg.goal_center_offset),
        dtype=current_center.dtype,
        device=current_center.device,
    )
    return current_center + goal_center_offset


def _matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    if rotation_matrix.ndim != 3 or rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (B, 3, 3), got {tuple(rotation_matrix.shape)}")

    m = rotation_matrix
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = torch.zeros((m.shape[0], 4), dtype=m.dtype, device=m.device)

    positive_trace = trace > 0.0
    if positive_trace.any():
        s = torch.sqrt(trace[positive_trace] + 1.0) * 2.0
        quat[positive_trace, 3] = 0.25 * s
        quat[positive_trace, 0] = (m[positive_trace, 2, 1] - m[positive_trace, 1, 2]) / s
        quat[positive_trace, 1] = (m[positive_trace, 0, 2] - m[positive_trace, 2, 0]) / s
        quat[positive_trace, 2] = (m[positive_trace, 1, 0] - m[positive_trace, 0, 1]) / s

    cond_x = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]) & (~positive_trace)
    if cond_x.any():
        s = torch.sqrt(1.0 + m[cond_x, 0, 0] - m[cond_x, 1, 1] - m[cond_x, 2, 2]) * 2.0
        quat[cond_x, 3] = (m[cond_x, 2, 1] - m[cond_x, 1, 2]) / s
        quat[cond_x, 0] = 0.25 * s
        quat[cond_x, 1] = (m[cond_x, 0, 1] + m[cond_x, 1, 0]) / s
        quat[cond_x, 2] = (m[cond_x, 0, 2] + m[cond_x, 2, 0]) / s

    cond_y = (m[:, 1, 1] > m[:, 2, 2]) & (~positive_trace) & (~cond_x)
    if cond_y.any():
        s = torch.sqrt(1.0 + m[cond_y, 1, 1] - m[cond_y, 0, 0] - m[cond_y, 2, 2]) * 2.0
        quat[cond_y, 3] = (m[cond_y, 0, 2] - m[cond_y, 2, 0]) / s
        quat[cond_y, 0] = (m[cond_y, 0, 1] + m[cond_y, 1, 0]) / s
        quat[cond_y, 1] = 0.25 * s
        quat[cond_y, 2] = (m[cond_y, 1, 2] + m[cond_y, 2, 1]) / s

    cond_z = (~positive_trace) & (~cond_x) & (~cond_y)
    if cond_z.any():
        s = torch.sqrt(1.0 + m[cond_z, 2, 2] - m[cond_z, 0, 0] - m[cond_z, 1, 1]) * 2.0
        quat[cond_z, 3] = (m[cond_z, 1, 0] - m[cond_z, 0, 1]) / s
        quat[cond_z, 0] = (m[cond_z, 0, 2] + m[cond_z, 2, 0]) / s
        quat[cond_z, 1] = (m[cond_z, 1, 2] + m[cond_z, 2, 1]) / s
        quat[cond_z, 2] = 0.25 * s

    return quat_unit(quat)


def _axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1)
    safe_angle = angle.unsqueeze(-1).clamp_min(1e-8)
    axis = axis_angle / safe_angle
    return quat_from_angle_axis(angle, axis)


def _nlerp_batch(start_quat: torch.Tensor, end_quat: torch.Tensor, num_steps: int) -> torch.Tensor:
    if num_steps <= 0:
        return start_quat.new_zeros(start_quat.shape[0], 0, 4)

    start = quat_unit(start_quat)
    end = quat_unit(end_quat)
    same_hemisphere = (start * end).sum(dim=-1, keepdim=True) >= 0.0
    end = torch.where(same_hemisphere, end, -end)

    alphas = torch.linspace(1.0 / num_steps, 1.0, num_steps, device=start.device, dtype=start.dtype)
    interpolated = (1.0 - alphas[None, :, None]) * start[:, None, :] + alphas[None, :, None] * end[:, None, :]
    return quat_unit(interpolated.reshape(-1, 4)).reshape(start.shape[0], num_steps, 4)


def _interpolate_positions(start_pos: torch.Tensor, end_pos: torch.Tensor, num_steps: int) -> torch.Tensor:
    if num_steps <= 0:
        return start_pos.new_zeros(start_pos.shape[0], 0, 3)
    alphas = torch.linspace(1.0 / num_steps, 1.0, num_steps, device=start_pos.device, dtype=start_pos.dtype)
    return (1.0 - alphas[None, :, None]) * start_pos[:, None, :] + alphas[None, :, None] * end_pos[:, None, :]


def _interpolate_scalars(start_values: torch.Tensor, end_values: torch.Tensor, num_steps: int) -> torch.Tensor:
    if num_steps <= 0:
        return start_values.new_zeros(start_values.shape[0], 0, start_values.shape[1])
    alphas = torch.linspace(1.0 / num_steps, 1.0, num_steps, device=start_values.device, dtype=start_values.dtype)
    return (1.0 - alphas[None, :, None]) * start_values[:, None, :] + alphas[None, :, None] * end_values[:, None, :]


def _resample_linear_sequence(sequence: torch.Tensor, target_steps: int) -> torch.Tensor:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if sequence.shape[0] == target_steps:
        return sequence
    if sequence.shape[0] == 1:
        return sequence.repeat(target_steps, 1)

    src_positions = torch.linspace(
        0,
        sequence.shape[0] - 1,
        target_steps,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    left_index = torch.floor(src_positions).to(torch.long)
    right_index = torch.clamp(left_index + 1, max=sequence.shape[0] - 1)
    alpha = (src_positions - left_index.to(sequence.dtype)).unsqueeze(-1)
    return (1.0 - alpha) * sequence[left_index] + alpha * sequence[right_index]


def _resample_quaternion_sequence(sequence: torch.Tensor, target_steps: int) -> torch.Tensor:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if sequence.shape[0] == target_steps:
        return quat_unit(sequence)
    if sequence.shape[0] == 1:
        return quat_unit(sequence.repeat(target_steps, 1))

    src_positions = torch.linspace(
        0,
        sequence.shape[0] - 1,
        target_steps,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    left_index = torch.floor(src_positions).to(torch.long)
    right_index = torch.clamp(left_index + 1, max=sequence.shape[0] - 1)
    alpha = (src_positions - left_index.to(sequence.dtype)).unsqueeze(-1)

    start = quat_unit(sequence[left_index])
    end = quat_unit(sequence[right_index])
    same_hemisphere = (start * end).sum(dim=-1, keepdim=True) >= 0.0
    end = torch.where(same_hemisphere, end, -end)
    return quat_unit((1.0 - alpha) * start + alpha * end)


@dataclass
class HistoryStep:
    object_points: torch.Tensor
    action_points: torch.Tensor


@dataclass
class TrajectoryParticleBatch:
    translation_action: torch.Tensor
    rotation_action: torch.Tensor
    target_hand_q: torch.Tensor

    @property
    def actor_action(self) -> torch.Tensor:
        return torch.cat([self.translation_action, self.rotation_action, self.target_hand_q], dim=-1)


class WorldModelPSOPlanner:
    def __init__(self, cfg: DictConfig, env):
        self.cfg = cfg
        self.env = env
        self.pso_cfg = _get_pso_cfg(cfg)
        self.wm_cfg = _get_wm_cfg(cfg)
        self.wm_setup = _resolve_world_model_setup(cfg, self.wm_cfg)
        self.world_model = self.wm_setup["model"]
        self.world_model_device = next(self.world_model.parameters()).device
        self.history_horizon = int(self.wm_setup["history_horizon"])
        self.future_horizon = int(self.wm_setup["future_horizon"])
        self.num_contact_points = int(self.wm_setup["num_contact_points"])
        self.outside_distance_threshold = float(self.wm_setup["outside_distance_threshold"])
        self.env_index = int(self.pso_cfg.env_index)
        self.approach_steps = int(self.pso_cfg.approach_steps)
        self.close_hand_steps = int(self.pso_cfg.close_hand_steps)
        self.lift_steps = int(self.pso_cfg.lift_steps)
        default_wm_lift_steps = max(1, (self.lift_steps + 1) // 2)
        self.wm_lift_steps = int(
            default_wm_lift_steps if self.pso_cfg.wm_lift_steps is None else self.pso_cfg.wm_lift_steps
        )
        self.search_steps = int(
            self.close_hand_steps
            if self.pso_cfg.search_steps is None
            else self.pso_cfg.search_steps
        )

        if self.env_index < 0 or self.env_index >= env.num_envs:
            raise ValueError(f"pso.env_index={self.env_index} is invalid for num_envs={env.num_envs}")
        if env.arm_controller != "pose":
            raise NotImplementedError(
                f"run_world_model_pso.py currently expects task.env.armController=pose, got {env.arm_controller!r}"
            )
        if self.search_steps <= 0:
            raise ValueError("pso.search_steps must be positive.")
        if self.approach_steps < 0:
            raise ValueError("pso.approach_steps must be non-negative.")
        if self.close_hand_steps <= 0:
            raise ValueError("pso.close_hand_steps must be positive.")
        if self.lift_steps <= 0:
            raise ValueError("pso.lift_steps must be positive.")
        if self.wm_lift_steps <= 0:
            raise ValueError("pso.wm_lift_steps must be positive.")

        self.pp_model, self.model_mapping = _build_ppmodel_and_mapping(
            env,
            self.wm_setup,
            preprocess_device=str(self.wm_cfg.preprocess_device),
        )
        if self.model_mapping["missing_joint_names"]:
            print("Warning: PPModel joints missing in env and set to zero:", self.model_mapping["missing_joint_names"])

        env_joint_to_pp_joint = {
            int(env_idx): int(pp_idx)
            for pp_idx, env_idx in zip(
                self.model_mapping["model_joint_indices"].tolist(),
                self.model_mapping["env_joint_indices"].tolist(),
            )
        }
        self.env_arm_indices = _as_long_tensor(env.arm_dof_indices, device=env.device)
        self.env_active_hand_indices = _as_long_tensor(env.active_hand_dof_indices, device=env.device)
        self.env_passive_hand_indices = _as_long_tensor(env.passive_hand_dof_indices, device=env.device)
        self.env_mimic_parent_indices = _as_long_tensor(env.mimic_parent_dof_indices, device=env.device)
        self.env_hand_indices = _as_long_tensor(env.hand_dof_indices, device=env.device)
        self.pp_arm_indices = torch.tensor(
            [env_joint_to_pp_joint[int(idx)] for idx in self.env_arm_indices.tolist()],
            dtype=torch.long,
            device=self.pp_model.device,
        )

        self.hand_action_slice = slice(int(env.hand_dof_start_idx), int(env.num_actions))
        self.eef_link_name = str(env.hand_specific_cfg["eef_link"])
        self.t_ref = int(env.T_ref)
        self.t_ref_start_lifting = int(env.T_ref_start_lifting)
        self.interpolation_step_scale = float(env.cfg["env"]["interpolationStepScale"])
        self.reference_env_index = min(self.env_index, int(env.tracking_reference["wrist_quat"].shape[0]) - 1)

        self.arm_lower_limits = env.robot_dof_lower_limits[self.env_arm_indices]
        self.arm_upper_limits = env.robot_dof_upper_limits[self.env_arm_indices]
        self.active_hand_lower_limits = env.robot_dof_lower_limits[self.env_active_hand_indices]
        self.active_hand_upper_limits = env.robot_dof_upper_limits[self.env_active_hand_indices]
        self.hand_lower_limits = env.robot_dof_lower_limits[self.env_hand_indices]
        self.hand_upper_limits = env.robot_dof_upper_limits[self.env_hand_indices]
        self.arm_velocity_limit = float(env.act_max_ang_vel_arm * env.dt * env.decimation)
        self.hand_velocity_limit = float(env.act_max_ang_vel_hand * env.dt * env.decimation)
        self.clip_actions = float(env.clip_actions)
        self.hand_q_dim = int(self.env_active_hand_indices.numel())
        self.solution_dim = 6 + self.hand_q_dim

        self.position_margin = torch.tensor(
            list(self.pso_cfg.position_margin),
            dtype=torch.float32,
            device=env.device,
        )
        self.position_surface_noise = torch.tensor(
            list(self.pso_cfg.position_surface_noise),
            dtype=torch.float32,
            device=env.device,
        )
        self.contact_link_names = self._resolve_contact_link_names()
        self.contact_link_indices = torch.tensor(
            [self.pp_model.link_names.index(link_name) for link_name in self.contact_link_names],
            dtype=torch.long,
            device=self.pp_model.device,
        )
        self.contact_point_mask = torch.isin(self.pp_model.link_indices, self.contact_link_indices)
        print("[pso] contact_reward_mode=pointcloud")
        print(f"[pso] contact_links={self.contact_link_names}")
        print(
            "[pso] trajectory_mode=short_grasp_then_lift "
            f"reference_grasp_index={self._get_reference_grasp_index()} "
            f"search_steps={self.search_steps} "
            f"wm_lift_steps={self.wm_lift_steps} "
            f"eval_steps={self.search_steps + self.wm_lift_steps}"
        )

    def _resolve_contact_link_names(self):
        palm_link = str(self.env.hand_specific_cfg.get("palm_link", "")).strip()
        fingertip_links = [str(name) for name in self.env.hand_specific_cfg.get("fingertips_link", [])]

        candidate_names = []
        if palm_link and palm_link in self.pp_model.all_link_names:
            candidate_names.extend(self.pp_model._collect_subtree_link_names(palm_link))
        else:
            for link_name in fingertip_links:
                if link_name in self.pp_model.all_link_names:
                    candidate_names.extend(self.pp_model._collect_subtree_link_names(link_name))

        filtered = [link_name for link_name in self.pp_model.link_names if link_name in set(candidate_names)]
        if filtered:
            return filtered

        return list(self.pp_model.link_names)

    def _map_robot_q_to_ppmodel_q(self, robot_q: torch.Tensor) -> torch.Tensor:
        original_shape = robot_q.shape[:-1]
        flat_robot_q = robot_q.reshape(-1, robot_q.shape[-1])
        pp_q = torch.zeros(
            (flat_robot_q.shape[0], self.pp_model.dof),
            dtype=self.pp_model.dtype,
            device=self.pp_model.device,
        )
        if self.model_mapping["model_joint_indices"].numel() > 0:
            pp_q[:, self.model_mapping["model_joint_indices"]] = flat_robot_q[
                :, self.model_mapping["env_joint_indices"]
            ].to(device=self.pp_model.device, dtype=self.pp_model.dtype)
        return pp_q.reshape(*original_shape, self.pp_model.dof)

    @torch.no_grad()
    def _compute_hand_points(self, robot_q: torch.Tensor) -> torch.Tensor:
        pp_q = self._map_robot_q_to_ppmodel_q(robot_q)
        flat_pp_q = pp_q.reshape(-1, pp_q.shape[-1])
        pp_state = self.pp_model.get_state(flat_pp_q)
        hand_points = pp_state.points
        if hand_points.ndim == 2:
            hand_points = hand_points.unsqueeze(0)
        hand_points = hand_points[:, self.contact_point_mask]
        return hand_points.to(dtype=torch.float32).reshape(*pp_q.shape[:-1], hand_points.shape[-2], hand_points.shape[-1])

    def _compute_action_points(self, hand_points: torch.Tensor, object_points: torch.Tensor) -> torch.Tensor:
        action_points = []
        for batch_index in range(hand_points.shape[0]):
            action_points.append(
                _select_contact_points_from_hand_points(
                    hand_points[batch_index],
                    object_points[batch_index],
                    num_contact_points=self.num_contact_points,
                    outside_distance_threshold=self.outside_distance_threshold,
                )
            )
        return torch.stack(action_points, dim=0)

    def _compute_eef_pose(self, robot_q: torch.Tensor) -> torch.Tensor:
        pp_q = self._map_robot_q_to_ppmodel_q(robot_q)
        flat_pp_q = pp_q.reshape(-1, pp_q.shape[-1])
        frame_status = self.pp_model.pk_chain.forward_kinematics(flat_pp_q)
        transform = frame_status[self.eef_link_name].get_matrix()
        position = transform[:, :3, 3]
        quaternion = _matrix_to_quaternion(transform[:, :3, :3])
        return torch.cat([position, quaternion], dim=-1).reshape(*pp_q.shape[:-1], 7)

    def _compute_arm_delta(self, robot_q: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
        pp_q = self._map_robot_q_to_ppmodel_q(robot_q)
        flat_pp_q = pp_q.reshape(-1, pp_q.shape[-1])
        frame_status = self.pp_model.pk_chain.forward_kinematics(flat_pp_q)
        transform = frame_status[self.eef_link_name].get_matrix()
        current_position = transform[:, :3, 3]
        current_quaternion = _matrix_to_quaternion(transform[:, :3, :3])

        pos_error = target_pose[:, :3] - current_position
        orn_error = orientation_error(quat_unit(target_pose[:, 3:7]), current_quaternion)
        dpose = torch.cat([pos_error, orn_error], dim=-1).unsqueeze(-1).to(dtype=self.pp_model.dtype)

        jacobian = compute_link_jacobians(
            self.pp_model.pk_chain,
            flat_pp_q,
            frame_status,
            [self.eef_link_name],
        )[self.eef_link_name][:, :, self.pp_arm_indices]
        jacobian_t = jacobian.transpose(1, 2)
        damping = 0.1
        regularizer = torch.eye(6, dtype=self.pp_model.dtype, device=self.pp_model.device).unsqueeze(0) * (damping ** 2)
        arm_delta = (jacobian_t @ torch.inverse(jacobian @ jacobian_t + regularizer) @ dpose).squeeze(-1)
        return arm_delta.to(device=robot_q.device, dtype=robot_q.dtype)

    def capture_history_step(self) -> HistoryStep:
        object_points = self.env.transform_obj_pcl_2_world()[self.env_index].detach().clone().to(
            device=self.pp_model.device,
            dtype=torch.float32,
        )
        current_robot_q = self.env.robot_dof_pos[self.env_index].detach().clone().unsqueeze(0)
        current_hand_points = self._compute_hand_points(current_robot_q)[0].to(device=self.pp_model.device)
        current_action_points = self._compute_action_points(
            current_hand_points.unsqueeze(0),
            object_points.unsqueeze(0),
        )[0]
        return HistoryStep(
            object_points=object_points.contiguous(),
            action_points=current_action_points.contiguous(),
        )

    def _build_reference_action_sequence(self, sequence_length: int) -> torch.Tensor:
        original_progress = self.env.progress_buf.clone()
        sequence = []
        max_progress = int(self.env.max_episode_length - 1)
        try:
            for offset in range(sequence_length):
                self.env.progress_buf[self.env_index] = min(
                    int(original_progress[self.env_index].item()) + offset,
                    max_progress,
                )
                sequence.append(self.env.compute_reference_actions()[self.env_index].detach().clone())
        finally:
            self.env.progress_buf.copy_(original_progress)
        return torch.stack(sequence, dim=0)

    def _get_search_anchor(
        self,
        current_wrist_pose: torch.Tensor,
        current_hand_q: torch.Tensor,
    ):
        del current_wrist_pose, current_hand_q
        reference = self._get_base_tracking_reference(device=self.env.device)
        grasp_index = self._get_reference_grasp_index()
        object_init_pos = self.env.object_init_states[self.env_index, 0:3].to(
            device=self.env.device,
            dtype=torch.float32,
        )
        reference_wrist_pose = torch.cat(
            [
                reference["wrist_initobj_pos"][grasp_index] + object_init_pos,
                reference["wrist_quat"][grasp_index],
            ],
            dim=-1,
        )
        reference_hand_q = reference["hand_qpos"][grasp_index]
        return reference_wrist_pose.detach().clone(), reference_hand_q.detach().clone()

    def _decode_reference_targets(self, reference_actions: torch.Tensor):
        grasp_index = 0
        reference_pregrasp_pose = reference_actions[grasp_index, :7]
        reference_hand_q = scale(
            reference_actions[grasp_index, self.hand_action_slice],
            self.active_hand_lower_limits,
            self.active_hand_upper_limits,
        )
        return reference_pregrasp_pose, reference_hand_q

    def _safe_normalize_quaternion(self, quat: torch.Tensor, fallback_quat: torch.Tensor | None = None) -> torch.Tensor:
        norms = torch.linalg.norm(quat, dim=-1, keepdim=True)
        normalized = quat / norms.clamp_min(1e-8)
        if fallback_quat is None:
            fallback_quat = quat.new_tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).repeat(quat.shape[0], 1)
        return torch.where(norms > 1e-8, normalized, fallback_quat)

    def _compute_position_bounds(self, object_points: torch.Tensor):
        object_xyz = object_points[..., :3].to(device=self.env.device)
        position_lower = object_xyz.amin(dim=0) - self.position_margin
        position_upper = object_xyz.amax(dim=0) + self.position_margin
        return position_lower, position_upper

    def _initialize_particles(
        self,
        current_object_points: torch.Tensor,
        anchor_wrist_pose: torch.Tensor,
        anchor_hand_q: torch.Tensor,
    ):
        num_particles = int(self.pso_cfg.num_particles)
        device = self.env.device
        particles = torch.zeros((num_particles, self.solution_dim), dtype=torch.float32, device=device)
        velocities = torch.zeros_like(particles)
        object_xyz = current_object_points[..., :3].to(device=device, dtype=torch.float32)
        position_lower, position_upper = self._compute_position_bounds(current_object_points)
        object_center = object_xyz.mean(dim=0, keepdim=True)
        object_init_pos = self.env.object_init_states[self.env_index, 0:3].to(device=device, dtype=torch.float32)
        base_reference = self._get_base_tracking_reference(device=device)
        grasp_index = self._get_reference_grasp_index()
        base_grasp_wrist_world = base_reference["wrist_initobj_pos"][grasp_index] + object_init_pos

        sampled_indices = torch.randint(object_xyz.shape[0], (num_particles,), device=device)
        sampled_surface_points = object_xyz[sampled_indices]
        outward = sampled_surface_points - object_center
        outward_norm = torch.linalg.norm(outward, dim=-1, keepdim=True)
        random_outward = torch.randn_like(outward)
        random_outward = random_outward / torch.linalg.norm(random_outward, dim=-1, keepdim=True).clamp_min(1e-8)
        outward_dir = torch.where(outward_norm > 1e-8, outward / outward_norm.clamp_min(1e-8), random_outward)
        sampled_noise = torch.randn((num_particles, 3), device=device) * self.position_surface_noise
        sampled_prep_world = tensor_clamp(
            sampled_surface_points
            + outward_dir * float(self.pso_cfg.surface_offset_distance)
            + sampled_noise,
            position_lower,
            position_upper,
        )
        translation_range = self.env.randomize_tracking_reference_range[0:3].to(device=device, dtype=torch.float32)
        safe_translation_range = torch.where(
            translation_range.abs() > 1e-6,
            translation_range,
            torch.ones_like(translation_range),
        )
        particles[:, 0:3] = tensor_clamp(
            (sampled_prep_world - base_grasp_wrist_world.unsqueeze(0)) / safe_translation_range.unsqueeze(0),
            -torch.ones(3, device=device, dtype=torch.float32),
            torch.ones(3, device=device, dtype=torch.float32),
        )

        anchor_hand_q_batch = anchor_hand_q.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(num_particles, 1)
        particles[:, 3:6] = tensor_clamp(
            torch.randn((num_particles, 3), device=device, dtype=torch.float32) * float(self.pso_cfg.init_rot_std),
            -torch.ones(3, device=device, dtype=torch.float32),
            torch.ones(3, device=device, dtype=torch.float32),
        )
        particles[:, 6:] = tensor_clamp(
            anchor_hand_q_batch + torch.randn(num_particles, self.hand_q_dim, device=device) * float(self.pso_cfg.init_hand_std),
            self.active_hand_lower_limits,
            self.active_hand_upper_limits,
        )
        particles = self._clamp_particles(particles, position_lower, position_upper, anchor_wrist_pose[3:7].unsqueeze(0).repeat(num_particles, 1))
        return particles, velocities, position_lower, position_upper, anchor_wrist_pose[3:7].unsqueeze(0).repeat(num_particles, 1)

    def _clamp_particles(
        self,
        particles: torch.Tensor,
        position_lower: torch.Tensor,
        position_upper: torch.Tensor,
        fallback_quat: torch.Tensor,
    ) -> torch.Tensor:
        particles[:, 0:3] = torch.clamp(particles[:, 0:3], -1.0, 1.0)
        particles[:, 3:6] = torch.clamp(particles[:, 3:6], -1.0, 1.0)
        particles[:, 6:] = tensor_clamp(
            particles[:, 6:],
            self.active_hand_lower_limits,
            self.active_hand_upper_limits,
        )
        return particles

    def _decode_particles(
        self,
        particles: torch.Tensor,
    ) -> TrajectoryParticleBatch:
        return TrajectoryParticleBatch(
            translation_action=torch.clamp(particles[:, 0:3], -1.0, 1.0),
            rotation_action=torch.clamp(particles[:, 3:6], -1.0, 1.0),
            target_hand_q=tensor_clamp(
                particles[:, 6:],
                self.active_hand_lower_limits,
                self.active_hand_upper_limits,
            ),
        )

    def _focus_particles_around_global_best(
        self,
        particles: torch.Tensor,
        velocities: torch.Tensor,
        costs: torch.Tensor,
        global_best: torch.Tensor,
        global_best_cost: torch.Tensor,
        iteration_idx: int,
        position_lower: torch.Tensor,
        position_upper: torch.Tensor,
        fallback_quat: torch.Tensor,
    ):
        if not torch.isfinite(global_best_cost):
            return particles, velocities

        resample_ratio = float(self.pso_cfg.focus_resample_ratio)
        if resample_ratio <= 0.0 or particles.shape[0] <= 1:
            return particles, velocities

        total_iterations = max(1, int(self.pso_cfg.num_iterations))
        start_iter = int(total_iterations * float(self.pso_cfg.focus_start_ratio))
        start_iter = min(max(start_iter, 0), total_iterations - 1)
        current_iter = iteration_idx + 1
        if current_iter <= start_iter:
            return particles, velocities

        num_focus = min(
            particles.shape[0] - 1,
            max(1, int(round(particles.shape[0] * resample_ratio))),
        )
        worst_indices = torch.topk(costs, k=num_focus, largest=True).indices

        if total_iterations - start_iter <= 1:
            progress = 1.0
        else:
            progress = float(current_iter - start_iter) / float(total_iterations - start_iter)
        progress = max(0.0, min(1.0, progress))
        focus_scale = max(float(self.pso_cfg.focus_min_scale), 1.0 - progress)

        focused_particles = global_best.unsqueeze(0).repeat(num_focus, 1)
        focused_velocities = torch.zeros_like(focused_particles)

        focused_particles[:, 0:3] = focused_particles[:, 0:3] + (
            torch.randn((num_focus, 3), device=particles.device, dtype=particles.dtype)
            * float(self.pso_cfg.focus_position_std)
            * focus_scale
        )

        focused_particles[:, 3:6] = focused_particles[:, 3:6] + (
            torch.randn((num_focus, 3), device=particles.device, dtype=particles.dtype)
            * float(self.pso_cfg.focus_rotation_std)
            * focus_scale
        )

        focused_particles[:, 6:] = focused_particles[:, 6:] + (
            torch.randn((num_focus, self.hand_q_dim), device=particles.device, dtype=particles.dtype)
            * float(self.pso_cfg.focus_hand_std)
            * focus_scale
        )

        focused_particles = self._clamp_particles(
            focused_particles,
            position_lower=position_lower,
            position_upper=position_upper,
            fallback_quat=fallback_quat,
        )
        particles[worst_indices] = focused_particles
        velocities[worst_indices] = focused_velocities
        return particles, velocities

    def _repeat_pose(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        num_steps: int,
    ):
        if num_steps <= 0:
            return pos.new_zeros((pos.shape[0], 0, 7))
        return torch.cat(
            [
                pos[:, None, :].repeat(1, num_steps, 1),
                quat[:, None, :].repeat(1, num_steps, 1),
            ],
            dim=-1,
        )

    def _repeat_hand_q(
        self,
        hand_q: torch.Tensor,
        num_steps: int,
    ):
        if num_steps <= 0:
            return hand_q.new_zeros((hand_q.shape[0], 0, hand_q.shape[1]))
        return hand_q[:, None, :].repeat(1, num_steps, 1)

    def _concat_nonempty_sequences(self, sequences):
        nonempty = [sequence for sequence in sequences if sequence.shape[1] > 0]
        if not nonempty:
            raise ValueError("Expected at least one non-empty sequence segment.")
        return torch.cat(nonempty, dim=1)

    def _get_base_tracking_reference(self, device) -> Dict[str, torch.Tensor]:
        ref_index = self.reference_env_index
        return {
            "wrist_initobj_pos": self.env.tracking_reference["wrist_initobj_pos"][ref_index].to(device=device, dtype=torch.float32),
            "wrist_quat": self.env.tracking_reference["wrist_quat"][ref_index].to(device=device, dtype=torch.float32),
            "hand_qpos": self.env.tracking_reference["hand_qpos"][ref_index].to(device=device, dtype=torch.float32),
        }

    def _get_reference_grasp_index(self) -> int:
        raw_index = self.pso_cfg.get("reference_grasp_index", None)
        if raw_index is None:
            grasp_index = int(self.t_ref_start_lifting) - 1
        else:
            grasp_index = int(raw_index)
        return min(max(grasp_index, 0), int(self.t_ref) - 1)

    def _slice_reference_segment(
        self,
        sequence: torch.Tensor,
        *,
        end_index: int,
        length: int,
    ) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Reference segment length must be positive.")
        start_index = max(0, end_index - length + 1)
        segment = sequence[start_index : end_index + 1]
        if segment.shape[0] < length:
            pad = segment[0:1].repeat(length - segment.shape[0], *([1] * (segment.ndim - 1)))
            segment = torch.cat([pad, segment], dim=0)
        return segment

    def _build_project_tracking_trajectories(
        self,
        trajectory_particles: TrajectoryParticleBatch,
    ):
        num_particles = trajectory_particles.translation_action.shape[0]
        device = trajectory_particles.translation_action.device
        base_reference = self._get_base_tracking_reference(device=device)
        grasp_index = self._get_reference_grasp_index()
        search_steps = int(self.search_steps)
        lift_steps = int(self.wm_lift_steps)

        wrist_initobj_pos = self._slice_reference_segment(
            base_reference["wrist_initobj_pos"],
            end_index=grasp_index,
            length=search_steps,
        ).unsqueeze(0).repeat(num_particles, 1, 1)
        wrist_quat = self._slice_reference_segment(
            base_reference["wrist_quat"],
            end_index=grasp_index,
            length=search_steps,
        ).unsqueeze(0).repeat(num_particles, 1, 1)
        reference_hand_start = self._slice_reference_segment(
            base_reference["hand_qpos"],
            end_index=grasp_index,
            length=search_steps,
        )[0]

        rotation_scale = self.env.randomize_tracking_reference_range[3:6].to(device=device, dtype=torch.float32)
        rand_quat = quat_from_euler_xyz(
            trajectory_particles.rotation_action[:, 0] * rotation_scale[0],
            trajectory_particles.rotation_action[:, 1] * rotation_scale[1],
            trajectory_particles.rotation_action[:, 2] * rotation_scale[2],
        ).unsqueeze(1).expand(-1, search_steps, -1)
        wrist_quat = quat_mul(rand_quat, wrist_quat)
        wrist_initobj_pos = quat_apply(
            rand_quat.reshape(-1, 4),
            wrist_initobj_pos.reshape(-1, 3),
        ).reshape(num_particles, search_steps, 3)

        translation_scale = self.env.randomize_tracking_reference_range[0:3].to(device=device, dtype=torch.float32)
        wrist_initobj_pos = wrist_initobj_pos + (
            trajectory_particles.translation_action * translation_scale
        ).unsqueeze(1)

        reference_hand_start = reference_hand_start.to(device=device, dtype=torch.float32)
        hand_qpos = _interpolate_scalars(
            reference_hand_start.unsqueeze(0).repeat(num_particles, 1),
            trajectory_particles.target_hand_q,
            search_steps,
        )
        lift_hand_qpos = trajectory_particles.target_hand_q[:, None, :].repeat(1, lift_steps, 1)
        lift_offsets = (
            torch.arange(1, lift_steps + 1, device=device, dtype=torch.float32)
            * float(self.pso_cfg.lift_step_height)
        )
        lift_pos = wrist_initobj_pos[:, -1:, :].repeat(1, lift_steps, 1)
        lift_pos[:, :, 2] = lift_pos[:, :, 2] + lift_offsets.unsqueeze(0)
        lift_quat = wrist_quat[:, -1:, :].repeat(1, lift_steps, 1)

        wrist_initobj_pos = torch.cat([wrist_initobj_pos, lift_pos], dim=1)
        wrist_quat = torch.cat([wrist_quat, lift_quat], dim=1)
        hand_qpos = torch.cat([hand_qpos, lift_hand_qpos], dim=1)
        hand_qpos = tensor_clamp(hand_qpos, self.active_hand_lower_limits, self.active_hand_upper_limits)

        object_init_pos = self.env.object_init_states[self.env_index, 0:3].to(device=device, dtype=torch.float32)
        wrist_world_pos = wrist_initobj_pos + object_init_pos.view(1, 1, 3)
        wrist_pose_sequence = torch.cat([wrist_world_pos, wrist_quat], dim=-1)
        return wrist_pose_sequence, hand_qpos

    def _build_execution_trajectories(
        self,
        current_wrist_pose: torch.Tensor,
        current_hand_q: torch.Tensor,
        trajectory_particles: TrajectoryParticleBatch,
    ):
        tracking_wrist_pose_sequence, tracking_hand_q_sequence = self._build_project_tracking_trajectories(
            trajectory_particles
        )
        num_particles = tracking_wrist_pose_sequence.shape[0]
        current_pose_batch = current_wrist_pose.to(
            device=tracking_wrist_pose_sequence.device,
            dtype=tracking_wrist_pose_sequence.dtype,
        ).unsqueeze(0).repeat(num_particles, 1)
        reaching_plan_ee, _ = batch_linear_interpolate_poses(
            current_pose_batch,
            tracking_wrist_pose_sequence[:, 0],
            max_trans_step=0.04 * self.interpolation_step_scale,
            max_rot_step=0.1 * self.interpolation_step_scale,
        )
        reaching_wrist_pose = reaching_plan_ee[:, 1:]
        reaching_hand_q = tracking_hand_q_sequence[:, 0:1].repeat(1, reaching_wrist_pose.shape[1], 1)
        wrist_pose_sequence = self._concat_nonempty_sequences([reaching_wrist_pose, tracking_wrist_pose_sequence])
        hand_q_sequence = self._concat_nonempty_sequences([reaching_hand_q, tracking_hand_q_sequence])
        return wrist_pose_sequence, hand_q_sequence

    def _build_world_model_trajectories(
        self,
        trajectory_particles: TrajectoryParticleBatch,
    ):
        return self._build_project_tracking_trajectories(trajectory_particles)

    def _apply_hand_configuration(self, robot_q: torch.Tensor, target_hand_q: torch.Tensor) -> torch.Tensor:
        robot_q = robot_q.clone()
        robot_q[:, self.env_active_hand_indices] = target_hand_q
        if bool(self.env.have_passive_joints):
            robot_q[:, self.env_passive_hand_indices] = (
                robot_q[:, self.env_mimic_parent_indices] * self.env.mimic_multipliers.unsqueeze(0)
            )
        robot_q[:, self.env_hand_indices] = tensor_clamp(
            robot_q[:, self.env_hand_indices],
            self.hand_lower_limits,
            self.hand_upper_limits,
        )
        return robot_q

    def _solve_robot_q_for_pose(
        self,
        initial_robot_q: torch.Tensor,
        target_pose: torch.Tensor,
        target_hand_q: torch.Tensor,
    ) -> torch.Tensor:
        robot_q = self._apply_hand_configuration(initial_robot_q, target_hand_q)
        for _ in range(int(self.pso_cfg.ik_iterations)):
            arm_delta = self._compute_arm_delta(robot_q, target_pose)
            robot_q[:, self.env_arm_indices] = tensor_clamp(
                robot_q[:, self.env_arm_indices] + arm_delta,
                self.arm_lower_limits,
                self.arm_upper_limits,
            )
            robot_q = self._apply_hand_configuration(robot_q, target_hand_q)
        return robot_q

    def _build_grasp_action_points(
        self,
        current_robot_q: torch.Tensor,
        current_object_points: torch.Tensor,
        wrist_pose_sequence: torch.Tensor,
        hand_q_sequence: torch.Tensor,
    ) -> torch.Tensor:
        num_particles, horizon, _ = wrist_pose_sequence.shape
        if current_robot_q.ndim == 1:
            robot_q = current_robot_q.unsqueeze(0).repeat(num_particles, 1)
        elif current_robot_q.ndim == 2 and current_robot_q.shape[0] == num_particles:
            robot_q = current_robot_q.clone()
        else:
            raise ValueError(
                "current_robot_q must have shape (D,) or (B, D) matching wrist_pose_sequence batch size."
            )
        object_points_batch = current_object_points.unsqueeze(0).repeat(num_particles, 1, 1).to(
            device=self.pp_model.device
        )
        rollout_action_points = []

        for timestep in range(horizon):
            robot_q = self._solve_robot_q_for_pose(
                robot_q,
                wrist_pose_sequence[:, timestep],
                hand_q_sequence[:, timestep],
            )
            hand_points = self._compute_hand_points(robot_q).to(device=self.pp_model.device, dtype=torch.float32)
            action_points = self._compute_action_points(hand_points, object_points_batch)
            rollout_action_points.append(action_points)

        return torch.stack(rollout_action_points, dim=1)

    def _predict_object_rollout(
        self,
        object_history: torch.Tensor,
        action_history: torch.Tensor,
        future_action_points: torch.Tensor,
    ):
        if future_action_points.ndim != 4:
            raise ValueError(
                f"future_action_points must have shape (B, T, N, C), got {tuple(future_action_points.shape)}"
            )

        predicted_chunks = []
        rolling_object_history = object_history
        rolling_action_history = action_history
        start_step = 0

        while start_step < future_action_points.shape[1]:
            end_step = min(start_step + self.future_horizon, future_action_points.shape[1])
            chunk = future_action_points[:, start_step:end_step]
            chunk_len = end_step - start_step
            if chunk_len < self.future_horizon:
                pad_count = self.future_horizon - chunk_len
                pad_chunk = chunk[:, -1:].repeat(1, pad_count, 1, 1)
                chunk = torch.cat([chunk, pad_chunk], dim=1)

            prediction = self.world_model(
                rolling_object_history,
                rolling_action_history,
                chunk,
            )
            predicted_chunk = prediction.object_future_points[:, :chunk_len]
            predicted_chunks.append(predicted_chunk)

            rolling_object_history = torch.cat([rolling_object_history, predicted_chunk], dim=1)[:, -self.history_horizon :]
            rolling_action_history = torch.cat(
                [rolling_action_history, future_action_points[:, start_step:end_step]],
                dim=1,
            )[:, -self.history_horizon :]
            start_step = end_step

        return torch.cat(predicted_chunks, dim=1)

    def _build_env_action(self, wrist_pose: torch.Tensor, hand_q: torch.Tensor) -> torch.Tensor:
        env_action = self.env.no_op_action[self.env_index].detach().clone()
        env_action[:7] = wrist_pose
        env_action[self.hand_action_slice] = unscale(
            hand_q.unsqueeze(0),
            self.active_hand_lower_limits,
            self.active_hand_upper_limits,
        )[0]
        return torch.clamp(env_action, -self.clip_actions, self.clip_actions)

    def _build_env_action_sequence(
        self,
        wrist_pose_sequence: torch.Tensor,
        hand_q_sequence: torch.Tensor,
    ) -> torch.Tensor:
        env_actions = []
        for timestep in range(wrist_pose_sequence.shape[0]):
            env_actions.append(self._build_env_action(wrist_pose_sequence[timestep], hand_q_sequence[timestep]))
        return torch.stack(env_actions, dim=0)

    def _resample_plan_sequences(
        self,
        wrist_pose_sequence: torch.Tensor,
        hand_q_sequence: torch.Tensor,
        target_steps: int,
    ):
        if target_steps <= 0:
            raise ValueError("target_steps must be positive.")
        if wrist_pose_sequence.shape[0] == target_steps:
            return wrist_pose_sequence, hand_q_sequence

        resampled_pos = _resample_linear_sequence(wrist_pose_sequence[:, :3], target_steps)
        resampled_quat = _resample_quaternion_sequence(wrist_pose_sequence[:, 3:7], target_steps)
        resampled_hand_q = _resample_linear_sequence(hand_q_sequence, target_steps)
        return torch.cat([resampled_pos, resampled_quat], dim=-1), resampled_hand_q

    def _compute_pointcloud_sdf(
        self,
        contact_points: torch.Tensor,
        current_object_points: torch.Tensor,
    ):
        contact_xyz = contact_points[..., :3]
        object_xyz = current_object_points[..., :3].to(device=contact_xyz.device, dtype=contact_xyz.dtype)
        contact_distance = torch.cdist(
            contact_xyz,
            object_xyz.unsqueeze(0).expand(contact_xyz.shape[0], -1, -1),
        ).amin(dim=-1)
        hand_radius = float(self.pso_cfg.hand_contact_radius)
        return contact_distance - hand_radius

    def _compute_contact_band_reward(
        self,
        contact_points: torch.Tensor,
        current_object_points: torch.Tensor,
    ):
        sdf = self._compute_pointcloud_sdf(contact_points, current_object_points)
        reward_scale = float(
            self.pso_cfg.get(
                "contact_reward_scale",
                self.pso_cfg.get("sdf_reward_scale", 0.01),
            )
        )
        contact_margin = float(self.pso_cfg.contact_margin)
        inside_error = torch.clamp(-sdf, min=0.0)
        outside_error = torch.clamp(sdf - contact_margin, min=0.0)
        band_error = inside_error + outside_error
        per_point_reward = torch.exp(-band_error / max(reward_scale, 1e-6))
        return per_point_reward.mean(dim=-1), per_point_reward

    def _compute_contact_distribution_reward(
        self,
        contact_points: torch.Tensor,
        current_object_points: torch.Tensor,
    ) -> torch.Tensor:
        contact_xyz = contact_points[..., :3]
        object_com = current_object_points[..., :3].mean(dim=0, keepdim=True).to(
            device=contact_xyz.device,
            dtype=contact_xyz.dtype,
        )
        _, per_point_reward = self._compute_contact_band_reward(contact_points, current_object_points)
        directions = contact_xyz - object_com.unsqueeze(1)
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-6)
        directions = directions / direction_norm
        weights = per_point_reward.unsqueeze(-1)
        weighted_mean_direction = (directions * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        spread_reward = 1.0 - torch.linalg.norm(weighted_mean_direction, dim=-1)
        quality_reward = per_point_reward.mean(dim=-1)
        return spread_reward.clamp(0.0, 1.0) * quality_reward

    def _get_support_height(self, *, device, dtype) -> torch.Tensor:
        if hasattr(self.env, "table_heights") and self.env.table_heights is not None:
            return self.env.table_heights[self.env_index].detach().to(device=device, dtype=dtype)
        return torch.tensor(0.0, device=device, dtype=dtype)

    def _compute_ground_collision_penalty(
        self,
        points: torch.Tensor,
        support_height: torch.Tensor,
    ) -> torch.Tensor:
        point_z = points[..., 2]
        tolerance = float(self.pso_cfg.get("ground_penetration_tolerance", 0.0))
        penetration = torch.clamp((support_height - tolerance) - point_z, min=0.0)
        reduction_dims = tuple(range(1, penetration.ndim))
        if not reduction_dims:
            return penetration
        return penetration.mean(dim=reduction_dims)

    def _evaluate_costs(
        self,
        prediction,
        current_object_points: torch.Tensor,
        contact_point_sequence: torch.Tensor,
    ) -> torch.Tensor:
        predicted_future_centers = prediction[..., :3].mean(dim=2)
        current_center = current_object_points[..., :3].mean(dim=0)
        final_center = predicted_future_centers[:, -1]
        final_lift = final_center[:, 2] - current_center[2]
        grasp_contact_points = contact_point_sequence[:, 0]
        grasp_contact_weight = float(
            self.pso_cfg.get("grasp_contact_weight", self.pso_cfg.get("contact_reward_weight", self.pso_cfg.contact_distance_weight))
        )
        lift_contact_weight = float(
            self.pso_cfg.get("lift_contact_weight", self.pso_cfg.get("contact_reward_weight", self.pso_cfg.contact_distance_weight))
        )
        contact_distribution_weight = float(self.pso_cfg.get("contact_distribution_weight", 0.0))
        ground_collision_weight = float(self.pso_cfg.get("ground_collision_weight", 0.0))
        support_height = self._get_support_height(device=prediction.device, dtype=prediction.dtype)

        grasp_contact_reward, _ = self._compute_contact_band_reward(
            contact_points=grasp_contact_points,
            current_object_points=current_object_points,
        )
        if contact_point_sequence.shape[1] > 1:
            lift_contact_rewards, _ = self._compute_contact_band_reward(
                contact_points=contact_point_sequence[:, 1:].reshape(
                    -1,
                    contact_point_sequence.shape[2],
                    contact_point_sequence.shape[3],
                ),
                current_object_points=current_object_points,
            )
            lift_contact_reward = lift_contact_rewards.reshape(contact_point_sequence.shape[0], -1).mean(dim=-1)
        else:
            lift_contact_reward = grasp_contact_reward
        contact_distribution_reward = self._compute_contact_distribution_reward(
            contact_points=grasp_contact_points,
            current_object_points=current_object_points,
        )
        object_ground_penalty = self._compute_ground_collision_penalty(prediction[..., :3], support_height)
        hand_ground_penalty = self._compute_ground_collision_penalty(contact_point_sequence[..., :3], support_height)
        ground_collision_penalty = object_ground_penalty + hand_ground_penalty

        return (
            -float(self.pso_cfg.lift_weight) * final_lift
            - contact_distribution_weight * contact_distribution_reward
            - grasp_contact_weight * grasp_contact_reward
            - lift_contact_weight * lift_contact_reward
            + ground_collision_weight * ground_collision_penalty
        )

    @torch.no_grad()
    def plan_action(self, history_buffer: Deque[HistoryStep]) -> Dict[str, torch.Tensor]:
        if len(history_buffer) < self.history_horizon:
            raise RuntimeError(
                f"Need at least {self.history_horizon} history steps, got {len(history_buffer)}"
            )

        latest_step = history_buffer[-1]
        current_object_points = latest_step.object_points.to(device=self.pp_model.device, dtype=torch.float32)
        current_robot_q = self.env.robot_dof_pos[self.env_index].detach().clone()
        current_wrist_pose = self._compute_eef_pose(current_robot_q.unsqueeze(0))[0].to(device=self.env.device)
        current_hand_q = current_robot_q[self.env_active_hand_indices]
        goal_center = _build_goal_center(current_object_points, self.pso_cfg).to(self.world_model_device)

        object_history = torch.stack(
            [step.object_points for step in list(history_buffer)[-self.history_horizon :]],
            dim=0,
        ).to(device=self.world_model_device, dtype=torch.float32)
        action_history = torch.stack(
            [step.action_points for step in list(history_buffer)[-self.history_horizon :]],
            dim=0,
        ).to(device=self.world_model_device, dtype=torch.float32)

        reference_pregrasp_pose, reference_hand_q = self._get_search_anchor(current_wrist_pose, current_hand_q)
        particles, velocities, position_lower, position_upper, fallback_quat = self._initialize_particles(
            current_object_points=current_object_points,
            anchor_wrist_pose=reference_pregrasp_pose,
            anchor_hand_q=reference_hand_q,
        )
        personal_best = particles.clone()
        personal_best_cost = torch.full(
            (particles.shape[0],),
            float("inf"),
            dtype=torch.float32,
            device=particles.device,
        )
        global_best = particles[0].clone()
        global_best_cost = torch.tensor(float("inf"), dtype=torch.float32, device=particles.device)

        best_wrist_pose_sequence = None
        best_hand_q_sequence = None
        best_wm_wrist_pose_sequence = None
        best_wm_hand_q_sequence = None
        best_prediction = None
        best_translation_action = None
        best_rotation_action = None
        best_target_hand_q = None
        particle_eval_batch_size = max(1, int(self.pso_cfg.particle_eval_batch_size))

        for iteration_idx in range(int(self.pso_cfg.num_iterations)):
            costs = torch.empty((particles.shape[0],), dtype=torch.float32, device=particles.device)
            iteration_best_cost = torch.tensor(float("inf"), dtype=torch.float32, device=particles.device)
            iteration_best_prediction = None
            iteration_best_translation_action = None
            iteration_best_rotation_action = None
            iteration_best_target_hand_q = None

            for batch_start in range(0, particles.shape[0], particle_eval_batch_size):
                batch_end = min(batch_start + particle_eval_batch_size, particles.shape[0])
                particle_batch = particles[batch_start:batch_end]

                trajectory_particles = self._decode_particles(particle_batch)
                wm_wrist_pose_sequence, wm_hand_q_sequence = self._build_world_model_trajectories(
                    trajectory_particles=trajectory_particles,
                )
                execution_wrist_pose_sequence, execution_hand_q_sequence = self._build_execution_trajectories(
                    current_wrist_pose=current_wrist_pose,
                    current_hand_q=current_hand_q,
                    trajectory_particles=trajectory_particles,
                )
                wm_initial_robot_q = self._solve_robot_q_for_pose(
                    current_robot_q.unsqueeze(0).repeat(particle_batch.shape[0], 1),
                    wm_wrist_pose_sequence[:, 0],
                    wm_hand_q_sequence[:, 0],
                )
                future_action_points = self._build_grasp_action_points(
                    current_robot_q=wm_initial_robot_q,
                    current_object_points=current_object_points,
                    wrist_pose_sequence=wm_wrist_pose_sequence,
                    hand_q_sequence=wm_hand_q_sequence,
                ).to(device=self.world_model_device, dtype=torch.float32)

                prediction = self._predict_object_rollout(
                    object_history.unsqueeze(0).repeat(particle_batch.shape[0], 1, 1, 1),
                    action_history.unsqueeze(0).repeat(particle_batch.shape[0], 1, 1, 1),
                    future_action_points,
                )
                batch_costs = self._evaluate_costs(
                    prediction=prediction,
                    current_object_points=current_object_points.to(self.world_model_device),
                    contact_point_sequence=future_action_points.to(self.world_model_device),
                ).to(device=particles.device)
                costs[batch_start:batch_end] = batch_costs

                batch_best_cost, batch_best_index = torch.min(batch_costs, dim=0)
                global_candidate_index = batch_start + int(batch_best_index.item())
                batch_best_translation_action = trajectory_particles.translation_action[batch_best_index].detach().clone()
                batch_best_rotation_action = trajectory_particles.rotation_action[batch_best_index].detach().clone()
                batch_best_target_hand_q = trajectory_particles.target_hand_q[batch_best_index].detach().clone()
                batch_best_prediction = prediction[batch_best_index].detach().clone()

                if batch_best_cost < iteration_best_cost:
                    iteration_best_cost = batch_best_cost
                    iteration_best_prediction = batch_best_prediction
                    iteration_best_translation_action = batch_best_translation_action
                    iteration_best_rotation_action = batch_best_rotation_action
                    iteration_best_target_hand_q = batch_best_target_hand_q

                if batch_best_cost < global_best_cost:
                    global_best_cost = batch_best_cost
                    global_best = particles[global_candidate_index].clone()
                    best_wrist_pose_sequence = execution_wrist_pose_sequence[batch_best_index].detach().clone()
                    best_hand_q_sequence = execution_hand_q_sequence[batch_best_index].detach().clone()
                    best_wm_wrist_pose_sequence = wm_wrist_pose_sequence[batch_best_index].detach().clone()
                    best_wm_hand_q_sequence = wm_hand_q_sequence[batch_best_index].detach().clone()
                    best_prediction = batch_best_prediction
                    best_translation_action = batch_best_translation_action
                    best_rotation_action = batch_best_rotation_action
                    best_target_hand_q = batch_best_target_hand_q

            if bool(self.pso_cfg.log_iteration_best) and iteration_best_prediction is not None:
                iteration_final_center = iteration_best_prediction[-1, :, :3].mean(dim=0).detach().cpu()
                iteration_lift = float(iteration_final_center[2].item() - current_object_points[..., 2].mean().item())
                print(
                    f"[pso_iter] iter={iteration_idx + 1}/{int(self.pso_cfg.num_iterations)} "
                    f"iter_best_cost={float(iteration_best_cost.detach().cpu().item()):.6f} "
                    f"global_best_cost={float(global_best_cost.detach().cpu().item()):.6f} "
                    f"pred_lift={iteration_lift:.6f} "
                    f"pred_final_center={iteration_final_center.tolist()}"
                )

            improved = costs < personal_best_cost
            personal_best[improved] = particles[improved]
            personal_best_cost[improved] = costs[improved]

            r1 = torch.rand_like(particles)
            r2 = torch.rand_like(particles)
            velocities = (
                float(self.pso_cfg.inertia) * velocities
                + float(self.pso_cfg.cognitive) * r1 * (personal_best - particles)
                + float(self.pso_cfg.social) * r2 * (global_best.unsqueeze(0) - particles)
            )
            particles = self._clamp_particles(
                particles + velocities,
                position_lower=position_lower,
                position_upper=position_upper,
                fallback_quat=fallback_quat,
            )
            particles, velocities = self._focus_particles_around_global_best(
                particles=particles,
                velocities=velocities,
                costs=costs,
                global_best=global_best,
                global_best_cost=global_best_cost,
                iteration_idx=iteration_idx,
                position_lower=position_lower,
                position_upper=position_upper,
                fallback_quat=fallback_quat,
            )

        if (
            best_wrist_pose_sequence is None
            or best_hand_q_sequence is None
            or best_wm_wrist_pose_sequence is None
            or best_wm_hand_q_sequence is None
            or best_prediction is None
            or best_translation_action is None
            or best_rotation_action is None
            or best_target_hand_q is None
        ):
            raise RuntimeError("PSO did not produce a valid best particle.")

        best_action_sequence = self._build_env_action_sequence(best_wrist_pose_sequence, best_hand_q_sequence)
        predicted_final_center = best_prediction[-1, :, :3].mean(dim=0).detach().cpu()
        return {
            "action": best_action_sequence[0].detach().clone(),
            "action_sequence": best_action_sequence.detach().clone(),
            "wrist_pose_sequence": best_wrist_pose_sequence.detach().clone(),
            "hand_q_sequence": best_hand_q_sequence.detach().clone(),
            "wm_wrist_pose_sequence": best_wm_wrist_pose_sequence.detach().clone(),
            "wm_hand_q_sequence": best_wm_hand_q_sequence.detach().clone(),
            "goal_center": goal_center.detach().cpu(),
            "predicted_final_center": predicted_final_center,
            "best_cost": float(global_best_cost.detach().cpu().item()),
            "best_trajectory_particle": {
                "translation_action": best_translation_action.detach().cpu(),
                "rotation_action": best_rotation_action.detach().cpu(),
                "target_hand_q": best_target_hand_q.detach().cpu(),
            },
            "best_translation_action": best_translation_action.detach().cpu(),
            "best_rotation_action": best_rotation_action.detach().cpu(),
            "best_target_hand_q": best_target_hand_q.detach().cpu(),
        }


def _initialize_reference_plan(cfg: DictConfig, env):
    pso_cfg = _get_pso_cfg(cfg)
    plan_source = str(pso_cfg.plan_source).lower()
    if plan_source == "policy":
        _apply_checkpoint_compat_overrides(cfg, env)
        runner = _build_policy_runner(cfg, env)
        policy = runner.actor_critic
        policy.eval()
        obs = env.reset_idx(torch.arange(env.num_envs))["obs"]
        states = env.get_state()
        with torch.no_grad():
            plan_actions = policy(obs, states, inference=True)
        env.generate_reaching_plan_idx(torch.arange(env.num_envs), actions=plan_actions)
        return policy

    if plan_source == "reference":
        env.reset_idx(torch.arange(env.num_envs))
        env.generate_reaching_plan_idx(torch.arange(env.num_envs), actions=None)
        return None

    if plan_source == "random":
        env.reset_idx(torch.arange(env.num_envs))
        return None

    raise ValueError(f"Unsupported pso.plan_source={plan_source!r}. Expected policy, reference, or random.")


def run_world_model_pso(cfg: DictConfig, env) -> None:
    planner = WorldModelPSOPlanner(cfg, env)
    pso_cfg = _get_pso_cfg(cfg)
    history_buffer: Deque[HistoryStep] = deque(maxlen=int(planner.history_horizon))

    env.reset_idx(torch.arange(env.num_envs))
    _initialize_reference_plan(cfg, env)

    history_buffer.append(planner.capture_history_step())
    warmup_steps = int(
        planner.history_horizon - 1
        if pso_cfg.warmup_steps is None
        else pso_cfg.warmup_steps
    )
    plan_source = str(pso_cfg.plan_source).lower()
    if plan_source == "random":
        for warmup_step in range(max(0, warmup_steps)):
            history_buffer.append(planner.capture_history_step())
            print(f"[warmup] step={warmup_step + 1}/{warmup_steps} mode=static")
    else:
        for warmup_step in range(max(0, warmup_steps)):
            action = env.compute_reference_actions()
            _, _, _, extras = env.step(action)
            history_buffer.append(planner.capture_history_step())
            print(
                f"[warmup] step={warmup_step + 1}/{warmup_steps} "
                f"success_rate={float(extras['current_successes'].float().mean().item()):.4f}"
            )

    total_steps = int(env.max_episode_length if pso_cfg.num_steps is None else pso_cfg.num_steps)
    execution_mode = str(pso_cfg.execution_mode).lower()
    if execution_mode not in {"realtime", "trajectory", "open_loop"}:
        raise ValueError(
            f"Unsupported pso.execution_mode={execution_mode!r}. Expected realtime, trajectory, or open_loop."
        )

    step = 0
    while step < total_steps:
        plan = planner.plan_action(history_buffer)
        if bool(pso_cfg.log_final_solution):
            print(
                "[pso_final] "
                f"best_cost={plan['best_cost']:.6f} "
                f"best_translation_action={plan['best_translation_action'].tolist()} "
                f"best_rotation_action={plan['best_rotation_action'].tolist()} "
                f"best_target_hand_q={plan['best_target_hand_q'].tolist()} "
                f"predicted_final_center={plan['predicted_final_center'].tolist()}"
            )
        if execution_mode == "realtime":
            planned_action_sequence = plan["action_sequence"][:1]
        else:
            if pso_cfg.execute_steps is None:
                execute_steps = int(plan["action_sequence"].shape[0])
            else:
                execute_steps = int(pso_cfg.execute_steps)
            if execute_steps <= 0:
                raise ValueError("pso.execute_steps must be positive when using trajectory/open_loop execution.")
            if execute_steps <= int(plan["action_sequence"].shape[0]):
                planned_action_sequence = plan["action_sequence"][:execute_steps]
            else:
                dense_wrist_pose_sequence, dense_hand_q_sequence = planner._resample_plan_sequences(
                    plan["wrist_pose_sequence"],
                    plan["hand_q_sequence"],
                    execute_steps,
                )
                planned_action_sequence = planner._build_env_action_sequence(
                    dense_wrist_pose_sequence,
                    dense_hand_q_sequence,
                )

        print(
            f"[plan_exec] mode={execution_mode} env={planner.env_index} "
            f"planned_steps={int(planned_action_sequence.shape[0])}"
        )

        final_extras = None
        for local_step, planned_action in enumerate(planned_action_sequence):
            if step >= total_steps:
                break

            action = env.no_op_action.clone()
            action[planner.env_index] = planned_action.to(device=action.device, dtype=action.dtype)

            _, _, _, extras = env.step(action)
            final_extras = extras
            history_buffer.append(planner.capture_history_step())

            if bool(pso_cfg.log_execution_steps) and step % max(1, int(pso_cfg.log_interval)) == 0:
                current_object_center = history_buffer[-1].object_points[..., :3].mean(dim=0).detach().cpu()
                print(
                    f"[pso_exec] step={step} chunk_step={local_step} env={planner.env_index} "
                    f"current_center={current_object_center.tolist()} "
                    f"success={bool(extras['current_successes'][planner.env_index].item() > 0.5)}"
                )

            step += 1
            if bool(env.reset_buf[planner.env_index].item()):
                print(f"[pso] env {planner.env_index} reached episode end at step={step - 1}")
                return

        if final_extras is not None:
            final_object_center = history_buffer[-1].object_points[..., :3].mean(dim=0).detach().cpu()
            print(
                "[pso_result] "
                f"final_center={final_object_center.tolist()} "
                f"success={bool(final_extras['current_successes'][planner.env_index].item() > 0.5)}"
            )

        if execution_mode == "realtime":
            continue
        if not bool(pso_cfg.replan_after_execution):
            print("[pso] trajectory execution finished; stopping because pso.replan_after_execution=False")
            break


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    set_np_formatting()

    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    _apply_pso_env_defaults(cfg)

    def create_isaacgym_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

    env = create_isaacgym_env()
    run_world_model_pso(cfg, env)


if __name__ == "__main__":
    main()
