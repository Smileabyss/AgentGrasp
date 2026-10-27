"""Evaluate latent grasp trajectories with the point-cloud world model.

The simulator supplies the object point cloud and pose. Latent candidates are
decoded in the object frame, transformed into the robot base frame, converted
to hand contact points with PPModel, and scored by the learned world model.

Run from this repository with the same Hydra overrides as `testing.search_latent_grasp`
and add, for example, ``wm.checkpoint=/path/to/world_model/best.pt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict

# Isaac Gym must be imported before PyTorch in this environment.
from isaacgym import gymapi  # noqa: F401

import torch

from testing import search_latent_grasp as latent_search


ARGS = latent_search.ARGS
_configure_object = latent_search._configure_object
_load_object_pose = latent_search._load_object_pose
from world_model.run_world_model_pso import HistoryStep, WorldModelPSOPlanner  # noqa: E402
from world_model.test_world_model import _get_wm_cfg  # noqa: E402
from memory.train_latent_flow import _make_flow  # noqa: E402
from memory.train_trajectory_ae import _build_model  # noqa: E402
from memory.train_trajectory_vqvae import rotation6d_to_matrix  # noqa: E402



def _world_to_base_pose(world_pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
    """Transform [B,T,7] world poses into the robot base coordinate frame."""
    base_q = base_pose[:, 3:7]
    base_r = _quat_to_matrix(base_q)
    delta = world_pose[..., :3] - base_pose[:, None, :3]
    position = torch.einsum("bti,bij->btj", delta, base_r)
    inverse_q = torch.cat((-base_q[:, :3], base_q[:, 3:4]), dim=-1)
    orientation = _quat_multiply(inverse_q[:, None].expand_as(world_pose[..., 3:7]), world_pose[..., 3:7])
    return torch.cat((position, orientation), dim=-1)


def _base_to_world_pose(base_trajectory: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
    base_r = _quat_to_matrix(base_pose[:, 3:7])
    position = torch.einsum("bti,bji->btj", base_trajectory[..., :3], base_r) + base_pose[:, None, :3]
    base_q = base_pose[:, 3:7][:, None].expand_as(base_trajectory[..., 3:7])
    orientation = _quat_multiply(base_q, base_trajectory[..., 3:7])
    return torch.cat((position, orientation), dim=-1)


def _quat_to_matrix(q):
    x, y, z, w = (q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def _quat_multiply(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack((w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2,
                        w1*w2 - x1*x2 - y1*y2 - z1*z2), dim=-1)


def _candidate_hand_q(env, decoded_hand):
    source_indices = env.hand_specific_cfg.get("reference_hand_source_indices")
    if source_indices is not None and decoded_hand.shape[-1] != len(env.active_hand_dof_indices):
        decoded_hand = decoded_hand.index_select(-1, torch.as_tensor(source_indices, device=decoded_hand.device))
    return torch.maximum(torch.minimum(decoded_hand, env.robot_dof_upper_limits[env.active_hand_dof_indices]),
                         env.robot_dof_lower_limits[env.active_hand_dof_indices])


def _render_viewer(env) -> bool:
    gym = env.gym
    if env.viewer is None or gym.query_viewer_has_closed(env.viewer):
        return False
    if env.device != "cpu":
        gym.fetch_results(env.sim, True)
    gym.step_graphics(env.sim)
    gym.draw_viewer(env.viewer, env.sim, True)
    gym.sync_frame_time(env.sim)
    return not gym.query_viewer_has_closed(env.viewer)


@torch.no_grad()
def _score_population(env, planner, flow, ae, bases, latent_shape, object_pose, object_points, base_pose, history, eef_offset):
    latent = flow.inverse(bases).reshape(bases.shape[0], *latent_shape)
    decoded = ae.decode(latent)
    # search_latent_grasp's decoded wrist poses are relative to the object.
    latent_search.torch = torch
    candidate_object_pose = object_pose.expand(bases.shape[0], -1)
    world_targets = latent_search._world_wrist(candidate_object_pose, decoded, rotation6d_to_matrix)
    world_targets = latent_search._apply_eef_offset(world_targets, eef_offset)
    targets = _world_to_base_pose(world_targets, base_pose)
    hand_q = _candidate_hand_q(env, decoded[..., 9:])

    # PPModel is rooted at the robot base, while the observed points/history and
    # world-model predictions use the simulation world frame. Robot base poses
    # in this grasp task are fixed; convert candidate hand points back to world.
    base_r = _quat_to_matrix(base_pose[:, 3:7])
    object_points_on_base_device = object_points.to(base_pose.device)
    delta = object_points_on_base_device[None, :, :3] - base_pose[:, None, :3]
    object_points_base = torch.einsum("bni,bij->bnj", delta, base_r)
    contacts = planner._build_grasp_action_points(
        env.robot_dof_default_pos.unsqueeze(0).repeat(bases.shape[0], 1), object_points_base[0],
        targets, hand_q,
    )
    contacts = torch.einsum("btni,bji->btnj", contacts, base_r) + base_pose[:, None, None, :3]
    oh = torch.stack([step.object_points for step in history], dim=0).unsqueeze(0).repeat(bases.shape[0], 1, 1, 1)
    ah = torch.stack([step.action_points for step in history], dim=0).unsqueeze(0).repeat(bases.shape[0], 1, 1, 1)
    prediction = planner._predict_object_rollout(
        oh.to(planner.world_model_device), ah.to(planner.world_model_device),
        contacts.to(planner.world_model_device),
    )
    cost = planner._evaluate_costs(
        prediction, object_points.to(planner.world_model_device),
        contacts.to(planner.world_model_device),
    )
    # World-model cost is the only optimization signal; lower cost is better.
    return cost, latent, decoded, targets, prediction, hand_q


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    import isaacgymenvs
    import tasks  # noqa: F401
    from isaacgymenvs.utils.utils import set_seed

    if cfg.task.env.armController != "pose":
        raise ValueError("Latent world-model search requires task.env.armController=pose")
    label, _ = _load_object_pose()
    _configure_object(cfg, label)
    # Isaac Gym is used only to read the initial object cloud/pose here. The
    # candidate population is evaluated entirely by the world model.
    cfg.num_envs = cfg.task.env.numEnvs = 1
    ae_ckpt = torch.load(ARGS.ae_checkpoint, map_location="cpu")
    flow_ckpt = torch.load(ARGS.flow_checkpoint, map_location="cpu")
    sequence_length = int(ae_ckpt["config"]["sequence_length"])
    cfg.task.env.episodeLength = sequence_length + ARGS.post_lift_steps + ARGS.hold_steps + 1
    cfg.task.env.enablePointCloud = True
    cfg.task.env.observationType = "eefpose+objinitpose+objpcl"
    cfg.task.env.randomizeTrackingReference = False
    cfg.task.env.randomizeGraspPose = False
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.force_render = False
    dual_arm = bool(cfg.task.hand_config.get("dualArm", cfg.task.hand_config.get("dualArmInspire", False)))
    if dual_arm:
        cfg.task.env.activeArm = 1
    cfg.headless = True
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env = isaacgymenvs.make(cfg.seed, cfg.task_name, 1, cfg.sim_device, cfg.rl_device,
                            cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, False,
                            cfg.force_render, cfg)
    device = torch.device(env.device)
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=device)
    object_pose_np = latent_search._load_object_pose()[1]
    randomize_object_pose = not ARGS.no_randomize_object_pose
    if ARGS.object_position is not None or ARGS.object_quaternion is not None:
        randomize_object_pose = False
    if randomize_object_pose:
        env.reset_idx(env_ids)
        object_poses = env.object_init_states[:, :7].detach().clone()
    else:
        env.reset_idx(env_ids, object_init_pose=object_pose_np)
        object_poses = torch.as_tensor(object_pose_np, dtype=torch.float32, device=device).view(1, 7).expand(env.num_envs, -1).clone()
    object_pose = object_poses[0]
    object_points = env.transform_obj_pcl_2_world()[0].detach().cpu().float()
    robot_handle = env.gym.find_actor_handle(env.envs[0], "robot")
    base_body_index = env.gym.find_actor_rigid_body_index(
        env.envs[0], robot_handle, "pelvis", gymapi.DOMAIN_SIM
    )
    base_pose = env.rigid_body_states.view(-1, 13)[base_body_index, :7].reshape(1, 7).detach().clone()
    right_eef_offset = cfg.task.hand_config.get("trajectoryEefOffsetByArm", {}).get("right", None)
    if right_eef_offset is not None:
        right_eef_offset = torch.as_tensor(right_eef_offset, dtype=torch.float32, device=device)

    wm_cfg = _get_wm_cfg(cfg)
    if not wm_cfg.checkpoint:
        raise ValueError("Set wm.checkpoint to a point-cloud world-model checkpoint")
    # The checkpoint was trained with FR3, but the requested simulation uses
    # G1. Build PPModel from the simulated robot so arm IK and hand geometry
    # both use the same joint names and coordinate frame as the environment.
    with open_dict(cfg.wm):
        cfg.wm.urdf_path = str(
            (Path(__file__).resolve().parents[1] / "assets" / env.hand_specific_cfg["robotAssetFile"]).resolve()
        )
        cfg.wm.root_link_names = "pelvis"
    # Older world-model planner expects these optional hand bookkeeping
    # tensors even when the selected hand has no passive/mimic joints.
    if not hasattr(env, "passive_hand_dof_indices"):
        env.passive_hand_dof_indices = []
    if not hasattr(env, "mimic_parent_dof_indices"):
        env.mimic_parent_dof_indices = []
    planner = WorldModelPSOPlanner(cfg, env)
    if getattr(env, "dual_arm", False):
        # Grasp.pre_physics_step uses right_base_link and solves waist + right
        # arm together. Match that same kinematic chain when scoring contacts.
        env_to_model = {
            int(env_index): int(model_index)
            for model_index, env_index in zip(
                planner.model_mapping["model_joint_indices"].tolist(),
                planner.model_mapping["env_joint_indices"].tolist(),
            )
        }
        ik_env_indices = [int(index) for index in env.right_ik_dof_indices]
        missing = [index for index in ik_env_indices if index not in env_to_model]
        if missing:
            names = [env.robot_dof_names[index] for index in missing]
            raise RuntimeError(f"G1 PPModel is missing simulation IK joints: {names}")
        planner.eef_link_name = str(env.hand_specific_cfg.get("right_eef_link", "right_base_link"))
        planner.env_arm_indices = torch.as_tensor(ik_env_indices, dtype=torch.long, device=env.device)
        planner.pp_arm_indices = torch.as_tensor(
            [env_to_model[index] for index in ik_env_indices],
            dtype=torch.long,
            device=planner.pp_model.device,
        )
        planner.arm_lower_limits = env.robot_dof_lower_limits[planner.env_arm_indices]
        planner.arm_upper_limits = env.robot_dof_upper_limits[planner.env_arm_indices]
    current_q = env.robot_dof_pos[0].detach().clone().unsqueeze(0)
    hand_points_base = planner._compute_hand_points(current_q)[0]
    base_r = _quat_to_matrix(base_pose[:, 3:7])
    hand_points_world = torch.einsum("ni,bji->bnj", hand_points_base, base_r)[0]
    hand_points_world = hand_points_world + base_pose[0, None, :3]
    initial_contacts = planner._compute_action_points(
        hand_points_world.unsqueeze(0), object_points.to(hand_points_world.device).unsqueeze(0)
    )[0]
    step = HistoryStep(
        object_points=object_points.to(planner.pp_model.device),
        action_points=initial_contacts.to(planner.pp_model.device),
    )
    history = [step] * planner.history_horizon

    ae = _build_model(ae_ckpt["config"]).to(device)
    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()
    flow = _make_flow(flow_ckpt["config"], device)
    flow.load_state_dict(flow_ckpt["model"])
    flow.eval()
    latent_shape = tuple(flow_ckpt["config"]["latent_shape"])
    latent_dim = int(flow_ckpt["config"]["latent_dim"])
    mean = torch.zeros(latent_dim, device=device)
    std = torch.full_like(mean, ARGS.initial_std)
    elite_n = max(1, int(np.ceil(ARGS.population_size * ARGS.elite_fraction)))
    best = None
    latent_search.torch = torch
    for iteration in range(ARGS.iterations):
        bases = (mean + std * torch.randn(ARGS.population_size, latent_dim, device=device)).clamp(
            -ARGS.max_base_abs, ARGS.max_base_abs)
        bases[0] = mean
        if dual_arm:
            env.active_arm_ids.fill_(False)
        wm_cost, latent, decoded, targets, prediction, hand_q = _score_population(
            env, planner, flow, ae, bases, latent_shape, object_pose,
            object_points, base_pose, history, right_eef_offset)
        fitness = -wm_cost
        elites = bases[torch.topk(fitness, elite_n).indices]
        mean = (1 - ARGS.update_rate) * mean + ARGS.update_rate * elites.mean(0)
        std = ((1 - ARGS.update_rate) * std + ARGS.update_rate * elites.std(0, unbiased=False)).clamp(
            ARGS.min_std, ARGS.max_std)
        idx = int(fitness.argmax())
        if best is None or fitness[idx] > best["fitness"]:
            best = {"fitness": fitness[idx].item(), "base": bases[idx].clone(),
                    "latent": latent[idx].clone(), "decoded": decoded[idx].clone(),
                    "base_trajectory": targets[idx].clone(),
                    "world_targets": latent_search._apply_eef_offset(
                        latent_search._world_wrist(
                            object_pose.expand(1, -1), decoded[idx:idx + 1], rotation6d_to_matrix
                        ), right_eef_offset
                    )[0].clone(),
                    "hand_targets": decoded[idx, :, 9:].clone(),
                    "hand_q": hand_q[idx].clone(), "prediction": prediction[idx].clone(),
                    "object_pose": object_pose.clone(),
                    "wm_cost": wm_cost[idx].item()}
        predicted_lift = prediction[idx, -1, :, 2].mean() - object_points[:, 2].mean()
        print(f"iter={iteration + 1}/{ARGS.iterations} best_wm_cost={best['wm_cost']:.5f} "
              f"candidate_wm_cost={float(wm_cost[idx]):.5f} "
              f"predicted_lift={predicted_lift.item():.4f}m "
              f"std={std.mean().item():.4f}")
        del bases, fitness, latent, decoded, targets, prediction, hand_q, wm_cost
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Best world-model result: cost={best['wm_cost']:.4f}, "
          f"predicted_lift={best['prediction'][-1, :, 2].mean().item() - object_points[:, 2].mean().item():.4f} m")
    print("Best trajectory (robot-base frame):", best["base_trajectory"].detach().cpu().numpy())

    search_gym, search_sim = env.gym, env.sim
    search_gym.destroy_sim(search_sim)
    del env
    import gc
    from isaacgymenvs.tasks.base import vec_task
    gc.collect()
    vec_task.EXISTING_SIM = None
    cfg.num_envs = cfg.task.env.numEnvs = 1
    cfg.task.env.episodeLength = max(100000, sequence_length + ARGS.post_lift_steps + ARGS.hold_steps + 1)
    cfg.headless = False
    replay_env = isaacgymenvs.make(cfg.seed, cfg.task_name, 1, cfg.sim_device, cfg.rl_device,
                                   cfg.graphics_device_id, False, cfg.multi_gpu, False,
                                   cfg.force_render, cfg)
    if dual_arm:
        replay_env.active_arm_ids.fill_(False)
    latent_search._replay_best_trajectory(
        replay_env, best["decoded"], best["object_pose"], rotation6d_to_matrix,
        gymapi, right_eef_offset, torch.zeros(3, device=device),
        prepared_world_targets=best["world_targets"],
        prepared_hand_targets=best["hand_targets"],
    )


if __name__ == "__main__":
    main()
